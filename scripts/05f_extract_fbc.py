"""
stage-1 (force BCs) ODB -> <name>_results.npz. Run with Abaqus' OWN Python (only it can open an .odb):
    abaqus python 05f_extract_fbc.py <job>.odb <out>.npz
Label-agnostic: every set is read BY NAME from the ODB (WALL, ANC_B, DFRAME, DSKIRT, DSKIN, DPLRP, DRP), so prototypes and repo runs work alike.
ALL steps on ONE total-time axis (05 kept only the last step). Saved arrays:
  times, step (frame -> step index), labels, X0 (reference coordinates), U (frames x nodes x 3, float32)
  sets: wall_nodes, anc_nodes, frame_nodes / skirt_nodes / skin_nodes, plate_rp, drp, frame_el + frame_conn, skirt_el + skirt_conn, skin_el + skin_conn
  RF of the plate reference nodes and the DRP (frames x n x 3): crimp force (radial sum), delivery reaction
  contact (every frame): cpress_labels + cpress, copen_labels + copen (sentinels dropped), cnormf_labels + cnormf (frames x n x 3, if CFORCE written)
  WALL element fields (frames with element output): elem_times, elem_labels, elem_ESEDEN, elem_EVOL, elem_S, elem_LE, elem_ELASE, elem_EMSF
  FRAME / SKIRT element fields: dev_times, frame_LEmax (max principal log strain), frame_Smises, frame_ELASE, frame_EMSF, skirt_LEmax, skirt_Smises, skirt_EMSF
  histories (total time): hist_<SET>_<VAR> energies per set (ASM = whole model), hist_n<label>_<VAR> node histories (DRP U, plate U/RF),
  hist_<VAR>_<SURFACE> whole-surface contact force/area (CFN1-3, CFNM, CAREA)
"""
import sys
import numpy as np
from abaqusConstants import MAX_PRINCIPAL, MISES
from odbAccess import openOdb

SENT=1e5 # |COPEN| beyond this value abaqus sentinel for nodes without a contact partner
odb_path,out_path=sys.argv[1],sys.argv[2]
o=openOdb(odb_path, readOnly=True);inst=o.rootAssembly.instances.values()[0]; iname=inst.name
steps=list(o.steps.values()); res={}

def nodes_of_elset(nm):
    s=set()
    for e in inst.elementSets[nm].elements:
        s.update(int(n) for n in e.connectivity)
    return np.array(sorted(s),dtype=np.int64)

def elset(nm):
    els=inst.elementSets[nm].elements
    return np.array([e.label for e in els], dtype=np.int64), np.array([list(e.connectivity) for e in els], dtype=np.int64)

def blocks(fo, nodal=True):
    """bulk access (much faster than looping over values): labels, data rows of this instance."""
    L, D = [], []
    for b in fo.bulkDataBlocks:
        if b.instance is not None and b.instance.name != iname:
            continue
        L.append(np.array(b.nodeLabels if nodal else b.elementLabels)); D.append(np.array(b.data).reshape(len(L[-1]), -1))
    if not L:
        return np.zeros(0, np.int64), np.zeros((0, 1))
    return np.concatenate(L), np.vstack(D)

def per_element(lab,dat):
    """average integration-point rows per element (C3D10M: 4 points; C3D8R7S4R: 1 point per section point row)"""
    u,inv=np.unique(lab,return_inverse=True); s=np.zeros((len(u), dat.shape[1])); np.add.at(s,inv,dat)
    return u, (s/np.bincount(inv)[:,None]).astype(np.float32)

labels=np.array([n.label for n in inst.nodes], dtype=np.int64);order=np.argsort(labels);labels=labels[order]
res["labels"] = labels; res["X0"] = np.array([n.coordinates for n in inst.nodes])[order]
idx = dict((int(l), i) for i, l in enumerate(labels))
res["wall_nodes"]=nodes_of_elset("WALL"); res["anc_nodes"]=nodes_of_elset("ANC_B")
for key, nm in (("frame","DFRAME"),("skirt","DSKIRT"),("skin","DSKIN")):
    res[key + "_el"], res[key + "_conn"] = elset(nm); res[key + "_nodes"] = np.unique(res[key + "_conn"])
res["plate_rp"]=np.array(sorted(int(n.label) for n in inst.nodeSets["DPLRP"].nodes),dtype=np.int64)
res["drp"] = np.array([int(n.label) for n in inst.nodeSets["DRP"].nodes], dtype=np.int64)
rp_rows = np.array([idx[int(l)] for l in np.concatenate([res["plate_rp"], res["drp"]])])

# frames on total time axis
frames, t0=[],0.0
for si, st in enumerate(steps):
    for fr in st.frames:
        if si>0 and fr.frameValue==0.0:
            continue  # the first frame of a later step repeats the last one
        frames.append((si, t0 + fr.frameValue, fr))
    t0+=st.timePeriod
res["times"]=np.array([t for _, t, _ in frames]); res["step"] = np.array([s for s, _, _ in frames])
U=np.zeros((len(frames),len(labels),3),np.float32); RF=np.zeros((len(frames),len(rp_rows),3),np.float32)
CP,CO,CN=[],[],[]
E=dict((k,[]) for k in ("ESEDEN","EVOL","S","LE","ELASE","EMSF")); Et,Dt=[],[]
D=dict((k,[]) for k in ("frame_LEmax","frame_Smises","frame_ELASE","frame_EMSF","skirt_LEmax","skirt_Smises","skirt_EMSF"))
wall_set,frame_set,skirt_set=inst.elementSets["WALL"],inst.elementSets["DFRAME"],inst.elementSets["DSKIRT"]
for k, (si, t, fr) in enumerate(frames):
    fo = fr.fieldOutputs
    l, d = blocks(fo["U"]); U[k, [idx[int(x)] for x in l]] = d[:, :3]
    if "RF" in fo.keys():
        l, d = blocks(fo["RF"]); m = dict((int(a), i) for i, a in enumerate(l))
        RF[k] = np.array([d[m[int(labels[r])], :3] if int(labels[r]) in m else np.zeros(3) for r in rp_rows])
    for store, prefix, vec in ((CP, "CPRESS", False), (CO, "COPEN", False), (CN, "CNORMF", True)):
        keys = [x for x in fo.keys() if x.startswith(prefix)]
        if keys:
            l, d = blocks(fo[keys[0]]); d = d[:, :3] if vec else d[:, 0]
            ok = np.all(np.abs(d) < SENT, axis=1) if vec else np.abs(d) < SENT
            store.append((k, l[ok], d[ok]))
    if "ESEDEN" in fo.keys():                                                # wall element output frames
        Et.append(t)
        for key in E:
            if key in fo.keys():
                lab, v = per_element(*blocks(fo[key].getSubset(region=wall_set), nodal=False)); E[key].append(v); res["elem_labels"] = lab
    if "LE" in fo.keys() and len(fo["LE"].getSubset(region=frame_set).values):   # device element output frames
        Dt.append(t)
        for key, var, setx, inv in (("frame_LEmax", "LE", frame_set, MAX_PRINCIPAL), ("frame_Smises", "S", frame_set, MISES),
                                     ("skirt_LEmax", "LE", skirt_set, MAX_PRINCIPAL), ("skirt_Smises", "S", skirt_set, MISES)):
            lab, v = per_element(*blocks(fo[var].getSubset(region=setx).getScalarField(invariant=inv), nodal=False)); D[key].append(v[:, 0])
            res[key.split("_")[0] + "_field_el"] = lab
        for key, var, setx in (("frame_ELASE", "ELASE", frame_set), ("frame_EMSF", "EMSF", frame_set), ("skirt_EMSF", "EMSF", skirt_set)):
            if var in fo.keys():
                lab, v = per_element(*blocks(fo[var].getSubset(region=setx), nodal=False)); D[key].append(v[:, 0])
res["U"] = U; res["RF_rp"]=RF
for store, name, vec in ((CP, "cpress", False), (CO, "copen", False), (CN, "cnormf", True)):
    if store:                                                                # one row per frame, one column per node ever reported (0 = none)
        allL = np.unique(np.concatenate([l for _, l, _ in store])); M = np.zeros((len(frames), len(allL)) + ((3,) if vec else ()), np.float32)
        for k, l, d in store:
            M[k, np.searchsorted(allL, l)] = d
        res[name + "_labels"], res[name] = allL, M
if Et:
    res["elem_times"] = np.array(Et)
    for key, v in E.items():
        if v:
            res["elem_" + key] = np.stack(v)
if Dt:
    res["dev_times"] = np.array(Dt)
    for key, v in D.items():
        if v:
            res[key] = np.stack(v)
print("fields: %d frames (%d wall element frames, %d device element frames), %d nodes" % (len(frames), len(Et), len(Dt), len(labels)))
# histories on total time axis

H,t0={},0.0
for st in steps:
    for reg in st.historyRegions.values():
        nm=reg.name
        if nm.startswith("Node "):
            tag="n" + nm.split(".")[-1]
        else:
            tag=nm.replace("ElementSet ","").replace("Assembly Assembly-1","ASM").replace(iname+".","").strip()
        for key,ho in reg.historyOutputs.items():
            d=np.array(ho.data)
            if d.ndim!=2 or not len(d):
                continue
            if " on surface " in key:     # whole-surface contact output: 'CFN1 on surface DSKIN_S'
                var,surf=key.split(" on surface "); name = "hist_%s_%s" % (var.strip(), surf.strip().split(".")[-1])
            else:
                name="hist_%s_%s" % (tag if tag != "PIBATCH" else "ASM", key)
            H.setdefault(name,[]).append(np.column_stack([d[:, 0] + t0, d[:, 1]]))
    t0+=st.timePeriod
for k, v in H.items():
    res[k]=np.vstack(v)
print("histories: %d (%s ...)" % (len(H), ", ".join(sorted(H)[:6])))
np.savez(out_path, **res); print("saved", out_path)
o.close()


