"""
Pulls the results out of an Abaqus result file (.odb) into a plain .npz that numpy can handle cleanly.

Run with Abaqus' OWN Python (the .odb can only be opened by the Abaqus libraries):
    abaqus python 05_extract_odb.py <job>.odb <out>.npz
Saves: times (frames), labels (node ids, sorted), U (frames x nodes x 3 displacement), RF (reaction force, if written),
       hist_<name>: energy histories (model level) and hist_wall_<name> (the WALL element set alone). 
"""

import sys
import numpy as np
from odbAccess import openOdb

odb_path,out_path=sys.argv[1],sys.argv[2]
odb=openOdb(odb_path, readOnly=True)
step=odb.steps[list(odb.steps.keys())[-1]]
times, labels,U,RF=[],None,[],[]

SENT=1e30 
ELEM_KEYS=("ESEDEN","EVOL","S","LE")

def element_field(frame,key):
    """WALL element output-> element labels, one row per element. Integration point fields (S,LE 4 points per C2D10M) are averaged per element; whole-element fields (ESEDEN, EVOL) already have one value
    """
    f=frame.fieldOutputs[key]
    lab=np.concatenate([b.elementLabels for b in f.bulkDataBlocks])
    dat=np.concatenate([np.asarray(b.data).reshape(len(b.elementLabels),-1) for b in f.bulkDataBlocks])
    u,inv=np.unique(lab,return_inverse=True)
    s=np.zeros((len(u), dat.shape[1])); np.add.at(s,inv,dat)
    return u, (s/np.bincount(inv)[:,None]).astype(np.float32)



def contact_field(frame,prefix):
    """(node labels, values) of a general contact output (CPRESS/COPEN) at the nodes Abaqus tracks"""
    keys=[k for k in frame.fieldOutputs.keys() if k.startswith(prefix)]
    if not keys:
        return None,None
    f=frame.fieldOutputs[keys[0]]
    lab=np.concatenate([b.nodeLabels for b in f.bulkDataBlocks]); dat=np.concatenate([b.data for b in f.bulkDataBlocks]).ravel()
    ok=np.abs(dat)<SENT
    return lab[ok],dat[ok]

cp_ever=set() # nodes that carried contact pressure in ANY frame (detachment = ever in contact but not at the end)

for frame in step.frames:
    if "U" not in frame.fieldOutputs:
        continue
    u=frame.fieldOutputs["U"]
    blocks=u.bulkDataBlocks # bulk access: the whole field as arrays, much faster than looping over values
    lab=np.concatenate([b.nodeLabels for b in blocks])
    dat=np.concatenate([b.data for b in blocks])
    order=np.argsort(lab)  # same node order in every frame
    if labels is None:
        labels=lab[order]
    U.append(dat[order].astype(np.float32)); times.append(frame.frameValue)
    if "RF" in frame.fieldOutputs:
        r=frame.fieldOutputs["RF"]
        rl=np.concatenate([b.nodeLabels for b in r.bulkDataBlocks]); rd = np.concatenate([b.data for b in r.bulkDataBlocks])
        RF.append(rd[np.argsort(rl)].astype(np.float32))
        cl,cv=contact_field(frame,"CPRESS")
        if cl is not None:
            cp_ever.update(cl[cv>1e-6].tolist()); cp_last=(cl,cv)


hist={}

for region_name, region in step.historyRegions.items():
    tag="wall_" if "WALL" in region_name.upper() else ("anc_" if "ANC_B" in region_name.upper() else "")  # the energies of the WALL element set get a prefix, model-level ones keep the plain name
    for key, out in region.historyOutputs.items():
        if key in ("ALLKE", "ALLIE", "ALLAE", "ALLVD", "ALLWK", "ETOTAL"):
            hist[tag+key]=np.array(out.data,dtype=float)

E_t,E_lab,E_dat=[],None,dict((k,[]) for k in ELEM_KEYS)
for frame in step.frames:    # frames that carry the wall element output (every ELEM_OUT_INTERVAL)
    if "ESEDEN" not in frame.fieldOutputs:
        continue
    E_t.append(frame.frameValue)
    for k in ELEM_KEYS:
        if k in frame.fieldOutputs:
            E_lab,v=element_field(frame,k); E_dat[k].append(v)

result={"times":np.array(times),"labels":labels,"U":np.stack(U)}

if E_t:
    result["elem_times"], result["elem_labels"]=np.array(E_t),E_lab #element label k 
    for k in ELEM_KEYS:
        if E_dat[k]:
            result["elem_"+k]=np.stack(E_dat[k])

if cp_ever or "cp_last" in globals():
    result["cpress_ever_labels"]=np.array(sorted(cp_ever),dtype=np.int64)
    result["cpress_last_labels"],result["cpress_last"]=cp_last[0],cp_last[1].astype(np.float32)
    ol,ov=contact_field(step.frames[-1],"COPEN")
    if ol is not None:
        result["copen_last_labels"],result["copen_last"]=ol,ov.astype(np.float32)

if RF:
    result["RF"] = np.stack(RF)
for k, v in hist.items():
    result["hist_" + k] = v
np.savez(out_path, **result)
print("frames %d, nodes %d, histories %s" % (len(times), len(labels), sorted(hist)))
odb.close()


