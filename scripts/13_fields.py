"""
Stress, strain, strain energy density and contact pressure of ONE run: a ParaView time series of the deformed FEA wall and the statistics the UQ compares.

From the run's won Abaqus input (exact C3D10M mesh) and results_npz: the frames carry the wall element output (every cfg.ELEM_OUT_INTERVAL-th frame, the first one = undeformed).
Cell data per tet: SED_kPa (ESEDEN), VM_kPa (Von Mises)  S_maxP_kPa / S_minP_kPa (max / min principal Cauchy stress),
                    LE_maxP_pct (max principal logarithmic strain, %), J (EVOL / EVOL at t = 0, exact), zone (1 = within ZONE_MM of the annulus)

Point data: displacement (mm), CPRESS_kPa (contact pressure; every frame if series was stored)
Statistics:  volume-weighted mean, p50, p95, p99, max + location of the max (angle about the cuff axis, height above the annulus plane); contact pressure peak / p95 / mean over the wall loaded nodes;
elements above the aortic rupture flag (taken as a conservative rule as no validation data is present,max principal stress 2.5 MPa)
Units: Abaqus mm-t-s -> stresses in MPa, reported in kPa. LE shear components are engineering strains (halved for the tensor).

Usage:  python 13_fields.py --id SYN_0001 --level L3 --tag cen_hard_ramp0.2_E0.01  [--no_series]
Writes  deploy_<level>_<tag>/post/<name>_fields.json  and  post/paraview/fields_####.vtu + fields.pvd
"""
import argparse
import importlib.util
import json
import numpy as np
from types import SimpleNamespace
import vtk
from scipy.spatial import cKDTree
from vtkmodules.util.numpy_support import numpy_to_vtk
import fea_config as cfg

KEYS=("SED_kPa","VM_kPa","S_maxP_kPa","S_minP_kPa","LE_maxP_pct","J")
RUPTURE_FLAG_KPA=2500.0
EXP_NODE0=2_000_000  # 03: expander node ids start above this (wall nodes are 1..N)

def load(name):
    """Import a pipeline script whose file name starts with a digit."""
    spec = importlib.util.spec_from_file_location(name, cfg.SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def read_wall(inp):
    """Wall nodes (*NODE, NSET=WALL_N) and C3D10M connectivity (*ELEMENT, ELSET=WALL) of the input file, 0-based"""
    nodes,elems, mode=[],[],None
    with open(inp) as fh:
        for line in fh:
            if line.startswith("*"):
                key=line.upper().replace(" ","")
                mode="n" if key.startswith("*NODE,NSET=WALL_N") else ("e" if key.startswith("*ELEMENT,TYPE=C3D10M") else None)
                continue
            if mode=="n":
                nodes.append([float(v) for v in line.split(",")[1:4]])
            elif mode=="e":
                elems.append([int(v) for v in line.split(",")[1:11]])
    return np.array(nodes), np.array(elems)-1

def tensor(v, shear_half):
    """Abaqus 3D order 11, 22, 33, 12, 13, 23 -> symmetric 3x3 tensors (shear halved for engineering strains)."""
    f = 0.5 if shear_half else 1.0
    t = np.zeros((len(v), 3, 3))
    t[:, 0, 0], t[:, 1, 1], t[:, 2, 2] = v[:, 0], v[:, 1], v[:, 2]
    t[:, 0, 1] = t[:, 1, 0] = f * v[:, 3]; t[:, 0, 2] = t[:, 2, 0] = f * v[:, 4]; t[:, 1, 2] = t[:, 2, 1] = f * v[:, 5]
    return t

def fields(res,k):
    """Cell fields of element frame k"""
    s=res["elem_S"][k]; ps=np.linalg.eigvalsh(tensor(s,False))   # ascending principal stresses (MPa)
    vm=np.sqrt(0.5*((s[:,0]-s[:,1])**2 + (s[:,1]-s[:,2])**2 + (s[:,2]-s[:,0])**2)+3*(s[:,3]**2+s[:,4]**2+s[:,5]**2))
    pe=np.linalg.eigvalsh(tensor(res["elem_LE"][k], True))
    return dict(SED_kPa=1e3*res["elem_ESEDEN"][k][:,0], VM_kPa=1e3*vm,S_maxP_kPa=1e3*ps[:,2],S_minP_kPa=1e3*ps[:,0],
                LE_maxP_pct=100*pe[:,2],J=res["elem_EVOL"][k][:,0]/res["elem_EVOL"][0][:,0])

def wstats(x,w):
    """Volume-weighted mean, percentiles and max statistics"""
    o=np.argsort(x);cw=np.cumsum(w[o])/w.sum()
    pct={f"p{q}": float(x[o][min(np.searchsorted(cw,q/100),len(x)-1)]) for q in (50,95,99)}
    return dict(mean=float(np.sum(x*w)/w.sum()), **pct, max=float(x.max()), min=float(x.min()))

def write_vtu(path,P,T10,cell,point):
    g=vtk.vtkUnstructuredGrid(); pts=vtk.vtkPoints(); pts.SetData(numpy_to_vtk(P,deep=True));g.SetPoints(pts)
    ca=vtk.vtkCellArray()

    for c in T10:
        ca.InsertNextCell(10,[int(i) for i in c])       # Abaqus C3D10 node order = VTK_QUADRATIC_TETRA order
    g.SetCells(vtk.VTK_QUADRATIC_TETRA,ca)
    for name, a in cell.items():
        arr=numpy_to_vtk(np.ascontiguousarray(a,dtype=np.float32),deep=True);arr.SetName(name); g.GetCellData().AddArray(arr)
    for name, a in point.items():
        arr=numpy_to_vtk(np.ascontiguousarray(a,dtype=np.float32),deep=True);arr.SetName(name);g.GetPointData().AddArray(arr)
    w=vtk.vtkXMLUnstructuredGridWriter(); w.SetFileName(str(path)); w.SetInputData(g); w.SetDataModeToBinary(); w.Write()

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id",default="SYN_0001"); ap.add_argument("--level", default="L3"); ap.add_argument("--tag", required=True)
    ap.add_argument("--no_series",action="store_true",help="statistics only (no ParaView series)")
    a=ap.parse_args(); run=cfg.RUNS/a.id; name=f"{a.id}_{a.level}_{a.tag}"; dep= run/f"deploy_{a.level}_{a.tag}"
    res=np.load(dep/f"{name}_results.npz"); info=json.loads((dep/f"{name}_info.json").read_text())
    assert "elem_S" in res.files, "No wall element output in this run: re-run it"
    X10,T10=read_wall(dep/f"{name}.inp"); n=len(X10)
    assert len(T10)==len(res["elem_labels"]) and np.array_equal(res["elem_labels"],np.arange(1,len(T10)+1)) , "element labels do not match the input"
    idx=np.searchsorted(res["labels"], np.arange(1,n+1))  # wall node k -> row in U
    # device zone + cuff-axis frame (same definitions as 06/10)
    qc=load("04_mesh_qc"); s10=load("10_compare_runs"); fc=load("fea_common")
    Xc,Tc,pd,mp=qc.read_mesh(run, a.id,a.level)
    ring=Xc[pd["tv_annulus"]==1]
    cen=X10[T10[:,:4]].mean(1); zone=cKDTree(ring).query(cen)[0]<cfg.ZONE_MM
    plan,_,_=s10.common(SimpleNamespace(id=a.id)); fr=fc.axis_plan(plan,info.get("axis_offset_e1e2_mm"))["frame"]
    c,nrm,e1,e2=(np.array(fr[k]) for k in ("centre","normal","e1","e2"))
    vol=res["elem_EVOL"][-1][:,0]   # current element volumes (weights)
    # contact pressure: series or last frame
    if "cpress" in res.files:
        cl,CP=res["cpress_labels"], res["cpress"]
    else:
        cl,CP=res["cpress_last_labels"], np.zeros((len(res["times"]), len(res["cpress_last_labels"])),np.float32); CP[-1]=res["cpress_last"]
    wall_c=cl<EXP_NODE0  # wall nodes only (the expander also reports CPRESS)

    def cpress_nodes(i):
        p=np.zeros(n,np.float32); p[cl[wall_c]-1]=CP[i,wall_c]; return 1e3*p

    ef=list(range(len(res["elem_times"])))
    uf=[int(np.argmin(np.abs(res["times"]-t))) for t in res["elem_times"]]  # U / contact frame of every element frame
    out={"id":a.id, "level":a.level, "tag": a.tag, "material":info.get("material"), "zone_mm":cfg.ZONE_MM, "zone_tets": int(zone.sum()),
    "units": "kPa, %, -", "rupture_flag_kPa": RUPTURE_FLAG_KPA, "cpress_series": "cpress" in res.files, "history": {}, "final": {}}
    if not a.no_series:
        pv_dir=dep/"post"/"paraview"; pv_dir.mkdir(parents=True, exist_ok=True); col=[]
    for k, i in zip(ef,uf):
        f=fields(res,k); f["zone"]=zone.astype(np.float32)
        p=cpress_nodes(i)
        out["history"][f"{res['elem_times'][k]:.4f}"] = {kk: float(np.percentile(f[kk][zone], 99)) for kk in KEYS if kk != "J"} | {"CPRESS_peak_kPa": float(p.max())}
        if not a.no_series:
            write_vtu(pv_dir / f"fields_{k:04d}.vtu", X10 + res["U"][i].astype(float)[idx], T10, f,
                      {"displacement": res["U"][i].astype(float)[idx], "CPRESS_kPa": p})
            col.append(f'<DataSet timestep="{res["elem_times"][k]:.6g}" file="fields_{k:04d}.vtu"/>')
    f=fields(res,ef[-1]); p=cpress_nodes(uf[-1]); Xd=X10 + res["U"][uf[-1]].astype(float)[idx]
    for region, m in (("zone", zone),("wall",np.ones(len(T10), bool))):
        out["final"][region]={kk: wstats(f[kk][m], vol[m]) for kk in KEYS}
    loc={}
    for kk in ("SED_kPa","VM_kPa","S_maxP_kPa","LE_maxP_pct"):
        e=int(np.argmax(f[kk])); v=Xd[T10[e,:4]].mean(0)-c
        loc[kk]=dict(element=e+1,theta_deg=float(np.degrees(np.arctan2(v @ e2, v @ e1))%360),z_mm=float(v@nrm),in_zone=bool(zone[e]))
    out["final"]["max_location"]=loc
    loaded=p[p>1e-6]
    out["final"]["CPRESS"]=dict(peak_kPa=float(p.max()), p95_kPa=float(np.percentile(loaded, 95)) if loaded.size else 0.0,
                          mean_loaded_kPa=float(loaded.mean()) if loaded.size else 0.0, loaded_nodes=int(loaded.size))
    out["final"]["above_rupture_flag_tets"]=int((f["S_maxP_kPa"]>RUPTURE_FLAG_KPA).sum())
    (dep / "post" / f"{name}_fields.json").write_text(json.dumps(out, indent=1))
    if not a.no_series:
        (pv_dir / "fields.pvd").write_text('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n'
                                           + "\n".join(col) + "\n</Collection>\n</VTKFile>\n")
    z = out["final"]["zone"]
    print(f"{name} (last frame, device zone {cfg.ZONE_MM:g} mm, {int(zone.sum()):,} tets; volume-weighted):")
    for kk in KEYS:
        print(f"  {kk:12s} mean {z[kk]['mean']:9.3f}  p95 {z[kk]['p95']:9.3f}  p99 {z[kk]['p99']:9.3f}  max {z[kk]['max']:9.3f}  min {z[kk]['min']:9.3f}")
    cp = out["final"]["CPRESS"]
    print(f"  contact pressure: peak {cp['peak_kPa']:.1f} kPa, p95 {cp['p95_kPa']:.1f}, mean {cp['mean_loaded_kPa']:.1f} over {cp['loaded_nodes']} loaded wall nodes"
          + ("" if out["cpress_series"] else "  (last frame only: rerun 05 for the series)"))
    print(f"  tets above the {RUPTURE_FLAG_KPA:g} kPa max-principal flag: {out['final']['above_rupture_flag_tets']}; "
          + "; ".join(f"max {kk} at {v['theta_deg']:.0f} deg, z {v['z_mm']:+.1f} mm" for kk, v in loc.items() if kk in ("S_maxP_kPa", "LE_maxP_pct")))
    print(f"wrote {dep / 'post' / (name + '_fields.json')}" + ("" if a.no_series else f" and {pv_dir / 'fields.pvd'} ({len(ef)} frames)"))


if __name__=="__main__":
    main()





