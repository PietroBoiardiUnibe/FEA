"""
The deployed thick wall for the computational mesh domain (IBAMR or other CFD/FSI solvers); the FEA displacement of the coarse
FEA wall is carried onto every node of the official 07a wall (cfg.WALLS/<id>_wall.vtu: inner surface = all SSM vertices, 4 mm grown outward)
with IDENTICAL connectivity.

Mapping   each fine node -> the coarse FEA tet that contains it (barycentric weights; nodes just outside every tet, e.g. on the decimated inner
          surface, are clamped onto the nearest tet) -> LINEAR (corner) interpolation of the FEA displacement. The weights are stored, so any frame
          (time series) can be mapped again without searching.
Checks    inverted fine tets (must be 0), J = V_after / V_before (p1, min, count < 0.2 and where), wall thickness (inner node -> outer surface)
          before/after by distance from the TV ring. Inverted tets -> FLAG
Series    BY DEFAULT (--no_series to skip): the thick wall at every FEA frame (--every k to thin out) as a ParaView time series next to 06's
heart / expander series: post/paraview/wall_####.vtu + wall.pvd (point data: displacement, displacement_mm, surface label).

Usage     python 06b_wall_deployed.py --id SYN_0001 --level L3 --tag cen_bonded [--every 1] [--no_series]
Writes    deploy_<level>_<tag>/post/<name>_wall_deployed.vtu (all labels + displacement, displacement_mm, thickness_ratio; cell data J),
          <name>_wall_map.npz (tet index + weights), <name>_wall_checks.json, paraview/wall_####.vtu + wall.pvd
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import vtk
from scipy.spatial import cKDTree 
from vtkmodules.util.numpy_support import numpy_to_vtk
import fea_config as cfg

J_LOW=0.2 # tets compressed below 20% of their volume are reported with their location to evaluate the issue

def load_module(name):
    spec = importlib.util.spec_from_file_location(name.replace("0", "w0"), Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def vol(X,T):
    return np.einsum("ij,ij->i",X[T[:,1]] - X[T[:,0]], np.cross(X[T[:,2]]-X[T[:,0]], X[T[:,3]]-X[T[:,0]]))/6

def locate(Xf,Xc,Tc,k=24):
    """Containing coarse tet + clamped barycentric weights of every fine node."""
    A=Xc[Tc]; Minv=np.linalg.inv(np.stack([A[:,1]-A[:,0], A[:,2]-A[:,0],A[:,3]-A[:,0]],axis=2))
    _, cand = cKDTree(A.mean(1)).query(Xf, k=k)
    bt = np.zeros(len(Xf), int); bL = np.zeros((len(Xf), 4)); bs = np.full(len(Xf), -np.inf)
    for j in range(cand.shape[1]):
        t = cand[:, j]; l = np.einsum("nij,nj->ni", Minv[t], Xf - A[t, 0]); L = np.column_stack([1 - l.sum(1), l]); s = L.min(1)
        b = s > bs; bt[b], bL[b], bs[b] = t[b], L[b], s[b]
    L = np.clip(bL, 0, None); L /= L.sum(1, keepdims=True)
    return bt, L, bs

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id",default="SYN_0001"); ap.add_argument("--level",default="L3"); ap.add_argument("--tag", default="")
    ap.add_argument("--every",type=int,default=1,help="ParaView series: every k-th FEA frame (the last one always); 1 = the same frames as 06")
    ap.add_argument("--no_series", action="store_true",help="skip the ParaView series of the thick wall")
    a=ap.parse_args()

    w03=load_module("03_write_inp") #read VTU orient, boundary faces
    run=cfg.RUNS/a.id; sfx=f"_{a.tag}" if a.tag else ""; name=f"{a.id}_{a.level}{sfx}"; dep=run/f"deploy_{a.level}{sfx}"
    post=dep/"post"; post.mkdir(exist_ok=True)
    res = np.load(dep / f"{name}_results.npz"); labels = res["labels"]
    Xc, Tc, _ = w03.read_vtu(run / "mesh" / f"{a.id}_wall_{a.level}.vtu"); Tc, _ = w03.orient(Xc, Tc)
    Xf,Tf,pdf=w03.read_vtu(cfg.WALLS/ f"{a.id}_wall.vtu"); Tf,_=w03.orient(Xf,Tf)
    print(f"{name}: official wall {len(Xf):,} nodes / {len(Tf):,} tets <- FEA wall {len(Xc):,} corner nodes / {len(Tc):,} tets")

    t0=time.time(); bt,L,bs=locate(Xf,Xc,Tc)
    np.savez(post / f"{name}_wall_map.npz", tet=bt, weights=L)
    Uc = res["U"][-1].astype(float)[np.searchsorted(labels, np.arange(1, len(Xc) + 1))]   # corner nodes = the first len(Xc) FEA nodes
    U = (L[:, :, None] * Uc[Tc[bt]]).sum(1)
    out = dict(id=a.id, level=a.level, tag=a.tag, nodes=len(Xf), tets=len(Tf), outside_clamped=int((bs < -1e-9).sum()), worst_overshoot=float(bs.min()))
    print(f"  mapping: {len(Xf) - out['outside_clamped']:,} nodes inside a FEA tet, {out['outside_clamped']:,} clamped (worst overshoot {bs.min():.3f}); "
          f"{time.time() - t0:.0f} s; max displacement {np.linalg.norm(U, axis=1).max():.2f} mm")

    J=vol(Xf+U, Tf)/vol(Xf,Tf)
    out.update(inverted=int((J<=0).sum()), J_min=float(J.min()), J_p1=float(np.percentile(J,1)), J_median=float(np.median(J)),
    J_max=float(J.max()), n_J_low=int((J<J_LOW).sum()))
    if out["n_J_low"]:                                                    # where (annulus frame: angle, axial height)
        plan = json.loads((cfg.PLANS / f"{a.id}_plan.json").read_text()); fr = plan["frame"]
        c, n, e1, e2 = (np.array(fr[k]) for k in ("centre", "normal", "e1", "e2"))
        cen = (Xf + U)[Tf[J < J_LOW]].mean(1) - c
        out["J_low_theta_deg"] = np.round(np.degrees(np.arctan2(cen @ e2, cen @ e1)) % 360).astype(int).tolist()
        out["J_low_axial_mm"] = np.round(cen @ n, 1).tolist()
    print(f"  validity: inverted tets {out['inverted']}; J min {out['J_min']:.2f}, p1 {out['J_p1']:.2f}, median {out['J_median']:.3f}, max {out['J_max']:.2f}; "
          f"J < {J_LOW}: {out['n_J_low']}" + (f" at theta {min(out['J_low_theta_deg'])}-{max(out['J_low_theta_deg'])} deg, axial "
                                              f"{min(out['J_low_axial_mm'])}..{max(out['J_low_axial_mm'])} mm" if out["n_J_low"] else ""))
    
    inner=np.flatnonzero(pdf["surface"]==1)
    el,loc=w03.boundary_faces(Tf,pdf["surface"]==2)
    tri=np.array([[Tf[e,k] for k in w03.FACES[f-1]] for e,f in zip(el,loc)])
    def thick(X):
        poly=vtk.vtkPolyData();pts=vtk.vtkPoints();pts.SetData(numpy_to_vtk(np.ascontiguousarray(X), deep=True)); poly.SetPoints(pts)
        ca=vtk.vtkCellArray()
        for q in tri:
            ca.InsertNextCell(3,[int(v) for v in q])
        poly.SetPolys(ca); d=vtk.vtkImplicitPolyDataDistance(); d.SetInput(poly)
        return np.abs(np.array([d.EvaluateFunction(p) for p in X[inner]]))
    th0, th1 = thick(Xf), thick(Xf + U); ratio = th1 / th0
    dring = cKDTree(Xf[inner][pdf["tv_annulus"][inner] == 1]).query(Xf[inner])[0]
    out["thickness_mm"] = dict(before_p1=float(np.percentile(th0, 1)), before_median=float(np.median(th0)), after_p1=float(np.percentile(th1, 1)),
                               after_median=float(np.median(th1)), ratio_min=float(ratio.min()))
    out["thickness_by_ring_distance"] = []
    for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 45), (45, 400)):
        m = (dring >= lo) & (dring < hi)
        out["thickness_by_ring_distance"].append(dict(band_mm=[lo, hi], before=float(np.median(th0[m])), after=float(np.median(th1[m])),
                                                      ratio_min=float(ratio[m].min()), ratio_max=float(ratio[m].max())))
    print(f"  thickness: median {out['thickness_mm']['before_median']:.2f} -> {out['thickness_mm']['after_median']:.2f} mm, p1 "
          f"{out['thickness_mm']['before_p1']:.2f} -> {out['thickness_mm']['after_p1']:.2f} mm, smallest ratio {ratio.min():.2f} "
          f"(by distance from the ring: " + ", ".join(f"{r['band_mm'][0]}-{r['band_mm'][1]} mm {r['ratio_min']:.2f}" for r in out["thickness_by_ring_distance"]) + ")")
    out["verdict"] = "PASS" if out["inverted"] == 0 else "FLAG: inverted tets -> IB consistency correction / fallback re-grow"
    print("  verdict:", out["verdict"])

    g=vtk.vtkUnstructuredGrid(); pts=vtk.vtkPoints(); pts.SetData(numpy_to_vtk(np.ascontiguousarray(Xf+U), deep=True)); g.SetPoints(pts)
    ca=vtk.vtkCellArray()
    for t in Tf:
        ca.InsertNextCell(4,[int(v) for v in t])
    g.SetCells(vtk.VTK_TETRA,ca)
    tr = np.full(len(Xf), np.nan); tr[inner] = ratio
    for nm, arr, dst in [("displacement", U, g.GetPointData()), ("displacement_mm", np.linalg.norm(U, axis=1), g.GetPointData()),
                         ("thickness_ratio", tr, g.GetPointData()), ("J", J, g.GetCellData())] + [(k, v, g.GetPointData()) for k, v in pdf.items()]:
        v = numpy_to_vtk(np.ascontiguousarray(arr), deep=True); v.SetName(nm); dst.AddArray(v)
    w=vtk.vtkXMLUnstructuredGridWriter(); w.SetFileName(str(post/f"{name}_wall_deployed.vtu")); w.SetInputData(g); w.Write()
    (post / f"{name}_wall_checks.json").write_text(json.dumps(out, indent=1))
    print("wrote", post / f"{name}_wall_deployed.vtu")
    if not a.no_series:                                                   # the same grid, one file per frame: only points and displacement change
        pdir = post / "paraview"; pdir.mkdir(exist_ok=True); corner = np.searchsorted(labels, np.arange(1, len(Xc) + 1))
        nfr = len(res["times"]); frames = sorted(set(list(range(0, nfr, a.every)) + [nfr - 1])); entries = []
        s_arr = numpy_to_vtk(np.ascontiguousarray(pdf["surface"]), deep=True); s_arr.SetName("surface")
        for f in frames:
            Uk=(L[:,:,None]*res["U"][f].astype(float)[corner][Tc[bt]]).sum(1)  # FEA frame -> every fine wall node (stored weights)
            gk=vtk.vtkUnstructuredGrid(); pk=vtk.vtkPoints(); pk.SetData(numpy_to_vtk(np.ascontiguousarray(Xf + Uk), deep=True)); gk.SetPoints(pk)
            gk.SetCells(vtk.VTK_TETRA,ca)
            for nm, arr in (("displacement", Uk), ("displacement_mm", np.linalg.norm(Uk, axis=1))):
                v = numpy_to_vtk(np.ascontiguousarray(arr), deep=True); v.SetName(nm); gk.GetPointData().AddArray(v)
            gk.GetPointData().AddArray(s_arr)
            w = vtk.vtkXMLUnstructuredGridWriter(); w.SetFileName(str(pdir / f"wall_{f:04d}.vtu")); w.SetInputData(gk); w.Write()
            entries.append((float(res["times"][f]), f"wall_{f:04d}.vtu"))
        (pdir / "wall.pvd").write_text('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n'
                                       + "".join(f'<DataSet timestep="{t:.6g}" file="{fn}"/>\n' for t, fn in entries) + "</Collection>\n</VTKFile>\n")
        print(f"ParaView series of the thick wall: {len(entries)} frames -> {pdir / 'wall.pvd'} (open together with heart.pvd / expander.pvd of 06)")

if __name__=="__main__":
    main()





