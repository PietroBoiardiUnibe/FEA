"""
From the Abaqus results to the deformed SSM surface that shall be prepared to run FSI with the IBAMR method.
Endpoint metrics, figures and ParaView visualization kits are alo generated.

Usage: python 06_deformed.py --id SYN_0001 --level L3 [--no_series]
Reads : runs/<id>/mesh/<id>_wall_<level>_map.npz, runs/<id>/device/expander.vtp, runs/<id>/deploy_<level>/<name>_results.npz and _info.json
Writes (runs/<id>/deploy_<level>/post/):
  <name>_deformed.vtp            final deformed SSM surface (ALL vertices, all labels, displacement as arrays)
  <name>_deployed.json           every number reported in the thesis: energies, ring, seal line, far field, volume
  <name>_summary.png             the figure for the methods/results chapter
  paraview/heart_####.vtp, expander_####.vtp, heart.pvd, expander.pvd   time series for ParaView
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import vtk
from scipy.spatial import cKDTree 
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import fea_config as cfg

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BANDS = (0, 5, 10, 20, 30, 45, 60, 200)                 # mm from the TV ring

def load_module(name,folder):
    spec=importlib.util.spec_from_file_location(name,folder/f"{name}.py")
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def read_polydata(path):
    r=vtk.vtkXMLPolyDataReader(); r.SetFileName(str(path)); r.Update()
    return r.GetOutput()

def write_polydata(path, poly):
    w=vtk.vtkXMLPolyDataWriter(); w.SetFileName(str(path)); w.SetInputData(poly); w.Write()

def make_poly(points, polys, arrays):
    """New polydata: given points, the connectivity of an existing surface (shared not copied) and named point arrays"""
    poly=vtk.vtkPolyData()
    pts=vtk.vtkPoints(); pts.SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64),deep=True)); poly.SetPoints(pts)
    poly.SetPolys(polys)
    for name, arr in arrays.items():
        a=numpy_to_vtk(np.ascontiguousarray(arr),deep=True); a.SetName(name); poly.GetPointData().AddArray(a)
    return poly

def surface_displacement(U,labels,coarse_nodes, faces, tri , bary):
    """FEA displacement of every original SSM vertex; labels are the sorted Abaqus nodes ids (mesh index +1);
    the coarse inner nodes carry the FEA displacements and each original vertex takes the barycentric average of the three corners of its coarse triangle"""
    Uc=U[np.searchsorted(labels,coarse_nodes+1)]
    return (bary[:,:,None]*Uc[faces[tri]]).sum(axis=1)

def energy_report(res,t_ramp):
    """Quasi-static check on the energies of the WALL alone as 
    the expander's point masses would dominate the whole-model kinetic energy"""
    t=res["hist_wall_ALLKE"][:,0];ke=res["hist_wall_ALLKE"][:,1]; ie=res["hist_wall_ALLIE"][:,1]
    ae,vd=res["hist_wall_ALLAE"][:,1],res["hist_wall_ALLVD"][:,1]
    late=t>=0.5 * t[-1]
    return dict(KE_over_IE_end_pct=float(100 * ke[-1] / ie[-1]), KE_over_IE_max_second_half_pct=float(100 * np.max(ke[late] / np.maximum(ie[late], 1e-30))),
                hourglass_over_IE_pct=float(100 * ae[-1] / ie[-1]), viscous_over_IE_pct=float(100 * vd[-1] / ie[-1]), max_internal_energy_mJ=float(ie.max()),
                t=t, ratio=100 * ke / np.maximum(ie, 1e-30), ie=ie)

def mesh_volume(P,F):
    """Volume enclosed by a triangle surface (divergence theorem)"""
    return abs(np.einsum("ij,ij->i", P[F[:, 0]], np.cross(P[F[:, 1]], P[F[:, 2]])).sum() / 6.0)

def ring_of(P,F,plan,tv,b07,polydata_from_arrays):
    """Radius r(theta) and height z(theta) of the TV ring around the plane axis, measured with the same tool as the planning phase (07b)"""
    tree=vtk.vtkOBBTree(); tree.SetDataSet(polydata_from_arrays(P,F)); tree.BuildLocator()
    fr=plan["frame"]; ann= {k: np.array(fr[k]) for k in ("centre", "normal", "e1", "e2")}
    th=np.array(plan["ring"]["theta"])
    r,z,X=b07.ring(tree,dict(ann,band=P[tv]),th,8)
    return th,r,z,X

def seal_line(X_final,poly_deformed, plan, nbin=120):
    """ chcks that there is a closed contact line around the device; for every angle bin the closest approach of the expander's final surface to the tissue.
    Signed distance.<0 expander inside the blood pool (gap),>0 inside the tissue with penetration"""
    dist=vtk.vtkImplicitPolyDataDistance();dist.SetInput(poly_deformed)
    sd=np.array([dist.EvaluateFunction(p) for p in X_final])
    fr=plan["frame"]; c,n,e1,e2= (np.array(fr[k]) for k in ("centre","normal","e1","e2"))
    rel=X_final-c; ax=rel @ n
    th=np.degrees(np.arctan2(rel@e2,rel@e1))%360
    b=np.round(th/(360.0/nbin)).astype(int) % nbin # nearest bin centre: the expander's generator lines sit exactly at multiples of 3 deg
    gap,z_lo,z_hi=np.full(nbin, np.nan), np.full(nbin, np.nan), np.full(nbin, np.nan)

    for j in range(nbin):
        m=b==j
        if m.any():
            gap[j]=np.abs(sd[m]).min()
            touch=m & (np.abs(sd)<=0.3)       # the part of the device the tissue grips at this angle
            if touch.any():
                 z_lo[j], z_hi[j] = ax[touch].min(), ax[touch].max()à
    return sd,gap,z_lo,z_hi

def far_field(P0,P1,tv):
    """Displacement against distance from the TV ring, before and after removing the best rigid motion of the far field (>45 mm far)"""
    d=cKDTree(P0[tv]).query(P0)[0]
    far=d>45
    A,B=P0[far],P1[far]; ca,cb=A.mean(0),B.mean(0)
    Uu,S,Vt=np.linalg.svd((A-ca).T @ (B-cb))
    D=np.eye(3); D[2,2]=np.sign(np.linalg.det(Vt.T@Uu.T)); R=Vt.T@ D @ Uu.T
    P1r=(P1-cb)@R +ca
    u,ur=np.linalg.norm(P1-P0,axis=1), np.linalg.norm(P1r-P0,axis=1)
    rows=[]

    for lo, hi in zip(BANDS[:-1],BANDS[1:]):
        m=(d>=lo) & (d< hi)
        rows.append(dict(band_mm=[lo, hi], n=int(m.sum()), median_mm=float(np.median(u[m])), p95_mm=float(np.percentile(u[m], 95)), p95_drift_removed_mm=float(np.percentile(ur[m], 95))))

    return rows, float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))),float(np.linalg.norm(cb-ca))

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id",default="SYN_0001")
    ap.add_argument("--level",default="L3",choices=sorted(cfg.INNER_FACES))
    ap.add_argument("--no_series",action="store_true",help="skip the ParaView time series (slow part)")
    a=ap.parse_args()
    run=cfg.RUNS/a.id
    name= f"{a.id}_{a.level}"
    dep=run/f"deploy_{a.level}"
    post=dep/"post"; post.mkdir(parents=True, exist_ok=True)

    mesh_io = load_module("mesh_io", cfg.SHAPE_SCRIPTS)
    polydata_from_arrays = mesh_io.polydata_from_arrays
    b07=load_module("07b_deploy_plan",cfg.SHAPE_SCRIPTS)
    res = np.load(dep / f"{name}_results.npz"); info = json.loads((dep / f"{name}_info.json").read_text())
    mp=np.load(run/"mesh"/f"{a.id}_wall_{a.level}_map.npz")
    plan = json.loads((cfg.PLANS / f"{a.id}_plan.json").read_text())
    dev = json.loads((run / "device" / "device.json").read_text())
    heart=read_polydata(cfg.HEARTS/f"{a.id}_RH.vtp")
    P0 = vtk_to_numpy(heart.GetPoints().GetData()).astype(float)
    F0 = vtk_to_numpy(heart.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    tv = vtk_to_numpy(heart.GetPointData().GetArray("tv_annulus")) == 1
    labels,times=res["labels"],res["times"]
    out= {"id": a.id, "level": a.level, "device_size_mm": dev["size_mm"], "fea_repo": info["fea_repo"], "shape_repo": info["shape_repo"]}

    #quasi static check
    en=energy_report(res,info["time"]["ramp_s"])
    out.update({k: v for k, v in en.items() if k not in ("t", "ratio", "ie")})
    print(f"wall energies: kinetic/internal at the end {out['KE_over_IE_end_pct']:.2f} %, max in the second half {out['KE_over_IE_max_second_half_pct']:.2f} %, "
          f"hourglass {out['hourglass_over_IE_pct']:.2f} %")

    # final deformed SSM surface with all labels being preserved
    Uf=surface_displacement(res["U"][-1].astype(float),labels,mp["coarse_nodes"],mp["faces"],mp["tri"],mp["bary"])
    P1=P0+Uf
    mag=np.linalg.norm(Uf,axis=1)
    keep={n: vtk_to_numpy(heart.GetPointData().GetArray(n)) for n in ("chamber", "tv_annulus", "patch")}
    write_polydata(post/f"{name}_deformed.vtp",make_poly(P1,heart.GetPolys(), dict(keep, displacement=Uf, displacement_mm=mag)))
    out.update(max_disp_mm=float(mag.max()), volume_before_ml=mesh_volume(P0, F0) / 1000, volume_after_ml=mesh_volume(P1, F0) / 1000)
    print(f"deformed surface: max displacement {mag.max():.2f} mm; blood-pool volume {out['volume_before_ml']:.1f} -> {out['volume_after_ml']:.1f} ml")

    # the ring before and after the deployment
    th,r0,_,_=ring_of(P0,F0,plan,tv,b07,polydata_from_arrays)
    _,r1,_,X1=ring_of(P1,F0,plan,tv,b07,polydata_from_arrays)
    out.update(ring_SD_before_mm=float(np.nanstd(r0)),ring_SD_after_mm=float(np.nanstd(r1)),
               ring_radius_after_mm=[float(np.nanmin(r1)),float(np.nanmean(r1)),float(np.nanmax(r1))],
               ring_perimeter_before_mm=float(plan["perimeter_mm"]), ring_perimeter_after_mm=float(np.linalg.norm(np.roll(X1, -1, axis=0) - X1, axis=1).sum()))
    out["ring_stretch_pct"]=100*(out["ring_perimeter_after_mm"]/out["ring_perimeter_before_mm"] -1)
    print(f"ring: SD {out['ring_SD_before_mm']:.2f} -> {out['ring_SD_after_mm']:.2f} mm, radius {out['ring_radius_after_mm'][0]:.2f}..{out['ring_radius_after_mm'][2]:.2f} mm, "
          f"perimeter {out['ring_perimeter_before_mm']:.1f} -> {out['ring_perimeter_after_mm']:.1f} mm ({out['ring_stretch_pct']:+.1f} %)")

    # find out if there is a closed contact line around device once expanded to verify absence of major PVL
    exp=read_polydata(run/"device"/"expander.vtp")
    E=vtk_to_numpy(exp.GetPoints().GetData()).astype(float)
    idx_exp=np.searchsorted(labels,info["expander_node0"]+1+np.arange(len(E)))
    X_final=E+res["U"][-1].astype(float)[idx_exp]
    deformed=polydata_from_arrays(P1,F0)
    sd,gap,z_lo,z_hi=seal_line(X_final, deformed,plan)
    out.update(seal_worst_angle_gap_mm=float(np.nanmax(gap)),seal_median_gap_mm=float(np.nanmedian(gap)),seal_angles_above_0p3mm=int((gap>0.3).sum()),seal_angles=int(len(gap)),contact_band_mm=[float(np.nanmin(z_lo)),float(np.nanmax(z_hi))],
               contact_band_median_height_mm=float(np.nanmedian(z_hi-z_lo)),max_penetration_mm=float(max(sd.max(),0.0)))
    print(f"seal line: worst-angle gap {out['seal_worst_angle_gap_mm']:.2f} mm, angles with gap > 0.3 mm: {out['seal_angles_above_0p3mm']} of {out['seal_angles']}, "
          f"contact band {out['contact_band_mm'][0]:+.1f}..{out['contact_band_mm'][1]:+.1f} mm (median height {out['contact_band_median_height_mm']:.1f} mm), penetration {out['max_penetration_mm']:.2f} mm")

    # verify that the deformation is local only
    rows,drift_deg, drift_mm= far_field(P0,P1,tv)
    out.update(far_field=rows,far_field_drift_deg=drift_deg,far_field_drift_mm=drift_mm)
    print("far field p95 (drift removed): " + ", ".join(f"{r['band_mm'][0]}-{r['band_mm'][1]} mm {r['p95_drift_removed_mm']:.2f}" for r in rows) + f" | drift {drift_deg:.2f} deg / {drift_mm:.2f} mm")
    (post / f"{name}_deployed.json").write_text(json.dumps(out, indent=1))

if __name__=="__main__":
    main()



                     

    