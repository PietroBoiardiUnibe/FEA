"""
The EVOQUE like device for one heart: a closed polymer cuff resembling the skirt without stent struts
scaled to the specific heart. The STL package is measures, the NOMINAL closed cuff is built (44mm model) and we prove it is watertight"""

import sys
from pathlib import Path
import numpy as np
import trimesh
import fea_config as cfg
import argparse
import json
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

def load(path):
    return trimesh.load(str(path),process=True) #stl is a triangle mix: merge duplicated vertices so that closedness can be tested

def measure(pkg):
    """Everything is taken from the STL package, nothing is typed by hand."""
    skirt = load(pkg / "EVOQUE_falda_exterior_real44_mm.stl")
    V = skirt.vertices
    r, z = np.hypot(V[:, 0], V[:, 1]), V[:, 2]
    z_env, r_env = [], []
    for a in np.arange(z.min(), z.max() + 0.25, 0.25):                  # outer envelope: widest radius in every 0.25 mm slice
        s = (z >= a) & (z < a + 0.25)
        if s.any():
            z_env.append(a + 0.125); r_env.append(r[s].max())
    leaves = [load(pkg / "valvas_escaladas_real44_mm" / f"leaf{k}_DEF_x1.4447_mm.stl") for k in (1, 2, 3)]
    edge = []
    for L in leaves:
        rl = np.hypot(L.vertices[:, 0], L.vertices[:, 1])
        edge.append(rl[rl > rl.max() - 0.3])                            # the leaflet attachment edge = its outermost band
    return dict(z_env=np.array(z_env), r_env=np.array(r_env), z_atrial=float(z.min()), z_vent=float(z.max()),
                r_in=float(np.median(np.concatenate(edge))), leaves=leaves)

def cuff_profile(m):
    """
    Closed (r, z) polygon: atrial cap -> outer envelope (atrial to ventricular) -> ventricular cap -> lumen wall back down
    """
    k=5
    r_smooth=np.convolve(np.pad(m["r_env"],k//2,mode="edge"), np.ones(k)/k, mode="valid") # remove the stair-steps of the slicing
    outer=np.stack([r_smooth, m["z_env"]],1)
    wall=[(m["r_in"],z) for z in np.arange(m["z_vent"]-0.5, m["z_atrial"]+0.4,-0.5)]
    return np.array([(m["r_in"], m["z_atrial"])] + [tuple(p) for p in outer] + [(m["r_in"], m["z_vent"])] + wall)

def revolve(profile, ntheta=180):
    """Closed polygon (r, z) -> closed solid of revolution: one ring of vertices per profile point and angle, quads split in triangles."""
    m=len(profile)
    th=np.linspace(0,2*np.pi,ntheta,endpoint=False)
    X=np.stack([profile[:, 0][None, :] * np.cos(th)[:, None], profile[:, 0][None, :] * np.sin(th)[:, None],
                  np.broadcast_to(profile[:, 1], (ntheta, m))], -1).reshape(-1, 3)
    F=[]
    for j in range(ntheta):
        j2=(j+1)%ntheta
        for i in range(m):
            i2 = (i + 1) % m
            a, b, c, d = j * m + i, j2 * m + i, j2 * m + i2, j * m + i2
            F += [[a, b, c], [a, c, d]]
    mesh=trimesh.Trimesh(X,np.array(F),process=True)
    trimesh.repair.fix_normals(mesh) # outward normals, positive volume
    return mesh

def choose_size(d_perim,d_max, phase):
    """EVOQUE size is taken from the official IFU table when anatomy allows: the smallest size whose perimeter-derived diameter range contains the measured TV annular ring
    and whose maximum annulus length is not exceeded. Returns (size or None, note)
    Need to validate results of deployment with supporting literature regarding percutaneous TTVR and pre-procedural planning and device sizing.
     The current dataset should be derived from diastolic or mid-diastolic cardiac CT frames """
    sizes=sorted(cfg.EVOQUE_SIZES)
    for D in sizes:
        lo,hi,max_len=cfg.EVOQUE_SIZES[D][phase]
        if lo<=d_perim<=hi and d_max<=max_len:
            return D, "within the IFU recommended range"
    if d_perim<cfg.EVOQUE_SIZES[sizes[0]][phase][0]:
        if cfg.SMALL_ANNULUS_POLICY=="oversize":
            return sizes[0], "BELOW the IFU range: smallest size, oversized (flag)"
        return None, "Below the IFU range: clinically ineligible for TTVR with the proposed device"
    return None, "Above the IFU range or the maximum allowable annular dimension: not treatable"


def expander_profile(m,size,t_leaf,spacing=1.0):
    """Open (r,z) outline of the expander= the outer cuff envelope scaled to the device size and moved outward by the leaflet thickness too.
    Resampled at equal arc length. The expander is what pushes aside radially the tissue and must leave enough room for juxtaposition of native leaflet layer"""
    s=size/44.0 #the measures STL geometry is the 44 mm model; sizes scale it linearly
    k=5
    r_smooth=np.convolve(np.pad(m["r_env"],k//2,mode="edge"),np.ones(k)/k,mode="valid")
    P=np.stack([r_smooth,m["z_env"]],1)*s
    T=np.gradient(P,axis=0); T/=np.linalg.norm(T,axis=1,keepdims=True)
    P=P+t_leaf*np.stack([T[:,1],-T[:,0]],1) #outward normal in the (r,z) plane with profile running atrial to ventricular
    arc=np.concatenate([[0],np.cumsum(np.hypot(*np.diff(P,axis=0).T))])
    ss=np.linspace(0,arc[-1],max(int(round(arc[-1]/spacing)),2)+1)
    prof=np.stack([np.interp(ss,arc,P[:,0]),np.interp(ss,arc,P[:,1])],1)
    band= prof[prof[:, 0] >= prof[:, 0].max() - 0.1, 1]         # the widest band: the landing reference
    return prof, 0.5*(band.min()+band.max()),s

def to_world(local,plan,z_shift):
    """Device coordinates (x,y,axial z; ventricular +z)-> heart coordinates on the TV axis of the 07b script's planning outcome (e1,e2,n is right handed)"""
    fr=plan["frame"]
    c,n,e1,e2=(np.array(fr[k]) for k in ("centre","normal","e1","e2"))
    return c+np.outer(local[:,0],e1) + np.outer(local[:,1],e2) + np.outer(local[:,2]-z_shift,n)

def write_expander(path,X_crimped,quads,X_final):
    poly=vtk.vtkPolyData(); pts=vtk.vtkPoints(); pts.SetData(numpy_to_vtk(np.ascontiguousarray(X_crimped),deep=True)); poly.SetPoints(pts)
    cells=vtk.vtkCellArray()
    for q in quads:
        cells.InsertNextCell(4,[int(v)  for v in q])
    poly.SetPolys(cells)
    for name, arr in (("final",X_final),("displacement",X_final-X_crimped)):
        a=numpy_to_vtk(np.ascontiguousarray(arr),deep=True);a.SetName(name);poly.GetPointData().AddArray(a)
        w=vtk.vtkXMLPolyDataWriter(); w.SetFileName(str(path));w.SetInputData(poly); w.Write()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--id",default="SYN_0001")
    ap.add_argument("--ntheta",type=int,default=120, help="nodes around the expander")
    a=ap.parse_args()

    m=measure(cfg.EVOQUE)
    assert abs(m["r_in"]-cfg.R_IN_NOMINAL)<0.05 and abs(m["r_env"].max()-cfg.R_MID_NOMINAL)<0.05, "fea_config.py device constants do not match the STL package"
    cuff=revolve(cuff_profile(m))
    print(f"nominal cuff: watertight {cuff.is_watertight}, faces {len(cuff.faces):,}, volume {cuff.volume:,.0f} mm3")
    plan=json.loads((cfg.PLANS/f"{a.id}_plan.json").read_text())

    #which EVOQUE sizing does the annulus get?
    size,note=choose_size(plan["d_perim_mm"],plan["d_max_mm"],cfg.SIZING_PHASE)
    print(f"{a.id}: perimeter-derived diameter {plan['d_perim_mm']} mm, max length {plan['d_max_mm']} mm ({cfg.SIZING_PHASE}) -> EVOQUE {size} mm, {note}")
    if size is None:
        raise SystemExit("No device for this heart")
    print(f"oversizing {100 * (size / plan['d_perim_mm'] - 1):.1f} % vs the perimeter-derived diameter")

    # expander outline, then the surface of revolution on TV axis
    prof, z_mid,s=expander_profile(m,size,cfg.NATIVE_LEAFLET_THICKNESS)
    r_wide=prof[:,0].max()
    n_prof,nt=len(prof),a.ntheta
    th=np.linspace(0,2*np.pi,nt,endpoint=False)
    final_local=np.stack([prof[:, 0][None, :] * np.cos(th)[:, None], prof[:, 0][None, :] * np.sin(th)[:, None], np.broadcast_to(prof[:, 1], (nt, n_prof))], -1).reshape(-1, 3)
    crimp=plan["R0_mm"]/r_wide # shrink radially so the widest part starts at the plan's start radius
    crimped_local=final_local*np.array([crimp,crimp,1.0])
    quads=[[j * n_prof + i, ((j + 1) % nt) * n_prof + i, ((j + 1) % nt) * n_prof + i + 1, j * n_prof + i + 1] for j in range(nt) for i in range(n_prof - 1)]
    X_final,X_crimped=to_world(final_local,plan,z_mid),to_world(crimped_local,plan,z_mid)
    print(f"scale s = {s:.3f}; expander widest radius {r_wide:.2f} mm (skirt {m['r_env'].max() * s:.2f} + leaflet {cfg.NATIVE_LEAFLET_THICKNESS:g}); starts at {plan['R0_mm']:.2f} mm (crimp {crimp:.3f}); "
          f"axial span {prof[0, 1] - z_mid:+.1f} .. {prof[-1, 1] - z_mid:+.1f} mm around the ring plane; {len(X_final):,} nodes, {len(quads):,} quads")

    # does the crimped expander start inside the blood pool clear of the wall ?
    r=vtk.vtkXMLPolyDataReader();r.SetFileName(str(cfg.HEARTS/f"{a.id}_RH.vtp"));r.Update()
    dist=vtk.vtkImplicitPolyDataDistance();dist.SetInput(r.GetOutput())
    sd=np.array([dist.EvaluateFunction(p) for p in X_crimped]) # < 0: inside the blood pool
    print(f"crimped expander vs the wall: signed distance max {sd.max():+.2f} mm (must be < 0), nodes inside the tissue: {int((sd >= 0).sum())}, closest clearance {-sd.max():.2f} mm")
    out=cfg.RUNS/a.id/"device"; out.mkdir(parents=True,exist_ok=True)
    write_expander(out/"expander.vtp",X_crimped,quads,X_final)
    (out / "device.json").write_text(json.dumps({"id": a.id, "size_mm": size, "sizing_phase": cfg.SIZING_PHASE, "note": note, "d_perim_mm": plan["d_perim_mm"], "d_max_mm": plan["d_max_mm"],
                                                 "oversizing_pct": 100 * (size / plan["d_perim_mm"] - 1), "scale": s, "leaflet_thickness_mm": cfg.NATIVE_LEAFLET_THICKNESS,
                                                 "expander_r_widest_mm": r_wide, "start_radius_mm": plan["R0_mm"], "z_band_mid_nominal_mm": z_mid}, indent=1))
    print("wrote", out / "expander.vtp")

if __name__=="__main__":
    main()


