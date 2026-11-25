"""
Physical acceptance of one deployment run (device-agnostic: skirt only pilot trials, 02b full device and later for the scanned real device)
PRIORITY 1 = no major unphysical PENETRATION; then quasi-static; then the anchors against the DEFORMED annulus; seal / ring / far field reported (from 06).

Penetration drops with refinement of mesh and is measured with exact geometry on the FEA's own wall nodes  (what the solver enforced), and again on the SSM vertices mapped by 06
(the dense SSM surface: the difference is the coarse-facet mapping error; the frozen CFD/IB deliverable is the DIRECT FEA surface of 06c):

  P1  Abaqus contact opening COPEN of every tracked node at the end (05 extension)                       FAIL if any node penetrates > PEN_FAIL
  P2  wall inner nodes vs the final cuff contact surface (skirt + 1 mm leaflet layer), signed distance     FAIL if any node is > PEN_FAIL inside
  P3  wall inner nodes vs every anchor body (arm tube r_arm around the deformed beam, ball r_ball)         FAIL if any node is > PEN_FAIL inside
  P4  anchor arms (cuff-contact part) and balls vs the final cuff contact surface (anchors through skirt)   FAIL if > PEN_FAIL inside
  P5  = P2 + P3 on the SSM vertices mapped by 06                                                          reported only (matters if the mapped surface is used)
Quasi-static: wall KE/IE mean over the ramp < 5 %, end < 1 %; anchor tips move < 0.5 mm during the hold.
Anchors: most atrial point of each anchor (arm surface / ball surface) vs the DEFORMED annulus (07b ring tool on the FEA's deformed inner surface):
  PASS >= 0 mm (at or below the annulus), BORDERLINE -0.5..0 mm (at the annulus within the FEA/IB resolution), FAIL < -0.5 mm (supra-annular).
Usage: python 07_checks.py --id SYN_0001 --level L3 --tag c175        Writes deploy_<level>_<tag>/post/<name>_checks.json; exit code 1 if any FAIL.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import vtk
from scipy.spatial import cKDTree 
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import fea_config as cfg
import fea_common as fc

PEN_FAIL=0.3 # mm: deeper penetration = unphysical (> 3 x the solver's COPEN of the frozen run, < 1/1.5 of the IB grid spacing)
COPEN_FAIL=0.2  # mm: Abaqus contact opening
ANCHOR_BORDER=-0.5 # mm: anchors between this and 0 mm = at the annulus within the resolution
KE_IE_MEAN,KE_IE_END,TIP_SETTLE=5.0,1.0,0.5
ANC_N0=2_200_000

def load_module(name, folder):
    spec = importlib.util.spec_from_file_location(name, folder / f"{name}.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def read_poly(path):
    r=vtk.vtkXMLPolyDataReader(); r.SetFileName(str(path)); r.Update(); return r.GetOutput()

def cuff_surface(cuff_file,c,n):
    """Final skirt cuff contact surface as triangles with OUTWARD normals (away from the device axis) -> vtkImplicitPolyDataDistance: > 0 outside, < 0 inside."""
    d=read_poly(cuff_file); X=vtk_to_numpy(d.GetPointData().GetArray("final")).astype(float)
    Q=vtk_to_numpy(d.GetPolys().GetConnectivityArray()).reshape(-1,4)
    T = np.vstack([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]])
    nrm = np.cross(X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]]); cen = X[T].mean(1) - c
    rad = cen - np.outer(cen @ n, n)
    flip = (nrm * rad).sum(1) < 0; T[flip] = T[flip][:, [0, 2, 1]]
    poly=vtk.vtkPolyData();pts=vtk.vtkPoints(); pts.SetData(numpy_to_vtk(np.ascontiguousarray(X),deep=True)); poly.SetPoints(pts)
    ca=vtk.vtkCellArray()
    for t in T:
        ca.InsertNextCell(3,[int(v) for v in t])
    poly.SetPolys(ca)
    f=vtk.vtkImplicitPolyDataDistance(); f.SetInput(poly)
    return f,X

def seg_dist(P, A, B):
    """Distance of every point P to the polyline segments A[i]-B[i] (minimum over segments)."""
    AB = B - A; L2 = np.maximum((AB ** 2).sum(1), 1e-12)
    t = np.clip(((P[:, None, :] - A[None]) * AB[None]).sum(2) / L2[None], 0, 1)
    return np.linalg.norm(P[:, None, :] - (A[None] + t[..., None] * AB[None]), axis=2).min(1)

def anchors_deformed(info,res):
    """deformed centre lines of the compliant anchors with last node as ball centre, arm/ball radii and the cuff-contact masks (02b)."""
    f = info.get("anchors") or info.get("compliant", {}).get("file", "")
    if not f:
        return None
    AB = np.load(f); nA = int(AB["n"]); shift = float(AB["shift"]) if "shift" in AB.files else max(float(AB[f"shift{k}"]) for k in range(nA))
    lab, U = res["labels"], res["U"][-1].astype(float); nid, C = ANC_N0, []
    for k in range(nA):
        C0 = AB[f"C{k}"] - shift * AB[f"er{k}"]; ids = np.arange(nid + 1, nid + 1 + len(C0)); nid += len(C0)
        C.append(C0 + U[np.searchsorted(lab, ids)])
    mask = AB["cuff_mask"] if "cuff_mask" in AB.files else np.ones((nA, len(C[0]) - 1), bool)
    return dict(C=C, final=[AB[f"C{k}"] for k in range(nA)], arm_r=float(AB["arm_r"]), ball_r=float(AB["ball_r"]), cuff_mask=mask)

def wall_vs_anchors(P,anc,kd=None):
    """Clearance  (mm, < 0 = inside) of the wall points P to each anchor body: arm tube (all segments) and ball."""
    kd=kd or cKDTree(P); out=[]
    for C in anc["C"]:
        near=np.unique(np.concatenate([np.array(j, int) for j in kd.query_ball_point(C, anc["arm_r"] + anc["ball_r"] + 2.0)] + [np.zeros(0, int)]))
        if not len(near):
            out.append((np.inf,np.inf)); continue
        da=seg_dist(P[near], C[:-2], C[1:-1]).min() - anc["arm_r"] 
        db=np.linalg.norm(P[near] - C[-1], axis=1).min() - anc["ball_r"]
        out.append((float(da),float(db)))
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", default="SYN_0001"); ap.add_argument("--level", default="L3"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    run = cfg.RUNS / a.id; sfx = f"_{a.tag}" if a.tag else ""; name = f"{a.id}_{a.level}{sfx}"; dep = run / f"deploy_{a.level}{sfx}"
    post = dep / "post"; post.mkdir(exist_ok=True)
    info = json.loads((dep / f"{name}_info.json").read_text()); res = np.load(dep / f"{name}_results.npz"); lab = res["labels"]; U = res["U"][-1].astype(float)
    plan = fc.axis_plan(json.loads((cfg.PLANS / f"{a.id}_plan.json").read_text()), info.get("axis_offset_e1e2_mm")); fr = plan["frame"]
    ann = {k: np.array(fr[k]) for k in ("centre", "normal", "e1", "e2")}; c, n = ann["centre"], ann["normal"]
    mp = np.load(run / "mesh" / f"{a.id}_wall_{a.level}_map.npz"); cn, F = mp["coarse_nodes"], mp["faces"]
    r=vtk.vtkXMLUnstructuredGridReader(); r.SetFileName(str(run/"mesh"/f"{a.id}_wall_{a.level}.vtu")); r.Update(); g=r.GetOutput()
    X=vtk_to_numpy(g.GetPoints().GetData()).astype(float); tv=  vtk_to_numpy(g.GetPointData().GetArray("tv_annulus"))[cn]==1
    W0 = X[cn]; W1 = W0 + U[np.searchsorted(lab, cn + 1)]                      # FEA inner (blood-pool) surface before / after
    heart = read_poly(cfg.HEARTS / f"{a.id}_RH.vtp"); S0 = vtk_to_numpy(heart.GetPoints().GetData()).astype(float)
    S1 = S0 + (mp["bary"][:, :, None] * U[np.searchsorted(lab, cn + 1)][F[mp["tri"]]]).sum(1)   # SSM vertices mapped as in 06
    cuff_file = Path(info.get("device", run / "device" / "expander.vtp"))
    out = dict(id=a.id, level=a.level, tag=a.tag, verdict={}, criteria=dict(pen_fail_mm=PEN_FAIL, copen_fail_mm=COPEN_FAIL, anchor_border_mm=ANCHOR_BORDER,
               ke_ie_mean_pct=KE_IE_MEAN, ke_ie_end_pct=KE_IE_END, tip_settle_mm=TIP_SETTLE))
    V = out["verdict"]
    print(f"{name}: physical acceptance (penetration first)")

    # --- P1 abaqus contact opening
    if "copen_last" in res.files:
        cl,cv=res["copen_last_labels"], res["copen_last"]
        w,d=cv[cl<2_000_000], cv[cl>=2_000_000]
        out["P1_copen"]=dict(wall_min_mm=float(w.min()) if len(w) else 0.0, device_min_mm=float(d.min()) if len(d) else 0.0,
                            n_over_0p1=int((cv < -0.1).sum()), n_over_fail=int((cv < -COPEN_FAIL).sum()))
        V["P1 COPEN"] = "PASS" if out["P1_copen"]["n_over_fail"] == 0 else "FAIL"
        print(f"  P1 COPEN: worst wall {-min(w.min(), 0):.3f} mm, device {-min(d.min(), 0):.3f} mm; nodes > 0.1 mm {out['P1_copen']['n_over_0p1']}, "
              f"> {COPEN_FAIL} mm {out['P1_copen']['n_over_fail']} -> {V['P1 COPEN']}")
    else:
        V["P1 COPEN"] = "NOT AVAILABLE"; print("  P1 COPEN: not in the results (05 without the contact extension)")

    # P2 wall vs skirt of device
    f_cuff,Xcuff=cuff_surface(cuff_file, c,n)
    zc = (Xcuff - c) @ n
    def cuff_pen(P):                                                           # only wall points within the cuff's axial range and near it
        z = (P - c) @ n; m = (z > zc.min() + 0.5) & (z < zc.max() - 0.5)
        m &= cKDTree(Xcuff).query(P)[0] < 4.0
        sd = np.array([f_cuff.EvaluateFunction(p) for p in P[m]])
        return sd, int(m.sum())
    sd,nm=cuff_pen(W1)
    out["P2_wall_in_cuff"]=dict(nodes_checked=nm, deepest_mm=float(max(-sd.min(),0)),n_over_0p1=int((sd < -0.1).sum()), n_over_fail=int((sd < -PEN_FAIL).sum()))
    V["P2 wall in cuff"] = "PASS" if out["P2_wall_in_cuff"]["n_over_fail"] == 0 else "FAIL"
    print(f"  P2 FEA wall nodes inside the cuff: deepest {out['P2_wall_in_cuff']['deepest_mm']:.2f} mm ({nm} nodes at the cuff; > 0.1 mm "
          f"{out['P2_wall_in_cuff']['n_over_0p1']}, > {PEN_FAIL} mm {out['P2_wall_in_cuff']['n_over_fail']}) -> {V['P2 wall in cuff']}")

    # P3 walls vs anchors and P4 anchors vs cuff
    anc=anchors_deformed(info,res)
    if anc:
        cl3=wall_vs_anchors(W1,anc)
        worst=min(min(x) for x in cl3)
        out["P3_wall_in_anchors"]=dict(per_anchor_arm_ball_mm=cl3,worst_mm=float(worst))
        V["P3 wall in anchors"] = "PASS" if worst >= -PEN_FAIL else "FAIL"
        print("  P3 FEA wall nodes vs anchor bodies (clearance, < 0 = inside): " + ", ".join(f"A{k} {min(x):+.2f}" for k, x in enumerate(cl3)) + f" -> {V['P3 wall in anchors']}")
        p4=[]
        for k, C in enumerate(anc["C"]):
            m=anc["cuff_mask"][k]; mids=0.5*(C[:-1] + C[1:])[m]
            arm = min([f_cuff.EvaluateFunction(p) for p in mids] or [np.inf]) - anc["arm_r"]
            ball = f_cuff.EvaluateFunction(C[-1]) - anc["ball_r"]
            p4.append((float(arm), float(ball)))
        worst4=min(min(x) for x in p4)
        out["P4_anchors_in_cuff"]=dict(per_anchor_arm_ball_mm=p4, worst_mm=float(worst4))
        V["P4 anchors through skirt"] = "PASS" if worst4 >= -PEN_FAIL else "FAIL"
        print("  P4 anchor arms/balls vs the cuff contact surface: " + ", ".join(f"A{k} {min(x):+.2f}" for k, x in enumerate(p4)) + f" -> {V['P4 anchors through skirt']}")

    # P5 same but on SSM vertices mapped by 06
    sd5, _ = cuff_pen(S1)
    out["P5_mapped"] = dict(cuff_deepest_mm=float(max(-sd5.min(), 0)), cuff_n_over_fail=int((sd5 < -PEN_FAIL).sum()))
    if anc:
        cl5 = wall_vs_anchors(S1, anc); out["P5_mapped"]["anchors_worst_mm"] = float(min(min(x) for x in cl5))
        out["P5_mapped"]["anchors_per_anchor_mm"] = cl5
    print(f"  P5 mapped SSM surface: cuff deepest {out['P5_mapped']['cuff_deepest_mm']:.2f} mm"
          + (f", anchors worst {out['P5_mapped']['anchors_worst_mm']:+.2f} mm" if anc else "") + " -> reported (deliverable = direct FEA surface)")

    # quasi static check
    t,ke,ie=res["hist_wall_ALLKE"][:,0], res["hist_wall_ALLKE"][:,1],res["hist_wall_ALLIE"][:,1]
    t_ramp = info["time"]["ramp_s"]; m = (ie > 0.05 * ie.max()) & (t <= t_ramp + 1e-9); q = 100 * ke / np.maximum(ie, 1e-12)
    qs=dict(wall_mean_ramp_pct=float(q[m].mean()), wall_end_pct=float(q[-1]),wall_peak_pct=float(q[m].max()))
    ok=qs["wall_mean_ramp_pct"]<KE_IE_MEAN and qs["wall_end_pct"]<KE_IE_END
    if anc:
        hold=[k for k, tf in enumerate(res["times"]) if tf>=t_ramp - 1e-9]; nid,mov=ANC_N0,[]
        AB=np.load(info.get("anchors") or info["compliant"]["file"])
        for k in range(int(AB["n"])):
            nid += len(AB[f"C{k}"]); tip = res["U"][hold][:, np.searchsorted(lab, nid)].astype(float); mov.append(float(np.linalg.norm(tip - tip[-1], axis=1).max()))
        qs["anchor_tip_motion_hold_mm"]=max(mov); ok=ok and max(mov)< TIP_SETTLE
        if "hist_anc_ALLKE" in res.files:
            qs["anchor_KE_IE_end_pct"]=float(100*res["hist_anc_ALLKE"][-1,1]/max(res["hist_anc_ALLIE"][-1,1],1e-12))
    out["quasi_static"] = qs; V["quasi-static"] = "PASS" if ok else "FAIL"
    print(f"  quasi-static: wall KE/IE ramp {qs['wall_mean_ramp_pct']:.2f} %, end {qs['wall_end_pct']:.2f} %"
          + (f"; anchor tips move {qs['anchor_tip_motion_hold_mm']:.3f} mm in the hold" if anc else "") + f" -> {V['quasi-static']}")

    # anchors vs the deformed annulus
    if anc:
        sys.path.append(str(cfg.SHAPE_SCRIPTS)); b07= load_module("07b_deploy_plan",cfg.SHAPE_SCRIPTS);mio=load_module("mesh_io",cfg.SHAPE_SCRIPTS)
        th=np.array(plan["ring"]["theta"])
        tree=vtk.vtkOBBTree(); tree.SetDataSet(mio.polydata_from_arrays(W1,F)); tree.BuildLocator()
        rr,zz,_=b07.ring(tree,dict(ann,band=W1[tv]),th,8)
        assert np.isfinite(rr).all(), "ring rays missed the wall (run in the shapeworks env, VTK 9.5.1)"
        o = np.argsort(np.mod(th, 2 * np.pi)); zring = lambda tt: np.interp(np.mod(tt, 2 * np.pi), np.mod(th, 2 * np.pi)[o], zz[o], period=2 * np.pi)
        marg = []
        for C in anc["C"]:
            rel = C - c; z = rel @ n; tt = np.arctan2(rel @ ann["e2"], rel @ ann["e1"])
            marg.append(float(min(((z[:-1] - anc["arm_r"]) - zring(tt[:-1])).min(), (z[-1] - anc["ball_r"]) - zring(tt[-1]))))
        w=min(marg); V["anchors at/below annulus"]="PASS" if w>=0 else ("BORDERLINE" if w>=ANCHOR_BORDER else "FAIL")
        tipd=[float(np.linalg.norm(C[-1] - Cf[-1])) for C, Cf in zip(anc["C"], anc["final"])]
        out["anchors"] = dict(margin_mm=marg, n_ge_1mm=int(sum(m_ >= 1 for m_ in marg)), tip_deflection_mm=tipd, ring_mean_mm=float(np.nanmean(rr)), ring_SD_mm=float(np.nanstd(rr)))
        print("  anchors vs the deformed annulus (mm, + = ventricular): " + ", ".join(f"A{k} {v:+.2f}" for k, v in enumerate(marg))
              + f" -> {V['anchors at/below annulus']}; tip deflection {min(tipd):.2f}-{max(tipd):.2f} mm")

    # seal/ring/far field from script 06
    d06=post/f"{name}_deployed.json"
    if d06.exists():
        j=json.loads(d06.read_text())
        out["from_06"] = {k: j[k] for k in ("seal_angles_above_0p3mm", "seal_worst_angle_gap_mm", "ring_SD_after_mm", "ring_stretch_pct", "far_field_drift_mm", "max_disp_mm") if k in j}
        print("  from 06: " + ", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in out["from_06"].items()))
        # clinical plausibility
        #input is the sizing rule(device D vs the pre-deployment perimeter derived diameter against the clinical reference over-sizing)
        #Output= what the FEA does to annulus (perimeter and area change)an UPPER bound, the prescribed cuff reaches 100 % of nominal (clinical frames ~95 %).
        sz,cl=j.get("sizing"), cfg.CLINICAL
        if sz:
            ov=sz["oversizing_vs_pre"]["perimeter_derived_pct"]; lo,hi=cl["oversizing_range_pct"]
            out["clinical"] = dict(input_sizing_oversizing_pct=ov, clinical_oversizing_mean_pct=cl["oversizing_mean_pct"], clinical_oversizing_range_pct=[lo, hi],
                               output_perimeter_stretch_pct=sz["perimeter_stretch_pct"], output_area_increase_pct=sz["area_increase_pct"],
                               output_Dp_post_mm=sz["post"]["Dp_mm"], device_D_mm=sz["d_device_mm"], leaflet_layer_mm=cfg.NATIVE_LEAFLET_THICKNESS, source=cl["source"])
            V["sizing rule vs clinic (flag)"] = "WITHIN" if lo <= ov <= hi else "OUTSIDE"
            print(f"  clinical, INPUT: sizing oversizing {ov:.1f} % (Penta 2025: {cl['oversizing_mean_pct']:g} %, {lo:g}-{hi:g} %) -> {V['sizing rule vs clinic (flag)']}")
            print(f"  clinical, OUTPUT (upper bound): annulus perimeter +{sz['perimeter_stretch_pct']:.1f} %, area +{sz['area_increase_pct']:.1f} %, deployed Dp "
                  f"{sz['post']['Dp_mm']:.1f} mm = device {sz['d_device_mm']:g} + 2 x {cfg.NATIVE_LEAFLET_THICKNESS:g} mm leaflet layer {'(consistent)' if abs(sz['post']['Dp_mm'] - sz['d_device_mm'] - 2 * cfg.NATIVE_LEAFLET_THICKNESS) < 1.0 else '(CHECK)'}")
    else:
        print("  from 06: not available (run 06 first for seal line, ring stretch and far field)")

    fails = [k for k, v in V.items() if v == "FAIL"]; out["overall"] = "FAIL" if fails else ("BORDERLINE" if "BORDERLINE" in V.values() else "PASS")
    (post / f"{name}_checks.json").write_text(json.dumps(out, indent=1))
    print(f"OVERALL: {out['overall']}" + (f" ({', '.join(fails)})" if fails else "") + f"  -> {post / (name + '_checks.json')}")
    sys.exit(1 if fails else 0)

if __name__=="__main__":
    main()
