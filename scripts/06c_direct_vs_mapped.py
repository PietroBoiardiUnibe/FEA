"""
Standard Output for CFD/FSI computational domain definition: the FEA own's deformed mesh is passed through, plus the evidence that skipping the map back onto
the dense SSM vertices from the initially generated VTP surface mesh and the interpolation of displacements back onto the thick VTU tet wall, costs no accuracy.

1) Writes for every level of the variant:  post/<name>_deployed_fea_wall.vtu (deformed FEA wall, corner tets, every label of 01 + displacement) and
   post/<name>_deployed_fea_inner.vtp (its deformed inner surface = the blood-pool boundary handed to CFD/FSI).

2) Compares the DIRECT SURFACE with the mapped one of 06 (every SSM vertex moved with the barycentric FEA displacement of its coarse triangles).
Both carry the SAME displacement field and differ only in the geometry they start from (decimated vs original SSM surface),
so their distance equals the price of skipping the map step.surface distance both ways (mean / p95 / max, also in the device zone < ZONE_MM from the TV annulus), cavity-volume change of both against the
   mesh uncertainty U, tissue inside the deployed cuff, edge length of the direct surface in the device zone.
3) FIGURE to compare direct vs mapped cases: A distance per level vs the 0.3 mm seal criterion used, B cavity change direct vs mapped vs the U band of the finest mesh,
C map (angle x height about the cuff axis) of the distance on the device zone of the shown level.
Usage: python 06c_direct_vs_mapped.py --id SYN_0001 --tag cen_bonded_ramp0.2 [--levels L1 L2 L3 L4 L5] [--show L3]
"""

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import vtk
from scipy.spatial import cKDTree 
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import fea_common as fc
import fea_config as cfg
SEAL_MM=0.3 # seal gap criterion 0f script 06 (smallest judged geometric scale we have in chain)
INK, INK2, BLUE, ORANGE, RED, GRID, SURFACE, BAND = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#d03b3b", "#e1e0d9", "#fcfcfb", "#cde2fb"

def read_vtp(path):
    r=vtk.vtkXMLPolyDataReader(); r.SetFileName(str(path)); r.Update(); return r.GetOutput()

def polydata(P,F,arrays=None):
    pd=vtk.vtkPolyData(); pts=vtk.vtkPoints();pts.SetData(numpy_to_vtk(np.ascontiguousarray(P,float), deep=True)); pd.SetPoints(pts)
    ca=vtk.vtkCellArray();ca.SetCells(len(F), numpy_to_vtk(np.hstack([np.full((len(F), 1), 3), F]).ravel().astype(np.int64), deep=True,array_type=vtk.VTK_ID_TYPE)); pd.SetPolys(ca)
    for k , v in (arrays or {}).items():
        a=numpy_to_vtk(np.ascontiguousarray(v),deep=True); a.SetName(k); pd.GetPointData().AddArray(a)
    return pd

def write_wall(path,X,T,arrays):
    g=vtk.vtkUnstructuredGrid(); pts=vtk.vtkPoints(); pts.SetData(numpy_to_vtk(np.ascontiguousarray(X,float),deep=True)); g.SetPoints(pts)
    ca=vtk.vtkCellArray(); ca.SetCells(len(T), numpy_to_vtk(np.hstack([np.full((len(T),1),4),T]).ravel().astype(np.int64), deep=True,array_type=vtk.VTK_ID_TYPE));g.SetCells(vtk.VTK_TETRA,ca)

    for k, v in arrays.items():
        a=numpy_to_vtk(np.ascontiguousarray(v), deep=True); a.SetName(k); g.GetPointData().AddArray(a)
    w=vtk.vtkXMLUnstructuredGridWriter(); w.SetFileName(str(path)); w.SetInputData(g); w.Write()

def distance(P,F,Q): 
    """Unsigned distance of the points Q to the triangle surface (P,F)"""
    d=vtk.vtkImplicitPolyDataDistance(); d.SetInput(polydata(P,F)); out=vtk.vtkDoubleArray()
    d.FunctionValue(numpy_to_vtk(np.ascontiguousarray(Q,float),deep=True),out); return np.abs(vtk_to_numpy(out))

def volume(P,F):
     return abs(np.einsum("ij,ij->i", P[F[:, 0]], np.cross(P[F[:, 1]], P[F[:, 2]])).sum() / 6.0) / 1000.0     # ml

def inside_cuff(Q,cuff_final, c,n):
    """Depth (mm)  by which points lie inside the deployed cuff:  cuff = surface of revolution about the axis, radius r_cuff(z) from its final nodes."""
    zc = (cuff_final - c) @ n; rc = np.linalg.norm((cuff_final - c) - np.outer(zc, n), axis=1)
    bins = np.arange(zc.min(), zc.max() + 0.5, 0.5); r_of = np.array([rc[(zc >= b) & (zc < b + 0.5)].max() if ((zc >= b) & (zc < b + 0.5)).any() else np.nan for b in bins])
    zq = (Q - c) @ n; rq = np.linalg.norm((Q - c) - np.outer(zq, n), axis=1)
    ok = (zq > zc.min() + 0.5) & (zq < zc.max() - 0.5)                                      # ends excluded (the cuff rims round off)
    rcz = np.interp(zq[ok], bins + 0.25, r_of); return float(np.nanmax(rcz - rq[ok])) if ok.any() else float("nan")

def one_level(a,run,level,plan_raw):
    tag=f"{level}_{a.tag}"; dep = run / f"deploy_{tag}"; name = f"{a.id}_{tag}"
    info=json.loads((dep / f"{name}_info.json").read_text()); res = np.load(dep / f"{name}_results.npz")
    mp=np.load(run / "mesh" / f"{a.id}_wall_{level}_map.npz"); cn, Fc = mp["coarse_nodes"], mp["faces"]
    r=vtk.vtkXMLUnstructuredGridReader(); r.SetFileName(str(run/"mesh"/f"{a.id}_wall_{level}.vtu")); r.Update(); g=r.GetOutput()
    X=vtk_to_numpy(g.GetPoints().GetData()).astype(float); T=vtk_to_numpy(g.GetCells().GetConnectivityArray()).reshape(-1,4)
    labels= {g.GetPointData().GetArrayName(i): vtk_to_numpy(g.GetPointData().GetArray(i)) for i in range(g.GetPointData().GetNumberOfArrays())}
    U=res["U"][-1].astype(float)[np.searchsorted(res["labels"],np.arange(1,len(X)+1))]
    # corner nodes = the first len(X) FEA nodes
    # 1) the deliverable: the deformed FEA wall and its inner surface
    write_wall(dep/"post"/f"{name}_deployed_fea_wall.vtu",X+U,T,dict(labels, displacement=U,displacement_mm=np.linalg.norm(U, axis=1)))
    w=vtk.vtkXMLPolyDataWriter(); w.SetFileName(str(dep/"post"/f"{name}_deployed_fea_inner.vtp"))
    w.SetInputData(polydata(X[cn]+U[cn], Fc,{k: v[cn] for k,v in labels.items()}|
                            {"displacement": U[cn]})); w.Write()
    P0=X[cn];P1=P0+U[cn]
    heart = read_vtp(cfg.HEARTS / f"{a.id}_RH.vtp"); S0=vtk_to_numpy(heart.GetPoints().GetData()).astype(float)
    Fs=vtk_to_numpy(heart.GetPolys().GetConnectivityArray()).reshape(-1,3); ann_s=vtk_to_numpy(heart.GetPointData().GetArray("tv_annulus")) == 1
    S1=vtk_to_numpy(read_vtp(dep / "post" / f"{name}_deformed.vtp").GetPoints().GetData()).astype(float) #  MAPPED (06)
    ta = cKDTree(S0[ann_s]); zs = ta.query(S0)[0] < cfg.ZONE_MM; zd = ta.query(P0)[0] < cfg.ZONE_MM
    d_sd = distance(P1, Fc, S1); d_ds = distance(S1, Fs, P1)                                  # mapped -> direct surface, direct -> mapped surface
    E = np.concatenate([Fc[:, [0, 1]], Fc[:, [1, 2]], Fc[:, [2, 0]]]); el = np.linalg.norm(P1[E[:, 0]] - P1[E[:, 1]], axis=1); ez = zd[E[:, 0]] & zd[E[:, 1]]
    plan = fc.axis_plan(plan_raw, info.get("axis_offset_e1e2_mm")); fr = plan["frame"]; c, n, e1, e2 = (np.array(fr[k]) for k in ("centre", "normal", "e1", "e2"))
    fin = vtk_to_numpy(read_vtp(info["device"]).GetPointData().GetArray("final")).astype(float)
    vd0, vd1, vm0, vm1 = volume(P0, Fc), volume(P1, Fc), volume(S0, Fs), volume(S1, Fs)
    q=lambda x,p : float(np.percentile(x,p)); rel=S0-c
    row = dict(level=level, ssm_vertices=len(S0), fea_inner_vertices=len(P0),
               mapped_to_direct_mm=dict(mean=float(d_sd.mean()), p95=q(d_sd, 95), max=float(d_sd.max()), zone_p95=q(d_sd[zs], 95), zone_max=float(d_sd[zs].max())),
               direct_to_mapped_mm=dict(mean=float(d_ds.mean()), p95=q(d_ds, 95), max=float(d_ds.max()), zone_p95=q(d_ds[zd], 95), zone_max=float(d_ds[zd].max())),
               volume_ml=dict(direct_before=vd0, direct_after=vd1, mapped_before=vm0, mapped_after=vm1),
               cavity_change_ml=dict(direct=vd1 - vd0, mapped=vm1 - vm0, diff_pct=100 * ((vd1 - vd0) / (vm1 - vm0) - 1)),
               inside_cuff_mm=dict(direct=inside_cuff(P1, fin, c, n), mapped=inside_cuff(S1, fin, c, n)),
               edge_mm=dict(zone_median=float(np.median(el[ez])), zone_p95=q(el[ez], 95), all_median=float(np.median(el))))
    plot = dict(d=d_sd, zone=zs, theta=np.degrees(np.arctan2(rel @ e2, rel @ e1)) % 360, z=rel @ n)
    return row, plot

def figure(path, a, rows, plots, U, show):
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.6), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1, 0.9, 1.35], wspace=0.28))
    lv = [r["level"] for r in rows]; x = np.arange(len(lv))
    ax = axs[0]                                                                    # A: distance per level
    bp = ax.boxplot([plots[L]["d"] for L in lv], positions=x, widths=0.55, whis=(1, 99), showfliers=False, patch_artist=True, medianprops=dict(color=INK))
    for b in bp["boxes"]:
        b.set(facecolor=BAND, edgecolor=BLUE)
    ax.scatter(x, [plots[L]["d"].max() for L in lv], color=ORANGE, edgecolor=INK, zorder=5, label="largest distance")
    ax.axhline(SEAL_MM, color=RED, lw=1.2, ls="--"); ax.text(-0.45, SEAL_MM * 1.02, f"seal-gap criterion {SEAL_MM:g} mm", color=RED, ha="left", va="bottom", fontsize=8.5)
    for xi, r in zip(x, rows):
        ax.text(xi, r["mapped_to_direct_mm"]["max"] + 0.02, f"p95 {r['mapped_to_direct_mm']['p95']:.2f}\nmax {r['mapped_to_direct_mm']['max']:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([f"{L}\n{r['fea_inner_vertices']:,} vert." for L, r in zip(lv, rows)], fontsize=8.5)
    ax.set_ylim(0, max(SEAL_MM * 1.4, max(r["mapped_to_direct_mm"]["max"] for r in rows) * 1.35)); ax.set_ylabel("distance SSM vertex -> FEA surface (mm)")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title(f"A  How far the FEA surface is from the {rows[0]['ssm_vertices']:,} SSM vertices\n(after deployment; box 1-99 %, line = median)", loc="left", fontsize=10.5, fontweight="bold", color=INK)
    ax = axs[1]                                                                    # B: cavity change
    cd = [r["cavity_change_ml"]["direct"] for r in rows]; cm = [r["cavity_change_ml"]["mapped"] for r in rows]
    if np.isfinite(U):
        ax.axhspan(cm[0] * (1 - U / 100), cm[0] * (1 + U / 100), color=BAND, lw=0, label=f"mesh uncertainty of {lv[0]} (U = {U:.2f} %)")
    ax.scatter(x - 0.1, cm, s=80, color=BLUE, label="mapped (SSM vertices)", zorder=4); ax.scatter(x + 0.1, cd, s=80, marker="D", color=ORANGE, label="direct (FEA surface)", zorder=4)
    for xi, r in zip(x, rows):
        ax.text(xi, max(cd[xi], cm[xi]) + 0.03, f"{r['cavity_change_ml']['diff_pct']:+.2f} %", ha="center", va="bottom", fontsize=8, color=INK)
    lo = min(cd + cm + ([cm[0] * (1 - U / 100)] if np.isfinite(U) else [])); hi = max(cd + cm + ([cm[0] * (1 + U / 100)] if np.isfinite(U) else []))
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.18 * (hi - lo))
    ax.set_xticks(x); ax.set_xticklabels(lv); ax.set_ylabel("cavity volume change (ml)"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("B  The judged shape QoI: direct = mapped\n(label = direct vs mapped difference)", loc="left", fontsize=10.5, fontweight="bold", color=INK)
    ax = axs[2]; p = plots[show]; z = p["zone"]                                    # C: map of the distance in the device zone
    sc = ax.scatter(p["theta"][z], p["z"][z], c=p["d"][z], s=4, cmap="viridis", vmin=0, vmax=max(0.05, float(np.percentile(p["d"][z], 99.5))))
    cb = fig.colorbar(sc, ax=ax, pad=0.01); cb.set_label("distance (mm)")
    r = next(r for r in rows if r["level"] == show)
    ax.set(xlim=(0, 360), xlabel="angle about the cuff axis (deg)", ylabel="height above the annulus plane (mm)"); ax.set_xticks(range(0, 361, 90))
    ax.set_title(f"C  Where the difference is ({show}, device zone < {cfg.ZONE_MM:g} mm from the annulus)\np95 {r['mapped_to_direct_mm']['zone_p95']:.3f} mm, "
                 f"max {r['mapped_to_direct_mm']['zone_max']:.2f} mm, FEA edge {r['edge_mm']['zone_median']:.2f} mm", loc="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.suptitle(f"{a.id} ({a.tag}): the FEA's own deformed surface reproduces the SSM geometry -> handed to CFD/FSI directly", x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.subplots_adjust(top=0.83)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id",default="SYN_0001"); ap.add_argument("--tag",default="cen_bonded_ramp0.2")
    ap.add_argument("--levels", nargs="+", default=["L1", "L2", "L3", "L4", "L5"]); ap.add_argument("--show", default="L3", help="level of the map (panel C)")
    ap.add_argument("--out",default="",help="folder of the comparison json/png (default runs/<id>/compare)")
    a=ap.parse_args(); run=cfg.RUNS/a.id; plan_raw=json.loads((cfg.PLANS/f"{a.id}_plan.json").read_text())
    levels=[L for L in a.levels if (run / f"deploy_{L}_{a.tag}" / f"{a.id}_{L}_{a.tag}_results.npz").exists()]
    rows,plots=[],{}

    for L in levels:
        row, plot=one_level(a,run,L,plan_raw); rows.append(row); plots[L]=plot
        print(f"{L}: wrote post/{a.id}_{L}_{a.tag}_deployed_fea_wall.vtu + _deployed_fea_inner.vtp")
    mc=cfg.compare_dir(run,"mesh")/f"mesh_convergence_fea_{a.tag}.json"
    U = json.loads(mc.read_text())["gci_fine_triplet"]["cavity_change_ml"]["uncertainty_pct"] if mc.exists() else float("nan")
    print(f"{a.id} {a.tag}: DIRECT (FEA inner surface) vs MAPPED (SSM vertices, 06) after deployment; device zone = < {cfg.ZONE_MM:g} mm from the TV annulus")
    print(f"{'':5s} {'vertices':>8s} | {'mapped->direct (mm)':>27s} | {'direct->mapped (mm)':>27s} | {'cavity change (ml)':>25s} | {'inside cuff (mm)':>17s} | {'edge in zone':>12s}")
    print(f"{'level':5s} {'FEA':>8s} | {'mean':>5s} {'p95':>5s} {'max':>5s} {'zone95':>6s} {'zmax':>4s} | {'mean':>5s} {'p95':>5s} {'max':>5s} {'zone95':>6s} {'zmax':>4s} | "
          f"{'direct':>7s} {'mapped':>7s} {'diff %':>7s} | {'direct':>7s} {'mapped':>7s} | {'median':>6s} {'p95':>5s}")
    for r_ in rows:
        m, d, cv, ic, e = r_["mapped_to_direct_mm"], r_["direct_to_mapped_mm"], r_["cavity_change_ml"], r_["inside_cuff_mm"], r_["edge_mm"]
        print(f"{r_['level']:5s} {r_['fea_inner_vertices']:8,d} | {m['mean']:5.3f} {m['p95']:5.3f} {m['max']:5.2f} {m['zone_p95']:6.3f} {m['zone_max']:4.2f} | "
              f"{d['mean']:5.3f} {d['p95']:5.3f} {d['max']:5.2f} {d['zone_p95']:6.3f} {d['zone_max']:4.2f} | {cv['direct']:7.3f} {cv['mapped']:7.3f} {cv['diff_pct']:+7.2f} | "
              f"{ic['direct']:7.2f} {ic['mapped']:7.2f} | {e['zone_median']:6.2f} {e['zone_p95']:5.2f}")
    print(f"reference: mesh uncertainty of the cavity change at {levels[0]} U = {U:.2f} %; seal-gap criterion {SEAL_MM:g} mm; SSM surface {rows[0]['ssm_vertices']:,} vertices")
    out=Path(a.out) if a.out else cfg.compare_dir(run,"deliverable"); out.mkdir(exist_ok=True)
    (out / f"direct_vs_mapped_{a.tag}.json").write_text(json.dumps(dict(id=a.id, tag=a.tag, cavity_U_pct=U, levels=rows), indent=1))
    figure(out / f"direct_vs_mapped_{a.tag}.png", a, rows, plots, U, a.show if a.show in plots else levels[0])
    print("wrote", out / f"direct_vs_mapped_{a.tag}.json", "and .png")


if __name__=="__main__":
    main()


