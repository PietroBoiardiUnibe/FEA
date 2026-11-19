"""
This script is meant to handle the numerical credibility and trustworthiness of the FEA deployment: it includes mesh convergence and grid independence of results(GCI),
parameter sensitivity (friction, penalization strength, etc), check of quasi-static verification.
This is judged on the FEA solution only and physical acceptance (anchors ventricular position, seal and no PVL paths, skirt sizing, penetration)
belongs following scripts.

python 10_compare_runs.py --id SYN_0001 mesh --levels L2 L3 L4 L5 [--variant c175]     (finest first, at least three)
  python 10_compare_runs.py --id SYN_0001 sens --base L3_c175 --variants L3_c175_unbonded L3_c175_mu0.2 L3_c175_ramp0.2 [--mesh_ref L2_c175]
  (the runs are made by run_study.py, which prints these two commands)

The script is device agnostic: reads only run folders (info json and results npz) and the FEA meshes;works for the cuff-only pilot, the 02b full device
and any later device that keeps the 03 conventions (wall labels < 2,000,000, device labels >= 2,000,001, anchor beams from 2,200,001).

The QoIs are computed on each FEA's mesh OWN nodes and never after mapping back the displacement field from the coarsened mesh onto the SSM vertices of finer inner surface that 
was generated. the map adds a level-dependent error and is
verified separately). JUDGED QoIs are INTEGRAL quantities only to avoid noisy indicators that could spike, all judged in % of the reference (finest level / base run).

SHAPE: cavity volume change, area-weighted RMS displacement of inner surface, ring mean radius, ring perimeter stretch

MECHANICAL for context: wall strain energy, cuff radial force, anchor strain energy (full device, they can bend)

Pointwise / local quantities are DESCRIPTIVE only: largest displacement, ring out-of-roundness, mean anchor tip deflection, cuff-wall contact
area at the end, share of wall nodes that lost contact.

Mesh: Grid Convergence Index (Celik et al. 2008, J Fluids Eng 130:078001), Fs 1.25, r = (N_fine/N_coarse)^(1/3) from tet counts; convergence
ratio R = eps21/eps32 (0<R<1 converging, observed order capped at the formal 2; R<0 oscillatory, R>=1 drifting -> spread band= 1.25 x max deviation from the finest).

Selection: the coarsest level whose judged QoIs all stay within 5% of the finest.

Sensitivity: a parameter is irrelevant for the deployed shape if every shape QoI changes < 5% on the same mesh level

Quasi-static check of every run: wall KE/IE averaged over the ramp <5% and <1% at the end (peak being reported); full device also: anchor tips
settled in the hold (largest tip movement after the ramp < 0.5 mm = to be validated) and anchor KE/IE at the end (reported as sanity check)
"""
import argparse
import importlib.util
import json
import csv
import sys
import textwrap
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import numpy as np
import vtk
import matplotlib.pyplot as plt
from vtkmodules.util.numpy_support import vtk_to_numpy
import fea_config as cfg
import fea_common as fc
from scipy.spatial import cKDTree

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#d03b3b"                   # red = status 'critical', always shipped with a text label
LEVEL_RAMP = ["#0b2d57", "#104281", "#256abf", "#3987e5", "#86b6ef"]             # ordinal blue ramp, finest (dark) -> coarsest (light)
BAND = "#cde2fb"                                                      # acceptance band (blue step 100)

CRIT=5.0 # % of the reference value (finest level/ base run)
P_FORMAL=2.0 #formal order used to cap the observed order (quadratic tets with contact)
KE_IE_MEAN,KE_IE_END=5.0,1.0 #quasi static criterion (% of the wall internal total energy)
ANC_N0=2_200_000  #first node label of the compliant anchor beams (03 full-device writer)
QOIS=[("strain_energy_mJ","wall strain energy (mJ)"),
      ("radial_force_N","cuff radial force (N)"),
      ("cavity_change_ml","cavity volume change (ml)"),
      ("u_rms_area_mm","area-weighted RMS displacement of internal surface nodes (mm)")
      ]
SHAPE_QOIS=("cavity_change_ml", "u_rms_area_mm")   # the quantities that describe the deployed SHAPE
ANCHOR_QOI= ("anchor_energy_mJ", "anchor strain energy (mJ)")                           # integral; only when every compared run has compliant anchors
DESCRIPTIVE = [("max_disp_mm", "largest wall displacement (mm)"), ("ring_mean_mm", "ring mean radius (mm)"), ("ring_stretch_pct", "ring perimeter stretch (%)"),
               ("ring_SD_mm", "ring out-of-roundness SD (mm)"),
               ("anchor_tip_defl_mm", "mean anchor tip deflection (mm)"), ("contact_area_mm2", "wall area in contact at the end (mm2)"),
               ("detached_pct", "wall nodes that lost contact (%)"), ("lateral_force_N", "net lateral force on the cuff (N)")]

TIP_SETTLE_MM=0.5 # anchor tips must move less than this (maybe IB grid size could be better index)
# spacing during the hold after ramping up smoothly yhe radial deployment
IGNORE_IN_DIFF = ("name", "tag", "file", "device", "anchors", "repo", "n_", "total_N", "node0", "level")   # info keys that are bookkeeping, not parameters

def load_module(name, folder):
    spec=importlib.util.spec_from_file_location(name, folder/f"{name}.py")
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def mesh_volume(P,F):
    return abs(np.einsum("ij,ij->i", P[F[:, 0]], np.cross(P[F[:, 1]], P[F[:, 2]])).sum() / 6.0)

def zone_tets(run,id_,level):
    """Tets whose centroid lies within cfg.ZONE_MM of the level's own TV annulus nodes.
    Refinement ratios are measured HERE only, where the device radial expansion acts: "coarsen where possible, refine where necessary.
    L1 refines only this zone (local sizing) with a global refinement ratio from L2 of ~1.11, and a ratio in the zone of ~1.33;
    for the uniform levels zone and global ratios agree.
    """
    r=vtk.vtkXMLUnstructuredGridReader(); r.SetFileName(str(run/"mesh"/f"{id_}_wall_{level}.vtu")); r.Update(); g=r.GetOutput()
    X=vtk_to_numpy(g.GetPoints().GetData()); T=vtk_to_numpy(g.GetCells().GetConnectivityArray()).reshape(-1,4)
    ann=vtk_to_numpy(g.GetPointData().GetArray("tv_annulus"))==1
    return int((cKDTree(X[ann]).query(X[T].mean(1))[0]<cfg.ZONE_MM).sum())


def cuff_points(info,run):
    """Crimped cuff as in catheter delivery in the order of the job's expander labels:
    cuff-only runs use 02's expander while full device runs their own device file."""
    r=vtk.vtkXMLPolyDataReader(); r.SetFileName(info.get("device",str(run/"device"/"expander.vtp"))); r.Update(); d=r.GetOutput()
    P=vtk_to_numpy(d.GetPoints().GetData()).astype(float); part= d.GetPointData().GetArray("part")
    return P if part is None else P[vtk_to_numpy(part)==0]

def anchors_file(info):
    """The compliant-anchor file of a run ('' = no compliant anchors): 03 writes info['anchors'], the scratch writer info['compliant']['file']."""
    return info.get("anchors") or info.get("compliant", {}).get("file", "") or ""

def anchor_tips(info,res, frames=(-1,)):
    """Tip (ball centre) positions of every anchor at the given frames -> array (frames, anchors, 3) and the planned final tips (anchors, 3)"""
    AB=np.load(anchors_file(info));nA=int(AB["n"])
    shift = float(AB["shift"]) if "shift" in AB.files else max(float(AB[f"shift{k}"]) for k in range(nA))
    lab = res["labels"]; nid, ids, start, final = ANC_N0, [], [], []
    for k in range(nA):
        Cf = AB[f"C{k}"]; nid += len(Cf); ids.append(nid)                       # last centre-line node = ball centre
        start.append(Cf[-1] - shift * AB[f"er{k}"]); final.append(Cf[-1])
    idx = np.searchsorted(lab, ids)
    return np.array([np.array(start) + res["U"][f].astype(float)[idx] for f in frames]), np.array(final)


def quasi_static(res,info):
    """Wall KE/IE in % is used as a sanity check proxy: mean over the ramp (once the wall carries >5% of its final internal energy), end value, peak over the ramp"""
    t,ke,ie=res["hist_wall_ALLKE"][:,0],res["hist_wall_ALLKE"][:,1], res["hist_wall_ALLIE"][:,1]
    t_ramp=info.get("time",{}).get("ramp_s",cfg.T_RAMP)  # rate sensitivity check introduces a longer ramp
    m=(ie>0.05*ie.max()) & (t<=t_ramp + 1e-9); q=100* ke/np.maximum(ie,1e-12)
    out=dict(mean_ramp_pct=float(q[m].mean()), end_pct=float(q[-1]), peak_pct=float(q[m].max()))
    out["ok"]=bool(out["mean_ramp_pct"]<KE_IE_MEAN and out["end_pct"]<KE_IE_END)
    if anchors_file(info):   # anchors: settled in the hold (frames after the ramp) + KE/IE at the end
        hold=[ f for f , tf in enumerate(res["times"]) if tf>= t_ramp - 1e-9]
        tips,_= anchor_tips(info,res,hold)
        out["anchor_tip_motion_hold_mm"]=float(np.linalg.norm(tips-tips[-1],axis=2).max())
        if "hist_anc_ALLKE" in res.files:
            out["anchor_KE_IE_end_pct"]=float(100*res["hist_anc_ALLKE"][-1,1]/ max(res["hist_anc_ALLIE"][-1,1],1e-12))
        out["ok"]=bool(out["ok"] and out["anchor_tip_motion_hold_mm"]<TIP_SETTLE_MM)
    return out

def anchor_deflection(info,res):
    """Mean over the anchors of the tip distance from its planned final position: how much the compliant anchors gave way (descriptive)"""
    tips,final=anchor_tips(info,res)
    return float(np.linalg.norm(tips[0]-final,axis=1).mean())

def fea_qois(a_id,tag,run,plan, b07, polydata_from_arrays):
    """ All Quantities of Interest (QoIs) of ONE FEA RUN (tag = '<level>' or '<level>_<variant>'), on its own mesh: no mapping onto the SSM vertices. """
    level=tag.split("_")[0]; dep = run / f"deploy_{tag}"; name = f"{a_id}_{tag}"
    info=json.loads((dep/f"{name}_info.json").read_text())
    plan=fc.axis_plan(plan,info.get("axis_offset_e1e2_mm"))   # ring QoIs and the radial/lateral force about the CUFF axis of this run
    res=np.load(dep/f"{name}_results.npz"); labels=res["labels"]; U=res["U"][-1].astype(float)
    mp=np.load(run/"mesh"/f"{a_id}_wall_{level}_map.npz")
    r=vtk.vtkXMLUnstructuredGridReader(); r.SetFileName(str(run/"mesh"/f"{a_id}_wall_{level}.vtu")); r.Update(); g=r.GetOutput()
    X=vtk_to_numpy(g.GetPoints().GetData()).astype(float); tv_all=vtk_to_numpy(g.GetPointData().GetArray("tv_annulus"))
    cn,F=mp["coarse_nodes"],mp["faces"] # the FEA inner faces its nodes and triangles
    P0 = X[cn]; Ui = U[np.searchsorted(labels, cn + 1)]; P1 = P0 + Ui; tv = tv_all[cn] == 1
    A = 0.5 * np.linalg.norm(np.cross(P0[F[:, 1]] - P0[F[:, 0]], P0[F[:, 2]] - P0[F[:, 0]]), axis=1)
    u2=(np.linalg.norm(Ui,axis=1)**2)[F].mean(1) # |u|^2 averaged on each triangle

    q=dict(strain_energy_mJ=float(res["hist_wall_ALLIE"][-1,1]),
           cavity_change_ml=float((mesh_volume(P1,F)-mesh_volume(P0,F))/1000.0),
           u_rms_area_mm=float(np.sqrt((A*u2).sum()/A.sum())),
           max_disp_mm=float(np.linalg.norm(U[labels<=info["n_nodes"]],axis=1).max())
           )
    fr = plan["frame"]; c, n = np.array(fr["centre"]), np.array(fr["normal"])
    E = cuff_points(info, run)
    ie = np.searchsorted(labels, info["expander_node0"] + 1 + np.arange(len(E)))
    Xe = E + U[ie]; rel = Xe - c; rad = rel - np.outer(rel @ n, n); rad /= np.linalg.norm(rad, axis=1, keepdims=True)
    q["radial_force_N"]=float(abs(np.einsum("ij,ij->i", res["RF"][-1].astype(float)[ie], rad).sum()))
    F_cuff=res["RF"][-1].astype(float)[ie].sum(0) # net reaction on the cuff nodes: its in-plane part = how far the axis is from force balance
    q["lateral_force_N"]=float(np.hypot(F_cuff@np.array(fr["e1"]), F_cuff@ np.array(fr["e2"])))
    if anchors_file(info):
        q["anchor_tip_defl_mm"]=anchor_deflection(info,res)
        if "hist_anc_ALLIE" in res.files:
            q[ANCHOR_QOI[0]]=float(res["hist_anc_ALLIE"][-1,1])
    if "cpress_last_labels" in res.files: # contact (05 extension): area of the inner-surface nodes pressed at the end
        Av=np.bincount(F.ravel(), np.repeat(A/3,3), minlength=len(cn))
        pressed = dict(zip(res["cpress_last_labels"].tolist(), (res["cpress_last"] > 1e-6).tolist()))
        q["contact_area_mm2"]=float(sum(Av[i] for i , nd in enumerate(cn+1) if pressed.get(int(nd),False)))
        ever= res["cpress_ever_labels"]; ever = ever[ever < 2_000_000]
        q["detached_pct"]=float(100*np.mean([not pressed.get(int(nd),False) for nd in ever])) if len(ever) else 0.0
    br=fc.band_ring(P0[tv],P1[tv],plan)
    q.update(ring_mean_mm=br["mean_r"],ring_SD_mm=br["sd_r"],ring_stretch_pct=br["stretch_pct"]) #DESCRIPTIVE ONLY
    qs=quasi_static(res,info)
    anc = (f"; anchors: tip motion in the hold {qs['anchor_tip_motion_hold_mm']:.3f} mm" + (f", KE/IE end {qs['anchor_KE_IE_end_pct']:.2f} %" if "anchor_KE_IE_end_pct" in qs else "")
           if "anchor_tip_motion_hold_mm" in qs else "")
    print(f"  {tag:22s} quasi-static: wall KE/IE mean over ramp {qs['mean_ramp_pct']:.2f} %, end {qs['end_pct']:.2f} %, peak {qs['peak_pct']:.2f} %{anc} "
          f"-> {'OK' if qs['ok'] else 'NOT QUASI-STATIC (wall mean < %g %%, end < %g %%; tips < %g mm)' % (KE_IE_MEAN, KE_IE_END, TIP_SETTLE_MM)}")
    return dict(q=q, info=info, mp=mp, qs=qs)

def judged(runs):
    """The judged runs' common QoIs the anchor QoI joins only when all runs have compliant anchors) """
    return QOIS + ([ANCHOR_QOI] if all(ANCHOR_QOI[0] in r["q"] for r in runs) else [])


def change(k, new, ref):
    """Change used for the verdict: % of the reference value (every judged QoI is integral and judged in %)."""
    return 100 * (new - ref) / abs(ref)

def limit(k):
    """Tolerance of QoI k in its verdict unit. One value for all judged QoIs (5%); k stays an argument so a per-QoI tolerance can be added later"""
    return CRIT

def unit(k):
    """unit of verdict of QoI k (all %)"""
    return "%"

def gci3(phi,N,Fs=1.25):
    """Three-grid Grid Convergence Index (GCI);phi and N ordered fine, medium, coarse. Relative to the fine value (%). """
    f1,f2,f3=phi
    r21,r32= (N[0]/N[1])**(1/3),(N[1]/N[2])**(1/3)
    e21,e32=f2-f1,f3-f2
    out=dict(r21=float(r21),r32=float(r32),r_ok=bool(min(r21,r32)>=1.3),
             phi=list(map(float,phi)))
    spread=100*Fs*max(abs(f1-f2),abs(f1-f3))/abs(f1) # uncertainty used when the GCI is not applicable
    if abs(e21)<1e-12 or abs(e32)<1e-12:
        return dict(out, type="converged", R=np.nan, p=np.nan, ext=f1, uncertainty_pct=0.0, method="no change")
    R=e21/e32 
    if R<0:
        return dict(out,type="oscillatory",R=float(R),p=np.nan, ext=np.nan, uncertainty_pct=float(spread), method="spread band")
    if R>=1:
        return dict(out, type="diverging",R=float(R),p=np.nan, ext=np.nan,uncertainty_pct=float(spread),method="spread band")
    p=2.0
    for _ in range(100):
        p=abs(np.log(abs(e32/e21))+np.log((r21**p -1)/(r32**p-1)))/np.log(r21)  # observed order above the formal one = the coarse grid is not yet asymptotic: use the formal order (conservative, Roache 1998 / ASME V&V 20)
    p_used=min(p,P_FORMAL)
    ext=(r21**p_used*f1-f2)/(r21**p_used -1)
    gci=100*Fs*abs(f1-f2)/((r21**p_used -1)*abs(f1))
    return dict(out, type="converging", R=float(R), p=float(p), p_used=float(p_used), ext=float(ext), uncertainty_pct=float(gci),
                method="GCI" if p <= P_FORMAL else f"GCI (p capped at {P_FORMAL:g})")

def style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6); ax.set_axisbelow(True); ax.set_facecolor(SURFACE)


def common(a):
    plan = json.loads((cfg.PLANS / f"{a.id}_plan.json").read_text())
    sys.path.append(str(cfg.SHAPE_SCRIPTS))
    b07 = load_module("07b_deploy_plan", cfg.SHAPE_SCRIPTS); mio = load_module("mesh_io", cfg.SHAPE_SCRIPTS)
    return plan, b07, mio.polydata_from_arrays

def r_sequence(vals):
    """Convergence ratio R=eps21/eps32 of every consecutive triplet (fine medium and coarse) of a series ordered finest first:
    0<R<1 converging, R<0 oscillating, R>=1 diverging (the step between the finer grids is larger than between the coarser ones)."""
    out=[]
    for i in range(len(vals)-2):
        e21,e32=vals[1+i]-vals[i], vals[i+2]-vals[i+1]
        out.append(float(e21/e32) if abs(e32)>1e-12 else float("nan"))
    return out

def conv_type(R):
    if not np.isfinite(R):
        return "no change"
    return "converging" if 0<R<1 else ("oscillating" if R<0 else "diverging")

SHORT = {"strain_energy_mJ": "strain energy", "radial_force_N": "radial force", "cavity_change_ml": "cavity volume", "u_rms_area_mm": "RMS displ.", "anchor_energy_mJ": "anchor energy"}
R_FILL = {"converging": "#d7e8fb", "oscillating": "#ecebe6", "diverging": "#fde3d3", "no change": "#ffffff"}

def convergence_figure(path, out, levels, Q, unc, title=None, kind="judged"):
    """Grid convergence, one panel per QoI. x = degrees of freedom of the FEA wall (log scale, finer to the right), y = deviation from the finest level
    in %. Bands: tolerance, and (conservative rule) the tolerance left after the finest level's own uncertainty U. The bar on the finest level = U, the
    dashed orange line = Richardson extrapolation to h -> 0 (where the GCI applies). Below: values, R of every triplet, U."""
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    sel = out["selection"]["selected"]; cons = kind == "judged" and out["selection"].get("rule_name") == "conservative"
    ks = [k for k, _ in Q]; n = len(ks); dev = out["deviation_from_finest"]; nl = len(levels)
    D = np.array([out["mesh"][L]["dofs"] for L in levels], float)                                        # x position of every level (finest = largest)
    col = {L: LEVEL_RAMP[i] for i, L in enumerate(levels)}
    ncol = 4 if n == 4 else 3; nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(4.8 * ncol + 0.4, 3.4 * nrow + 0.33 * (n + 6) + 0.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.4 * nrow, 0.33 * (n + 6)], hspace=0.22)
    sg = gs[0].subgridspec(nrow, ncol, hspace=0.75, wspace=0.30)
    shared_y = max(1.6 * CRIT, 1.25 * max(abs(dev[k][L]) for k in ks for L in levels[1:]))
    for i, (k, name) in enumerate(Q):
        ax = fig.add_subplot(sg[i // ncol, i % ncol]); style(ax)
        v = np.array([out["values"][L][k] for L in levels], float); d = 100 * (v / v[0] - 1); g = out["gci_fine_triplet"][k]; u = unc[k]
        tol = limit(k); extra = u if cons else 0.0
        Y = shared_y if kind == "judged" else max(1.6 * tol, 1.3 * np.abs(d).max(), 1.15 * u)
        ax.axhspan(-tol, tol, color="#e6effb", lw=0, zorder=0)
        if cons and tol - u > 0:
            ax.axhspan(-(tol - u), tol - u, color=BAND, lw=0, zorder=0)
        ax.axhline(0, color=BASE, lw=1, zorder=1)
        ax.plot(D, d, color=INK2, lw=1.2, zorder=2)
        ax.errorbar(D[0], 0, yerr=u, color=INK, lw=1.8, capsize=5, zorder=3)
        if np.isfinite(g.get("ext", np.nan)):
            de = 100 * (g["ext"] / v[0] - 1)
            ax.axhline(de, color=ORANGE, lw=1.4, ls="--", zorder=2)
            ax.text(D[-1] / 1.2, de - 0.04 * Y, "extrapolated value (h → 0)", color=ORANGE, fontsize=7.5, va="top")
        for j, L in enumerate(levels):
            big = L == sel
            ax.scatter(D[j], d[j], s=190 if big else 95, color=col[L], edgecolor=ORANGE if big else "white", linewidth=2.6 if big else 1.4, zorder=5)
            if j:
                bad = abs(d[j]) + extra >= tol
                ax.text(D[j], d[j] + (1 if d[j] >= 0 else -1) * 0.11 * Y, f"{d[j]:+.1f}", ha="center", va="bottom" if d[j] >= 0 else "top",
                        fontsize=8.5, fontweight="bold", color=RED if bad else INK)
        ax.set_xscale("log"); ax.set_xlim(D[-1] / 1.3, D[0] * 1.3); ax.set_ylim(-Y, Y); ax.minorticks_off()
        ax.set_xticks(D); ax.set_xticklabels([f"{L}\n{x / 1000:.0f}k" for L, x in zip(levels, D)], fontsize=7.8)
        if i % ncol == 0:
            ax.set_ylabel("deviation from " + levels[0] + " (%)", fontsize=8.5, color=INK2)
        if i // ncol == nrow - 1 or i + ncol >= n:
            ax.set_xlabel("FEA wall degrees of freedom (log scale)  →  finer", fontsize=8, color=INK2)
        how = f"{conv_type(g['R'])} · R {g['R']:.2f} · " + (f"GCI {u:.2f} %" if g["method"].startswith("GCI") else f"spread band {u:.2f} %") if np.isfinite(g["R"]) else "no change between the fine levels"
        ax.text(0, 1.095, textwrap.fill(name, 40), transform=ax.transAxes, color=INK, fontsize=10, fontweight="bold", va="bottom")
        ax.text(0, 1.025, how, transform=ax.transAxes, color=INK2, fontsize=8.3, va="bottom")
    # table: values, R of every triplet, U of the finest level
    at = fig.add_subplot(gs[1]); at.axis("off"); at.set_xlim(0, 1); at.set_ylim(-3.6, n + 2.4)
    nr = nl - 2; x0 = 0.30; wv = 0.075; wr = 0.085; xs = [x0 + j * wv for j in range(nl)]; xr = [x0 + nl * wv + 0.015 + j * wr for j in range(nr)]; xu = xr[-1] + wr + 0.01
    at.text(0.0, n + 1.5, "QoI", fontsize=8.5, fontweight="bold", color=INK2, va="center")
    for j, L in enumerate(levels):
        at.text(xs[j] + wv / 2, n + 1.5, L, fontsize=8.5, fontweight="bold", color=INK2, ha="center", va="center")
    for j in range(nr):
        at.text(xr[j] + wr / 2, n + 1.5, f"R {levels[j]}-{levels[j + 2]}", fontsize=8.5, fontweight="bold", color=INK2, ha="center", va="center")
    at.text(xu + 0.04, n + 1.5, f"U({levels[0]})", fontsize=8.5, fontweight="bold", color=INK2, ha="center", va="center")
    for i, (k, name) in enumerate(Q):
        y = n - 0.5 - i + 0.5; v = [out["values"][L][k] for L in levels]; g = out["gci_fine_triplet"][k]
        at.text(0.0, y, name, fontsize=8.3, color=INK, va="center")
        for j in range(nl):
            at.text(xs[j] + wv / 2, y, f"{v[j]:.4g}", fontsize=8.3, color=INK, ha="center", va="center", fontweight="bold" if levels[j] == sel else "normal")
        for j, R in enumerate(r_sequence(v)):
            at.add_patch(plt.Rectangle((xr[j] + 0.004, y - 0.42), wr - 0.008, 0.84, color=R_FILL[conv_type(R)], lw=0))
            at.text(xr[j] + wr / 2, y, f"{R:.2f}" if np.isfinite(R) else "-", fontsize=8.3, color=INK, ha="center", va="center")
        at.text(xu + 0.04, y, f"{unc[k]:.2f} % " + ("GCI" if g["method"].startswith("GCI") else "band"), fontsize=8.3, color=INK, ha="center", va="center")
    notes = ([f"Bands: light blue = ±{CRIT:g} % tolerance around {levels[0]};  darker blue = tolerance left after the uncertainty U of {levels[0]} (a level passes when its marker is inside it)."
              if cons else f"Band: light blue = ±{CRIT:g} % around {levels[0]} (reference only, these QoIs are not judged)."]) + [
             f"Bar on {levels[0]} = U;  dashed orange line = Richardson extrapolation to h → 0 (only where R is in (0,1), p capped at {P_FORMAL:g});  bold value = selected level {sel}.",
             "R = (φ(i+1) - φ(i)) / (φ(i+2) - φ(i+1)) per triplet of consecutive levels:  blue 0<R<1 converging  ·  grey R<0 oscillating  ·  orange R≥1 diverging (finer step larger than coarser step).",
             f"U({levels[0]}) = GCI (Celik 2008, Fs 1.25) if the finest triplet converges, otherwise spread band = 1.25 x largest deviation of the three finest levels; non-nested meshes carry ~1 % generation noise."]
    for m_, t_ in enumerate(notes):
        at.text(0.0, -0.45 - 0.78 * m_, t_, fontsize=7.6, color=INK2, va="top")
    vt = f" ({out['variant']})" if out.get("variant") else ""
    rule = out["selection"].get("rule_name", "")
    fig.suptitle(title or (f"{out['id']}{vt}: grid convergence of the judged QoIs; selected level {sel} ({rule} rule: |deviation from {levels[0]}| + U({levels[0]}) < {CRIT:g} %)"
                           if rule == "conservative" else f"{out['id']}{vt}: grid convergence of the judged QoIs; selected level {sel}"),
                 x=0.01, y=0.995, ha="left", color=INK, fontsize=12.5, fontweight="bold")
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)


def selection_figure(path, out, levels, Q):
    """Which mesh? A: per level the worst QoI as deviation + finest-level uncertainty U (stacked, % of the tolerance) against the 100 % limit;
    B: every QoI x level; C: what each level costs."""
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    order = levels[::-1]; col = {L: LEVEL_RAMP[i] for i, L in enumerate(levels)}; sel = out["selection"]; dev = out["deviation_from_finest"]
    lab = dict(Q); ks = [k for k, _ in Q]; cons = sel.get("rule_name") == "conservative"
    U = {k: (out["gci_fine_triplet"][k]["uncertainty_pct"] if cons else 0.0) for k in ks}
    fig, axs = plt.subplots(1, 3, figsize=(19, 6.0), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1.15, 1.45, 0.8], wspace=0.55))

    ax = axs[0]; style(ax)                                              # A: stacked bars
    x = np.arange(len(order)); top = 100.0
    ax.axhspan(0, 100, color="#e6effb", lw=0); ax.axhline(100, color=INK2, lw=1, ls="--")
    ax.text(len(order) - 1, 103, f"tolerance ({CRIT:g} %)", color=INK2, fontsize=8.5, ha="center")      # above the empty reference slot
    for xi, L in zip(x, order):
        if L == levels[0]:
            ax.text(xi, 4, "reference\n(finest)", ha="center", color=INK2, fontsize=8.5); continue
        k = sel["worst_qoi"][L]; a = 100 * abs(dev[k][L]) / limit(k); b = 100 * U[k] / limit(k)
        isel = L == sel["selected"]
        ax.bar(xi, a, width=0.6, color=col[L], edgecolor=ORANGE if isel else col[L], linewidth=3 if isel else 0)
        ax.bar(xi, b, width=0.6, bottom=a, color="#cde2fb", edgecolor=ORANGE if isel else col[L], hatch="///", lw=3 if isel else 0.8)
        bad = a + b >= 100; top = max(top, a + b)
        ax.text(xi, a + b + 4, (f"{abs(dev[k][L]):.2f} + {U[k]:.2f}\n= {abs(dev[k][L]) + U[k]:.2f} %" if cons else f"{abs(dev[k][L]):.2f} %"),
                ha="center", va="bottom", color=RED if bad else INK, fontsize=9.5, fontweight="bold")
        ax.text(xi, -4, SHORT.get(k, lab[k].split(" (")[0]), ha="center", va="top", color=INK2, fontsize=8)
    xs = order.index(sel["selected"])
    ax.text(xs, top * 1.13, f"selected: {sel['selected']}" + (f"\n(plain rule: {sel['selected_plain_rule']})" if cons else ""), ha="center", color=ORANGE, fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"{L}\n{out['mesh'][L]['tets']:,} tets" for L in order], fontsize=8.5); ax.tick_params(axis="x", pad=16)
    ax.set(ylim=(-12, max(top, 100) * 1.45), ylabel="worst QoI: deviation" + (" + U(" + levels[0] + ")" if cons else "") + "  (% of the tolerance)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=INK2, label="|deviation from " + levels[0] + "|"),
                       plt.Rectangle((0, 0), 1, 1, facecolor="#cde2fb", edgecolor=INK2, hatch="///", label="finest-level uncertainty U")] if cons else
              [plt.Rectangle((0, 0), 1, 1, color=INK2, label="|deviation from " + levels[0] + "|")], frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Which mesh is good enough?", loc="left", color=INK, fontsize=11, fontweight="bold")

    ax = axs[1]                                                        # B: every QoI x level
    cols = [L for L in order if L != levels[0]]; ramp = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5"]
    for i, k in enumerate(ks):
        for j, L in enumerate(cols):
            tot = (abs(dev[k][L]) + U[k]) / limit(k); fail = tot >= 1
            ax.add_patch(plt.Rectangle((j, len(ks) - 1 - i), 0.96, 0.92, color="#fde3e3" if fail else ramp[min(int(tot * 4), 4)], ec=RED if fail else GRID, lw=1.8 if fail else 0.8))
            ax.text(j + 0.48, len(ks) - 1 - i + 0.55, ("FAIL " if fail else "") + f"{dev[k][L]:+.1f} %", ha="center", va="center", color=RED if fail else INK, fontsize=10, fontweight="bold")
            if cons:
                ax.text(j + 0.48, len(ks) - 1 - i + 0.25, f"+ U {U[k]:.1f}  =  {abs(dev[k][L]) + U[k]:.1f}", ha="center", va="center", color=INK2, fontsize=7.5)
    ax.set_xlim(0, len(cols)); ax.set_ylim(0, len(ks)); ax.set_xticks(np.arange(len(cols)) + 0.48); ax.set_xticklabels(cols, fontsize=10, fontweight="bold")
    ax.set_yticks(np.arange(len(ks)) + 0.46); ax.set_yticklabels([textwrap.fill(lab[k], 24) for k in ks][::-1], fontsize=8.5); ax.tick_params(length=0, colors=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Each QoI: deviation from {levels[0]}" + (" + finest-level uncertainty" if cons else "") + "\nshading = share of the tolerance, red frame = beyond it",
                 loc="left", color=INK, fontsize=11, fontweight="bold")

    ax = axs[2]; style(ax)                                              # C: cost
    d = [out["mesh"][L]["dofs"] for L in order]
    ax.bar(x, d, width=0.6, color=[col[L] for L in order])
    for xi, vv in zip(x, d):
        ax.text(xi, vv * 1.02, f"{vv / 1000:.0f}k", ha="center", color=INK, fontsize=9.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=10); ax.set(ylabel="degrees of freedom of the FEA wall", ylim=(0, max(d) * 1.15))
    ax.set_title("What each level costs", loc="left", color=INK, fontsize=11, fontweight="bold")
    failing = sorted(set(lab[sel["worst_qoi"][L]].split(" (")[0] for L in levels[1:] if sel["worst_fraction_of_limit"][L] >= 1))
    vt = f" ({out['variant']})" if out.get("variant") else ""
    fig.suptitle(f"{out['id']}{vt}: FEA mesh selection on FEA results only -> {sel['selected']}"
                 + (f" (coarser levels fail on {', '.join(failing)})" if failing else " (every level within tolerance)"),
                 x=0.01, ha="left", color=INK, fontsize=12.5, fontweight="bold")
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)

def mesh_mode(a,run):
    plan,b07, pfa=common(a)
    levels = a.levels; tags =[f"{L}_{a.variant}" if a.variant else L for L in levels]
    print(f"mesh study on {', '.join(tags)}")
    runs={L: fea_qois(a.id,t,run,plan,b07, pfa) for L,t in zip(levels, tags)}
    Q=judged(list(runs.values())); ks=[k for k,_ in Q]
    N=[zone_tets(run,a.id,L) for L in levels] # tets in the device zone (refinement ratio where the device acts)
    out = {"id": a.id, "variant": a.variant, "levels": levels, "criterion_pct_of_finest": CRIT,
           "qois_judged": ks, "qois_descriptive": [k for k, _ in DESCRIPTIVE if all(k in runs[L]["q"] for L in levels)],
           "descriptive": {k: {L: runs[L]["q"][k] for L in levels} for k, _ in DESCRIPTIVE if all(k in runs[L]["q"] for L in levels)},
           "refinement_ratios": {f"{levels[i + 1]}->{levels[i]}": float((N[i] / N[i + 1]) ** (1 / 3)) for i in range(len(levels) - 1)},
           "mesh":{L:{"tets":runs[L]["info"]["n_tets"],"tets_zone":N[i],"nodes":runs[L]["info"]["n_nodes"],"dofs":3*runs[L]["info"]["n_nodes"],"inner_vertices": int(len(runs[L]["mp"]["coarse_nodes"]))} for i,L in enumerate(levels)},
           "values": {L: runs[L]["q"] for L in levels}, "quasi_static": {L: runs[L]["qs"] for L in levels},
           "gci_fine_triplet": {k: gci3([runs[L]["q"][k] for L in levels[:3]], N[:3]) for k in ks}}
    if len(levels) >= 4:
        out["gci_coarse_triplet"] = {k: gci3([runs[L]["q"][k] for L in levels[1:4]], N[1:4]) for k in ks}
    f0={k:runs[levels[0]]["q"][k] for k in ks}
    out["deviation_from_finest"] = {k: {L: float(change(k, runs[L]["q"][k], f0[k])) for L in levels[1:]} for k in ks}
    # uncertainty of the finest level in the QoI's verdict unit (all % now; an absolute-unit QoI would be converted here)
    unc={k: out["gci_fine_triplet"][k]["uncertainty_pct"]*(abs(f0[k])/100 if unit(k)=="mm" else 1) for k in ks}
    out["verdict"] = {k: bool(unc[k] < limit(k)) for k in ks}
    # MESH SELECTION RULE : the COARSEST level from which every judged QoI stays within its limit of the finest level
    # (that level and every finer one must pass); the GCI quantifies how uncertain the finest level itself still is.
    dev=out["deviation_from_finest"]
    # conservative rule as default option: devation from the finest + finest level's own uncertainty must stay below the acceptance tolerance criterion,
    #the selected level is within tolerance of the converged (h -> 0) value, not only of a finest level that still carries error
    extra={k: (unc[k] if a.rule=="conservative" else 0.0) for k in ks}
    ratio={L: max((abs(dev[k][L])+extra[k])/limit(k) for k in ks) for L in levels[1:]}
    worst_k={L:max(ks,key=lambda k: (abs(dev[k][L])+extra[k])/ limit(k)) for L in levels[1:]}
    plain={L: max(abs(dev[k][L])/limit(k) for k in ks) for L in levels[1:]}    # the plain rule, reported for reference
    selected=levels[0]
    for L in levels[1:]:
        if ratio[L]<1.0:
            selected=L
        else:
            break
    sel_plain = levels[0]
    for L in levels[1:]:
        if plain[L] < 1.0:
            sel_plain = L
        else:
            break
    out["selection"]=dict(rule=(f"coarsest level whose QoIs (and those of every finer level) satisfy |deviation from the finest| "
                                f"+ finest-level uncertainty < {CRIT:g} %" if a.rule == "conservative" else
                                f"coarsest level whose QoIs (and those of every finer level) all stay within {CRIT:g} % of the finest level"),
                            rule_name=a.rule, selected_plain_rule=sel_plain,
                            worst_fraction_of_limit=ratio, worst_qoi=worst_k, selected=selected)
    dst=run/"compare";dst.mkdir(exist_ok=True); sfx= f"_{a.variant}" if a.variant else ""
    (dst / f"mesh_convergence_fea{sfx}.json").write_text(json.dumps(out, indent=1))
    lab = dict(Q)
    print("worst deviation from the finest level, per level: " + "; ".join(
        f"{L} {abs(dev[worst_k[L]][L]):.2f} {unit(worst_k[L])} ({lab[worst_k[L]]})" for L in levels[1:]))
    print(f"SELECTED MESH LEVEL: {selected}   (rule '{a.rule}'" + (f"; plain |deviation| < {CRIT:g} % rule would give {sel_plain})" if a.rule == "conservative" else ")"))
    print("refinement ratios " + ", ".join(f"{k} {v:.3f}" for k, v in out["refinement_ratios"].items())
          + ("  (>= 1.3 recommended by Celik 2008: " + ("met" if min(out["refinement_ratios"].values()) >= 1.3 else "NOT MET for the coarsest pair") + ")"))
    print(f"{'QoI (FEA mesh only)':38s} {' / '.join(levels):28s}   fine triplet: type, R, p, method, uncertainty | deviation of {selected} | verdict")
    for k in ks:
        g = out["gci_fine_triplet"][k]; vals = " / ".join(f"{runs[L]['q'][k]:.4g}" for L in levels)
        d_sel = dev[k].get(selected, 0.0)
        print(f"  {lab[k]:36s} {vals:28s}   {g['type']:11s} R {g['R']:6.2f} p {g['p']:5.2f} {g['method']:20s} {unc[k]:5.2f} {unit(k):2s} | "
              f"{d_sel:+.2f} {unit(k)} | {'PASS' if out['verdict'][k] else 'FAIL'}")

    # descriptive only QoIs: the same convergence analysis (GCI/spread band, deviation from the finest for interpretation only and completeness)
    # never used as metrics officially to pick DoFs
    Qd=[(k,lb) for k,lb in DESCRIPTIVE if all(k in runs[L]["q"] and np.isfinite(runs[L]["q"][k])for L in levels) and abs(runs[levels[0]]["q"][k])>1e-9]
    kd = [k for k, _ in Qd]; fd0 = {k: runs[levels[0]]["q"][k] for k in kd}
    gd = {k: gci3([runs[L]["q"][k] for L in levels[:3]], N[:3]) for k in kd}
    devd = {k: {L: float(change(k, runs[L]["q"][k], fd0[k])) for L in levels[1:]} for k in kd}
    uncd = {k: gd[k]["uncertainty_pct"] for k in kd}
    out["descriptive_convergence"] = {k: dict(gci_fine_triplet=gd[k], deviation_from_finest=devd[k]) for k in kd}
    (dst / f"mesh_convergence_fea{sfx}.json").write_text(json.dumps(out, indent=1))
    for k, lb in Qd:
        print(f"  (descriptive) {lb:36s} " + " / ".join(f"{runs[L]['q'][k]:.4g}" for L in levels)
              + f"   {gd[k]['type']:11s} R {gd[k]['R']:6.2f} {gd[k]['method']:20s} {uncd[k]:6.2f} % | {devd[k].get(selected, 0.0):+.2f} % (not judged)")
    for k, lb in [(k, lb) for k, lb in DESCRIPTIVE if k not in kd and all(k in runs[L]["q"] for L in levels)]:
        print(f"  (descriptive) {lb:36s} " + " / ".join(f"{runs[L]['q'][k]:.4g}" for L in levels) + "   (finest value 0: no % analysis)")
    selection_figure(dst / f"mesh_selection_fea{sfx}.png", out, levels, Q)
    convergence_figure(dst / f"mesh_convergence_fea{sfx}.png", out, levels, Q, unc)
    if Qd:
        vt=f"({a.variant})" if a.variant else ""
        convergence_figure(dst / f"mesh_convergence_descriptive_fea{sfx}.png", dict(out, gci_fine_triplet=gd, deviation_from_finest=devd), levels, Qd, uncd,
                           title=f"{a.id}{vt}: DESCRIPTIVE QoIs, same grid-convergence analysis (reported, NOT evaluated for numerical credibility); orange = level selected on the judged QoIs ({selected})",
                           kind="descriptive")

def flat(d,pre=""):
    """Flatten a nested info dict into {'a.b' value }"""
    out={}
    for k, v in d.items():
        if isinstance(v,dict):
            out.update(flat(v,f"{pre}{k}."))
        elif not isinstance(v,list):
            out[f"{pre}{k}"]=v
    return out

def what_changed(base_info, other_info):
    """the parameters that differ between two runs, read from their info files"""
    b,o=flat(base_info),flat(other_info)
    keys=[k for k in sorted(set(b)| set(o)) if not any(s in k.split(".")[-1] for s in IGNORE_IN_DIFF) and b.get(k)!=o.get(k)]
    diffs = [f"{k} {b.get(k)} -> {o.get(k)}" for k in keys]
    if base_info.get("device") != other_info.get("device"):                       # different device file = different sizing/landing
        diffs.append(f"device {Path(str(base_info.get('device'))).parent.name} -> {Path(str(other_info.get('device'))).parent.name}")
    SAME = {"device": "axis_rule", "axis_offset_e1e2_mm": "axis_rule"}           # a new axis rule comes with its own device folder: ONE factor
    groups = {SAME.get(g, g) for g in (d.split(" ")[0].split(".")[0] for d in diffs)}  # 'sub_support.k' and 'sub_support.on' = one parameter
    if len(groups) > 1:
        print(f"  WARNING: {len(groups)} parameters differ ({', '.join(sorted(groups))}): the effect cannot be attributed to one of them")
    return ", ".join(diffs) or "no parameter difference found in the info files"

def sens_figure(path, a, out, Q):
    """One row per variant (what changed, read from the info files), one column per QoI: signed change vs the base run on the same mesh.
    Decisive columns (shape + anchors) framed red when beyond tolerance; mechanical columns are context; right: verdict per variant."""
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    ks = [k for k, _ in Q if k in SHAPE_QOIS] + [k for k, _ in Q if k == ANCHOR_QOI[0]] + [k for k, _ in Q if k not in SHAPE_QOIS and k != ANCHOR_QOI[0]]
    lab = dict(Q); tags = list(out["variants"]); nr = len(tags) + (1 if "mesh_uncertainty" in out else 0)
    H = 2.6 + 1.1 * nr; fig = plt.figure(figsize=(19, H), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[len(ks), 2.6], wspace=0.05)
    ax = fig.add_subplot(gs[0]); ramp = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5"]
    rows = [("mesh_unc", f"mesh uncertainty |{a.base} - {a.mesh_ref}|\n(context: smaller effects are not resolvable)")] if "mesh_uncertainty" in out else []
    rows += [(t, t + "\n" + textwrap.fill(out["variants"][t]["changed"], 38)) for t in tags]
    for i, (t, _) in enumerate(rows):
        yy = len(rows) - 1 - i
        for j, k in enumerate(ks):
            decisive = k in SHAPE_QOIS
            v = out["mesh_uncertainty"][k] if t == "mesh_unc" else out["variants"][t]["change"][k]
            fr_ = abs(v) / limit(k); fail = fr_ >= 1 and t != "mesh_unc"
            c = "#f4f3ef" if t == "mesh_unc" else ("#fde3e3" if fail and decisive else ramp[min(int(fr_ * 4), 4)] if decisive else "#efeee9")
            ax.add_patch(plt.Rectangle((j, yy), 0.96, 0.9, color=c, ec=RED if (fail and decisive) else GRID, lw=1.8 if (fail and decisive) else 0.8))
            txt = (f"{v:+.2f} mm" if unit(k) == "mm" else f"{v:+.1f} %") if t != "mesh_unc" else (f"{v:.2f} mm" if unit(k) == "mm" else f"{v:.1f} %")
            ax.text(j + 0.48, yy + 0.45, txt, ha="center", va="center", fontsize=10, fontweight="bold",
                    color=RED if (fail and decisive) else (INK if decisive and t != "mesh_unc" else INK2))
    ax.set_xlim(0, len(ks)); ax.set_ylim(0, len(rows)); ax.set_xticks(np.arange(len(ks)) + 0.48)
    ax.set_xticklabels([("SHAPE\n" if k in SHAPE_QOIS else "mechanical\n") + lab[k].replace(" (", "\n(") for k in ks],
                       fontsize=8.5); ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(rows)) + 0.45); ax.set_yticklabels([r[1] for r in rows][::-1], fontsize=8.5); ax.tick_params(length=0, colors=INK2)
    for s in ax.spines.values():
        s.set_visible(False)

    ax = fig.add_subplot(gs[1]); style(ax, "x")                         # verdict per variant
    yv = np.arange(len(tags))[::-1] + 0.45
    for y_, t in zip(yv, tags):
        w = 100 * out["variants"][t]["worst_share_of_tolerance"]; rel = out["variants"][t]["relevant"]
        qs = out["quasi_static"][t]; wb = min(w, 200)                          # bars beyond 200 % are clipped, the label keeps the value
        ax.barh(y_, wb, height=0.5, color=RED if rel else ORANGE)
        ax.text(wb + 4, y_ + 0.12, ("RELEVANT" if rel else "not relevant") + f"  ({w:.0f} %)", va="center", fontsize=9, fontweight="bold",
                color=RED if rel else INK)
        ax.text(wb + 4, y_ - 0.17, f"KE/IE {qs['mean_ramp_pct']:.1f} % {'ok' if qs['ok'] else 'NOT quasi-static'}", va="center", fontsize=7.5,
                color=INK2 if qs["ok"] else RED)
    ax.axvline(100, color=INK2, lw=1.2, ls="--"); ax.text(100, len(rows) - 0.25, " tolerance", color=INK2, fontsize=8)
    ax.set(xlim=(0, 330), ylim=(0, len(rows))); ax.set_yticks([]); ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xlabel("worst decisive change, % of its tolerance", fontsize=8.5, color=INK2)
    ax.set_title("Verdict", loc="left", color=INK, fontsize=10.5, fontweight="bold")
    n_rel = sum(out["variants"][t]["relevant"] for t in tags)
    fig.suptitle(f"{a.id}: sensitivity of the deployed shape to modelling parameters (base {a.base}; tolerance {CRIT:g} % on shape QoIs, "
                 f"integral QoIs) -> {n_rel} of {len(tags)} parameter(s) relevant",
                 x=0.01, y=1 + 0.8 / H, ha="left", va="bottom", color=INK, fontsize=12, fontweight="bold")   # above the column headers
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)

def sens_mode(a,run):
    plan, b07,pfa=common(a)
    print(f"sensitivity: base {a.base}; variants {','.join(a.variants)}" +(f"; mesh reference {a.mesh_ref}" if a.mesh_ref else ""))
    base=fea_qois(a.id,a.base,run,plan,b07,pfa)
    var={t: fea_qois(a.id,t,run,plan,b07,pfa) for t in a.variants}
    ref=fea_qois(a.id,a.mesh_ref,run,plan,b07,pfa) if a.mesh_ref else None
    Q=judged([base] + list(var.values())); ks=[k for k,_ in Q]; lab=dict(Q)
    out={"id": a.id, "base": a.base, "mesh_ref": a.mesh_ref, "criterion_pct": CRIT,
           "quasi_static": {a.base: base["qs"], **{t: v["qs"] for t, v in var.items()}}, "variants": {}}
    if ref:
        out["mesh_uncertainty"]={k: float(abs(change(k,base["q"][k], ref["q"][k]))) for k in ks}
    for t,v in var.items():
        ch={k:float(change(k,v["q"][k], base["q"][k])) for k in ks}
        decide=[k for k in ks if k in SHAPE_QOIS]
        #decision rule:  a parameter matters only if a SHAPE QoI changes by more than CRIT % on the same mesh level
        # (integral QoIs only); mechanical QoIs and the mesh uncertainty are context.
        relevant=any(abs(ch[k])>= limit(k) for k in decide)
        out["variants"][t] = dict(changed=what_changed(base["info"], v["info"]), base={k: base["q"][k] for k in ks}, value={k: v["q"][k] for k in ks},
                                  change=ch, relevant=bool(relevant), worst_share_of_tolerance=float(max(abs(ch[k]) / limit(k) for k in decide)))
        print(f"\n{t} vs {a.base}: {out['variants'][t]['changed']}")
        for k in ks:
            kind="[shape]" if k in SHAPE_QOIS else "[mech.]"
            ctx = f" | mesh uncertainty {out['mesh_uncertainty'][k]:.2f} {unit(k)}" if ref else ""
            print(f"  {kind} {lab[k]:36s} {base['q'][k]:.4g} -> {v['q'][k]:.4g}  change {ch[k]:+.2f} {unit(k)}"
                  f"{'  BEYOND TOLERANCE' if abs(ch[k]) >= limit(k) else ''}{ctx}")
        print(f"  VERDICT: {'RELEVANT: must be modelled / chosen with care' if relevant else 'not relevant for the deployed shape'} "
              f"(worst decisive change = {100 * out['variants'][t]['worst_share_of_tolerance']:.0f} % of its tolerance)")
    dst = run / "compare"; dst.mkdir(exist_ok=True)
    (dst / f"sens_fea_{a.base}.json").write_text(json.dumps(out, indent=1))
    sens_figure(dst / f"sens_fea_{a.base}.png", a, out, Q)


FACTOR_ORDER={"axis":["centroid","mec"],"contact":["bonded","hard"]}


def factors(info):
    """The swept factors of one run, read from its info file (never from the tag)"""
    return dict(axis=info.get("axis_rule","mec"), contact=info.get("cuff_wall_contact","-"), mu=float(info.get("friction",0.0)),
                penalty=float(info.get("penalty_scale",1.0)))

def main_effects(fac,ch,ks):
    """Mean change of each QoI when ONE factor changes and the others are equal  (matched pairs over the whole sweep)"""
    out={}
    tags=list(fac)
    for f, ref in(("axis","centroid"), ("contact","bonded"), ("mu",0.0), ("penalty",1.0)):
        pairs=[(t,u) for t in tags for u in tags if fac[t][f]==ref and fac[u][f]!= ref and all(fac[t][g]==fac[u][g] for g in fac[t] if g!=f) and not (f=="contact" and fac[u]["mu"]!=0.0)]
        if pairs:
            out[f]={k: float(np.mean([abs(ch[u][k]-ch[t][k]) for t, u in pairs])) for k in ks} | {"pairs": len(pairs)}
    return out

def overview_figure(path, a, tags,fac,ch,d6,runs,Q,me):
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    ks = [k for k, _ in Q]; lab = dict(Q); n = len(tags)
    fig = plt.figure(figsize=(20, 1.6 + 0.42 * n), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[4.2, 1.0], wspace=0.16)
    ax = fig.add_subplot(gs[0]); ax.axis("off")
    cols = [("axis", 0.07), ("contact", 0.07), ("μ", 0.04), ("penalty", 0.055)] + [(k, 0.095) for k in ks] + \
           [("open bins", 0.08), ("worst gap", 0.08), ("penetration", 0.08), ("KE/IE ramp", 0.08)]
    x = np.cumsum([0] + [w for _, w in cols]); W = x[-1]; ax.set_xlim(-0.30, W); ax.set_ylim(-0.6, n + 2.2)
    heads = ["axis", "contact", "μ", "penalty"] + [("SHAPE\n" if k in SHAPE_QOIS else "mechanical\n") + SHORT.get(k, lab[k].split(" (")[0]) for k in ks] + \
            ["open seal bins\n(> 0.3 mm)", "worst seal\ngap (mm)", "tissue inside\ncuff (mm)", "KE/IE over\nramp (%)"]
    for (c, w), x0, h in zip(cols, x, heads):
        ax.text(x0 + w / 2, n + 0.7, h, ha="center", va="center", fontsize=8.3, color=INK2, fontweight="bold")
    ax.text(x[4] + (x[4 + len(ks)] - x[4]) / 2, n + 1.8, f"change vs base {a.base} (%)", ha="center", fontsize=9, color=INK, fontweight="bold")
    ax.text(x[4 + len(ks)] + (W - x[4 + len(ks)]) / 2, n + 1.8, "from 06 (absolute)", ha="center", fontsize=9, color=INK, fontweight="bold")
    for i, t in enumerate(tags):
        y = n - 1 - i; f = fac[t]; base = t == a.base
        if base:
            ax.add_patch(plt.Rectangle((-0.30, y + 0.02), W + 0.30, 0.96, color="#fde7da", lw=0, zorder=0))
        if i and f["axis"] != fac[tags[i - 1]]["axis"]:
            ax.plot([-0.30, W], [y + 1, y + 1], color=INK2, lw=1.2)
        ax.text(-0.29, y + 0.5, t, fontsize=7.6, color=INK, va="center", fontweight="bold" if base else "normal")
        vals = [f["axis"].replace("centroid", "cen"), f["contact"], f"{f['mu']:g}", f"x{f['penalty']:g}"]
        for (c, w), x0, v in zip(cols[:4], x[:4], vals):
            ax.text(x0 + w / 2, y + 0.5, v, ha="center", va="center", fontsize=8.5, color=INK)
        for j, k in enumerate(ks):
            x0, w = x[4 + j], cols[4 + j][1]; v = ch[t][k]; r_ = abs(v) / limit(k); shape = k in SHAPE_QOIS; fail = shape and r_ >= 1
            colr = ("#fde3e3" if fail else ["#ffffff", "#e6effb", "#cde2fb", "#9ec5f4", "#6da7ec"][min(int(r_ * 5), 4)]) if shape else \
                   ["#ffffff", "#f4f3ef", "#ecebe6", "#e1e0d9", "#d3d2ca"][min(int(r_ / 2), 4)]
            ax.add_patch(plt.Rectangle((x0 + 0.003, y + 0.06), w - 0.006, 0.88, color=colr, ec=RED if fail else GRID, lw=1.6 if fail else 0.6))
            ax.text(x0 + w / 2, y + 0.5, "base" if base else f"{v:+.1f}", ha="center", va="center", fontsize=8.5, color=RED if fail else (INK if shape else INK2),
                    fontweight="bold" if shape else "normal")
        dd = d6[t]; ob, gp, pe, ke = dd.get("seal_angles_above_0p3mm"), dd.get("seal_worst_angle_gap_mm"), dd.get("max_penetration_mm"), runs[t]["qs"]["mean_ramp_pct"]
        for j, (v, s) in enumerate(((ob, f"{ob}" if ob is not None else "-"), (gp, f"{gp:.2f}" if gp is not None else "-"),
                                    (pe, f"{pe:.2f}" if pe is not None else "-"), (ke, f"{ke:.2f}"))):
            x0, w = x[4 + len(ks) + j], cols[4 + len(ks) + j][1]
            hot = (j == 0 and ob) or (j == 1 and gp is not None and gp > 0.3) or (j == 3 and ke >= KE_IE_MEAN)
            ax.add_patch(plt.Rectangle((x0 + 0.003, y + 0.06), w - 0.006, 0.88, color="#fde7da" if hot else "#ffffff", ec=GRID, lw=0.6))
            ax.text(x0 + w / 2, y + 0.5, s, ha="center", va="center", fontsize=8.5, color=INK)
    ax.text(-0.29, -0.45, f"Shape QoIs (blue): shading = share of the {CRIT:g} % tolerance, red frame = beyond it; mechanical QoIs (grey) are context. "
                          "Seal columns: orange = open bins / gap above 0.3 mm. All runs at the same mesh level, so mesh error cancels in the changes.",
            fontsize=8, color=INK2, va="center")
    ax2 = fig.add_subplot(gs[1]); style(ax2, "x")                       # main effects
    fs = list(me); yy = np.arange(len(fs))[::-1]; hb = 0.8 / len(ks)
    for j, k in enumerate(ks):
        ax2.barh(yy + (len(ks) / 2 - j - 0.5) * hb, [me[f][k] for f in fs], height=hb * 0.9, color=BLUE if k in SHAPE_QOIS else "#b9b8b0",
                 label=SHORT.get(k, lab[k].split(" (")[0]))
    ax2.axvline(CRIT, color=RED, lw=1, ls="--"); ax2.text(CRIT, len(fs) - 0.45, f" {CRIT:g} %", color=RED, fontsize=8)
    ax2.set_yticks(yy); ax2.set_yticklabels([f"{f}\n({me[f]['pairs']} pairs)" for f in fs], fontsize=9)
    ax2.set_xlabel("mean |change| when only this factor changes (%)", fontsize=8.5, color=INK2)
    ax2.set_title("Main effect of each factor", loc="left", fontsize=11, fontweight="bold", color=INK)
    ax2.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.09), ncol=2)
    fig.suptitle(f"{a.id}: sensitivity sweep, {len(tags)} runs ({a.pattern}) -> which modelling choices change the deployed shape", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)



def overview_mode(a,run):
    """All runs of a factorial sweep in ONE figure:
    rows = runs grouped by factor (read from the info files), columns = judged QoIs as % change vs the base
    run on the same mesh, then seal / penetration / quasi-static numbers of 06; right: main effect of each factor (mean |change| over matched pairs).
    """
    plan, b07, pfa=common(a)
    tags=sorted(p.name[len("deploy_"):] for p in run.glob(f"deploy_{a.pattern}") if (p / f"{a.id}_{p.name[len('deploy_'):]}_results.npz").exists())
    assert a.base in tags, f"base {a.base} not among the runs matching {a.pattern}"
    runs={t: fea_qois(a.id,t,run,plan,b07,pfa) for t in tags}
    Q=judged(list(runs.values())); Q=[q for q in Q if q[0] in SHAPE_QOIS]+[q for q in Q if q[0] not in SHAPE_QOIS]; ks=[k for k ,_ in Q]
    fac={t: factors(runs[t]["info"]) for t in tags}
    key=lambda t: (FACTOR_ORDER["axis"].index(fac[t]["axis"]) if fac[t]["axis"] in FACTOR_ORDER["axis"] else 9,
                     FACTOR_ORDER["contact"].index(fac[t]["contact"]) if fac[t]["contact"] in FACTOR_ORDER["contact"] else 9, fac[t]["mu"],
                     {1.0: 0, 0.1: 1, 10.0: 2}.get(fac[t]["penalty"], 3))
    tags=sorted(tags, key=key)
    ch={t:{k: float(change(k,runs[t]["q"][k], runs[a.base]["q"][k])) for k in ks} for t in tags}
    d6={}
    for t in tags:
        p = run / f"deploy_{t}" / "post" / f"{a.id}_{t}_deployed.json"
        d6[t] = json.loads(p.read_text()) if p.exists() else {}
    me=main_effects(fac,ch,ks)
    rows = [dict(tag=t, **fac[t], **{f"change_pct_{k}": ch[t][k] for k in ks}, **{k: runs[t]["q"][k] for k in ks},
                 seal_open_bins=d6[t].get("seal_angles_above_0p3mm"), seal_worst_gap_mm=d6[t].get("seal_worst_angle_gap_mm"),
                 penetration_mm=d6[t].get("max_penetration_mm"), ring_SD_after_mm=d6[t].get("ring_SD_after_mm"),
                 KE_IE_ramp_pct=runs[t]["qs"]["mean_ramp_pct"], quasi_static=runs[t]["qs"]["ok"]) for t in tags]
    dst = run / "compare"; dst.mkdir(exist_ok=True)
    with open(dst / f"sens_overview_{a.base}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (dst / f"sens_overview_{a.base}.json").write_text(json.dumps(dict(base=a.base, pattern=a.pattern, main_effects=me, runs=rows), indent=1))
    print(f"overview of {len(tags)} runs vs {a.base}: main effect = mean |change| when only that factor changes")
    for f, v in me.items():
        print(f"  {f:8s} ({v['pairs']} pairs): " + ", ".join(f"{k} {v[k]:.2f} %" for k in ks))
    overview_figure(dst / f"sens_overview_{a.base}.png", a, tags, fac, ch, d6, runs, Q, me)
    print("wrote", dst / f"sens_overview_{a.base}.png", "+ .csv + .json")


def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", default="SYN_0001")
    sub=ap.add_subparsers(dest="mode", required=True)
    m=sub.add_parser("mesh"); m.add_argument("--levels",nargs="+", default=["L1", "L2", "L3", "L4", "L5"], help="finest first, at least three")
    m.add_argument("--rule", default="conservative", choices=("conservative","finest"), help="selection: |dev| + finest-level uncertainty < tolerance (conservative) or |dev| < tolerance (finest)")
    m.add_argument("--variant",default="",help="run tag after the level (e.g. cen_bonded -> deploy_L2_cen_bonded ...); empty = plain level folders")
    s=sub.add_parser("sens"); s.add_argument("--base",default="L3", help="run tag of the base run, e.g. L3_cen_bonded")
    s.add_argument("--variants",nargs="+", required=True, help="run tags of the variants, each differing from the base in ONE parameter")
    s.add_argument("--mesh_ref", default="", help="finest-level run of the same configuration: its difference to the base is shown as context")
    o=sub.add_parser("overview"); o.add_argument("--base", default="L3_cen_bonded_ramp0.2", help="run tag every other run is compared with")
    o.add_argument("--pattern",default="L3_*_ramp0.2", help="run tags of the sweep (glob on deploy_<tag>)")
    a=ap.parse_args()
    run=cfg.RUNS/a.id
    if a.mode=="mesh":
        assert len(a.levels)>=3, "The GCI needs at least three mesh levels"
        mesh_mode(a,run)
    elif a.mode=="overview":
        overview_mode(a,run)
    else:
        sens_mode(a,run)


if __name__=="__main__":
    main()