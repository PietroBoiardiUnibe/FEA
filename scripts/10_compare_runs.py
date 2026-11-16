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
LEVEL_RAMP = ["#104281", "#256abf", "#3987e5", "#86b6ef"]             # ordinal blue ramp, finest (dark) -> coarsest (light)
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
        return dict(out, type="drifting",R=float(R),p=np.nan, ext=np.nan,uncertainty_pct=float(spread),method="spread band")
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

def convergence_figure(path,out,levels,Q,unc):
    """Classic grid-convergence plot where each QoI is plotted against the FEA wall DoFs in log scale.
    Relative QoIs share one y-scale (+-YSPAN % of the finest value) so the panels compare visually. The anchor based QoIs are in absolute mm
    Acceptance band, selected level highlighted, the finest point carries its own numerical uncertainty due to discretization error left (GCI or spread band)
    extrapolate dvalue is also reported where applicable."""
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    order=levels[::-1]; col= {L: LEVEL_RAMP[i] for i,L in enumerate(levels)}; sel=out["selection"]["selected"]
    D=np.array([out["mesh"][L]["dofs"] for L in order], float); ks=[k for k,_ in Q]; dev=out["deviation_from_finest"]
    rel=[k for k in ks if unit(k)=="%"]
    YSPAN = max(1.6 * CRIT, 1.35 * max(abs(dev[k][L]) for k in rel for L in levels[1:]))   # common half-range of the relative panels, %
    nrow = int(np.ceil(len(ks) / 3))
    fig = plt.figure(figsize=(19, 4.8 * nrow), facecolor=SURFACE)
    gs = fig.add_gridspec(nrow, 4, hspace=0.55, wspace=0.62)
    axs = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(len(ks))]
    xs = D[order.index(sel)]
    for n_, (ax, (k, lab)) in enumerate(zip(axs, Q)):
        style(ax)
        v = np.array([out["values"][L][k] for L in order]); fine = out["values"][levels[0]][k]; g = out["gci_fine_triplet"][k]
        mm = unit(k) == "mm"; a = 1.0 if mm else abs(fine) / 100; lim = limit(k)
        span = max(1.6 * lim, 1.35 * max(abs(dev[k][L]) for L in levels[1:])) if mm else YSPAN
        ax.axhspan(fine - lim * a, fine + lim * a, color=BAND, lw=0, zorder=0)
        ax.axhline(fine, color=BASE, lw=0.8, zorder=1)
        ax.axvspan(xs / 1.2, xs * 1.2, color="#fde7da", lw=0, zorder=0)                       # selected level column
        if g["method"].startswith("GCI"):
            ax.axhline(g["ext"], color=ORANGE, lw=1.4, ls="--", zorder=1)
            ax.text(D[0] / 1.25, g["ext"] - 0.04 * span * a, "extrapolated value (h -> 0)", color=ORANGE, fontsize=7.5, va="top")
        ax.plot(D, v, color=INK2, lw=1.3, zorder=2)
        ax.errorbar(D[-1], fine, yerr=unc[k] * (1 if mm else a), color=INK, lw=1.6, capsize=5, zorder=3)   # uncertainty of the finest level
        for L, x, y in zip(order, D, v):
            big = L == sel
            ax.scatter(x, y, s=150 if big else 80, color=col[L], edgecolor=ORANGE if big else "white", linewidth=2.4 if big else 1.4, zorder=4)
            if L != levels[0]:
                d = dev[k][L]; fail = abs(d) >= lim
                ax.text(x, y + (1 if d >= 0 else -1) * 0.12 * span * a, f"{d:+.2f} mm" if mm else f"{d:+.1f} %", ha="center",
                        va="bottom" if d >= 0 else "top", fontsize=8, fontweight="bold", color=RED if fail else INK)
        ax.text(D[0] / 1.3, fine + lim * a, f" +{lim:g} {unit(k)}", color=INK2, fontsize=7.5, va="bottom", ha="left")
        ax.text(D[0] / 1.3, fine - lim * a, f" -{lim:g} {unit(k)}", color=INK2, fontsize=7.5, va="top", ha="left")
        ax.set_xscale("log"); ax.set_xlim(D[0] / 1.35, D[-1] * 1.45); ax.set_ylim(fine - span * a, fine + span * a)
        ax.set_xticks(D); ax.set_xticklabels([f"{L}\n{d / 1000:.0f}k DOF" for L, d in zip(order, D)], fontsize=8); ax.minorticks_off()
        if g["method"].startswith("GCI"):
            how = f"converging (R {g['R']:.2f}) -> GCI {unc[k]:.2f} {unit(k)}" + (" (p capped at 2)" if "capped" in g["method"] else "")
        else:
            how = f"{g['type']} (R {g['R']:.2f}) -> spread band {unc[k]:.2f} {unit(k)}"
        ax.set_title(f"{lab}\n", loc="left", color=INK, fontsize=10, fontweight="bold")
        ax.text(0, 1.03, how, transform=ax.transAxes, color=INK2, fontsize=8.3)
        if n_ % 3 == 0:
            ax.set_ylabel(f"QoI value  (axis = +-{YSPAN:.0f} % of finest)" if not mm else "QoI value (mm, absolute)", fontsize=8, color=INK2)
        if n_ >= len(ks) - 3:
            ax.set_xlabel("FEA wall degrees of freedom (log scale)  ->  finer", fontsize=8, color=INK2)
    y0, y1 = axs[0].get_ylim()
    axs[0].text(xs, y1 - 0.04 * (y1 - y0), f"selected {sel}", ha="center", va="top", color=ORANGE, fontsize=8.5, fontweight="bold")

    ax = fig.add_subplot(gs[:max(nrow - 1, 1), 3]); style(ax, "x")    # summary: both numbers per QoI as a share of their tolerance
    y = np.arange(len(ks))[::-1]
    for yi, k in zip(y, ks):
        g = out["gci_fine_triplet"][k]; u = 100 * unc[k] / limit(k); d3 = 100 * abs(dev[k].get(sel, 0.0)) / limit(k)
        ax.barh(yi + 0.17, u, height=0.32, color=BLUE if g["method"].startswith("GCI") else "#86b6ef")
        ax.barh(yi - 0.17, d3, height=0.32, color=ORANGE)
        ax.text(u + 2, yi + 0.17, f"{unc[k]:.2f} {unit(k)}", va="center", color=INK, fontsize=7.5)
        ax.text(d3 + 2, yi - 0.17, f"{abs(dev[k].get(sel, 0.0)):.2f} {unit(k)}", va="center", color=INK, fontsize=7.5)
    ax.axvline(100, color=INK2, lw=1.2, ls="--"); ax.text(100, y[0] + 0.55, " limit", color=INK2, fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels([lab.split(" (")[0] for _, lab in Q], fontsize=8)
    ax.set(xlim=(0, 150)); ax.set_xlabel(f"share of the tolerance used (100 % = {CRIT:g} %)", fontsize=8, color=INK2)
    ax.set_title("Both checks below the limit", loc="left", color=INK, fontsize=10, fontweight="bold")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE, label=f"uncertainty of {levels[0]}: GCI"),
                       plt.Rectangle((0, 0), 1, 1, color="#86b6ef", label=f"uncertainty of {levels[0]}: spread band"),
                       plt.Rectangle((0, 0), 1, 1, color=ORANGE, label=f"|{sel} - {levels[0]}| (selection)")],
              frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=1)

    ax = fig.add_subplot(gs[nrow - 1, 3]); ax.axis("off")              # how to read + method, in words
    rr = ", ".join(f"{k} {v:.2f}" for k, v in out["refinement_ratios"].items())
    txt = (f"How to read\n"
           f"- one panel per QoI, value vs mesh size; relative\n  panels span +-{YSPAN:.0f} % of the finest value\n"
           f"- blue band = tolerance around {levels[0]} (finest)\n"
           f"- point labels = deviation from {levels[0]} (red = beyond)\n"
           f"- orange ring + column = selected level ({sel})\n"
           f"- bar on {levels[0]} = its own numerical uncertainty\n\n"
           f"Method\n"
           f"- QoIs on each FEA mesh's own nodes (no mapping)\n"
           f"- GCI (Grid Convergence Index), Fs 1.25, r from tets\n  ({rr})\n"
           f"- R = eps21/eps32: 0<R<1 -> GCI (p capped at 2)\n  R<0 or R>=1 -> spread band = 1.25 x max|dev|\n"
           f"- selection: coarsest level whose QoIs all\n  stay within tolerance of {levels[0]}")
    ax.text(0, 0.98 if nrow > 2 else 0.98, txt, va="top", ha="left", color=INK2, fontsize=8.6, linespacing=1.35)
    vt = f" ({out['variant']})" if out.get("variant") else ""
    fig.suptitle(f"{out['id']}{vt}: grid convergence of the FEA deployment: every QoI inside its tolerance band from {sel} on; "
                 f"finest-level uncertainty <= {max(100 * unc[k] / limit(k) for k in ks):.0f} % of the tolerance",
                 x=0.01, ha="left", color=INK, fontsize=12.5, fontweight="bold")
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)

def selection_figure(path,out,levels,Q):
    """One figure that tells which mesh is used in the FEA. Per-level worst deviation as a share of its tolerance, which QoI drives it ans the cost"""
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    order = levels[::-1]; col = {L: LEVEL_RAMP[i] for i, L in enumerate(levels)}; sel = out["selection"]; dev = out["deviation_from_finest"]
    lab=dict(Q); ks=[k for k,_ in Q]
    fig, axs = plt.subplots(1, 3, figsize=(19, 6.0), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1.05, 1.45, 0.9]))

    ax = axs[0]; style(ax)                                              # A: worst deviation per level, % of the tolerance
    x = np.arange(len(order)); w = [100 * sel["worst_fraction_of_limit"].get(L, 0.0) for L in order]
    ax.axhspan(0, 100, color=BAND, lw=0); ax.axhline(100, color=INK2, lw=1, ls="--")
    ax.text(-0.45, 102, f"tolerance ({CRIT:g} %)", color=INK2, fontsize=8.5)
    ax.bar(x, w, width=0.6, color=[col[L] for L in order])
    for xi, v, L in zip(x, w, order):
        if L == levels[0]:
            ax.text(xi, 5, "reference\n(finest)", ha="center", color=INK2, fontsize=8.5)
        else:
            k = sel["worst_qoi"][L]
            ax.text(xi, v + 4, f"{abs(dev[k][L]):.2f} {unit(k)}", ha="center", color=RED if v >= 100 else INK, fontsize=10, fontweight="bold")
            ax.text(xi, v + 15, lab[k].split(" (")[0], ha="center", color=INK2, fontsize=7.5)
    xs = order.index(sel["selected"])
    ax.annotate(f"selected: {sel['selected']}\n(coarsest level\nwithin tolerance)", (xs + 0.05, w[xs] + 25), xytext=(xs + 0.55, max(w) * 0.85),
                ha="center", color=INK, fontsize=9.5, fontweight="bold", arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.set_xticks(x); ax.set_xticklabels([f"{L}\n{out['mesh'][L]['tets']:,} tets" for L in order], fontsize=9)
    ax.set(ylim=(0, max(max(w), 100) * 1.35), ylabel="worst deviation from the finest level (% of its tolerance)")
    ax.set_title("Which mesh is good enough?\nworst QoI deviation from the finest level", loc="left", color=INK, fontsize=11, fontweight="bold")

    ax = axs[1]                                                        # B: every QoI x level (the table behind panel A)
    cols = [L for L in order if L != levels[0]]
    ramp = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5"]
    for i, k in enumerate(ks):
        for j, L in enumerate(cols):
            fr_ = abs(dev[k][L]) / limit(k); fail = fr_ >= 1
            c = ramp[min(int(fr_ * 4), 4)] if not fail else "#fde3e3"
            ax.add_patch(plt.Rectangle((j, len(ks) - 1 - i), 0.96, 0.92, color=c, ec=RED if fail else GRID, lw=1.8 if fail else 0.8))
            ax.text(j + 0.48, len(ks) - 1 - i + 0.46, ("FAIL " if fail else "") + (f"{dev[k][L]:+.2f} mm" if unit(k) == "mm" else f"{dev[k][L]:+.1f} %"),
                    ha="center", va="center", color=RED if fail else INK, fontsize=9.5, fontweight="bold")
    ax.set_xlim(0, len(cols)); ax.set_ylim(0, len(ks)); ax.set_xticks(np.arange(len(cols)) + 0.48); ax.set_xticklabels(cols, fontsize=10, fontweight="bold")
    ax.set_yticks(np.arange(len(ks)) + 0.46); ax.set_yticklabels([lb for _, lb in Q][::-1], fontsize=9); ax.tick_params(length=0, colors=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Each QoI: deviation from {levels[0]} (finest)\nshading = share of the tolerance, red frame = beyond it", loc="left", color=INK, fontsize=11, fontweight="bold")

    ax = axs[2]; style(ax)                                              # C: what each level costs
    d = [out["mesh"][L]["dofs"] for L in order]
    ax.bar(x, d, width=0.6, color=[col[L] for L in order])
    for xi, v in zip(x, d):
        ax.text(xi, v * 1.03, f"{v / 1000:.0f}k", ha="center", color=INK, fontsize=9.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=10); ax.set(ylabel="degrees of freedom of the FEA wall", ylim=(0, max(d) * 1.15))
    ax.set_title("What each level costs", loc="left", color=INK, fontsize=11, fontweight="bold")
    failing = sorted(set(lab[sel["worst_qoi"][L]].split(" (")[0] for L in levels[1:] if sel["worst_fraction_of_limit"][L] >= 1))
    vt = f" ({out['variant']})" if out.get("variant") else ""
    fig.suptitle(f"{out['id']}{vt}: FEA mesh selection on FEA results only -> {sel['selected']}"
                 + (f" (coarser levels fail on {', '.join(failing)})" if failing else " (every level within tolerance)"),
                 x=0.01, ha="left", color=INK, fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(path, dpi=160, facecolor=SURFACE); plt.close(fig)

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
    ratio={L: max(abs(dev[k][L])/limit(k) for k in ks) for L in levels[1:]}
    worst_k = {L: max(ks, key=lambda k: abs(dev[k][L]) / limit(k)) for L in levels[1:]}
    selected=levels[0]

    for L in levels[1:]:
        if ratio[L]<1.0:
            selected=L
        else:
            break
    out["selection"]=dict(rule=f"coarsest level whose QoIs (and those of every finer level) all stay within {CRIT:g} % "
                                 f"of the finest level",
                            worst_fraction_of_limit=ratio, worst_qoi=worst_k, selected=selected)
    dst=run/"compare";dst.mkdir(exist_ok=True); sfx= f"_{a.variant}" if a.variant else ""
    (dst / f"mesh_convergence_fea{sfx}.json").write_text(json.dumps(out, indent=1))
    lab = dict(Q)
    print("worst deviation from the finest level, per level: " + "; ".join(
        f"{L} {abs(dev[worst_k[L]][L]):.2f} {unit(worst_k[L])} ({lab[worst_k[L]]})" for L in levels[1:]))
    print(f"SELECTED MESH LEVEL: {selected}")
    print("refinement ratios " + ", ".join(f"{k} {v:.3f}" for k, v in out["refinement_ratios"].items())
          + ("  (>= 1.3 recommended by Celik 2008: " + ("met" if min(out["refinement_ratios"].values()) >= 1.3 else "NOT MET for the coarsest pair") + ")"))
    print(f"{'QoI (FEA mesh only)':38s} {' / '.join(levels):28s}   fine triplet: type, R, p, method, uncertainty | deviation of {selected} | verdict")
    for k in ks:
        g = out["gci_fine_triplet"][k]; vals = " / ".join(f"{runs[L]['q'][k]:.4g}" for L in levels)
        d_sel = dev[k].get(selected, 0.0)
        print(f"  {lab[k]:36s} {vals:28s}   {g['type']:11s} R {g['R']:6.2f} p {g['p']:5.2f} {g['method']:20s} {unc[k]:5.2f} {unit(k):2s} | "
              f"{d_sel:+.2f} {unit(k)} | {'PASS' if out['verdict'][k] else 'FAIL'}")
    for k, lb in [(k, lb) for k, lb in DESCRIPTIVE if all(k in runs[L]["q"] for L in levels)]:   # only those every run has
        print(f"  (descriptive) {lb:30s} " + " / ".join(f"{runs[L]['q'][k]:.4g}" for L in levels))
    selection_figure(dst / f"mesh_selection_fea{sfx}.png", out, levels, Q)
    convergence_figure(dst / f"mesh_convergence_fea{sfx}.png", out, levels, Q, unc)

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


def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", default="SYN_0001")
    sub=ap.add_subparsers(dest="mode", required=True)
    m=sub.add_parser("mesh"); m.add_argument("--levels",nargs="+", default=["L1", "L2", "L3", "L4", "L5"], help="finest first, at least three")
    m.add_argument("--variant",default="",help="run tag after the level (e.g. cen_bonded -> deploy_L2_cen_bonded ...); empty = plain level folders")
    s=sub.add_parser("sens"); s.add_argument("--base",default="L3", help="run tag of the base run, e.g. L3_cen_bonded")
    s.add_argument("--variants",nargs="+", required=True, help="run tags of the variants, each differing from the base in ONE parameter")
    s.add_argument("--mesh_ref", default="", help="finest-level run of the same configuration: its difference to the base is shown as context")
    a=ap.parse_args()
    run=cfg.RUNS/a.id
    if a.mode=="mesh":
        assert len(a.levels)>=3, "The GCI needs at least three mesh levels"
        mesh_mode(a,run)
    else:
        sens_mode(a,run)

if __name__=="__main__":
    main()