"""
Uncertainty Quantification of the deployed state (ASME V&V 40 context, ASME V&V 20 numerics), built
on the sensitivity sweep for both device-skirt axis rules, measured on the DIRECT FEA surface.

1 NUMERICAL: U_num (bias of the discrete solution), on the base <level>_cen_bonded_<ramp> = the only configuration with a full grid:
     mesh     u_mesh = |phi_L3 - phi_L1| + U(L1)       U = GCI / spread band of L1-L2-L3 (Celik 2008); same conservative rule as the mesh choice
     penalty  u_pen  = |phi(x10) - phi(x1)|            distance to a stiffer, better converged contact (x0.1 = under-converged, shown only)
     time     u_time = |phi(ramp 0.1) - phi(ramp 0.2)|  full difference (upper bound); also bounds the inertia added by mass scaling
     U_num = u_mesh + u_pen + u_time      
     MEC family (no grid): same RELATIVE U_num for integral QoIs, same ABSOLUTE U_num for local geometric features (stated assumption).
    
2 MODELLING INPUT: MODELLING INPUT with unknown truth: the tissue-skirt interaction {bonded, hard mu 0, hard mu 0.2}. Propagated as an INTERVAL over the
   alternatives at the frozen numerics (penalty x1): [min, max]. The FULL interval of a family = [min - U_num, max + U_num].

3 PROTOCOL CHOICE (decision variable, not an uncertainty): cuff axis {area centroid, MEC} -> two scenarios, each with its full interval.
   Robust decision = the intervals do not overlap; otherwise the PAIRED comparison (9 pairs that differ only by the axis: shared numerical
   and input errors cancel in each difference).
4 SENSITIVITY vs NUMERICS: effect of every factor (mean |change| over matched pairs) / U_num. < 1 -> not resolvable -> freeze at default;
   > 1 -> a real choice: justify it and carry it (input interval or scenario).
5 MATERIAL: not propagated for shape: prescribed cuff + homogeneous neo-Hookean wall -> scaling E scales stresses/energies/forces but not the
   displacement field (only the support springs break the invariance; shown insensitive). Energy/force QoIs are for RELATIVE comparison only.
6 VALIDATION (light): sizing outputs vs clinical CT (CLINICAL below; add papers there).
7 Side note: fixed mass scaling acceptability (added mass, stable increment, KE/IE, hourglass, energy balance, inertia equivalence).
Usage: python 11_uq.py --id SYN_0001 [--level L3] [--ramp ramp0.2] [--out DIR]
Needs the 06 json (direct surface) of every run, L1/L2_cen_bonded_<ramp> and <level>_cen_bonded (ramp 0.1 s).
Writes compare/uq_runs_<level>_<ramp>.png (every run, every QoI), uq_decision_<level>_<ramp>.png (budget, factor matrix, axis decision,
validation, mass scaling) and uq_<level>_<ramp>.json 
"""
import argparse
import importlib.util
import json
import re
import matplotlib
from pathlib import Path
from types import SimpleNamespace
matplotlib.use("Agg")
import numpy as np
import vtk
import fea_config as cfg
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from vtkmodules.util.numpy_support import vtk_to_numpy

HERE = Path(__file__).resolve().parent
INK, INK2, GRID, SURFACE, RED = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb", "#d03b3b"
FAM = {"cen": ("area centroid (CT planning)", "#2a78d6", "#cde2fb"), "mec": ("MEC (07b circle centre)", "#eb6834", "#fbd9c6")}
CONT = ["bonded", "hard", "hard_mu0.2"]; PEN = ["", "_pen0.1", "_pen10"]
CLABEL = {"bonded": "bonded", "hard": "hard mu 0", "hard_mu0.2": "hard mu 0.2"}; PLABEL = {"": "penalty x1", "_pen0.1": "penalty x0.1", "_pen10": "penalty x10"}
VARIANTS = [(c + p, f"{CLABEL[c]}, {PLABEL[p]}") for c in CONT for p in PEN]
FAST = ("fast", "frozen reference, ramp 0.1 s (time check)")
# key, label, transfer of U_num to MEC (rel | abs), role in the decision
QOIS = [("penetration_mm", "max tissue inside the cuff (mm)", "abs", "PRIORITY 1: lower"),
        ("worst_gap_mm", "worst seal gap (mm)", "abs", "PVL: trade-off accepted"),
        ("open_bins", "open seal bins > 0.3 mm (of 120)", "abs", "PVL: trade-off accepted"),
        ("perimeter_stretch_pct", "annulus perimeter stretch (%)", "abs", "clinical sizing"),
        ("area_increase_pct", "annulus area increase (%)", "abs", "clinical sizing"),
        ("u_rms_area_mm", "area-weighted RMS wall displacement (mm)", "rel", "priority 2: lower"),
        ("cavity_change_ml", "cavity volume change (ml)", "rel", "shape"),
        ("sed_zone_mean_kPa", "isochoric SED, device-zone mean (kPa)", "rel", "priority 2: lower (x E)"),
        ("strain_energy_mJ", "wall strain energy (mJ)", "rel", "priority 2: lower (x E)"),
        ("radial_force_N", "cuff radial force (N)", "rel", "context (x E)")]
SHORT = dict(penetration_mm="tissue in cuff", worst_gap_mm="worst gap", open_bins="open bins", perimeter_stretch_pct="perim. stretch",
             area_increase_pct="area increase", u_rms_area_mm="RMS displ.", cavity_change_ml="cavity change", sed_zone_mean_kPa="SED zone",
             strain_energy_mJ="strain energy", radial_force_N="radial force")
NONNEG = ("penetration_mm", "worst_gap_mm", "open_bins")
# clinical comparators (validation-light); add papers here
CLINICAL = cfg.CLINICAL


def load(name):
    sp = importlib.util.spec_from_file_location("m_" + name.replace(".", "_"), HERE / f"{name}.py")
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def sed_stats(run, sid, tag, info, res):
    """ISOCHORIC strain-energy density of every wall tet, W_iso = C10 (I1bar - 3) (run's own neo-Hookean C10; MPa = mJ/mm3, reported in kPa),
    from the corner-node displacements (F constant per tet = mean deformation of the C3D10M element). The volumetric part (J - 1)^2 / D1 is NOT
    usable from corner nodes (nu 0.49: the element enforces volume in an averaged sense; corner J scatters and the term would sum to ~7 x the
    Abaqus energy). Check: sum(W_iso V0) against the Abaqus wall internal energy. Exact SED needs *ELEMENT OUTPUT ESEDEN in 03 (future runs)."""
    level = tag.split("_")[0]
    r = vtk.vtkXMLUnstructuredGridReader(); r.SetFileName(str(run / "mesh" / f"{sid}_wall_{level}.vtu")); r.Update(); g = r.GetOutput()
    X = vtk_to_numpy(g.GetPoints().GetData()).astype(float); T = vtk_to_numpy(g.GetCells().GetConnectivityArray()).reshape(-1, 4)
    ann = vtk_to_numpy(g.GetPointData().GetArray("tv_annulus")) == 1
    x = X + res["U"][-1].astype(float)[np.searchsorted(res["labels"], np.arange(1, len(X) + 1))]
    Dm = np.stack([X[T[:, k]] - X[T[:, 0]] for k in (1, 2, 3)], 2); Ds = np.stack([x[T[:, k]] - x[T[:, 0]] for k in (1, 2, 3)], 2)
    F = Ds @ np.linalg.pinv(Dm); J = np.linalg.det(F); I1 = np.einsum("nij,nij->n", F, F)            # I1 = tr(F^T F)
    W = info["material"]["C10"] * (np.clip(J, 1e-9, None) ** (-2 / 3) * I1 - 3)
    V0 = np.abs(np.linalg.det(Dm)) / 6; zone = cKDTree(X[ann]).query(X[T].mean(1))[0] < cfg.ZONE_MM
    E = float((W * V0).sum()); IE = float(res["hist_wall_ALLIE"][-1, 1])
    return dict(sed_zone_mean_kPa=1000 * float((W * V0)[zone].sum() / V0[zone].sum()), sed_p99_kPa=1000 * float(np.percentile(W[zone], 99)),
                sed_max_kPa=1000 * float(W.max()), sed_integral_mJ=E, sed_integral_vs_ALLIE=E / IE)


def solver_log(dep, name):
    """Fixed mass scaling as Abaqus applied it (.sta table): added mass (% of the model mass; scaled once at the step start, constant), stable
    increment at the start and its minimum (it falls below the target when elements shrink: smaller increments, NO extra mass), increments."""
    rows = [l.split() for l in (dep / f"{name}.sta").read_text(errors="ignore").splitlines()
            if re.match(r"\s+\d+\s+\d\.\d+E[+-]\d+\s+\d\.\d+E[+-]\d+\s+\d\d:\d\d:\d\d", l)]
    return dict(added_mass_pct=float(rows[-1][-1]), dt_start_s=float(rows[0][4]), dt_min_s=min(float(r[4]) for r in rows), increments=int(rows[-1][0]))


def values(s10, a, run, t, plan, b07, pfa):
    """Every QoI of one run: shape/energies from script 10 (FEA mesh), features + sizing from the 06 json (direct surface), SED, solver checks."""
    dep = run / f"deploy_{t}"; name = f"{a.id}_{t}"
    r = s10.fea_qois(a.id, t, run, plan, b07, pfa); q = r["q"]; res = np.load(dep / f"{name}_results.npz")
    d6 = json.loads((dep / "post" / f"{name}_deployed.json").read_text())
    if d6.get("surface") != "direct":
        raise SystemExit(f"{t}: the 06 json is not on the direct surface -> rerun: python 06_deformed.py --id {a.id} --level {t.split('_')[0]} "
                         f"--tag {t.split('_', 1)[1]} --no_series")
    sz = d6["sizing"]
    v = dict(penetration_mm=d6["max_penetration_mm"], worst_gap_mm=d6["seal_worst_angle_gap_mm"], open_bins=d6["seal_angles_above_0p3mm"],
             perimeter_stretch_pct=sz["perimeter_stretch_pct"], area_increase_pct=sz["area_increase_pct"],
             perimeter_pre_mm=sz["pre"]["perimeter_mm"], perimeter_post_mm=sz["post"]["perimeter_mm"],
             oversizing_planned_perimeter_pct=sz["planned_07b"]["perimeter_derived_pct"], oversizing_planned_area_pct=sz["planned_07b"]["area_pct"],
             cavity_change_ml=q["cavity_change_ml"], u_rms_area_mm=q["u_rms_area_mm"], strain_energy_mJ=q["strain_energy_mJ"],
             radial_force_N=q["radial_force_N"], lateral_force_N=q["lateral_force_N"])
    v.update(sed_stats(run, a.id, t, r["info"], res))
    et, ie, wk = res["hist_ETOTAL"][:, 1], res["hist_ALLIE"][:, 1], res["hist_ALLWK"][:, 1]
    v["checks"] = dict(solver_log(dep, name), KE_IE_ramp_pct=r["qs"]["mean_ramp_pct"], KE_IE_end_pct=r["qs"]["end_pct"],
                       hourglass_pct=float(100 * res["hist_wall_ALLAE"][-1, 1] / res["hist_wall_ALLIE"][-1, 1]),
                       energy_balance_pct=float(100 * np.abs(et).max() / max(np.abs(wk).max(), ie.max())))
    return v


def mesh_U(s10, f, N):
    """U(L1) in QoI units from the L1-L2-L3 triplet (script 10's gci3: GCI if converging, spread band otherwise)."""
    spread = 1.25 * max(abs(f[1] - f[0]), abs(f[2] - f[0]))
    if abs(f[0]) < 1e-9:
        return spread, "spread band (finest value 0)"
    g = s10.gci3(f, N)
    if g["type"] == "converged" and spread > 0:                     # one step zero, the other not (e.g. integer bins 6, 6, 7): use the spread
        return spread, "spread band (one step zero)"
    return g["uncertainty_pct"] / 100 * abs(f[0]), f"{g['method']}, {g['type']} R {g['R']:.2f}"


def factor_effects(val, tags, base, fast, l1, k):
    """Mean / max |change| of QoI k when ONE factor changes and everything else is equal (matched pairs over the sweep)."""
    pairs = {"axis (centroid -> MEC)": ("decision", [(tags["cen"][v], tags["mec"][v]) for v, _ in VARIANTS]),
             "contact law (bonded -> hard mu 0)": ("input", [(tags[f]["bonded" + p], tags[f]["hard" + p]) for f in FAM for p in PEN]),
             "friction (hard: mu 0 -> 0.2)": ("input", [(tags[f]["hard" + p], tags[f]["hard_mu0.2" + p]) for f in FAM for p in PEN]),
             "penalty x1 -> x10": ("numerical", [(tags[f][c], tags[f][c + "_pen10"]) for f in FAM for c in CONT]),
             "penalty x1 -> x0.1 (under-converged)": ("numerical", [(tags[f][c], tags[f][c + "_pen0.1"]) for f in FAM for c in CONT]),
             "ramp 0.2 -> 0.1 s": ("numerical", [(base, fast)]), "mesh L3 -> L1": ("numerical", [(base, l1)])}
    out = {}
    for name, (kind, P) in pairs.items():
        d = [abs(val[b][k] - val[a][k]) for a, b in P]; out[name] = dict(kind=kind, mean=float(np.mean(d)), max=float(np.max(d)), n=len(d))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", default="SYN_0001"); ap.add_argument("--level", default="L3"); ap.add_argument("--ramp", default="ramp0.2")
    ap.add_argument("--out", default="")
    ap.add_argument("--ref", default="hard", choices=CONT, help="contact law of the frozen reference; U_num is computed on it (needs its L1/L2 grid + ramp-0.1 run)")
    a = ap.parse_args(); run = cfg.RUNS / a.id
    s10 = load("10_compare_runs"); plan, b07, pfa = s10.common(SimpleNamespace(id=a.id))
    tg = lambda ax, v, L=a.level: f"{L}_{ax}_{v}_{a.ramp}"
    tags = {ax: {v: tg(ax, v) for v, _ in VARIANTS} for ax in FAM}; fast = f"{a.level}_cen_{a.ref}"; tags["cen"]["fast"] = fast
    refs = {ax: tags[ax][a.ref] for ax in FAM}; base = refs["cen"]; grid = [tg("cen", a.ref, "L1"), tg("cen", a.ref, "L2"), base]
    need = sorted(set(grid) | {t for d in tags.values() for t in d.values()})
    missing = [t for t in need if not (run / f"deploy_{t}" / f"{a.id}_{t}_results.npz").exists()]; assert not missing, f"missing runs: {missing}"
    val = {t: values(s10, a, run, t, plan, b07, pfa) for t in need}
    N = [s10.zone_tets(run, a.id, L) for L in ("L1", "L2", a.level)]
    unc = {}
    for k, lab, kind, role in QOIS:
        f = [val[t][k] for t in grid]; U1, how = mesh_U(s10, f, N)
        um = abs(f[2] - f[0]) + U1; up = abs(val[tags["cen"][a.ref + "_pen10"]][k] - f[2]); ut = abs(val[fast][k] - f[2])
        U = um + up + ut
        o = dict(label=lab, kind=kind, role=role, grid=f, U_L1=U1, mesh_method=how, u_mesh=um, u_pen=up, u_time=ut, U_num=U,
                 U_num_rss=float(np.sqrt(um ** 2 + up ** 2 + ut ** 2)), effects=factor_effects(val, tags, base, fast, grid[0], k))
        for fm in FAM:
            ref = val[refs[fm]][k]; Uf = U * abs(ref) / abs(f[2]) if kind == "rel" and abs(f[2]) > 1e-12 else U
            alt = [val[tags[fm][c]][k] for c in CONT]
            lo = min(alt) - Uf; lo = max(lo, 0.0) if k in NONNEG else lo                      # a penetration / gap / bin count cannot be < 0
            o[fm] = dict(ref=ref, U_num=Uf, input_lo=min(alt), input_hi=max(alt), lo=lo, hi=max(alt) + Uf,
                         input_alternatives=dict(zip(CONT, alt)))
        c, m = o["cen"], o["mec"]
        d = [val[tags["mec"][v]][k] - val[tags["cen"][v]][k] for v, _ in VARIANTS]
        o.update(pair_diffs=d, mec_higher=int(sum(x > 0 for x in d)), mec_lower=int(sum(x < 0 for x in d)), pairs=len(d),
                 overlap=bool(not (m["lo"] > c["hi"] or m["hi"] < c["lo"])),
                 mass_scaling_effect_est=ut * max(val[t]["checks"]["added_mass_pct"] for t in need) / 300.0)
        if not o["overlap"]:
            o["verdict"] = "centroid (robust)" if m["lo"] > c["hi"] else "MEC (robust)"
        else:
            o["verdict"] = "centroid (paired)" if o["mec_higher"] == len(d) else ("MEC (paired)" if o["mec_lower"] == len(d) else "undecided")
        unc[k] = o
    dst = Path(a.out) if a.out else cfg.compare_dir(run, "uq"); stem = f"{a.level}_{a.ref}_{a.ramp}"; dst.mkdir(exist_ok=True)
    (dst / f"uq_{stem}.json").write_text(json.dumps(dict(base=base, refs=refs, tags=tags, grid=grid, zone_tets=N, clinical=CLINICAL, uq=unc,
                                                         values=val), indent=1))
    print(f"\nUQ ({a.level}, {a.ramp}, direct FEA surface). U_num from {base} = mesh + penalty + time; interval = input alternatives +- U_num")
    print(f"{'QoI':40s} {'centroid: ref [full interval]':>30s} {'MEC: ref [full interval]':>30s} {'overlap':>8s} {'pairs':>6s}  verdict")
    for k, o in unc.items():
        c, m = o["cen"], o["mec"]
        print(f"{o['label']:40s} {c['ref']:8.4g} [{c['lo']:7.4g}, {c['hi']:7.4g}]   {m['ref']:8.4g} [{m['lo']:7.4g}, {m['hi']:7.4g}]   "
              f"{str(o['overlap']):>6s} {o['mec_higher']}/{o['pairs']}  {o['verdict']}")
    print("\nfactor effect / U_num (mean over matched pairs):")
    for name in unc[QOIS[0][0]]["effects"]:
        print(f"  {name:38s} " + "  ".join(f"{SHORT[k]} {unc[k]['effects'][name]['mean'] / max(unc[k]['U_num'], 1e-12):5.2f}" for k in unc))
    ck = [val[t]["checks"] for t in need]
    print("\nchecks over all runs: " + ", ".join(f"{k} {min(c[k] for c in ck):.3g}..{max(c[k] for c in ck):.3g}" for k in ck[0]))
    sd = [val[t]["sed_integral_vs_ALLIE"] for t in need]; print(f"SED check: sum(W_iso V0) / Abaqus wall internal energy = {min(sd):.3f}..{max(sd):.3f}")
    print("\nclinical sizing of every run (planned oversizing 07b; perimeter / area change on the direct surface):")
    for t in need:
        v = val[t]; print(f"  {t:34s} oversizing {v['oversizing_planned_perimeter_pct']:.1f} % (perimeter) / {v['oversizing_planned_area_pct']:.1f} % (area) | "
                          f"perimeter {v['perimeter_pre_mm']:.1f} -> {v['perimeter_post_mm']:.1f} mm ({v['perimeter_stretch_pct']:+.1f} %), area {v['area_increase_pct']:+.1f} %")
    fig_runs(dst / f"uq_runs_{stem}.png", a, val, unc, tags)
    fig_decision(dst / f"uq_decision_{stem}.png", a, val, unc, need)
    print("wrote", dst / f"uq_runs_{stem}.png", "+ uq_decision + json")


def fmt(k, v):
    return f"{v:.0f}" if k == "open_bins" else (f"{v:.3f}" if abs(v) < 1 else f"{v:.3g}")


def fig_runs(path, a, val, unc, tags):
    """Every run, every QoI: rows = sweep, two markers per row (centroid above, MEC below). Dark band = frozen reference +- U_num,
    light band = full interval (input alternatives +- U_num). Filled marker = outside the reference +- U_num (a real effect)."""
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    fig = plt.figure(figsize=(30, 15.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 5, hspace=0.30, wspace=0.08, top=0.86, bottom=0.04, left=0.11, right=0.995)
    rows = VARIANTS + [FAST]; Y = np.arange(len(rows))[::-1].astype(float)
    for i, (k, lab, kind, role) in enumerate(QOIS):
        ax = fig.add_subplot(gs[i // 5, i % 5]); o = unc[k]; xs = []
        for f_, (fname, col, light) in FAM.items():
            p = o[f_]
            ax.axvspan(p["lo"], p["hi"], color=light, alpha=0.40, lw=0, zorder=0)
            ax.axvspan(p["ref"] - p["U_num"], p["ref"] + p["U_num"], color=light, alpha=0.95, lw=0, zorder=0)
            ax.axvline(p["ref"], color=col, lw=1.3, zorder=1); xs += [p["lo"], p["hi"]]
        for j, (v, _) in enumerate(rows):
            for f_, off in (("cen", 0.17), ("mec", -0.17)):
                t = tags[f_].get(v)
                if t is None:
                    continue
                p = o[f_]; x, u, col = val[t][k], p["U_num"], FAM[f_][1]; real = abs(x - p["ref"]) > u; xs += [x - u, x + u]
                ax.errorbar(x, Y[j] + off, xerr=u, fmt="none", ecolor=col, elinewidth=1.0, capsize=2.5, alpha=0.6, zorder=2)
                ax.scatter(x, Y[j] + off, s=80 if v == a.ref else 44, marker="D" if v == a.ref else "o",
                           facecolor=col if (real or v == a.ref) else "white", edgecolor=col, linewidth=1.6, zorder=3)
        for yb in (Y[2] - 0.5, Y[5] - 0.5, Y[8] - 0.5):
            ax.axhline(yb, color=GRID, lw=1.0)
        if k == "worst_gap_mm":
            ax.axvline(0.3, color=RED, lw=1, ls="--"); ax.text(0.3, len(rows) - 0.55, " 0.3 mm", color=RED, fontsize=8)
        lo, hi = min(xs), max(xs); pad = 0.05 * (hi - lo + 1e-9); ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_yticks(Y); ax.set_yticklabels([r for _, r in rows] if i % 5 == 0 else [], fontsize=9.5)
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_facecolor(SURFACE); ax.tick_params(axis="x", labelsize=8.5, colors=INK2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        c, m = o["cen"], o["mec"]
        ax.set_title(f"{lab}\n[{role}]  ->  {o['verdict']}\ncentroid {fmt(k, c['ref'])} ± {c['U_num']:.2g}  [{fmt(k, c['lo'])}, {fmt(k, c['hi'])}]\n"
                     f"MEC {fmt(k, m['ref'])} ± {m['U_num']:.2g}  [{fmt(k, m['lo'])}, {fmt(k, m['hi'])}]", loc="left", fontsize=9.3, fontweight="bold", color=INK)
    h = [plt.Line2D([], [], marker="D", ls="", color=FAM["cen"][1], ms=8, label="frozen reference, centroid"),
         plt.Line2D([], [], marker="D", ls="", color=FAM["mec"][1], ms=8, label="frozen reference, MEC"),
         plt.Line2D([], [], marker="o", ls="", markerfacecolor=INK2, markeredgecolor=INK2, ms=7, label="run OUTSIDE reference ± U_num: real effect"),
         plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK2, ms=7, label="run inside: numerical noise"),
         plt.Rectangle((0, 0), 1, 1, color=FAM["cen"][2], alpha=0.95, label="reference ± U_num (numerical)"),
         plt.Rectangle((0, 0), 1, 1, color=FAM["cen"][2], alpha=0.40, label="full interval: contact alternatives ± U_num")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.005, 0.955), ncol=6, frameon=False, fontsize=10)
    fig.suptitle(f"{a.id}: every run of the sweep around the two frozen references ({a.level}, {a.ramp.replace('ramp', 'ramp ')} s, direct FEA surface)  ·  "
                 "error bar = ± U_num of the run", x=0.005, y=0.985, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.savefig(path, dpi=130, facecolor=SURFACE); plt.close(fig)


def fig_decision(path, a, val, unc, need):
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    fig = plt.figure(figsize=(26, 17), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.55], height_ratios=[1, 0.9], hspace=0.28, wspace=0.16, top=0.93, bottom=0.03, left=0.10, right=0.99)
    ks = [q[0] for q in QOIS]
    # A: what the uncertainty is made of
    ax = fig.add_subplot(gs[0, 0]); yy = np.arange(len(ks))[::-1]; cols = {"mesh": "#104281", "penalty": "#9ec5f4", "time": "#1baf7a"}
    for y, k in zip(yy, ks):
        o = unc[k]; b = abs(o["grid"][2]) if abs(o["grid"][2]) > 1e-12 else 1.0; x0 = 0
        for (nm, c), u in zip(cols.items(), (o["u_mesh"], o["u_pen"], o["u_time"])):
            ax.barh(y + 0.17, 100 * u / b, left=x0, color=c, height=0.32, label=nm if y == yy[0] else None); x0 += 100 * u / b
        w = 100 * (o["cen"]["input_hi"] - o["cen"]["input_lo"]) / b
        ax.barh(y - 0.17, w, color="#e9a23b", height=0.32, label="contact-law interval width" if y == yy[0] else None)
        ax.text(max(x0, w) + 1.5, y, f"U_num {o['U_num']:.2g} ({x0:.0f} %)   input {o['cen']['input_hi'] - o['cen']['input_lo']:.2g} ({w:.0f} %)",
                va="center", fontsize=8.3, color=INK2)
    ax.set_yticks(yy); ax.set_yticklabels([unc[k]["label"] for k in ks], fontsize=8.8)
    ax.set_xlim(0, max(100 * max(unc[k]["U_num"], unc[k]["cen"]["input_hi"] - unc[k]["cen"]["input_lo"]) / max(abs(unc[k]["grid"][2]), 1e-12) for k in ks) * 1.6)
    ax.set_xlabel("% of the frozen centroid value", fontsize=9, color=INK2); ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_facecolor(SURFACE)
    ax.set_title("A  Numerical uncertainty (U_num = mesh + penalty + time) vs\n    the modelling-input interval (bonded / hard mu 0 / hard mu 0.2), centroid",
                 loc="left", fontsize=11, fontweight="bold", color=INK)
    # B: factor effect / U_num matrix
    ax = fig.add_subplot(gs[0, 1]); ax.axis("off"); fac = list(unc[ks[0]]["effects"]); nr, nc = len(fac), len(ks)
    frozen = {"axis (centroid -> MEC)": "CENTROID: CT planning, force-balanced; MEC = held pose",
              "contact law (bonded -> hard mu 0)": "HARD mu 0: lift-off allowed (PVL physics), no crushed tet elements; bonded/mu 0.2 = interval",
              "friction (hard: mu 0 -> 0.2)": "mu 0: not resolvable (<U_num)",
              "penalty x1 -> x10": "x1 (converged): part of U_num", "penalty x1 -> x0.1 (under-converged)": "x0.1 rejected (direction only)",
              "ramp 0.2 -> 0.1 s": "0.2 s: part of U_num", "mesh L3 -> L1": "L3: part of U_num"}
    x0, cw, rh = 0.21, 0.054, 1.0 / (nr + 2.2)
    for j, k in enumerate(ks):
        ax.text(x0 + (j + 0.5) * cw, 1 - 0.6 * rh, SHORT[k].replace(" ", "\n"), ha="center", va="center", fontsize=8.6, fontweight="bold", color=INK2,
                transform=ax.transAxes)
    ax.text(x0 + nc * cw + 0.01, 1 - 0.6 * rh, "frozen choice (why)", ha="left", va="center", fontsize=8.8, fontweight="bold", color=INK2, transform=ax.transAxes)
    for i, nm in enumerate(fac):
        y = 1 - (i + 1.6) * rh; kind = unc[ks[0]]["effects"][nm]["kind"]
        ax.text(0.0, y, nm, ha="left", va="center", fontsize=9, color=INK, fontweight="bold" if kind != "numerical" else "normal", transform=ax.transAxes)
        ax.text(0.0, y - 0.32 * rh, f"{kind}, {unc[ks[0]]['effects'][nm]['n']} pair(s)", ha="left", va="center", fontsize=7.6, color=INK2, transform=ax.transAxes)
        for j, k in enumerate(ks):
            r = unc[k]["effects"][nm]["mean"] / max(unc[k]["U_num"], 1e-12)
            col = "#f4f3ef" if r < 1 else ("#fde3d3" if r < 3 else ("#f6b48f" if r < 10 else "#eb6834"))
            ax.add_patch(plt.Rectangle((x0 + j * cw + 0.002, y - 0.45 * rh), cw - 0.004, 0.9 * rh, color=col, transform=ax.transAxes, lw=0))
            ax.text(x0 + (j + 0.5) * cw, y, f"{r:.1f}", ha="center", va="center", fontsize=9, color=INK if r >= 1 else INK2,
                    fontweight="bold" if r >= 1 else "normal", transform=ax.transAxes)
        ax.text(x0 + nc * cw + 0.01, y, frozen.get(nm, ""), ha="left", va="center", fontsize=8.6, color=INK, transform=ax.transAxes)
        if i in (2,):
            ax.plot([0, 1], [y - 0.55 * rh] * 2, color=INK2, lw=0.8, transform=ax.transAxes)
    ax.text(0.0, 1 - (nr + 1.9) * rh, "cell = mean |change| over the matched pairs / U_num.   grey < 1: not resolvable (numerical noise) -> freeze at default;   "
            "orange >= 1: a real effect -> justify the choice and carry it (interval or scenario).", fontsize=8.6, color=INK2, transform=ax.transAxes)
    ax.set_title("B  Sensitivity judged by the numerical uncertainty: effect of every factor / U_num", loc="left", fontsize=11, fontweight="bold", color=INK)
    # C: axis decision
    ax = fig.add_subplot(gs[1, 0]); ax.axis("off")
    cell = [[SHORT[k], f"{fmt(k, unc[k]['cen']['ref'])} [{fmt(k, unc[k]['cen']['lo'])}, {fmt(k, unc[k]['cen']['hi'])}]",
             f"{fmt(k, unc[k]['mec']['ref'])} [{fmt(k, unc[k]['mec']['lo'])}, {fmt(k, unc[k]['mec']['hi'])}]", "yes" if unc[k]["overlap"] else "no",
             f"{unc[k]['mec_higher']}/{unc[k]['pairs']}", unc[k]["verdict"]] for k in ks]
    tb = ax.table(cellText=cell, colLabels=["QoI", "centroid  ref [full interval]", "MEC  ref [full interval]", "overlap", "MEC higher\n(pairs)", "lower"],
                  cellLoc="center", colWidths=[0.15, 0.24, 0.24, 0.09, 0.11, 0.17], bbox=[0, 0.12, 1, 0.80])
    tb.auto_set_font_size(False); tb.set_fontsize(9)
    for (r_, c_), cl in tb.get_celld().items():
        cl.set_edgecolor(GRID)
        if r_ == 0:
            cl.set_text_props(fontweight="bold", color=INK2); cl.set_facecolor("#f4f3ef")
        elif c_ == 5:
            t = cl.get_text().get_text(); cl.set_facecolor(FAM["cen"][2] if t.startswith("centroid") else (FAM["mec"][2] if t.startswith("MEC") else "white"))
    ax.text(0, 0.07, "robust = the full intervals (every contact alternative ± U_num) do not overlap; paired = they overlap, but MEC - centroid has the same sign\n"
            "in all 9 pairs that differ only by the axis (shared numerical and input errors cancel). User priority: penetration first, then displacement / SED;\n"
            "more PVL accepted. Energies, SED and force scale with E (placeholder material): compare them only relatively.", transform=ax.transAxes, fontsize=8.6, color=INK2, va="top")
    ax.set_title("C  Axis rule: the two scenarios with their full uncertainty", loc="left", fontsize=11, fontweight="bold", color=INK)
    # D: validation light + mass scaling
    ax = fig.add_subplot(gs[1, 1]); ax.axis("off"); ck = [val[t]["checks"] for t in need]
    rng = lambda key, f="{:.2f}": f"{f.format(min(c[key] for c in ck))} - {f.format(max(c[key] for c in ck))}"
    b = val[f"{a.level}_cen_{a.ref}_{a.ramp}"]; ps = unc["perimeter_stretch_pct"]["cen"]
    L1m = val[[t for t in need if t.startswith("L1")][0]]["checks"]["added_mass_pct"]
    est = max(unc[k]["mass_scaling_effect_est"] / max(abs(unc[k]["grid"][2]), 1e-12) * 100 for k in ks)
    left = [("VALIDATION (light): " + CLINICAL["source"], ""),
            ("planned oversizing (perimeter-derived, 07b) = INPUT", f"{b['oversizing_planned_perimeter_pct']:.1f} %   vs clinical {CLINICAL['oversizing_mean_pct']:g} % "
             f"(range {CLINICAL['oversizing_range_pct'][0]:g}-{CLINICAL['oversizing_range_pct'][1]:g} %)  -> inside"),
            ("annulus perimeter after / before (model, centroid)", f"{1 + b['perimeter_stretch_pct'] / 100:.3f}  [{1 + ps['lo'] / 100:.3f}, {1 + ps['hi'] / 100:.3f}]"),
            ("frame perimeter (node 3) / pre-CT annulus perimeter (clinical)", f"{CLINICAL['frame_node3_perimeter_mm'] / CLINICAL['annulus_pre_perimeter_mm']:.3f}  "
             "(different measure: device frame, end-systole)"),
            ("reading", "the prescribed cuff expands fully -> the model gives the UPPER bound of annular stretch;\n"
             "clinical frames conform to the annulus. Model-form question for the compliant-device step.")]
    right = [("SIDE NOTE: fixed mass scaling (DT 1e-5 s) is acceptable", ""),
             ("added mass", f"{b['checks']['added_mass_pct']:.2f} % at {a.level}  (L1 {L1m:.2f} %), constant (scaled once at step start)"),
             ("stable increment start / minimum", f"{min(c['dt_start_s'] for c in ck):.2g} / {min(c['dt_min_s'] for c in ck):.2g} s  (smaller = safer, no extra mass)"),
             ("KE/IE ramp mean | end | hourglass/IE", f"{rng('KE_IE_ramp_pct')} % (< 5) | {rng('KE_IE_end_pct', '{:.3f}')} % (< 1) | {rng('hourglass_pct')} % (< 5)"),
             ("energy balance |ETOTAL| / max(work, IE)", rng("energy_balance_pct") + " %"),
             ("inertia equivalence (inertia ~ m / T^2)", f"ramp 0.2 -> 0.1 s = inertia x4 moved QoIs by u_time;\n+{b['checks']['added_mass_pct']:.2f} % mass -> "
              f"<= {est:.3f} % of any QoI")]
    for col_x, lines in ((0.0, left), (0.53, right)):
        y = 0.97
        for kk, vv in lines:
            ax.text(col_x, y, kk, fontsize=9 if vv else 10, color=INK2 if vv else INK, fontweight="normal" if vv else "bold", transform=ax.transAxes, va="top")
            if vv:
                ax.text(col_x, y - 0.045, vv, fontsize=9.6, color=INK, fontweight="bold", transform=ax.transAxes, va="top"); y -= 0.06 * (1 + vv.count("\n"))
            y -= 0.075
    ax.set_title("D  Validation against clinical CT (light)  ·  mass-scaling side note", loc="left", fontsize=11, fontweight="bold", color=INK)
    fig.suptitle(f"{a.id}: uncertainty quantification and decision ({a.level}, {a.ramp.replace('ramp', 'ramp ')} s, direct FEA surface; "
                 "Roy & Oberkampf 2011 intervals, U_num after V&V 20 / Celik 2008)", x=0.005, y=0.985, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.savefig(path, dpi=125, facecolor=SURFACE); plt.close(fig)


if __name__ == "__main__":
    main()