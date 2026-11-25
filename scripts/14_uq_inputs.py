"""
Input / model-form UQ summary of the frozen cuff-only model + the run index of the whole study.

1 Numerical band of uncertainty U_num of the frozen FEA baseline (L3_cen_hard_ramp0.2) for every QoI, same recipe as 11:
u_mesh = |phi_L3 - phi_L1| + U(L1) (Celik GCI / spread band on L1-L2-L3), u_pen = |phi(x10) - phi(x1)|, u_time = |phi(ramp 0.1) - phi(ramp 0.2)|
   QoIs: shape / sizing / seal (06 json + script 10 on the FEA mesh) AND tissue loading from 13 (SED, von Mises, principal stress and log strain,
   contact pressure; device-zone, volume-weighted). Local geometric QoIs carry an ABSOLUTE U_num, integral / mechanical ones a RELATIVE one.
2 INPUT cases (model forms, stiffer annulus, ...): every QoI per case, interval [min, max] over the cases (baseline excluded unless listed),
   widened by U_num: [min - U, max + U] (Roy & Oberkampf 2011: epistemic inputs as intervals).
3 Per-run GATE (credibility checks): quasi-static KE/IE, added mass, hourglass, energy balance, penetration <= 0.3 mm, leaflet-layer room
   (1 mm - penetration) >= 0.7 mm, mesh QC after deformation (04), exact J range and elements above the 2.5 MPa flag (13).
4 RUN_INDEX.csv: one row per deploy folder of the heart (stage, what changed, status, gate values, max values + locations).
Usage:  python 14_uq_inputs.py --id SYN_0001 --cases L3_cen_hard_ramp0.2_lawNHra L3_cen_hard_ramp0.2_lawHGO ... [--labels "NH RA" "HGO" ...]
Writes  compare/07_uq_inputs/uq_inputs_<base>.json/.csv/.png  and  compare/RUN_INDEX.csv
"""

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import fea_config as cfg
import fea_common as fc


INK, INK2, GRID, BLUE, ORANGE, RED, GREEN = "#0b0b0b", "#52514e", "#e1e0d9", "#2a78d6", "#eb6834", "#d03b3b", "#1a7f37"
LAYER_MM, PEN_TOL, ROOM_MIN = 1.0, 0.3, 0.7                      # native-leaflet layer, penetration tolerance, minimum room left for the layer
RESOLUTION = {"open_bins": 1.0}                                   # smallest resolvable change of a count QoI (one 3-deg seal bin)
# key, label, kind of U_num (abs | rel), source
SHAPE = [("penetration_mm", "max penetration (mm)", "abs"), ("leaflet_room_mm", "leaflet-layer room (mm)", "abs"),
         ("worst_gap_mm", "worst seal gap (mm)", "abs"), ("open_bins", "open seal bins > 0.3 mm", "abs"),
         ("perimeter_stretch_pct", "annulus perimeter stretch (%)", "abs"), ("area_increase_pct", "annulus area increase (%)", "abs"),
         ("u_rms_area_mm", "RMS wall displacement (mm)", "rel"), ("cavity_change_ml", "cavity change (ml)", "rel"),
         ("strain_energy_mJ", "wall strain energy (mJ)", "rel"), ("radial_force_N", "cuff radial force (N)", "rel")]
FIELDS = [("SED_kPa", "p99", "SED p99 (kPa)"), ("SED_kPa", "mean", "SED mean (kPa)"), ("VM_kPa", "p99", "von Mises p99 (kPa)"),
          ("S_maxP_kPa", "p99", "max principal stress p99 (kPa)"), ("S_maxP_kPa", "max", "max principal stress max (kPa)"),
          ("S_minP_kPa", "min", "min principal stress min (kPa)"), ("LE_maxP_pct", "p99", "max principal log strain p99 (%)"),
          ("LE_maxP_pct", "max", "max principal log strain max (%)")]
CPRESS = [("peak_kPa", "contact pressure peak (kPa)"), ("p95_kPa", "contact pressure p95 (kPa)")]

def load(name):
    spec = importlib.util.spec_from_file_location("m_" + name, cfg.SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def stage(tag):
    """What a run is for, from its tag (tag convention of the study)."""
    L, rest = tag.split("_", 1) if "_" in tag else (tag, "")
    if any(k in rest for k in ("_law", "_band", "_E")): return "UQ input"
    if "_depth" in rest: return "scenario: landing depth"
    if "crimp" in rest: return "UQ boundary condition"
    if rest in ("cen_hard_pen10_ramp0.2", "cen_hard"): return "U_num companion"
    if L != "L3": return "mesh grid"
    return "sensitivity (V&V)"


def run_values(s10, sid, run, t, plan, b07, pfa):
    """Every QoI of one run (shape: 06 + script 10; fields: 13 json) + its gate checks; None for what is missing."""
    dep = run / f"deploy_{t}"; name = f"{sid}_{t}"; v, ck = {}, {}
    j6 = dep / "post" / f"{name}_deployed.json"
    if j6.exists():
        d6 = json.loads(j6.read_text()); sz = d6["sizing"]
        v.update(penetration_mm=d6["max_penetration_mm"], leaflet_room_mm=LAYER_MM - max(d6["max_penetration_mm"], 0.0),
                 worst_gap_mm=d6["seal_worst_angle_gap_mm"], open_bins=d6["seal_angles_above_0p3mm"],
                 perimeter_stretch_pct=sz["perimeter_stretch_pct"], area_increase_pct=sz["area_increase_pct"])
        q = s10.fea_qois(sid, t, run, plan, b07, pfa)
        v.update({k: q["q"][k] for k in ("u_rms_area_mm", "cavity_change_ml", "strain_energy_mJ", "radial_force_N")})
        ck.update(KE_IE_ramp_pct=q["qs"]["mean_ramp_pct"], KE_IE_end_pct=q["qs"]["end_pct"])
        res = np.load(dep / f"{name}_results.npz")
        ck.update(fc.solver_log(dep, name), hourglass_pct=float(100 * res["hist_wall_ALLAE"][-1, 1] / res["hist_wall_ALLIE"][-1, 1]),
                  energy_balance_pct=float(100 * np.abs(res["hist_ETOTAL"][:, 1]).max() / max(np.abs(res["hist_ALLWK"][:, 1]).max(), res["hist_ALLIE"][:, 1].max())))
    j13 = dep / "post" / f"{name}_fields.json"
    if j13.exists():
        f = json.loads(j13.read_text()); z = f["final"]["zone"]
        v.update({f"{k}_{s}": z[k][s] for k, s, _ in FIELDS})
        v.update({f"CPRESS_{k}": f["final"]["CPRESS"][k] for k, _ in CPRESS})
        ck.update(J_min=z["J"]["min"], J_max=z["J"]["max"], above_flag_tets=f["final"]["above_rupture_flag_tets"])
        v["_max_location"] = f["final"]["max_location"]
    L = t.split("_")[0]; qc = cfg.compare_dir(run, "qc") / f"mesh_qc_{t.split('_', 1)[1]}.csv" if "_" in t else None
    if qc and qc.exists():
        row = [r for r in csv.DictReader(open(qc)) if r["level"] == L and r["state"] == "end"]
        if row:
            ck.update(qc_verdict=row[0]["verdict"], qc_inverted=int(row[0]["inverted"]), qc_sicn_min=float(row[0]["sicn_min"]), qc_J_min=float(row[0]["J_min"]))
    test = lambda x, ok: None if x is None else bool(ok(x))                   # None = output not available for this run
    qs = None if "KE_IE_ramp_pct" not in ck else (ck["KE_IE_ramp_pct"], ck["KE_IE_end_pct"])
    gate = {"quasi-static": test(qs, lambda q: q[0] < 5 and q[1] < 1), "added mass < 2 %": test(ck.get("added_mass_pct"), lambda x: x < 2),
            "hourglass < 5 %": test(ck.get("hourglass_pct"), lambda x: x < 5), "energy balance < 1 %": test(ck.get("energy_balance_pct"), lambda x: x < 1),
            f"penetration <= {PEN_TOL} mm": test(v.get("penetration_mm"), lambda x: x <= PEN_TOL),
            f"leaflet room >= {ROOM_MIN} mm": test(v.get("leaflet_room_mm"), lambda x: x >= ROOM_MIN),
            "mesh QC: 0 inverted": test(ck.get("qc_inverted"), lambda x: x == 0), "no tet above 2.5 MPa": test(ck.get("above_flag_tets"), lambda x: x == 0)}
    return v, ck, gate


def keys():
    """(key, label, kind) of every QoI."""
    return SHAPE + [(f"{k}_{s}", lab, "rel") for k, s, lab in FIELDS] + [(f"CPRESS_{k}", lab, "rel") for k, lab in CPRESS]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", default="SYN_0001"); ap.add_argument("--base", default="L3_cen_hard_ramp0.2")
    ap.add_argument("--cases", nargs="*", default=[], help="run tags of the input / model-form cases"); ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--out", default="")
    a = ap.parse_args(); run = cfg.RUNS / a.id; sid = a.id
    s10= load("10_compare_runs")
    plan, b07, pfa = s10.common(SimpleNamespace(id=sid))
    L1, L2 = (a.base.replace("L3_", f"{L}_", 1) for L in ("L1", "L2"))
    comp = {"pen": "L3_cen_hard_pen10_ramp0.2", "time": "L3_cen_hard"}
    tags = [a.base, L1, L2, *comp.values(), *a.cases]
    V = {}
    for t in dict.fromkeys(tags):
        if (run / f"deploy_{t}").exists():
            V[t] = run_values(s10, sid, run, t, plan, b07, pfa)
    N = [s10.zone_tets(run, sid, L) for L in ("L1", "L2", "L3")]
    labels = dict(zip(a.cases, a.labels)) if a.labels else {c: c.replace(a.base + "_", "") for c in a.cases}
    unc, rows = {}, []
    for k, lab, kind in keys():
        b = V[a.base][0].get(k)
        if b is None:
            continue
        f = [V.get(t, ({},))[0].get(k) for t in (L1, L2, a.base)]
        if None in f:
            um, how = None, "grid values missing"
        else:
            U1, how = fc.mesh_U(s10, f, N); um = abs(f[2] - f[0]) + U1
        up = abs(V[comp["pen"]][0][k] - b) if comp["pen"] in V and k in V[comp["pen"]][0] else None
        ut = abs(V[comp["time"]][0][k] - b) if comp["time"] in V and k in V[comp["time"]][0] else None
        parts = [x for x in (um, up, ut) if x is not None]; U = max(float(sum(parts)), RESOLUTION.get(k, 0.0)) if parts else None
        complete = um is not None and up is not None and ut is not None
        cv = {c: V[c][0].get(k) for c in a.cases if c in V and V[c][0].get(k) is not None}
        o = dict(label=lab, kind=kind, base=b, U_num=U, U_num_pct=(100 * U / abs(b) if U is not None and abs(b) > 1e-12 else None),
                 U_num_complete=complete, u_mesh=um, u_pen=up, u_time=ut, mesh_method=how, cases=cv)
        if cv:
            lo, hi = min(cv.values()), max(cv.values()); Ur = U or 0.0
            sc = (lambda x: Ur * abs(x) / abs(b)) if kind == "rel" and abs(b) > 1e-12 else (lambda x: Ur)
            o.update(input_lo=lo, input_hi=hi, lo=lo - sc(lo), hi=hi + sc(hi), case_min=min(cv, key=cv.get), case_max=max(cv, key=cv.get))
        unc[k] = o
        rows.append(dict(qoi=lab, base=b, U_num=U, U_num_pct=o["U_num_pct"], complete=complete, **{labels.get(c, c): x for c, x in cv.items()},
                         interval_lo=o.get("lo"), interval_hi=o.get("hi")))
    out = Path(a.out) if a.out else cfg.compare_dir(run, "uqin"); out.mkdir(parents=True, exist_ok=True); stem = f"uq_inputs_{a.base}"
    gates = {t: dict(checks=V[t][1], gate=V[t][2], max_location=V[t][0].get("_max_location")) for t in V}
    (out / f"{stem}.json").write_text(json.dumps(dict(base=a.base, cases=a.cases, labels=labels, companions=comp, uq=unc, gates=gates), indent=1, default=float))
    with open(out / f"{stem}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r))); w.writeheader(); w.writerows(rows)
    # figure: one panel per QoI (baseline +- U_num, every case +- U_num), gate table
    ks = [k for k in unc]; nc = 6; nr = int(np.ceil(len(ks) / nc)) + 1
    fig = plt.figure(figsize=(3.3 * nc, 2.7 * nr)); gs = fig.add_gridspec(nr, nc, hspace=0.75, wspace=0.45)
    names = ["baseline"] + [labels[c] for c in a.cases if c in V]
    for i, k in enumerate(ks):
        o = unc[k]; ax = fig.add_subplot(gs[i // nc, i % nc]); U = o["U_num"] or 0.0
        vals = [o["base"]] + [o["cases"].get(c, np.nan) for c in a.cases if c in V]
        err = [U] + [(U * abs(x) / abs(o["base"]) if o["kind"] == "rel" and abs(o["base"]) > 1e-12 else U) for x in vals[1:]]
        ax.bar(range(len(vals)), vals, color=[INK2] + [BLUE] * (len(vals) - 1), yerr=err, capsize=3, ecolor=INK)
        ax.set_xticks(range(len(vals))); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
        ax.set_title(o["label"] + ("" if o["U_num_complete"] else "\n(U_num partial)"), fontsize=8.5, loc="left")
        ax.tick_params(labelsize=7); [ax.spines[s].set_visible(False) for s in ("top", "right")]
        if k == "penetration_mm": ax.axhline(PEN_TOL, color=RED, ls="--", lw=1)
        if k == "leaflet_room_mm": ax.axhline(ROOM_MIN, color=RED, ls="--", lw=1)
    ax = fig.add_subplot(gs[nr - 1, :]); ax.axis("off")
    gnames = list(next(iter(gates.values()))["gate"])
    lines = ["GATE per run (credibility checks; x = failed, - = not available):", f"{'run':42s} " + " | ".join(g[:18] for g in gnames)]
    for t, g in gates.items():
        lines.append(f"{t:42s} " + " | ".join(f"{'-' if g['gate'][n] is None else ('ok' if g['gate'][n] else 'x'):^18s}" for n in gnames))
    ml = V[a.base][0].get("_max_location") or {}
    lines.append("baseline maxima: " + "; ".join(f"{kk} at {m['theta_deg']:.0f} deg, z {m['z_mm']:+.1f} mm (element {m['element']})" for kk, m in ml.items()))
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=7.5, color=INK)
    fig.suptitle(f"{sid}: input / model-form UQ around {a.base}  (bars = value +- U_num of the baseline; relative QoIs scaled)", x=0.01, ha="left", fontsize=12, weight="bold")
    fig.savefig(out / f"{stem}.png", dpi=140, bbox_inches="tight")
    # run index of the whole heart
    idx = []
    for d in sorted(run.glob("deploy_*")):
        t = d.name[len("deploy_"):]; name = f"{sid}_{t}"; inf = d / f"{name}_info.json"
        if not inf.exists():
            continue
        info = json.loads(inf.read_text()); sta = d / f"{name}.sta"
        st = "completed" if sta.exists() and "COMPLETED SUCCESSFULLY" in sta.read_text(errors="ignore") else "not completed"
        r = dict(tag=t, stage=stage(t), status=st, contact=info.get("cuff_wall_contact"), friction=info.get("friction"), penalty=info.get("penalty_scale", 1.0),
                 ramp_s=info["time"]["ramp_s"], dt_target_s=info["time"].get("dt_target_s"), material=(info.get("material") or {}).get("model"),
                 E_MPa=(info.get("material") or {}).get("E_MPa"), axis=info.get("axis_rule"))
        if st == "completed" and t in V:
            v, ck, g = V[t]
        elif st == "completed":
            try:
                v, ck, g = run_values(s10, sid, run, t, plan, b07, pfa)
            except Exception as e:                                         # old runs without some outputs: keep the row, note why
                v, ck, g = {}, {"note": str(e)[:80]}, {}
        else:
            v, ck, g = {}, {}, {}
        r.update({k: v.get(k) for k in ("penetration_mm", "leaflet_room_mm", "worst_gap_mm", "S_maxP_kPa_max", "LE_maxP_pct_max", "CPRESS_peak_kPa")})
        r.update({k: ck.get(k) for k in ("KE_IE_ramp_pct", "KE_IE_end_pct", "added_mass_pct", "hourglass_pct", "qc_verdict", "J_min", "above_flag_tets", "note")})
        r["gate_passed"] = (all(x for x in g.values() if x is not None) if g else None)
        ml = v.get("_max_location") or {}
        r["max_S_maxP_at"] = (f"{ml['S_maxP_kPa']['theta_deg']:.0f} deg, z {ml['S_maxP_kPa']['z_mm']:+.1f} mm" if "S_maxP_kPa" in ml else None)
        idx.append(r)
    ri = (Path(a.out) if a.out else run / "compare") / "RUN_INDEX.csv"
    with open(ri, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in idx for k in r))); w.writeheader(); w.writerows(idx)
    print(f"U_num of {a.base} (" + ", ".join(f"{n}: {'ok' if comp[c] in V else 'MISSING'}" for c, n in (("pen", "penalty x10"), ("time", "ramp 0.1"))) + "):")
    for k, o in unc.items():
        s = f"  {o['label']:36s} base {o['base']:9.4g}  U_num {o['U_num'] if o['U_num'] is not None else float('nan'):8.3g}"
        s += f" ({o['U_num_pct']:.1f} %)" if o["U_num_pct"] is not None else ""
        s += "" if o["U_num_complete"] else "  [partial]"
        if "lo" in o:
            s += f" | cases [{o['input_lo']:.4g}, {o['input_hi']:.4g}] -> interval [{o['lo']:.4g}, {o['hi']:.4g}] (max: {labels.get(o['case_max'], o['case_max'])})"
        print(s)
    def verdict(g):
        bad = [n for n, ok in g.items() if ok is False]; na = [n for n, ok in g.items() if ok is None]
        return ("PASS" if not bad else "FAIL " + ", ".join(bad)) + (f" (not available: {', '.join(na)})" if na else "")
    print("gates: " + "; ".join(f"{t}: {verdict(g['gate'])}" for t, g in gates.items()))
    print(f"wrote {out / (stem + '.json')} / .csv / .png and {ri} ({len(idx)} runs)")


if __name__ == "__main__":
    main()
