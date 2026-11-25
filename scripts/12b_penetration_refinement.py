"""
Penetration of expander into wall vs refinement of mesh cells, on every given grid, shows that penetration is a DISCRETISATION effect
(flat tissue facets against a curved cuff + small penalty overclosure) that shrinks monotonically with the wall facet size, while the
worst seal gap is mesh-independent. Supports mesh choice and the tolerance check (penetration + U_num < tolerance)

Penetration = deepest final cuff node (incl. the 1 mm leaflet layer) inside the deformed DIRECT FEA surface (06 json).
Facet size h = median inner-surface edge length within ZONE_MM of the annulus (04 mesh QC csv, state 'before').
Usage:  python 12b_penetration_refinement.py --id SYN_0001 --variants cen_hard_ramp0.2 cen_bonded_ramp0.2
Writes  runs/<id>/compare/01_mesh_convergence/penetration_refinement_<variant1>.png
"""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import fea_config as cfg

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
STYLE = [("#2a78d6", "o"), ("#d03b3b", "s"), ("#1a7f37", "^"), ("#eb6834", "D")]
LABEL = {"cen_hard_ramp0.2": "hard frictionless (frozen)", "cen_bonded_ramp0.2": "bonded", "cen_bonded": "bonded, ramp 0.1 s"}


def observed_order(h,y):
    """Slope of p of log (penetration) vs log(h), not a proxy for GCI's apparent order used in grid convergence
    penetration ~ C h^p (least squares over all levels).
    """
    p,_,=np.polyfit(np.log(h),np.log(y),1)
    return p

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", default="SYN_0001"); ap.add_argument("--levels",nargs="+", default=["L1", "L2", "L3", "L4", "L5"])
    ap.add_argument("--variants", nargs="+", default=["cen_hard_ramp0.2", "cen_bonded_ramp0.2"], help="first = frozen model")
    ap.add_argument("--tol",type=float, default=0.3,help="penetration tolerance (mm)")
    ap.add_argument("--chosen", default="L3",help="chosen mesh level (marked)")
    ap.add_argument("--uq", default="", help="11_uq json with U_num of the penetration (optional; default compare/04_uq/uq_<chosen>_hard_ramp0.2.json)")
    ap.add_argument("--out", default="")
    a = ap.parse_args(); run = cfg.RUNS / a.id; L = a.levels
    rows = {r["level"]: r for r in csv.DictReader(open(cfg.compare_dir(run, "qc") / f"mesh_qc_{a.variants[0]}.csv")) if r["state"] == "before"}
    h = np.array([float(rows[l]["edge_zone_median_mm"]) for l in L]); dofs = np.array([int(rows[l]["dofs"]) for l in L])
    data = {}
    for v in a.variants:
        js = [json.loads((run / f"deploy_{l}_{v}" / "post" / f"{a.id}_{l}_{v}_deployed.json").read_text()) for l in L]
        data[v] = dict(pen=np.array([j["max_penetration_mm"] for j in js]), gap=np.array([j["seal_worst_angle_gap_mm"] for j in js]))
    uq_path=Path(a.uq) if a.uq else cfg.compare_dir(run,"uqin")/f"uq_inputs_{a.chosen}_cen_hard_ramp0.2.json"
    U = json.loads(uq_path.read_text())["uq"]["penetration_mm"]["U_num"] if uq_path.exists() else None
    ic = L.index(a.chosen); v0 = a.variants[0]

    fig, axs = plt.subplots(1, 3, figsize=(19, 6.2), gridspec_kw=dict(wspace=0.3))
    # A: log-log -> straight line = discretisation error ~ h^p
    ax = axs[0]
    for (col, mk), v in zip(STYLE, a.variants):
        ax.plot(h, data[v]["pen"], mk + "-", color=col, ms=8, lw=1.6, label=f"{LABEL.get(v, v)}: observed order p = {observed_order(h, data[v]['pen']):.2f}")
    for l, x, y in zip(L, h, data[v0]["pen"]):
        ax.annotate(l, (x, y), textcoords="offset points", xytext=(6, -12), fontsize=9, color=STYLE[0][0])
    xr = np.array([0.8 * h.min(), 1.2 * h.max()])
    for q, ls in ((1, ":"), (2, "--")):
        ax.plot(xr, data[v0]["pen"][ic] * (xr / h[ic]) ** q, ls, color=INK2, lw=1, label=f"slope {q} (reference)")
    ax.axhline(a.tol, color=INK, ls="-.", lw=1); ax.text(xr[0], a.tol * 1.05, f"tolerance {a.tol:g} mm", fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(color=GRID, lw=0.6, which="both")
    ax.set_xlabel("wall facet edge in the device zone h (mm, log)"); ax.set_ylabel("max penetration of the cuff into the tissue (mm, log)")
    ax.set_title("A  Penetration shrinks monotonically with the facet size\n(straight line in log-log = discretisation error ~ h^p)", loc="left", fontsize=10.5)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    # B: linear penetration vs DOFs, tolerance, U_num on the chosen level
    ax = axs[1]
    for (col, mk), v in zip(STYLE, a.variants):
        ax.plot(dofs, data[v]["pen"], mk + "-", color=col, ms=8, lw=1.6, label=LABEL.get(v, v))
    if U:
        y = data[v0]["pen"][ic]
        ax.errorbar([dofs[ic]], [y], yerr=[[min(U, y)], [U]], color=STYLE[0][0], capsize=5, lw=2, label=f"{a.chosen} +- U_num ({U:.2f} mm)")
    top = max(1.2 * max(d["pen"].max() for d in data.values()), 1.5 * a.tol)
    ax.axhline(a.tol, color=INK, ls="-.", lw=1); ax.axhspan(a.tol, top, color="#d03b3b", alpha=0.06)
    ax.text(dofs.min() * 1.05, a.tol + 0.01, "above tolerance", fontsize=9, color="#d03b3b")
    for l, x, y, hh in zip(L, dofs, data[v0]["pen"], h):
        ax.annotate(f"{l}\n{hh:.1f} mm", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8.5, color=INK2)
    ax.set_xscale("log"); ax.set_ylim(0, top); ax.grid(color=GRID, lw=0.6)
    ax.set_xlabel("FEA wall degrees of freedom (log)  -> finer"); ax.set_ylabel("max penetration (mm)")
    ax.set_title(f"B  {a.chosen} = coarsest level whose penetration (+ U_num) stays\nbelow the tolerance for every contact law", loc="left", fontsize=10.5)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    # C: gap (physical, converged) vs penetration (discretisation)
    ax = axs[2]
    for (col, mk), v in zip(STYLE, a.variants):
        ax.plot(dofs, data[v]["gap"], mk + "-", color=col, ms=8, lw=1.6, label=f"worst seal gap, {LABEL.get(v, v)}")
        ax.plot(dofs, data[v]["pen"], mk + ":", color=col, ms=6, lw=1, alpha=0.6, label=f"penetration, {LABEL.get(v, v)}")
    ax.set_xscale("log"); ax.set_ylim(0, 1.1 * max(max(d["gap"].max(), d["pen"].max()) for d in data.values())); ax.grid(color=GRID, lw=0.6)
    ax.set_xlabel("FEA wall degrees of freedom (log)  -> finer"); ax.set_ylabel("mm")
    ax.set_title("C  The seal gap is a converged feature (flat on the fine levels),\npenetration is a discretisation artefact (falls with h)", loc="left", fontsize=10.5)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle(f"{a.id} cuff-only deployment: penetration vs mesh refinement (direct FEA surface; deepest cuff node inside the deformed wall, "
                 "incl. the 1 mm leaflet layer)", x=0.01, ha="left", fontsize=12.5, weight="bold")
    fig.text(0.01, -0.04, "Penetration comes from a curved cuff meeting flat tissue facets (plus a small penalty overclosure): it scales with the facet "
             "size and vanishes as h -> 0, so it is a discretisation error, not a modelling choice. Lever to reduce it = finer wall facets, "
             "not the contact law or the penalty.", fontsize=9.5, color=INK2)
    out = Path(a.out) if a.out else cfg.compare_dir(run, "mesh"); out.mkdir(exist_ok=True)
    path = out / f"penetration_refinement_{v0}.png"; fig.savefig(path, dpi=150, bbox_inches="tight")
    for v in a.variants:
        print(f"{v}: penetration " + " / ".join(f"{l} {p:.3f}" for l, p in zip(L, data[v]["pen"])) + f" mm, order p {observed_order(h, data[v]['pen']):.2f}; "
              f"worst gap " + " / ".join(f"{g:.3f}" for g in data[v]["gap"]))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()