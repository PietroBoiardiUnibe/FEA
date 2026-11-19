"""
run_batch.py -- run the deployment chain for a LIST of jobs, one after the other; finished jobs are skipped, so it can be stopped and restarted.
Per job:  01 (FEA mesh, only if that level is missing) -> 03 (input) -> Abaqus -> 05 (extract) -> 06 (deformed surface) -> 06b / 07 if they exist.

  python run_batch.py matrix [--axes cen mec]        ALL cross-checks of the cuff-only pilot at L3: axis rule x cuff-wall contact (bonded | hard)
                                                      x friction (hard only: 0 | 0.2) x ramp (0.1 | 0.2 s)  = 12 jobs for two axes, cheapest first
  python run_batch.py grid --levels L1 L2 L3 L4 L5      the SAME model on several mesh levels (finest first); model via --base TAG --opts "03 options"
  python run_batch.py custom --level L3 --job TAG "03 options" [--job TAG "03 options" ...]
  python run_batch.py summary [--level L3]           one table (+ compare/matrix_summary_<level>.csv) of every finished run of the heart
  options:  --id SYN_0001   --full (full device: 02b)   --dry (print the plan only)   --redo (repeat finished jobs)   --fast (no ParaView series)
Tags: <axis>_<contact>[_mu<f>][_ramp<s>], axis cen = centroid (default rule), mec = 07b minimum enclosing circle (device_mec folder, made if missing).
"""
import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import fea_config as cfg

HERE = Path(__file__).resolve().parent
MIN_PER_JOB = {"L1":37,"L2": 26, "L3": 12, "L4": 6, "L5": 3}                              # rough Abaqus minutes at 4 cpus (ramp 0.2 s: x1.7)

def matrix(axes, ramp, penalties=(0.1, 10.0)):
    """FULL FACTORIAL of the cuff-only cross-checks at one mesh level and ramp, where applicable:
    axis rule (cen | mec) x cuff-wall contact (bonded | hard) x friction (hard only: 0 | 0.2) x contact penalty scale (1 | each of penalties).
    The ramp is always passed to 03 explicitly; tags: <axis>_<contact>[_mu<f>][_pen<p>][_ramp<s>] (no _ramp for the legacy 0.1 s)."""
    jobs = []
    for ax in axes:
        for contact in ("bonded", "hard"):
            for mu in ((0.0,) if contact == "bonded" else (0.0, 0.2)):           # friction only matters where the wall can slide
                for pen in (1.0,) + tuple(penalties):
                    tag = (f"{ax}_{contact}" + (f"_mu{mu:g}" if mu else "") + (f"_pen{pen:g}" if pen != 1.0 else "")
                           + (f"_ramp{ramp:g}" if ramp != 0.1 else ""))
                    opts = " ".join(x for x in (f"--contact {contact}" if contact == "hard" else "", f"--friction {mu:g}" if mu else "",
                                                f"--penalty {pen:g}" if pen != 1.0 else "", f"--ramp {ramp:g}",
                                                "--device_dir device_mec" if ax == "mec" else "") if x)
                    jobs.append((tag, opts, (contact == "hard") + (mu > 0) + (pen != 1.0) + (ax == "mec")))
    return [(t, o) for t, o, _ in sorted(jobs, key=lambda j: j[2])]               # fewest changed factors first (stable)

def run(cmd, cwd=HERE):
    p = subprocess.run([str(c) for c in cmd], cwd=str(cwd), capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def job(a, level, tag, opts):
    name = f"{a.id}_{level}_{tag}"; dep = cfg.RUNS / a.id / f"deploy_{level}_{tag}"; sta = dep / f"{name}.sta"
    if sta.exists() and "COMPLETED SUCCESSFULLY" in sta.read_text(errors="ignore") and (dep / f"{name}_results.npz").exists() and not a.redo:
        print(f"  {name}: finished before, skipped (--redo to repeat)"); return "skipped"
    if a.dry:
        print(f"  [plan] {name}: 03 {opts}  (~{MIN_PER_JOB[level]:.0f} min)"); return "planned"
    t0 = time.time(); print(f"  {name}: ", end="", flush=True)
    if not (cfg.RUNS / a.id / "mesh" / f"{a.id}_wall_{level}.vtu").exists():
        run([sys.executable, "01_fea_mesh.py", "--id", a.id, "--level", level]); print("mesh, ", end="", flush=True)
    code, out = run([sys.executable, "03_write_inp.py", "--id", a.id, "--level", level, "--tag", tag] + (["--full"] if a.full else []) + shlex.split(opts))
    if code:
        print("03 FAILED\n" + out[-1500:]); return "03 failed"
    print("input, Abaqus ", end="", flush=True)
    for k in range(10):                                                          # licence server busy / unreachable -> wait and retry
        out = run([cfg.ABAQUS, f"job={name}", f"input={name}.inp", f"cpus={cfg.CPUS}", "interactive", "ask=off"], cwd=dep)[1]
        if not any(m in out for m in ("is not available", "Cannot connect to license")):
            break
        print("(licence busy, waiting 2 min) ", end="", flush=True); time.sleep(120)
    if not (sta.exists() and "COMPLETED SUCCESSFULLY" in sta.read_text(errors="ignore")):
        print("NOT COMPLETED: look at the .sta / .msg in", dep); return "abaqus failed"
    run([cfg.ABAQUS, "python", HERE / "05_extract_odb.py", dep / f"{name}.odb", dep / f"{name}_results.npz"], cwd=dep)
    if (dep / f"{name}_results.npz").exists():                                  # extracted: the Abaqus restart/scratch files are never used
        for ext in ("abq", "pac", "stt", "mdl", "res", "sel", "prt"):
            (dep / f"{name}.{ext}").unlink(missing_ok=True)
    common = ["--id", a.id, "--level", level, "--tag", tag]
    code, out6 = run([sys.executable, "06_deformed.py"] + common + (["--no_series"] if a.fast else []))
    line = " | ".join(l.strip() for l in out6.splitlines() if l.startswith(("seal line", "ring:")))
    for script, extra in (("06b_wall_deployed.py", ["--no_series"] if a.fast else ["--every", "3"]), ("07_checks.py", [])):
        if (HERE / script).exists():
            c, o = run([sys.executable, script] + common + extra)
            if script == "07_checks.py":
                line += " | 07: " + next((l.split("OVERALL:")[1].split("->")[0].strip() for l in o.splitlines() if "OVERALL:" in l), "failed")
    print(f"done in {(time.time() - t0) / 60:.0f} min\n      {line}"); return "completed"


def summary(a):
    """One row per finished run: what was changed, seal, ring, push, quasi-static energies (from the results), 07 verdict."""
    rows = []
    for d in sorted((cfg.RUNS / a.id).glob(f"deploy_{a.level}_*")):
        tag = d.name[len(f"deploy_{a.level}_"):]; name = f"{a.id}_{a.level}_{tag}"; j6, j7, npz = d / "post" / f"{name}_deployed.json", d / "post" / f"{name}_checks.json", d / f"{name}_results.npz"
        if not (j6.exists() and npz.exists()):
            continue
        r = json.loads(j6.read_text()); info = json.loads((d / f"{name}_info.json").read_text()); res = np.load(npz)
        t, ke, ie = res["hist_wall_ALLKE"][:, 0], res["hist_wall_ALLKE"][:, 1], res["hist_wall_ALLIE"][:, 1]
        m = (ie > 0.05 * ie.max()) & (t <= info["time"]["ramp_s"] + 1e-9); q = 100 * ke / np.maximum(ie, 1e-12)
        rows.append(dict(tag=tag, axis=info.get("axis_rule", "mec"), contact=info.get("cuff_wall_contact", "-"), mu=info["friction"], penalty=info.get("penalty_scale",1.0), ramp_s=info["time"]["ramp_s"],
                         seal_open_bins=r["seal_angles_above_0p3mm"], seal_worst_gap_mm=round(r["seal_worst_angle_gap_mm"], 2), ring_SD_mm=round(r["ring_SD_after_mm"], 2),
                         ring_stretch_pct=round(r["ring_stretch_pct"], 1), max_disp_mm=round(r["max_disp_mm"], 1), KE_IE_ramp_pct=round(float(q[m].mean()), 2),
                         KE_IE_end_pct=round(float(q[-1]), 2), quasi_static="ok" if q[m].mean() < 5 and q[-1] < 1 else "NO",
                         check_07=json.loads(j7.read_text())["overall"] if j7.exists() else "-"))
    if not rows:
        print("no finished run with a 06 result found"); return
    print(f"{'tag':26s} {'axis':9s} {'contact':8s} {'mu':>4s} {'ramp':>5s} | {'open':>4s} {'gap':>5s} | {'ringSD':>6s} {'stretch':>7s} {'|u|max':>6s} | {'KE/IE ramp':>10s} {'end':>5s} {'q-static':>8s} | 07")
    for x in rows:
        print(f"{x['tag']:26s} {x['axis']:9s} {x['contact']:8s} {x['mu']:4.1f} {x['ramp_s']:5.2f} | {x['seal_open_bins']:4d} {x['seal_worst_gap_mm']:5.2f} | {x['ring_SD_mm']:6.2f} "
              f"{x['ring_stretch_pct']:6.1f}% {x['max_disp_mm']:6.1f} | {x['KE_IE_ramp_pct']:9.2f}% {x['KE_IE_end_pct']:4.2f}% {x['quasi_static']:>8s} | {x['check_07']}")
    out = cfg.RUNS / a.id / "compare"; out.mkdir(exist_ok=True)
    with open(out / f"matrix_summary_{a.level}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("table ->", out / f"matrix_summary_{a.level}.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("study", choices=("matrix", "grid", "custom", "summary")); ap.add_argument("--id", default="SYN_0001")
    ap.add_argument("--level", default="L3"); ap.add_argument("--levels", nargs="+", default=["L1", "L2", "L3", "L4", "L5"]); ap.add_argument("--axes", nargs="+", default=["cen", "mec"], choices=("cen", "mec"))
    ap.add_argument("--base", default="cen_bonded", help="grid: tag of the model run on every level"); ap.add_argument("--opts", default="", help="grid: 03 options of that model")
    ap.add_argument("--job", nargs=2, action="append", metavar=("TAG", "OPTS"), default=[])
    ap.add_argument("--penalties", type=float, nargs="*", default=[0.1,10.0], help="matrix: contact penalty scale factors besides 1 (none: --penalties)")
    ap.add_argument("--ramp", type=float, default=cfg.T_RAMP, help="matrix: ramp (s) of every job, default = fea_config T_RAMP")
    ap.add_argument("--full", action="store_true"); ap.add_argument("--dry", action="store_true"); ap.add_argument("--redo", action="store_true"); ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    if a.study == "summary":
        return summary(a)
    for dev, cmd in ((("device_full" if a.full else "device"), ["02b_device_full.py" if a.full else "02_device.py"]),
                     *((("device_mec", ["02_device.py", "--axis", "mec", "--out", "device_mec"]),) if a.study == "matrix" and "mec" in a.axes and not a.full else ())):
        if not (cfg.RUNS / a.id / dev / ("anchors_beam.npz" if dev == "device_full" else "expander.vtp")).exists() and not a.dry:
            print(f"{dev} missing: running {cmd[0]}"); print(run([sys.executable] + cmd + ["--id", a.id])[1][-500:])
    jobs = ([(a.level, t, o) for t, o in matrix(a.axes, a.ramp, a.penalties)] if a.study == "matrix" else
            [(L, a.base, a.opts) for L in a.levels] if a.study == "grid" else [(a.level, t, o) for t, o in a.job])
    print(f"{a.study} on {a.id}: {len(jobs)} jobs" + (" (DRY RUN)" if a.dry else ""))
    status = {f"{L}_{t}": job(a, L, t, o) for L, t, o in jobs}
    if not a.dry:
        out = cfg.RUNS / a.id / "compare"; out.mkdir(exist_ok=True); (out / f"batch_{a.study}.json").write_text(json.dumps(status, indent=1))
    print("status:", status)
    if a.study == "grid":
        print(f"next: python 10_compare_runs.py --id {a.id} mesh --levels {' '.join(a.levels)} --variant {a.base}")
    elif a.study == "matrix":
        m=matrix(a.axes,a.ramp, a.penalties)
        print(f"next: python run_batch.py summary --level {a.level}      then   python 10_compare_runs.py --id {a.id} sens --base {a.level}_{m[0][0]} --variants "
              + " ".join(f"{a.level}_{t}" for t, _ in m[1:]) + "   (one factor per comparison: vary --base, see the notes)")



if __name__ == "__main__":
    main()