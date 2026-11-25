"""
The whole input/model form UQ of the frozen skirt only model in ONE command.
estartable: relaunch the same command and
everything already finished is skipped (Abaqus runs AND outputs), only what is missing is made.
 1  run_batch custom: one job per wall model form (frozen setup L3, hard frictionless contact, ramp 0.2 s time-scaled by 03)
    -> Abaqus -> 05, 06, 06b, 07, 04 mesh QC, 13 fields, 15b ParaView views
 2  14_uq_inputs: U_num of the frozen base, model-form intervals +- U_num, per-run gate, RUN_INDEX.csv   (only once EVERY case is complete)
 3  run_batch summary: one table of every finished run of the level
Usage:  python run_uq.py [--id SYN_0001] [--level L3] [--dry] [--wait_h 10]
"""
import argparse
import subprocess
import sys
import fea_config as cfg
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE,OPTS="cen_hard_ramp0.2", "--contact hard --ramp 0.2"           # the frozen V&V setup: only the wall law changes
CASES = [("lawFUNG", "FUNG_kolawole", "Fung (Kolawole)"),               # cheapest first: time scale x1.45
         ("lawHGO", "HGO_wang", "HGO (Wang, nominal)"),                # x5.4
         ("lawNH", "NH_nordsletten", "NH (Nordsletten)")]              # x12.95 (~1 h)
NEEDED = ("_deployed.json", "_checks.json", "_fields.json")             # 06, 07, 13: what 14 reads


def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", default="SYN_0001"); ap.add_argument("--level", default="L3")
    ap.add_argument("--dry", action="store_true"); ap.add_argument("--wait_h",type=float,default=10.0)
    a=ap.parse_args(); py=sys.executable
    batch=[py, HERE / "run_batch.py", "custom", "--id", a.id, "--level", a.level, "--wait_h", a.wait_h] + (["--dry"] if a.dry else [])
    for suffix, mat,_ in CASES:
        batch += ["--job", f"{BASE}_{suffix}", f"{OPTS} --mat {mat}"]
    print("=== 1  Abaqus runs + every output (finished work is skipped)", flush=True)
    subprocess.run([str(c) for c in batch], cwd=HERE)
    if a.dry:
        return
    names = [f"{a.level}_{BASE}_{s}" for s, _, _ in CASES]
    missing = [n for n in names for e in NEEDED if not (cfg.RUNS / a.id / f"deploy_{n}" / "post" / f"{a.id}_{n}{e}").exists()]
    if missing:
        print(f"NOT COMPLETE: {sorted(set(missing))} -> relaunch the same command (licence back / script fixed)"); sys.exit(1)
    print("=== 2  UQ summary (14): U_num, intervals, gate, RUN_INDEX", flush=True)
    subprocess.run([py, str(HERE / "14_uq_inputs.py"), "--id", a.id, "--cases", *names, "--labels", *[l for *_, l in CASES]], cwd=HERE)
    print("=== 3  table of every finished run", flush=True)
    subprocess.run([py, str(HERE / "run_batch.py"), "summary", "--id", a.id, "--level", a.level], cwd=HERE)

if __name__=="__main__":
    main()

    