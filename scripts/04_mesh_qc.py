"""
Size and element quality of FEA wall meshes, BEFORE the simulation (every level) and AFTER it too (deformed tets of every run of one variant).
It is device agnostic; works for any heart and level, measured on the corned nodes (linear tets): the C3D10M midisde nodes start at mid-edge, so the corner tet IS the element shape 
before the run; after the run it is the corner-node approximation of the deformed element.

Indices:
1) SICN: signed inverse condition number of the element Jacobian wrt the regular tet. 1 regular, -> 0 flat, < 0 inverted (stiffness conditioning).
2) Shape:  V / V of the regular tet with the same circumsphere = Abaqus' 'tetrahedral quality measure'; skewness = 1 - shape
3) Aspect-Ratio: ongest edge / shortest height, normalised to 1 for the regular tet (Abaqus 'Verify mesh' warns above 10)
4) Dihedral: smallest / largest dihedral angle (deg): slivers approach 0 / 180 deg
5)J: deformed / undeformed volume (after only): < 0 inverted, small = crushed
6) Size: tets, corner nodes, quadratic nodes, DOFs, inner-surface vertices and edge length in the device zone, layers across the wall

After: also Abaqus' own count of distorted elements (.dat) and the added mass of the fixed mass scaling (.sta).
Usage:  python 04_mesh_qc.py --id SYN_0001                          before, every level found
        python 04_mesh_qc.py --id SYN_0001 --tag cen_bonded_ramp0.2  + after, every level that has deploy_<level>_<tag>
Writes  runs/<id>/mesh/mesh_qc.csv + .png (before)   and   runs/<id>/compare/mesh_qc_<tag>.csv + .png (before + after)
"""
import argparse
import csv
import json
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import vtk
from scipy.spatial import cKDTree 
from vtkmodules.util.numpy_support import vtk_to_numpy

import fea_config as cfg


SICN_MIN,ASPECT_P99, J_MIN, MASS_PCT=0.1,10.0,0.2,5.0 # acceptance (flags; 0 inverted is the gate). Dihedral angles for sliver identification is reported but not made a rule
DIHEDRAL=(5.0,170.0) #only for the reported share of tets outside range (crushed wall due to radial expansion near TV annulus)
W_INV=np.linalg.inv(np.array([[1, 0.5, 0.5], [0, np.sqrt(3) / 2, np.sqrt(3) / 6], [0, 0, np.sqrt(2 / 3)]])) #regular tet with unit edges
INK, INK2, BLUE, ORANGE, RED, GRID, SURFACE = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#d03b3b", "#e1e0d9", "#fcfcfb"

def tet_metrics(X,T):
    """Quality related indices of every tet, with corner nodes X, connectivity T"""
    A,B,C,D=(X[T[:,i]] for i in range (4))
    Jm=np.stack([B-A,C-A,D-A],axis=2) # edge vectors as columns
    V=np.linalg.det(Jm)/6 #volume
    M=Jm@W_INV; Mi=np.linalg.inv(M)       # map from the regular tet
    sicn=np.sign(V)*3/(np.linalg.norm(M,axis=(1,2))*np.linalg.norm(Mi,axis=(1,2)))
    faces=[(B, C, D), (A, C, D), (A, B, D), (A, B, C)]
    Af = np.stack([0.5 * np.linalg.norm(np.cross(q - p, r - p), axis=1) for p, q, r in faces], 1)
    a,b,c=B-A,C-A,D-A
    num=(np.einsum("ij,ij->i", a, a)[:, None] * np.cross(b, c) + np.einsum("ij,ij->i", b, b)[:, None] * np.cross(c, a)
           + np.einsum("ij,ij->i", c, c)[:, None] * np.cross(a, b))
    R=np.linalg.norm(num,axis=1)/(12*np.abs(V)+1e-30)
    shape=np.abs(V)/(8*np.sqrt(3)/27*R**3)
    E=np.stack([np.linalg.norm(p - q, axis=1) for p, q in ((A, B), (A, C), (A, D), (B, C), (B, D), (C, D))], 1)
    aspect=E.max(1)/(3*np.abs(V)[:,None]/Af).min(1)/np.sqrt(3/2)
    P4 = [A, B, C, D]; nrm = []                                                          # outward unit normal of each face (away from its opposite vertex)
    for i in range(4):
        f = [j for j in range(4) if j != i]
        n = np.cross(P4[f[1]] - P4[f[0]], P4[f[2]] - P4[f[0]]); n = n / np.linalg.norm(n, axis=1, keepdims=True)
        nrm.append(-n * np.sign(np.einsum("ij,ij->i", n, P4[i] - P4[f[0]]))[:, None])
    dih = np.stack([180 - np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", nrm[i], nrm[j]), -1, 1))) for i in range(4) for j in range(i + 1, 4)], 1)
    return dict(V=V, sicn=sicn, shape=shape, aspect=aspect,dmin=dih.min(1), dmax=dih.max(1))


def summary(m,V0=None):
    """The numbers of one mesh state with the acceptance flag"""
    p=lambda x,q: float(np.percentile(x,q))
    out_dih=(m["dmin"]<DIHEDRAL[0])|(m["dmax"]>DIHEDRAL[1])
    s = dict(inverted=int((m["V"] <= 0).sum()),sicn_min=float(m["sicn"].min()), sicn_p1=p(m["sicn"],1), sicn_median=p(m["sicn"],50),
             shape_min=float(m["shape"].min()), shape_below_0p02=int((m["shape"] < 0.02).sum()), skewness_max=float(1 - m["shape"].min()),
             aspect_p99=p(m["aspect"], 99), aspect_max=float(m["aspect"].max()), aspect_above_10=int((m["aspect"] > 10).sum()),
             dihedral_min=float(m["dmin"].min()), dihedral_max=float(m["dmax"].max()), dihedral_outside_pct=float(100 * out_dih.mean()))
    flags = [f for f, bad in (("inverted", s["inverted"] > 0), (f"SICN<{SICN_MIN:g}", s["sicn_min"] < SICN_MIN),
                              (f"aspect p99>{ASPECT_P99:g}", s["aspect_p99"] > ASPECT_P99)) if bad]

    if V0 is not None:
        J=m["V"]/V0; s.update(J_min=float(J.min()), J_p1=p(J,1), J_max=float(J.max()))
        flags+= [f"J<{J_MIN:g}"] if s["J_min"] < J_MIN else []
    s["verdict"]="FAIL: inverted" if s["inverted"] else ("PASS" if not flags else "FLAG:" + ",".join(flags))
    return s


def read_mesh(run,sid,level):
    r=vtk.vtkXMLUnstructuredGridReader(); r.SetFileName(str(run/"mesh"/f"{sid}_wall_{level}.vtu")); r.Update(); g=r.GetOutput()
    X=vtk_to_numpy(g.GetPoints().GetData()).astype(float); T=vtk_to_numpy(g.GetCells().GetConnectivityArray()).reshape(-1,4)
    pd={k: vtk_to_numpy(g.GetPointData().GetArray(k)) for k in ("surface", "tv_annulus")}
    mp=np.load(run/"mesh"/ f"{sid}_wall_{level}_map.npz")
    return X,T,pd,mp

def size(X,T,pd,mp):
    """Degrees of Freedom (DoFs) and spacing per-level"""
    edges=np.unique(np.sort(np.concatenate([T[:,[i,j]] for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))]),1),axis=0)
    cn,F=mp["coarse_nodes"],mp["faces"]; P=X[cn]
    zone=cKDTree(X[pd["tv_annulus"]==1]).query(P)[0]<cfg.ZONE_MM
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); el = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1); ez = zone[E[:, 0]] & zone[E[:, 1]]
    V=np.abs(np.linalg.det(np.stack([X[T[:,k]]-X[T[:,0]] for k in (1,2,3)], axis=2)))/6
    h = np.sqrt(2 / 3) * (6 * np.sqrt(2) * V) ** (1 / 3)                                  # height of the regular tet with the same volume
    tz = cKDTree(X[pd["tv_annulus"] == 1]).query(X[T].mean(1))[0] < cfg.ZONE_MM
    nq=len(X)+len(edges)

    return dict(tets=len(T), corner_nodes=len(X),quadratic_nodes=nq,dofs=3*nq, inner_vertices=len(cn),
    edge_zone_median_mm=float(np.median(el[ez])),edge_all_median_mm=float(np.median(el)),
                layers_zone=float(cfg.WALL_THICKNESS / np.median(h[tz])), layers_all=float(cfg.WALL_THICKNESS / np.median(h)))

def abaqus_logs(dep,name):
    """Abaqus' own distorted element count (datacheck) and the added mass of the fixed mass scaling parameters to keep target time stepping length (first increment, %)"""
    dat=dep/ f"{name}.dat"; sta = dep / f"{name}.sta"; d = m = float("nan")
    if dat.exists():
        hit=re.search(r"(\d+) elements are distorted", dat.read_text(errors="ignore")); d=int(hit.group(1)) if hit else 0
    if sta.exists():
        rows=[l.split() for l in sta.read_text(errors="ignore").splitlines() if re.match(r"\s+\d+\s+\d\.\d+E[+-]\d+", l)]
        m = float(rows[0][-1]) if rows else float("nan")
    return d,m

def figure(path, sid, title, rows, metrics_by):
    """A: DOFs; B-D: SICN, Aspect ratio, smaller dihedral angle per level, boxes p1-p99, dots=worst; E: verdict table"""
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    levels = [r["level"] for r in rows if r["state"] == "before"]; states = sorted({r["state"] for r in rows}, key=lambda s: ["before", "end of ramp", "end"].index(s))
    col = {"before": BLUE, "end of ramp": "#86b6ef", "end": ORANGE}
    fig, axs = plt.subplots(2, 3, figsize=(19, 9.5), facecolor=SURFACE, gridspec_kw=dict(hspace=0.45, wspace=0.28))
    ax = axs[0, 0]; dofs = [next(r for r in rows if r["level"] == L)["dofs"] for L in levels]
    ax.bar(levels, dofs, color=BLUE); ax.set_yscale("log"); ax.set_title("Size: degrees of freedom of the FEA wall", loc="left", fontsize=11, fontweight="bold", color=INK)
    for x, (L, v) in enumerate(zip(levels, dofs)):
        r0 = next(r for r in rows if r["level"] == L)
        ax.text(x, v * 1.15, f"{v / 1000:.0f}k\n{r0['tets']:,} tets\nedge {r0['edge_zone_median_mm']:.2f} mm\n~{r0['layers_zone']:.1f} layers", ha="center", fontsize=8, color=INK)
    ax.set_ylim(min(dofs) / 2, max(dofs) * 12)
    for ax, key, lab, lim, logy in ((axs[0, 1], "sicn", "SICN (1 = regular, < 0 = inverted)", SICN_MIN, False), (axs[0, 2], "aspect", "aspect ratio (1 = regular)", ASPECT_P99, True),
                                    (axs[1, 0], "dmin", "smallest dihedral angle (deg, reported)", None, False)):
        w = 0.8 / len(states)
        for i, st in enumerate(states):
            data = [metrics_by[(L, st)][key] for L in levels if (L, st) in metrics_by]; xs = [levels.index(L) + (i - (len(states) - 1) / 2) * w for L in levels if (L, st) in metrics_by]
            if not data:
                continue
            bp = ax.boxplot(data, positions=xs, widths=w * 0.85, whis=(1, 99), showfliers=False, patch_artist=True, medianprops=dict(color=INK))
            for b in bp["boxes"]:
                b.set(facecolor=col[st], alpha=0.75, edgecolor=col[st])
            worst = [d.max() if key == "aspect" else d.min() for d in data]; ax.scatter(xs, worst, color=col[st], edgecolor=INK, s=28, zorder=5)
        if lim is not None:
            ax.axhline(lim, color=RED, lw=1, ls="--")
        ax.set_xticks(range(len(levels))); ax.set_xticklabels(levels)
        if logy:
            ax.set_yscale("log")
        ax.set_title(lab, loc="left", fontsize=11, fontweight="bold", color=INK); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax = axs[1, 1]
    if any(r["state"] != "before" for r in rows):
        for i, st in enumerate([s for s in states if s != "before"]):
            data = [metrics_by[(L, st)]["J"] for L in levels if (L, st) in metrics_by]; xs = [levels.index(L) + (i - 0.5) * 0.4 for L in levels if (L, st) in metrics_by]
            bp = ax.boxplot(data, positions=xs, widths=0.34, whis=(1, 99), showfliers=False, patch_artist=True, medianprops=dict(color=INK))
            for b in bp["boxes"]:
                b.set(facecolor=col[st], alpha=0.75, edgecolor=col[st])
            ax.scatter(xs, [d.min() for d in data], color=col[st], edgecolor=INK, s=28, zorder=5)
        ax.axhline(J_MIN, color=RED, lw=1, ls="--"); ax.set_xticks(range(len(levels))); ax.set_xticklabels(levels); ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_title("J = deformed / undeformed volume", loc="left", fontsize=11, fontweight="bold", color=INK)
    else:
        ax.axis("off")
    ax = axs[1, 2]; ax.axis("off"); y = 0.98
    ax.text(0, y, "verdict (0 inverted = gate; the rest are flags)", fontsize=10, fontweight="bold", color=INK, va="top"); y -= 0.08
    for r in rows:
        extra = f"  | Abaqus distorted {r.get('abaqus_distorted', '-')}, mass +{r.get('mass_change_pct', float('nan')):.2f} %" if r["state"] == "end" else ""
        ax.text(0, y, f"{r['level']} {r['state']:11s} {r['verdict']}{extra}", fontsize=8, color=RED if r["verdict"].startswith(("FAIL", "FLAG")) else INK,
                va="top", family="monospace"); y -= 0.055
    handles = [plt.Rectangle((0, 0), 1, 1, color=col[s], label=s) for s in states]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9, ncol=len(states))
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, f"Boxes: 1st-99th percentile with median; dots = worst element. Dashed red: acceptance (SICN > {SICN_MIN:g}, aspect p99 < {ASPECT_P99:g}, "
                          f"J > {J_MIN:g}; dihedral angles reported only). Corner-node (linear) tets.", fontsize=8, color=INK2)
    fig.savefig(path, dpi=140, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id",default="SYN_0001"); ap.add_argument("--tag",default="",help="variant whose runs are checked after the simulation" )
    ap.add_argument("--levels", nargs="+", default=None, help="default: every level found in runs/<id>/mesh"); ap.add_argument("--out",default="")
    a=ap.parse_args(); run=cfg.RUNS/a.id
    levels=a.levels or sorted({p.stem.split("_wall_")[1] for p in (run/"mesh").glob(f"{a.id}_wall_L*.vtu")})
    rows,metrics_by=[],{}

    for L in levels:
        X,T,pd,mp=read_mesh(run, a.id, L); sz=size(X,T,pd,mp); m0=tet_metrics(X,T)
        rows.append(dict(level=L,state="before",**sz,**summary(m0))); metrics_by[(L,"before")]=m0
        dep=run/f"deploy_{L}_{a.tag}"; name = f"{a.id}_{L}_{a.tag}"
        if a.tag and (dep/f"{name}_results.npz"). exists():
            res=np.load(dep / f"{name}_results.npz"); info=json.loads((dep / f"{name}_info.json").read_text())
            idx=np.searchsorted(res["labels"], np.arange(1,len(X)+1)); t= res["times"]; d_ab, mass=abaqus_logs(dep,name)
            for st,f in (("end of ramp", int(np.argmin(np.abs(t-info["time"]["ramp_s"])))),("end",len(t)-1)):
                m = tet_metrics(X + res["U"][f].astype(float)[idx], T); m["J"] = m["V"] / m0["V"]
                rows.append(dict(level=L, state=st, **sz, **summary(m, m0["V"]), **({"abaqus_distorted": d_ab, "mass_change_pct": mass} if st == "end" else {})))
                metrics_by[(L, st)] = m
    out=Path(a.out) if a.out else (run / ("compare" if a.tag else "mesh")); out.mkdir(exist_ok=True)
    stem=f"mesh_qc_{a.tag}" if a.tag else "mesh_qc"
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with open(out / f"{stem}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"{'level':5s} {'state':11s} {'DOFs':>8s} {'tets':>7s} {'edge':>5s} {'layers':>6s} | {'inv':>3s} {'SICN min':>8s} {'p1':>5s} | {'shape<.02':>9s} {'aspect p99':>10s} {'max':>6s} | "
          f"{'dih min':>7s} {'max':>6s} {'out%':>5s} | {'J min':>5s} | verdict")
    for r in rows:
        print(f"{r['level']:5s} {r['state']:11s} {r['dofs']:8,d} {r['tets']:7,d} {r['edge_zone_median_mm']:5.2f} {r['layers_zone']:6.1f} | {r['inverted']:3d} {r['sicn_min']:8.3f} "
              f"{r['sicn_p1']:5.2f} | {r['shape_below_0p02']:9d} {r['aspect_p99']:10.2f} {r['aspect_max']:6.1f} | {r['dihedral_min']:7.1f} {r['dihedral_max']:6.1f} "
              f"{r['dihedral_outside_pct']:5.2f} | {r.get('J_min', float('nan')):5.2f} | {r['verdict']}"
              + (f"  [Abaqus distorted {r['abaqus_distorted']}, mass +{r['mass_change_pct']:.2f} %]" if r["state"] == "end" else ""))
    figure(out / f"{stem}.png", a.id, f"{a.id}: FEA wall mesh quality" + (f" before and after deployment ({a.tag})" if a.tag else " before the simulation"), rows, metrics_by)
    print("wrote", out / f"{stem}.csv", "and", out / f"{stem}.png")

if __name__=="__main__":
    main()





       