"""
Where the worst crushed wall tet element is in the WHOLE deployed shape? Whole-heart views of the run that shows the defect and of the
frozen run(s): same element, same mesh. Backs the contact-law choice (bonded crushes an annulus tet, hard frictionless does not).

Worst element= smallest volume ratio J (deformed/undeformed volume, corner nodes end of the run hold)  in the FIRST run of --tags;
the same element is shown in every other run (all runs of one level share the mesh).
Usage:  python 12_crushed_tet_location.py --id SYN_0001 --level L3 --tags cen_bonded_ramp0.2 cen_hard_ramp0.2
Writes  runs/<id>/compare/02_mesh_qc/crushed_tet_location_<level>_<tag1>_vs_<tag2>.png
"""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import fea_config as cfg

J_MIN=0.2       # 04_mesh_qc acceptance: crushed below
RED, GREEN, WALL, CUFF, RING = "#d03b3b", "#1a7f37", "#d9d6cc", "#2a78d6", "#eb6834"

def load(name):
    """Import a pipeline script whose file name starts with a digit (e.g. 04_mesh_qc)."""
    spec = importlib.util.spec_from_file_location(name, cfg.SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def final_state(run,sid,level,tag,X,T,V0,qc):
    """Deformed corner nodes J of every tet, final cuff surface and cuff-axis offset of one run"""
    dep=run/ f"deploy_{level}_{tag}"; name = f"{sid}_{level}_{tag}"
    res=np.load(dep/f"{name}_results.npz")
    idx = np.searchsorted(res["labels"], np.arange(1, len(X) + 1))              # wall nodes are labelled 1..N
    Xd = X + res["U"][-1].astype(float)[idx]   #adding disp
    J=qc.tet_metrics(Xd,T)["V"]/V0;  cuff = pv.read(str(sorted((dep / "post" / "paraview").glob("expander_*.vtp"))[-1]))   # final cuff frame written by 06
    info=json.loads((dep / f"{name}_info.json").read_text())
    return Xd,J,cuff, info["axis_offset_e1e2_mm"]

def tet_grid(P,cells):
    return pv.UnstructuredGrid(np.c_[np.full(len(cells),4),cells].ravel(), np.full(len(cells), pv.CellType.TETRA),P)

def render(Xd,T,ring,cuff,e,ok,centre,normal,side,view):
    """Off-screen picture: translucent wall, cuff, annulus nodes, element e (+ halo, the tet itself is ~1.5 mm)."""
    col=GREEN if ok else RED; tc=Xd[T[e]].mean(0)
    p=pv.Plotter(off_screen=True, window_size=(900,900)); p.set_background("white")
    p.add_mesh(tet_grid(Xd,T).extract_surface(), color=WALL, opacity=0.18)
    p.add_mesh(cuff,color=CUFF,opacity=0.55)
    p.add_mesh(pv.PolyData(Xd[ring]), color=RING,point_size=4,render_points_as_spheres=True)
    p.add_mesh(tet_grid(Xd, T[[e]]), color=col, show_edges=True, edge_color="black", line_width=2)
    p.add_mesh(pv.Sphere(radius=3.0, center=tc), color=col, opacity=0.35)
    if view == "side":
        eye, up = centre + 170 * side + 20 * normal, normal
    else:
        eye, up = centre + 170 * normal, (tc - centre) / np.linalg.norm(tc - centre)
    p.camera_position = [tuple(eye), tuple(centre), tuple(up)]
    img = p.screenshot(return_img=True); p.close()
    return img

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", default="SYN_0001"); ap.add_argument("--level", default="L3")
    ap.add_argument("--tags", nargs="+", default=["cen_bonded_ramp0.2", "cen_hard_ramp0.2"], help="first = run with the defect, then the runs to compare")
    ap.add_argument("--tet",type=int,default=None,help="element index (0-based) to show; default = worst J of the first tag")
    ap.add_argument("--out",default="")
    a=ap.parse_args(); run=cfg.RUNS/a.id
    qc, s10 = load("04_mesh_qc"), load("10_compare_runs")
    X, T, pd, mp = qc.read_mesh(run, a.id, a.level); V0 = qc.tet_metrics(X, T)["V"]
    plan, _, _ = s10.common(SimpleNamespace(id=a.id)); fr = plan["frame"]
    normal, e1, e2 = (np.array(fr[k]) for k in ("normal", "e1", "e2"))
    ring = pd["tv_annulus"] == 1
    states = {t: final_state(run, a.id, a.level, t, X, T, V0, qc) for t in a.tags}
    e = a.tet if a.tet is not None else int(np.argmin(states[a.tags[0]][1]))
    fig, axs = plt.subplots(len(a.tags), 2, figsize=(13, 6.6 * len(a.tags)), squeeze=False)
    for row, t in enumerate(a.tags):
        Xd, J, cuff, off = states[t]
        centre = np.array(fr["centre"]) + off[0] * e1 + off[1] * e2              # point on the cuff axis of this run
        r = Xd[T[e]].mean(0) - centre; r -= np.dot(r, normal) * normal; r /= np.linalg.norm(r)
        side = np.cross(normal, r)                                                # camera direction: sees the tet from the side
        ok = J[e] >= J_MIN
        for col, view in enumerate(("side", "axis")):
            ax = axs[row, col]; ax.imshow(render(Xd, T, ring, cuff, e, ok, centre, normal, side, view)); ax.axis("off")
            ax.set_title(f"{t}: {'side view (whole heart)' if view == 'side' else 'view along the cuff axis'}\n"
                         f"element {e}: J = {J[e]:.2f} (run minimum {J.min():.2f})", loc="left", fontsize=11, color=GREEN if ok else RED)
    off0 = states[a.tags[0]][3]; c0 = np.array(fr["centre"]) + off0[0] * e1 + off0[1] * e2
    v = X[T[e]].mean(0) - c0                                                       # undeformed position of the element
    theta = np.degrees(np.arctan2(np.dot(v, e2), np.dot(v, e1))) % 360; z = np.dot(v, normal)
    fig.suptitle(f"{a.id}, {a.level}: wall element {e} in the whole deployed model ({theta:.0f} deg about the cuff axis, "
                 f"{z:+.1f} mm from the annulus plane, undeformed)\n"
                 f"grey = FEA wall (deformed), blue = cuff (final), orange = TV-annulus nodes, "
                 f"element + halo: red = crushed (J < {J_MIN}), green = acceptable", x=0.02, ha="left", fontsize=12)
    out = Path(a.out) if a.out else cfg.compare_dir(run, "qc"); out.mkdir(exist_ok=True)
    path = out / f"crushed_tet_location_{a.level}_{'_vs_'.join(a.tags)}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(path, dpi=130)
    print(f"element {e}: " + ", ".join(f"{t} J {states[t][1][e]:.3f}" for t in a.tags) + f"\nwrote {path}")


if __name__ == "__main__":
    main()



