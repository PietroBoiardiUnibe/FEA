"""
Paraview time series of the compliant anchors seen in EVOQUE device (beams are not in the 06/13 scripts series): one tube (arm radius) along every deformed beam centre line and a sphere at each ball tip, for every output frame. Positions = start of centre line (final shape shifted inward bt the common start shift) +
the Abaqus displacement of the beam nodes (05 results)

Usage:  python 15_anchor_series.py --id SYN_0001 --level L3 --tag cen_hard_ramp0.2_full
Writes  deploy_<level>_<tag>/post/paraview/anchors_####.vtp + anchors.pvd (open together with fields.pvd, expander.pvd, heart.pvd)
"""
import argparse
import json
from pathlib import Path
import numpy as np
import vtk
import fea_config as cfg
from vtkmodules.util.numpy_support import numpy_to_vtk

ANC_N0=2_200_000 #beam node ids starting above this

def frame_polydata(C,arm_r,ball_r):
    """Tubes along the centre lines C (list of (n,3) arrays, last node=ball centre)+ balls; cell data 'anchor'= anchor index"""
    app=vtk.vtkAppendPolyData()
    for k, P in enumerate(C):
        pts=vtk.vtkPoints(); pts.SetData(numpy_to_vtk(np.ascontiguousarray(P[:-1]),deep = True))
        line=vtk.vtkPolyLine(); line.GetPointIds().SetNumberOfIds(len(P)-1)
        for i in range(len(P)-1):
            line.GetPointIds().SetId(i,i)
        cells=vtk.vtkCellArray(); cells.InsertNextCell(line)
        pl=vtk.vtkPolyData(); pl.SetPoints(pts); pl.SetLines(cells)
        tube=vtk.vtkTubeFilter(); tube.SetInputData(pl); tube.SetRadius(arm_r);tube.SetNumberOfSides(12); tube.CappingOn(); tube.Update()
        ball=vtk.vtkSphereSource(); ball.SetCenter(*P[-1]); ball.SetRadius(ball_r); ball.SetThetaResolution(16); ball.SetPhiResolution(12); ball.Update()

        for part in (tube.GetOutput(), ball.GetOutput()):
            arr=numpy_to_vtk(np.full(part.GetNumberOfCells(),k,np.int32), deep=True); arr.SetName("anchor")
            part.GetCellData().AddArray(arr); app.AddInputData(part)
    app.Update()
    return app.GetOutput()

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id",default="SYN_0001");ap.add_argument("--level",default="L3"); ap.add_argument("--tag",required=True)
    a=ap.parse_args(); run=cfg.RUNS/a.id; name= f"{a.id}_{a.level}_{a.tag}"; dep = run / f"deploy_{a.level}_{a.tag}"
    info=json.loads((dep / f"{name}_info.json").read_text())
    assert info.get("anchors"), "not a full device run for the FEA (no anchors of bioprosthesis are found in info data sheet)"
    AB=np.load(info["anchors"]);nA=int(AB["n"]); shift=float(AB["shift"])
    res=np.load(dep/f"{name}_results.npz"); lab = res["labels"]
    start,ids,nid=[],[],ANC_N0
    for k in range(nA):
        C0 = AB[f"C{k}"] - shift * AB[f"er{k}"]; start.append(C0)
        ids.append(np.searchsorted(lab, np.arange(nid + 1, nid + 1 + len(C0)))); nid += len(C0)
    out = dep / "post" / "paraview"; out.mkdir(parents=True, exist_ok=True); col = []
    for f,t in enumerate(res["times"]):
        U=res["U"][f].astype(float)
        C=[C0+U[i] for C0, i in zip(start, ids)]
        w=vtk.vtkXMLPolyDataWriter(); w.SetFileName(str(out/f"anchors_{f:04d}.vtp")); w.SetInputData(frame_polydata(C,float(AB["arm_r"]),float(AB["ball_r"])))
        w.SetDataModeToBinary(); w.Write()
        col.append(f'<DataSet timestep="{t:.6g}" file="anchors_{f:04d}.vtp"/>')
    (out / "anchors.pvd").write_text('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n' + "\n".join(col)
                                     + "\n</Collection>\n</VTKFile>\n")
    tip = [np.linalg.norm(C0[-1] + res["U"][-1].astype(float)[i[-1]] - AB[f"C{k}"][-1]) for k, (C0, i) in enumerate(zip(start, ids))]
    print(f"{nA} anchors, {len(res['times'])} frames; final ball centre vs planned deployed position: " + ", ".join(f"A{k} {d:.2f}" for k, d in enumerate(tip)) + " mm")
    print(f"wrote {out / 'anchors.pvd'}")

if __name__=="__main__":
    main()
    