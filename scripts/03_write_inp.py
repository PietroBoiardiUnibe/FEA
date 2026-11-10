"""Abaqus/Explicit input script for the virtual deployment of the device in one synthetically generated heart from SSM pipeline
Wall: the coarse FEA wall of step 1 (linear tets-> quadratic C3D10M, midside nodes added for better bending numerical discretization), neo-Hookean (placeholder material)
Lids: wall nodes under the vessel caps (patches 10,11,12) are fixed: the vessels are tethered
Support: graded normal-only springs on the outer wall (SPRINGA to fixed ground nodes): none near the TV ring, full far from it enough.
Expander:  the crimped cuff outline of step 2 (SFM3D4R surface + point masses to avoid penetration), every node moved to its final position with a smooth step.
Contact: general frictionless contact between expander and inner SSM derived surface, hard, pressure developed only at contact, no tangential shear from friction coefficient
Solver: Abaqus/Explicit time stepping, quasi-static checks: smooth-step ramp and hold, fixed mass scaling to raise necessary time step, wall energies for the quasi-static check (internal vs kinetic)


Usage: python 03_write_inp.py --id SYN_0001 --level L3
Writes: runs/<id>/deploy_<level>/<id>_<level>.inp and _info.json
"""
import argparse
from email.policy import default
import json
import subprocess
import numpy as np
import vtk
from scipy.spatial import cKDTree
from vtkmodules.util.numpy_support import vtk_to_numpy
import fea_config as cfg

EXP_NODE0,EXP_ELEM0=2_000_000, 5_000_000 # expander ids: nodes from 2,000,001, quads from 5,000,001 , masses from 5,100,001
SPR_NODE0,SPR_ELEM0=3_000_000,6_000_000 #ground nodes and spring elements of the graded normal only support to dampen far field deformation
GROUND_OFFSET=10.0 ## mm: ground nodes sit this far outside the wall along outward normal
FACES=((0,1,2),(0,3,1),(1,3,2),(2,3,0))  # Abaqus tet faces S1..S4 (0-based corner indices)
EDGES=((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))  # C3D10 mid-side nodes 5..10 = edges 1-2, 2-3, 3-1, 1-4, 2-4, 3-4

def read_vtu(path):
    r=vtk.vtkXMLUnstructuredGridReader(); r.SetFileName(str(path)); r.Update(); g=r.GetOutput()
    X=vtk_to_numpy(g.GetPoints().GetData()).astype(float)
    T=vtk_to_numpy(g.GetCells().GetConnectivityArray()).reshape(-1,4).astype(np.int64)
    pd={g.GetPointData().GetArrayName(i): vtk_to_numpy(g.GetPointData().GetArray(i)) for i in range(g.GetPointData().GetNumberOfArrays())}
    return X,T,pd

def orient(X,T):
    """Right-handed tets (positive volume): swap T and how many were flipped"""
    v = np.einsum("ij,ij->i", X[T[:, 1]] - X[T[:, 0]], np.cross(X[T[:, 2]] - X[T[:, 0]], X[T[:, 3]] - X[T[:, 0]]))
    flip = v < 0
    T = T.copy(); T[flip] = T[flip][:, [0, 2, 1, 3]]
    return T, int(flip.sum())


def quadratic(X,T):
    """add one mid-side node per edge (shared between tets)-> X10, T10 (10 node ids per tet, 0-based)."""
    e=np.sort(np.stack([T[:,a] for a,b in EDGES]+[T[:,b] for a,b in EDGES], axis=1).reshape(len(T),2,6).transpose(0,2,1).reshape(-1,2),axis=1)
    uniq, inv=np.unique(e,axis=0,return_inverse=True)
    inv=inv.reshape(len(T),6)
    mid=0.5*(X[uniq[:, 0]] + X[uniq[:, 1]])
    return np.vstack([X, mid]), np.hstack([T, len(X) + inv])

def boundary_faces(T,ok):
    """faces that belong to one tet only and whose 3 corners all satisfy ok[...]-> tet index, Abaqus face number 1.4"""
    keys,el,loc=[],[],[]
    for f, (a,b,c) in enumerate(FACES):
        keys.append(np.sort(T[:,[a,b,c]],axis=1)); el.append(np.arange(len(T)));loc.append(np.full(len(T),f+1))
    keys,el,loc=np.vstack(keys),np.concatenate(el), np.concatenate(loc)
    _, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    sel = (cnt[inv.ravel()] == 1) & ok[keys].all(axis=1)
    return el[sel],loc[sel]

def lines(ids,per=16):
    ids=list(map(int,ids))
    return "\n".join(", ".join(map(str, ids[i:i + per])) for i in range(0, len(ids), per))

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id",default="SYN_0001")
    ap.add_argument("--level",default="L3",choices=sorted(cfg.INNER_FACES))
    a=ap.parse_args()
    run=cfg.RUNS/a.id
    name=f"{a.id}_{a.level}"
    out=run/f"deploy_{a.level}"
    out.mkdir(parents=True,exist_ok=True)

    # wall
    X,T,pd=read_vtu(run/"mesh"/f"{a.id}_wall_{a.level}.vtu")
    T,n_flip=orient(X,T)
    X10,T10=quadratic(X,T)
    surf,patch=pd["surface"],pd["patch"]
    inner_idx=np.flatnonzero(surf==1)
    d_cap,j=cKDTree(X[inner_idx]).query(X10)  # every node (incl. midside) -> nearest inner (SSM) node
    lid=np.isin(patch[inner_idx][j],(10,11,12)) & (d_cap <= cfg.LID_DEPTH)
    el_in,loc_in=boundary_faces(T,surf==1)
    print(f"wall: {len(T):,} tets ({n_flip:,} re-oriented), {len(X):,} corner -> {len(X10):,} nodes; "
          f"inner contact faces {len(el_in):,}; lid nodes {int(lid.sum()):,}")

if __name__=="__main__":
    main()




