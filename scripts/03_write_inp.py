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

def read_expander(path):
    """expander of step 2: points=crimped start, "displacement" array = move to the final expanded and deployed shape mimicking self-expansion from catheter delivery"""
    r=vtk.vtkXMLPolyDataReader(); r.SetFileName(str(path)); r.Update(); p=r.GetOutput()
    E=vtk_to_numpy(p.GetPoints().GetData()).astype(float)
    Q=vtk_to_numpy(p.GetPolys().GetConnectivityArray()).reshape(-1,4).astype(np.int64)
    U=vtk_to_numpy(p.GetPointData().GetArray("displacement")).astype(float)
    return E,Q,U

def git_hash(folder):
    """Short commit hash of a repository: written into every input file so a result can be traced to the code that made it."""
    try:
        return subprocess.run(["git", "-C", str(folder), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def outer_support(X,T,T10,outer):
    """
    Tributary area and outward unit normal of every MIDSIDE node on the outer surface. A quadratic triangle carries a uniform pressure
    on its midside nodes only (A/3 each, corners 0): the consistent nodal split. k_node = k_area * A_node.
    """
    el,loc=boundary_faces(T,outer)
    area,nrm={},{}
    for e,f in zip(el,loc):
        a,b,c=FACES[f-1]; d=({0,1,2,3}-{a,b,c}).pop()
        pa,pb,pc=X[T[e,a]],X[T[e,b]],X[T[e,c]]
        n = np.cross(pb - pa, pc - pa); A = 0.5 * np.linalg.norm(n); n = n / np.linalg.norm(n)
        if np.dot(n,(pa+pb+pc)/3-X[T[e,d]])<0:
            n=-n  # outward = away from the tet's 4th corner
        for u,v in ((a,b),(b,c),(c,a)):
            m = T10[e, 4 + [i for i, (p, q) in enumerate(EDGES) if {p, q} == {u, v}][0]]
            area[m] = area.get(m, 0.0) + A / 3.0; nrm[m] = nrm.get(m, 0.0) + n * A / 3.0
    ids=np.array(sorted(area));A=np.array([area[i] for i in ids]); N=np.array([nrm[i] for i in ids])
    return ids, A, N / np.linalg.norm(N, axis=1, keepdims=True)

def grade(d,d_on,ramp):
    """Smooth step 0->1 of the distance d from the TV ring: no support within d_on, full support beyond d_on + ramp """
    t=np.clip((d-d_on)/ramp,0.0,1.0)
    return t*t*(3-2*t)

def support_blocks(X10,ids,A,N,weight,k_area,nbin=8):
    """SPRINGA from each supported node to a fixed ground node GROUND_OFFSET mm outside of domain.,
    along the normal (ABAQUS/EXPLICIT has no grounded SPRING1). Abaqus takes one stiffness per element set, so nodes are grouped in nbin classes of similar weighted area. """
    keep=weight>1e-3
    ids,A,N=ids[keep],(A*weight)[keep],N[keep]
    cls=np.minimum((np.argsort(np.argsort(A))*nbin)//len(A),nbin-1)
    G=X10[ids] +GROUND_OFFSET*N
    w = ["*NODE, NSET=SPR_G"] + [f"{SPR_NODE0 + i + 1}, {g[0]:.6f}, {g[1]:.6f}, {g[2]:.6f}" for i, g in enumerate(G)]
    for c in range(nbin):
        sel = np.flatnonzero(cls == c)
        w.append(f"*ELEMENT, TYPE=SPRINGA, ELSET=SUP{c}")
        w.append("\n".join(f"{SPR_ELEM0 + int(i) + 1}, {int(ids[i]) + 1}, {SPR_NODE0 + int(i) + 1}" for i in sel))
        w.append(f"*SPRING, ELSET=SUP{c}\n,\n{k_area * A[sel].mean():.6e}")
    w.append("*BOUNDARY\nSPR_G, ENCASTRE")
    return w, int(keep.sum()), float((k_area*A).sum())

# not anymore model data, but rather the history of what happens to it


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
    # expander
    E,Q,U=read_expander(run/"device"/"expander.vtp")
    print(f"expander: {len(E):,} nodes, {len(Q):,} quads, radial move {np.linalg.norm(U, axis=1).min():.2f}..{np.linalg.norm(U, axis=1).max():.2f} mm")

    #material as neo Hookean simple hyperelasticity (no HGO, Ogden or Mooney Rivlin)
    mu=cfg.E_MPA/(2*(1+cfg.NU)) #shear modulus
    K=cfg.E_MPA/(3*(1-2*cfg.NU)) #bulk modulus resistance to compression
    C10,D1=mu/2.0,2.0/K
    print(f"neo-Hookean material properties and coefficients: mu {mu:.4f} MPa, K {K:.3f} MPa, C10 {C10:.4e}, D1 {D1:.4e}")

    #model data
    w=[f"*HEADING\n{name}: device deployment (FEA repo {git_hash(cfg.SCRIPTS)}). mm, tonne, s, N, MPa.",
         "*PREPRINT, ECHO=NO, MODEL=NO, HISTORY=NO, CONTACT=NO"]
    w += ["*NODE, NSET=WALL_N", "\n".join(f"{i + 1}, {x:.6f}, {y:.6f}, {z:.6f}" for i, (x, y, z) in enumerate(X10))]
    w += ["*ELEMENT, TYPE=C3D10M, ELSET=WALL", "\n".join(f"{i + 1}, " + ", ".join(map(str, row + 1)) for i, row in enumerate(T10))]
    w += ["*NSET, NSET=LIDS", lines(np.flatnonzero(lid) + 1)]
    for f in (1, 2, 3, 4):
        if (loc_in == f).any():
            w += [f"*ELSET, ELSET=INNER_S{f}", lines(el_in[loc_in == f] + 1)]
    w += ["*SURFACE, TYPE=ELEMENT, NAME=WALL_INNER"] + [f"INNER_S{f}, S{f}" for f in (1, 2, 3, 4) if (loc_in == f).any()]
    w += ["*NODE, NSET=EXP_N", "\n".join(f"{EXP_NODE0 + i + 1}, {x:.6f}, {y:.6f}, {z:.6f}" for i, (x, y, z) in enumerate(E))]
    w += ["*ELEMENT, TYPE=SFM3D4R, ELSET=EXP_E", "\n".join(f"{EXP_ELEM0 + i + 1}, " + ", ".join(str(EXP_NODE0 + k + 1) for k in q) for i, q in enumerate(Q))]
    w += ["*SURFACE SECTION, ELSET=EXP_E"]
    w += ["*ELEMENT, TYPE=MASS, ELSET=EXP_MASS", "\n".join(f"{EXP_ELEM0 + 100_000 + i + 1}, {EXP_NODE0 + i + 1}" for i in range(len(E)))]
    w += ["*MASS, ELSET=EXP_MASS", f"{cfg.EXPANDER_NODE_MASS:g},"]
    w += ["*SURFACE, TYPE=ELEMENT, NAME=EXP_SURF", "EXP_E, SPOS"]
    w += ["*MATERIAL, NAME=TISSUE", "*DENSITY", f"{cfg.DENSITY:g},", "*HYPERELASTIC, NEO HOOKE", f"{C10:.6e}, {D1:.6e}"]
    w += ["*SOLID SECTION, ELSET=WALL, MATERIAL=TISSUE"]
    #graded outer support
    n_spr,k_tot=0,0.0

    if cfg.SUPPORT_MODE!="none":
        ids,A,N=outer_support(X,T,T10,surf==2)
        weight=np.ones(len(ids))
        if cfg.SUPPORT_MODE=="normal_graded":
            d_ring=cKDTree(X[pd["tv_annulus"]==1]).query(X10[ids])[0] #distance of each outer node to the TV ring
            weight=grade(d_ring,cfg.SUPPORT_D_ON,cfg.SUPPORT_RAMP)
        blocks,n_spr,k_tot=support_blocks(X10,ids,A,N,weight,cfg.SUPPORT_K)
        w+=blocks
        print(f"support '{cfg.SUPPORT_MODE}': {len(ids):,} outer midside nodes, {n_spr:,} with a spring, total stiffness {k_tot:,.0f} N/mm")
    n_total=len(X10)+len(E) + n_spr
    print(f"nodes in the job: {n_total:,} (license cap {cfg.MAX_NODES:,})")
    assert n_total<cfg.MAX_NODES, "above the license node cap: must use a coarser mesh level, remove unnecessary refinement"

    #amplitude, contact property , fixed vessel lids as constraints
    w+=["*AMPLITUDE, NAME=RAMP,DEFINITION=SMOOTH STEP",f"0.,0.,{cfg.T_RAMP:g}, 1."]
    w+=["*SURFACE INTERACTION, NAME=CPROP","*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=HARD"]+([f"*FRICTION\n{cfg.FRICTION:g}"] if cfg.FRICTION > 0 else [])
    w+=["*BOUNDARY","LIDS,ENCASTRE"]

    #history data: the step
    t_total=cfg.T_RAMP+cfg.T_HOLD
    w+=["*STEP, NAME=DEPLOY", "*DYNAMIC, EXPLICIT", f", {t_total:g}",
          f"*FIXED MASS SCALING, DT={cfg.DT_TARGET:g}, TYPE=BELOW MIN", "*BOUNDARY, AMPLITUDE=RAMP"]
    w+=["\n".join(f"{EXP_NODE0+i+1}, {d+1}, {d+1}, {U[i,d]:.6f}" for i in range(len(E)) for d in range(3))]
    w += ["*CONTACT", "*CONTACT INCLUSIONS", "EXP_SURF, WALL_INNER", "*CONTACT PROPERTY ASSIGNMENT", ", , CPROP"]
    w += ["*OUTPUT, FIELD, NUMBER INTERVAL=30", "*NODE OUTPUT", "U, RF"]
    w += ["*OUTPUT, HISTORY, TIME INTERVAL=" + f"{t_total / 150:g}", "*ENERGY OUTPUT", "ALLKE, ALLIE, ALLAE, ALLVD, ALLWK, ETOTAL",
          "*ENERGY OUTPUT, ELSET=WALL", "ALLKE, ALLIE, ALLAE, ALLVD"]
    w+=["*END STEP"]
    (out / f"{name}.inp").write_text("\n".join(w) + "\n")

    #what the post processing needs to know about input
    info = {"id": a.id, "level": a.level, "name": name, "fea_repo": git_hash(cfg.SCRIPTS), "shape_repo": git_hash(cfg.SHAPE_SCRIPTS),
            "n_corner_nodes": len(X), "n_nodes": len(X10), "n_tets": len(T), "n_lid_nodes": int(lid.sum()), "n_expander_nodes": len(E),
            "n_springs": n_spr, "support": {"mode": cfg.SUPPORT_MODE, "k_area": cfg.SUPPORT_K, "d_on": cfg.SUPPORT_D_ON, "ramp": cfg.SUPPORT_RAMP, "total_N_per_mm": k_tot},
            "material": {"model": "neo-Hookean", "E_MPa": cfg.E_MPA, "nu": cfg.NU, "C10": C10, "D1": D1, "assumed": True},
            "time": {"ramp_s": cfg.T_RAMP, "hold_s": cfg.T_HOLD, "dt_target_s": cfg.DT_TARGET}, "friction": cfg.FRICTION, "expander_node0": EXP_NODE0}
    (out / f"{name}_info.json").write_text(json.dumps(info, indent=1))
    print(f"wrote {out / (name + '.inp')} ({(out / (name + '.inp')).stat().st_size / 1e6:.1f} MB)")



if __name__=="__main__":
    main()




