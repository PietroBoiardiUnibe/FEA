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
ANC_N0,ANC_E0=2_200_000, 5_300_000 #compliant anchors that can bend under loading conditions:beam nodes from 2,200,001, B31 
BALL_N0,BALL_E0=2_300_000,5_400_000 # rigid ball tips: surface nodes from 2.300,001, SFM3D3 from 5,400,001, ball mass/rotary inertia from 5,490,001
SKIN_N0,SKIN_E0, SKIN_M0=2_400_000,5_500_000,5_600_000 # arm contact skin: ring nodes from 2,400,001, SFM3D4R from 5,500,001, nodal masses from 5,600,001
SKIN_NR=12 #nodes per skin ring
SKIN_NODE_MASS=1e-9 # t per skin node: penalty contact needs a nodal mass, transferred to the beam node by the coupling
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

def anchor_blocks(AB):
    """Compliant anchors deform under bending in FEA: per anchor a B31 beam along the deployed centre line, started shifted radially inward by ONE common shift,
    and a rigid ball (SFM3D3 surface, *RIGID BODY on the tip node) with the mass and rotary inertia of a solid nitinol sphere.
    (without them the tip reference node has ~no rotary inertia and Explicit stops on 'excessive rotation').
    Root Node: radial translation back to the deployed position (prescribed in the step), rotations fixed (clamped to the cap).
    Returns keyword lines, root boundary conditions [(node, displacement)], number of nodes """
    nA,shift=int(AB["n"]),float(AB["shift"])
    w,bc,nid,eid,bnid,beid=[],[],ANC_N0,ANC_E0,BALL_N0,BALL_E0
    rb=float(AB["ball_r"]); mb=cfg.NITI_DENSITY*4/3*np.pi*rb**3; Ib=0.4*mb*rb**2
    snid,seid,skin_arm=SKIN_N0,SKIN_E0,[]
    for k in range(nA):
        er,et=AB[f"er{k}"],AB[f"et{k}"]; C=AB[f"C{k}"]-shift*er; B=AB[f"B{k}"]-shift*er
        ids=np.arange(nid+1,nid+1+len(C)); nid+=len(C)
        w+=[f"*NODE,NSET=ANC{k}_N","\n".join(f"{i}, {x:.6f}, {y:.6f}, {z:.6f}" for i,(x,y,z) in zip(ids,C))]
        w+=[f"*ELEMENT, TYPE=B31, ELSET=ANC{k}_B","\n".join(f"{eid+j+1}, {ids[j]}, {ids[j+1]}" for j in range(len(C)-1))]
        eid+=len(C)-1
        w+=[f"*BEAM SECTION, SECTION=CIRC, ELSET=ANC{k}_B, MATERIAL=NITI_EQ",f"{float(AB['arm_r']):.4f}",f"{et[0]:.6f}, {et[1]:.6f}, {et[2]:.6f}"]  # radius; n1 = tangential
        bids=np.arange(bnid+1,bnid+1+len(B)); bnid+=len(B)
        w+=[f"*NODE, NSET=BALL{k}_N","\n".join(f"{i}, {x:.6f}, {y:.6f}, {z:.6f}" for i,(x,y,z) in zip(bids,B))]
        w+=[f"*ELEMENT, TYPE=SFM3D3, ELSET=BALL{k}_E","\n".join(f"{beid+j+1}, "+", ".join(str(bids[v]) for v in f) for j,f in enumerate(AB["ball_faces"]))]
        beid+=len(AB["ball_faces"])
        w+=[f"*SURFACE SECTION, ELSET=BALL{k}_E",f"*RIGID BODY, REF NODE={ids[-1]}, ELSET=BALL{k}_E"]  # the ball follows the tip node
        w+=[f"*ELEMENT, TYPE=MASS, ELSET=BALLM{k}",f"{BALL_E0+90_000+2*k+1}, {ids[-1]}",f"*MASS, ELSET=BALLM{k}",f"{mb:.4e},"]
        w+=[f"*ELEMENT, TYPE=ROTARYI, ELSET=BALLI{k}",f"{BALL_E0+90_000+2*k+2}, {ids[-1]}",f"*ROTARY INERTIA, ELSET=BALLI{k}",f"{Ib:.4e}, {Ib:.4e}, {Ib:.4e}"]
        bc.append((int(ids[0]),shift*er))
        #CONTACT SKIN: Abaqus/Explicit caps the general contact radius of a beam at 0.6 x its element length,
        # so the true arm tube r_arm is given as a massless geometry tube of SFM3D4R faces whose rings are KINEMATICALLY COUPLED to the beam nodes: the beam carried the bending stiffness, the skin carries the contact geometry
        T=np.gradient(C,axis=0); T/=np.linalg.norm(T,axis=1,keepdims=True)
        ring=[j for j in range(len(C)) if np.linalg.norm(C[j]-C[-1])> rb-0.3]
        phi=np.linspace(0,2*np.pi,SKIN_NR,endpoint=False)
        rid={}
        for j in ring:
            b=np.cross(T[j],et); P=C[j]+float(AB["arm_r"])*(np.cos(phi)[:,None]*et+np.sin(phi)[:,None]*b)
            rid[j]=np.arange(snid+1,snid+1+SKIN_NR); snid+=SKIN_NR
            w+=[f"*NODE, NSET=SKIN{k}_{j}","\n".join(f"{i}, {x:.6f}, {y:.6f}, {z:.6f}" for i,(x,y,z) in zip(rid[j],P))]
            w+=[f"*SURFACE,TYPE=NODE,NAME=SKINS{k}_{j}",f"SKIN{k}_{j},",      # the ring follows the beam node's rigid motion (Explicit: *COUPLING + *KINEMATIC)
                f"*COUPLING, CONSTRAINT NAME=CPL{k}_{j}, REF NODE={ids[j]}, SURFACE=SKINS{k}_{j}","*KINEMATIC","1, 3"]
        for j in ring:
            if j+1 in rid:
                q=[(rid[j][i],rid[j][(i+1)%SKIN_NR],rid[j+1][(i+1)%SKIN_NR],rid[j+1][i]) for i in range(SKIN_NR)]
                ids_q=np.arange(seid+1,seid+1+len(q)); seid+=len(q)
                w+=["*ELEMENT, TYPE=SFM3D4R, ELSET=SKIN_E","\n".join(f"{e}, {a_}, {b_}, {c_}, {d_}" for e,(a_,b_,c_,d_) in zip(ids_q,q))]
                if AB["cuff_mask"][k][j]: skin_arm+=list(ids_q)

    w+=["*SURFACE SECTION, ELSET=SKIN_E","*ELSET, ELSET=SKIN_ARM",lines(skin_arm)]
    w+=["*NSET, NSET=SKIN_N, GENERATE",f"{SKIN_N0+1}, {snid}, 1","*ELEMENT, TYPE=MASS, ELSET=SKIN_MASS"]
    w+=["\n".join(f"{SKIN_M0+i}, {SKIN_N0+i}" for i in range(1,snid-SKIN_N0+1)),"*MASS, ELSET=SKIN_MASS",f"{SKIN_NODE_MASS:g},"]  # penalty contact needs nodal mass
    w+=["*ELSET, ELSET=ANC_B"]+[", ".join(f"ANC{k}_B" for k in range(nA))]
    w+=["*ELSET, ELSET=BALL_E"]+[", ".join(f"BALL{k}_E" for k in range(nA))]
    w+=["*SURFACE, TYPE=ELEMENT, NAME=SKIN_SURF","SKIN_E, SPOS","*SURFACE, TYPE=ELEMENT, NAME=SKINARM_SURF","SKIN_ARM, SPOS",
        "*SURFACE, TYPE=ELEMENT, NAME=BALL_SURF","BALL_E, SPOS"]
    w+=["*MATERIAL, NAME=NITI_EQ","*DENSITY",f"{cfg.NITI_DENSITY:g},","*ELASTIC",f"{cfg.ANCHOR_E_EFF:g}, {cfg.ANCHOR_NU:g}",f"*DAMPING, ALPHA={cfg.ANCHOR_ALPHA:g}"]
    return w,bc,(nid-ANC_N0)+(bnid-BALL_N0)+ (snid-SKIN_N0)




# not anymore model data, but rather the history of what happens to it

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id",default="SYN_0001")
    ap.add_argument("--level",default="L3",choices=sorted(cfg.INNER_FACES))
    ap.add_argument("--friction",type=float,default=cfg.FRICTION,help="Coulomb friction coefficient (0 = frictionless, the default)")
    ap.add_argument("--ramp",type=float,default=cfg.T_RAMP,help="smooth-step ramp duration in s (rate-sensitivity runs)")
    ap.add_argument("--tag",default="",help="variant label: deploy_<level>_<tag>")
    ap.add_argument("--full",action="store_true",help="full device of 02b:device_full/cuff.vtp + compliant anchors (default: pilot cuff only, device/expander.vtp)")
    ap.add_argument("--contact", default=cfg.CUFF_WALL_CONTACT,choices=("bonded","hard"), help="cuff-wall contact: bonded (no separation + rough) or hard (separation allowed, Coulomb --friction); anchors: always hard")
    ap.add_argument("--device_dir",default="device",help="cuff-only runs: runs/<id>/<device_dir>/expander.vtp (e.g. device_mec)")
    a=ap.parse_args()
    assert a.friction>=0, "friction coefficient must be >=0 (it applies to every non-bonded contact: anchors always, skirt-wall only when --contact hard)"
    
    run=cfg.RUNS/a.id
    device=run/"device_full"/"cuff.vtp" if a.full else run/a.device_dir/"expander.vtp"
    dev_rec=json.loads((device.parent/("device_full.json" if a.full else "device.json")).read_text())   # the device record: size, axis rule, axis offset
    anchors=run/"device_full"/"anchors_beam.npz" if a.full else None
    suffix=f"_{a.tag}" if a.tag else ""
    name=f"{a.id}_{a.level}{suffix}"
    out=run/f"deploy_{a.level}{suffix}"
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
    E,Q,U=read_expander(device)
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
    root_bc,n_anc=[],0
    if anchors:
        blocks,root_bc,n_anc=anchor_blocks(np.load(anchors))
        w+=blocks
        print(f"compliant anchors: {len(root_bc)} B31 beams + rigid balls, E_eff {cfg.ANCHOR_E_EFF:g} MPa, alpha {cfg.ANCHOR_ALPHA:g}/s, {n_anc:,} nodes")

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
    n_total=len(X10)+len(E) + n_spr +n_anc
    print(f"nodes in the job: {n_total:,} (license cap {cfg.MAX_NODES:,})")
    assert n_total<cfg.MAX_NODES, "above the license node cap: must use a coarser mesh level, remove unnecessary refinement"

    #amplitude, contact property , fixed vessel lids as constraints
    w+=["*AMPLITUDE, NAME=RAMP,DEFINITION=SMOOTH STEP",f"0.,0.,{a.ramp:g}, 1."]
    w+=["*SURFACE INTERACTION, NAME=CPROP","*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=HARD"]+([f"*FRICTION\n{a.friction:g}"] if a.friction > 0 else [])
    if a.contact=="bonded": #cuff wall: no lift off once touched and no slip (explicit general contact accepts NO SEPARATION only with ROUGH)
        w+=["*SURFACE INTERACTION, NAME=CPROP_BOND","*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=HARD, NO SEPARATION","*FRICTION, ROUGH"]
    w+=["*BOUNDARY","LIDS,ENCASTRE"]

    #history data: the step
    t_total=a.ramp + cfg.T_HOLD
    w+=["*STEP, NAME=DEPLOY", "*DYNAMIC, EXPLICIT", f", {t_total:g}",
          f"*FIXED MASS SCALING, DT={cfg.DT_TARGET:g}, TYPE=BELOW MIN", "*BOUNDARY, AMPLITUDE=RAMP"]
    w+=["\n".join(f"{EXP_NODE0+i+1}, {d+1}, {d+1}, {U[i,d]:.6f}" for i in range(len(E)) for d in range(3))]
    if root_bc: #anchor roots : radial translation back to the deployed position (same ramp), rotations held at 0
        w+=["\n".join(f"{nr}, {d+1}, {d+1}, {u[d]:.6f}" for nr,u in root_bc for d in range(3))]
        w+=["\n".join(f"{nr}, {d}, {d}, 0." for nr,_ in root_bc for d in (4,5,6))]
    pairs=["EXP_SURF, WALL_INNER"]+(["SKIN_SURF, WALL_INNER","BALL_SURF, WALL_INNER","SKINARM_SURF, EXP_SURF","BALL_SURF, EXP_SURF"] if root_bc else [])
    w+=["*CONTACT","*CONTACT INCLUSIONS"]+pairs+["*CONTACT PROPERTY ASSIGNMENT", ", , CPROP"]+(["EXP_SURF, WALL_INNER, CPROP_BOND"] if a.contact=="bonded" else [])
    w+=["*OUTPUT, FIELD, NUMBER INTERVAL=30", "*NODE OUTPUT", "U,UR,RF" if root_bc else "U,RF","*CONTACT OUTPUT, GENERAL CONTACT","CSTRESS,CDISP"]
    w += ["*OUTPUT, HISTORY, TIME INTERVAL=" + f"{t_total / 150:g}", "*ENERGY OUTPUT", "ALLKE, ALLIE, ALLAE, ALLVD, ALLWK, ETOTAL",
          "*ENERGY OUTPUT, ELSET=WALL", "ALLKE, ALLIE, ALLAE, ALLVD"] +(["*ENERGY OUTPUT,ELSET=ANC_B","ALLKE,ALLIE,ALLVD"] if root_bc else [])
    w+=["*END STEP"]
    (out / f"{name}.inp").write_text("\n".join(w) + "\n")

    #what the post processing needs to know about input
    info = {"id": a.id, "level": a.level, "name": name, "fea_repo": git_hash(cfg.SCRIPTS), "shape_repo": git_hash(cfg.SHAPE_SCRIPTS),
            "n_corner_nodes": len(X), "n_nodes": len(X10), "n_tets": len(T), "n_lid_nodes": int(lid.sum()), "n_expander_nodes": len(E),
            "n_springs": n_spr, "support": {"mode": cfg.SUPPORT_MODE, "k_area": cfg.SUPPORT_K, "d_on": cfg.SUPPORT_D_ON, "ramp": cfg.SUPPORT_RAMP, "total_N_per_mm": k_tot},
            "material": {"model": "neo-Hookean", "E_MPa": cfg.E_MPA, "nu": cfg.NU, "C10": C10, "D1": D1, "assumed": True},
            "time": {"ramp_s": a.ramp, "hold_s": cfg.T_HOLD, "dt_target_s": cfg.DT_TARGET}, "friction": a.friction,"tag":a.tag, "expander_node0": EXP_NODE0,
            "device":str(device),"anchors":str(anchors or ""), "cuff_wall_contact": a.contact, "axis_rule":dev_rec.get("axis_rule","mec"), "axis_offset_e1e2_mm":dev_rec.get("axis_offset_e1e2_mm",[0.0,0.0]), "n_device_nodes":len(E)+n_anc,
            "anchor_model":{"E_eff_MPa":cfg.ANCHOR_E_EFF,"nu":cfg.ANCHOR_NU,"alpha":cfg.ANCHOR_ALPHA,"density":cfg.NITI_DENSITY} if anchors else None}
    
    (out / f"{name}_info.json").write_text(json.dumps(info, indent=1))
    print(f"wrote {out / (name + '.inp')} ({(out / (name + '.inp')).stat().st_size / 1e6:.1f} MB)")



if __name__=="__main__":
    main()




