"""
SOLID hex mesh (C3D8R) of a device frame from its pattern.npz.  (02c replica or 02e microCT): the chart-pattern method.
Every strut = strip of physical width W around its centre line (chart width W R0 / r(s)), every junction = disk HEX_JUNCTION_R; gmsh (OCC)
fuses them -> CONFORMING junctions.  meshes all-quad (Frontal-Delaunay quads + Blossom + one subdivision);
degenerate quads (edge < 0.02 mm) dropped, folded quads untangled locally; quads extruded through the thickness T in HEX_NL layers along the
profile normal (outer) or RADIALLY (inner).Tips + paddles = structured hex sweeps.

QC BEFORE the run (frame_hex_qc.json + frame_hex.vtu with sj / part / layer for ParaView): corner scaled Jacobian (sj), edge length, aspect;
hexes with sj <= HEX_SJ_MIN are dropped (junction slivers) and counted.
Node indices refer to X (unused nodes of dropped hexes stay in X and are not written by 03c: labels are reproducible).
Usage: python 02d_frame_hex.py [--device replica_44] [--h 0.26] [--nl 3] [--tag ""]  -> cfg.DEVICES/<device>/frame_hex[_<tag>].npz
"""

import argparse
import json
import gmsh
import numpy as np
import fea_device as fd
import fea_config as cfg

def mesh_surface(struts, surf, prof,h,name):
    """gmsh 2-D all quad mesh of the strut pattern of one surface in its chart. Returns chart nodes, quads , R0, L, s_end"""
    R0=float(prof.r.max());L=2*np.pi*R0;s_end=float(prof.s[-1]);rs=prof.r_of_s
    gmsh.model.add(name);occ=gmsh.model.occ;shapes=[];junc=[]
    for sf,a,s,W in struts:
        if sf!= surf:
            continue
        x,y=R0*np.radians(a),np.asarray(s,float);r=rs(y)
        tx, ty = np.gradient(x) * r / R0, np.gradient(y); nn = np.hypot(tx, ty); nx, ny = -ty / nn, tx / nn
        hw = W / 2
        left = np.c_[x + nx * hw * R0 / r, y + ny * hw]; right = np.c_[x - nx * hw * R0 / r, y - ny * hw]
        for shift in (-L, 0.0, L):                                              # periodic copies across the chart seam
            Lf, Rt = left + [shift, 0], right + [shift, 0]
            if max(Lf[:, 0].max(), Rt[:, 0].max()) < -1 or min(Lf[:, 0].min(), Rt[:, 0].min()) > L + 1:
                continue
            pl = [occ.addPoint(px, py, 0) for px, py in Lf]; pr = [occ.addPoint(px, py, 0) for px, py in Rt]
            c1 = occ.addBSpline(pl) if len(pl) > 2 else occ.addLine(pl[0], pl[-1])
            c3 = occ.addBSpline(pr[::-1]) if len(pr) > 2 else occ.addLine(pr[-1], pr[0])
            c2 = occ.addLine(pl[-1], pr[-1]); c4 = occ.addLine(pr[0], pl[0])
            shapes.append((2, occ.addPlaneSurface([occ.addCurveLoop([c1, c2, c3, c4])])))
        junc += [(x[0], y[0]), (x[-1], y[-1])]
    junc = np.unique(np.round(np.array(junc), 3), axis=0); jr = cfg.HEX_JUNCTION_R
    for xj, yj in junc:
        for shift in (-L, 0.0, L):
            if -1 < xj + shift < L + 1:
                rx = jr * R0 / float(rs(yj))
                if rx >= jr:
                    shapes.append((2, occ.addDisk(xj + shift, yj, 0, rx, jr)))
                else:
                    shapes.append((2, occ.addDisk(xj + shift, yj, 0, jr, rx, zAxis=[0, 0, 1], xAxis=[0, 1, 0])))
    fused,_=occ.fuse(shapes[:1],shapes[1:])
    box=occ.addRectangle(0,0,0,L,s_end)
    occ.intersect(fused,[(2,box)])
    occ.synchronize()
    for o_ in ("Mesh.MeshSizeFromCurvature","Mesh.MeshSizeFromPoints","Mesh.MeshSizeExtendFromBoundary"):
        gmsh.option.setNumber(o_,0)
    gmsh.option.setNumber("Mesh.MeshSizeMin",1.2*h);gmsh.option.setNumber("Mesh.MeshSizeMax",2*h)
    gmsh.option.setNumber("Mesh.Algorithm",8); gmsh.option.setNumber("Mesh.RecombineAll",1)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm",1); gmsh.option.setNumber("Mesh.SubdivisionAlgorithm",1)
    gmsh.option.setNumber("Mesh.Smoothing",30); gmsh.option.setNumber("Mesh.RecombineOptimizeTopology",5)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Laplace2D",niter=10)
    tags,xyz,_=gmsh.model.mesh.getNodes(); idx={t: i for i, t in enumerate(tags)}; Q = []
    for et, el, nd in zip(*gmsh.model.mesh.getElements(2)):
        if et == 3:
            Q.append(np.array([idx[t] for t in nd]).reshape(-1, 4))
        elif len(nd):
            print(f"  WARNING {name}: element type {et} ({len(el)}) is not a quad")
    xy, Q = xyz.reshape(-1, 3)[:, :2], np.vstack(Q)
    e = np.stack([np.linalg.norm(xy[Q[:, (i + 1) % 4]] - xy[Q[:, i]], axis=1) for i in range(4)], 1)
    keep = e.min(1) > 0.02                                                     # degenerate slivers at clipped junction disks
    if (~keep).any():
        print(f"  {name}: dropped {(~keep).sum()} degenerate quads (min edge < 0.02 mm, area {0.5 * (e[~keep].max(1) * e[~keep].min(1)).sum():.4f} mm2)")
    Q = Q[keep]; used = np.unique(Q); new = -np.ones(len(xy), int); new[used] = np.arange(len(used))
    xy, Q = untangle(xy[used], new[Q], R0, rs, name)
    return xy, Q, R0, L, s_end, int((~keep).sum())

def quad_quality(P,Q):
    """minimum corner sine of each quad in its own orientation  (<= 0: non-convex / folded)."""
    q=[]
    for c in range(4):
        a, b, d = Q[:, c], Q[:, (c + 1) % 4], Q[:, (c + 3) % 4]
        e1, e2 = P[b] - P[a], P[d] - P[a]
        q.append((e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) / (np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) + 1e-30))
    q=np.array(q); return(q*np.sign(q.sum(0))).min(0)
    
def untangle(xy,Q,R0,rs,name,iters=60):
    """Local laplacian smoothing in metric space of the interior nodes around non convex quads; quad still folded are dropped before volumetric meshing"""
    edges = np.sort(np.vstack([Q[:, [i, (i + 1) % 4]] for i in range(4)]), 1)
    ue, cnt = np.unique(edges, axis=0, return_counts=True); bnd = np.zeros(len(xy), bool); bnd[ue[cnt == 1].ravel()] = True
    nbr = [[] for _ in range(len(xy))]
    for a, b in ue:
        nbr[a].append(b); nbr[b].append(a)
    r = rs(xy[:, 1]); P = np.c_[xy[:, 0] * r / R0, xy[:, 1]]; q = quad_quality(P, Q); n0 = int((q <= 0.05).sum())
    for _ in range(iters):
        bad = np.flatnonzero(q <= 0.05)
        if not len(bad):
            break
        move = set(Q[bad].ravel()); move |= set(j for i in list(move) for j in nbr[i]); move = [i for i in move if not bnd[i]]
        for i in move:
            P[i] = P[nbr[i]].mean(0)
        q = quad_quality(P, Q)
    keep = q > 0.0
    if (~keep).any():
        print(f"  {name}: {n0} non-convex quads before untangling, {(~keep).sum()} still folded -> dropped")
        Q = Q[keep]
    elif n0:
        print(f"  {name}: {n0} non-convex quads untangled")
    xy = np.c_[P[:, 0] * R0 / rs(P[:, 1]), P[:, 1]]
    used = np.unique(Q); new = -np.ones(len(xy), int); new[used] = np.arange(len(used))
    return xy[used], new[Q]


def extrude(xy,Q,R0,prof,T,NL,radial,pkg):
    """Chart quads swept to hexes: node layers at -T/2 .. +T/2 along the surface normal (or radial); package coordinates; positive volume enforced. Saves computational cost"""
    P,Nn=prof.eval(np.degrees(xy[:,0]/R0),xy[:,1]); P,Nn=np.asarray(P), np.asarray(Nn)
    if radial:
        Nn = Nn.copy(); Nn[:, 2] = 0.0; Nn /= np.linalg.norm(Nn, axis=1, keepdims=True)
    X = np.vstack([pkg(P + (-T / 2 + k * T / NL) * Nn) for k in range(NL + 1)]); n = len(xy)
    H = np.vstack([np.c_[Q + k * n, Q + (k + 1) * n] for k in range(NL)])
    v = np.einsum("ij,ij->i", np.cross(X[H[:, 1]] - X[H[:, 0]], X[H[:, 3]] - X[H[:, 0]]), X[H[:, 4]] - X[H[:, 0]])
    flip = v < 0; H[flip] = H[flip][:, [0, 3, 2, 1, 4, 7, 6, 5]]
    return X, H, np.repeat(np.arange(NL), len(Q)), n


def tip_hex(Pt,T,NL,nw,nseg,pkg):
    """Straight structured hex sweeps of the atrial tips + paddles"""
    lean = np.radians(float(Pt["tip_lean_deg"])); W, Ltip = float(Pt["tip_W"]), float(Pt["tip_len"]); Xs, Hs, ends, bases = [], [], [], []; off = 0
    for t in Pt["tip_ang"]:
        th = np.radians(t); er = np.array([np.cos(th), np.sin(th), 0.0]); et = np.array([-np.sin(th), np.cos(th), 0.0])
        d = np.sin(lean) * er + np.array([0, 0, np.cos(lean)]); nr = er - np.dot(er, d) * d; nr /= np.linalg.norm(nr)
        base = float(Pt["tip_r0"]) * er + np.array([0, 0, float(Pt["tip_z0"])])
        a = np.linspace(-W / 2, W / 2, nw + 1); b = np.linspace(-T / 2, T / 2, NL + 1); c = np.linspace(0, Ltip, nseg + 1)
        G = np.array([[[base + ci * d + ai * et + bi * nr for ai in a] for bi in b] for ci in c])   # [c, b, a, 3]
        idx = np.arange(G.shape[0] * G.shape[1] * G.shape[2]).reshape(G.shape[:3]) + off
        for i in range(nseg):
            for j in range(NL):
                for k in range(nw):
                    Hs.append([idx[i, j, k], idx[i, j, k + 1], idx[i, j + 1, k + 1], idx[i, j + 1, k], idx[i + 1, j, k], idx[i + 1, j, k + 1],
                               idx[i + 1, j + 1, k + 1], idx[i + 1, j + 1, k]])
        Xs.append(pkg(G.reshape(-1, 3))); bases.append(idx[0].ravel()); ends.append(idx[-1].ravel()); off += G.shape[0] * G.shape[1] * G.shape[2]
    X = np.vstack(Xs); H = np.array(Hs)
    v = np.einsum("ij,ij->i", np.cross(X[H[:, 1]] - X[H[:, 0]], X[H[:, 3]] - X[H[:, 0]]), X[H[:, 4]] - X[H[:, 0]])
    flip = v < 0; H[flip] = H[flip][:, [0, 3, 2, 1, 4, 7, 6, 5]]
    return X, H, np.concatenate(bases), np.concatenate(ends)

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=cfg.DEVICE); ap.add_argument("--h",type=float,default=cfg.HEX_H); ap.add_argument("--nl",type=int,default=cfg.HEX_NL)
    ap.add_argument("--tag", default="", help="variant name (e.g. nl2): writes frame_hex_<tag>.npz")
    a = ap.parse_args(); dev = cfg.DEVICES / a.device; stem = "frame_hex" + (f"_{a.tag}" if a.tag else "")
    (outer, inner), struts, Pt = fd.load_pattern(dev / "pattern.npz")
    T, NL = float(Pt["T"]), a.nl; pkg = lambda P: fd.to_pkg(P, float(Pt["rot_deg"]), float(Pt["z_nat_to_pkg"]))
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0)
    xo,Qo,R0o,Lo,so,do=mesh_surface(struts,0,outer,a.h,"outer")
    xi, Qi, R0i, Li, si, di = mesh_surface(struts, 1, inner, a.h, "inner")
    gmsh.finalize()
    Xo,Ho,lo,no=extrude(xo,Qo,R0o,outer,T,NL,False,pkg)
    Xi, Hi, li, ni = extrude(xi, Qi, R0i, inner, T, NL, True, pkg)
    nseg = int(np.ceil(float(Pt["tip_len"]) / a.h)); Xt, Ht, tb, te = tip_hex(Pt, T, NL, 3, nseg, pkg)
    X = np.vstack([Xo, Xi, Xt]); H = np.vstack([Ho, Hi + len(Xo), Ht + len(Xo) + len(Xi)])
    part = np.concatenate([np.zeros(len(Ho), int), np.ones(len(Hi), int), np.full(len(Ht), 2)])
    layer = np.concatenate([lo, li, np.zeros(len(Ht), int)])

    def chart_sets(xy, n, L, s_end, off):                                      # node sets per surface: seams, end circles, faces
        k = np.repeat(np.arange(NL + 1), n); x = np.tile(xy[:, 0], NL + 1); y = np.tile(xy[:, 1], NL + 1); ids = np.arange(len(k)) + off
        return dict(seam0=ids[np.abs(x) < 1e-4], seamL=ids[np.abs(x - L) < 1e-4], top=ids[np.abs(y) < 1e-4], bot=ids[np.abs(y - s_end) < 1e-4],
                    fout=ids[k == NL], fin=ids[k == 0])
    so_ = chart_sets(xo, no, Lo, so, 0); si_ = chart_sets(xi, ni, Li, si, len(Xo)); tb += len(Xo) + len(Xi); te += len(Xo) + len(Xi)
    # ---- QC before the run: scaled Jacobian, edges, aspect; drop degenerate hexes (sj <= HEX_SJ_MIN)
    sj=fd.hex_scaled_jacobian(X,H); keep=sj>cfg.HEX_SJ_MIN;el=fd.hex_edges(X,H);
    asp=el.max(1)/el.min(1)
    H, part, layer, sjk, aspk = H[keep], part[keep], layer[keep], sj[keep], asp[keep]; used = np.zeros(len(X), bool); used[H.ravel()] = True
    qc = dict(device=a.device, h=a.h, NL=NL, quads_dropped=dict(outer=do, inner=di), hex_total=int(len(sj)), hex_dropped=int((~keep).sum()),
              hex=int(len(H)), nodes=int(used.sum()), hex_outer=int((part == 0).sum()), hex_inner=int((part == 1).sum()), hex_tips=int((part == 2).sum()),
              sj_min=float(sjk.min()), sj_p1=float(np.percentile(sjk, 1)), sj_lt_0p2=int((sjk < 0.2).sum()), sj_lt_0p3=int((sjk < 0.3).sum()),
              edge_mm=[float(el[keep].min()), float(np.median(el[keep])), float(el[keep].max())], aspect_max=float(aspk.max()), aspect_p99=float(np.percentile(aspk, 99)),
              elements_across_width=float(0.70 / np.median(el[keep][:, :4])), layers_through_thickness=NL)
    qc["verdict"] = "PASS" if qc["hex_dropped"] < 0.005 * qc["hex_total"] and qc["sj_lt_0p2"] < 0.01 * qc["hex"] else "CHECK"
    np.savez(dev / f"{stem}.npz", X=X, H=H, part=part, layer=layer, sj=sjk, NL=NL, h=a.h, T=T,
             o_seam0=so_["seam0"], o_seamL=so_["seamL"], o_top=so_["top"], o_bot=so_["bot"], o_fout=so_["fout"], o_fin=so_["fin"],
             i_seam0=si_["seam0"], i_seamL=si_["seamL"], i_top=si_["top"], i_bot=si_["bot"], i_fout=si_["fout"], i_fin=si_["fin"], t_base=tb, t_end=te)
    (dev / f"{stem}_qc.json").write_text(json.dumps(qc, indent=1))
    fd.write_hex_vtu(dev / f"{stem}.vtu", X, H, dict(part=part.astype(np.int32), layer=layer.astype(np.int32), scaled_jacobian=sjk, aspect=aspk))
    print(f"{a.device} {stem}: {qc['hex']} C3D8R ({qc['hex_outer']} outer / {qc['hex_inner']} inner / {qc['hex_tips']} tips), {qc['nodes']} nodes; "
          f"dropped {qc['hex_dropped']} hexes (sj <= {cfg.HEX_SJ_MIN}); sj min {qc['sj_min']:.3f}, < 0.2: {qc['sj_lt_0p2']}; edge {qc['edge_mm'][0]:.3f}-"
          f"{qc['edge_mm'][1]:.3f}-{qc['edge_mm'][2]:.3f} mm, aspect p99 {qc['aspect_p99']:.2f} -> {qc['verdict']}")

if __name__=="__main__":
    main()


    