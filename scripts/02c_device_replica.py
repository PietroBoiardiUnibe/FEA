"""
02c_device_replica.py -- DEVICE PACKAGE of the appearance-model REPLICA (EVOQUE-like, 44 mm) for the stage-1 force-controlled BC.
Heart independent, package coordinates (device axis = z). Writes cfg.DEVICES/<out>/:
  pattern.npz  strut centre lines in the CHART (theta, s) of the two surfaces of revolution (outer frame profile, inner valve cylinder),
               widths W, thickness T, atrial tips + paddles -> meshed by 02d (02e writes the SAME interface for the microCT device)
  skirt.npz    S4R skirt grid on the fabric mid-surface (outer profile - strut half thickness - 0.02 - t/2 = package falda) + the 1 mm room
               skin (grid at fabric outer face + NATIVE_LEAFLET_THICKNESS, every 2nd skirt node) -> SYNTHETIC until the laser scan
  device.json  provenance (generator / synthetic / measured), radii, QC verdict;   *.vtp  centre lines, skirt, skin (ParaView)
Strut paths = the package generator (generar_frame_evoque.py, rebuilt here, not executed). Its Catmull-Rom profiles are forced z-monotone
(the inner profile has a ~0.03 mm loop at the bottom flare that folded hexes; the correction is reported).
Usage: python 02c_device_replica.py [--out replica_44] [--hs 0.5]
"""
import argparse
import json

import numpy as np
import vtk

import fea_config as cfg
import fea_device as fd

# ---- generator parameters (generar_frame_evoque.py, 44 mm natural geometry: atrium up)
N, R_IN, STRUT_W, STRUT_T = 9, 14.0, 0.70, 0.55
OUTER_PROFILE = [(14.0, 27.5), (16.0, 27.0), (18.2, 25.9), (20.0, 24.3), (21.2, 22.5), (21.85, 20.8), (22.0, 19.0), (22.0, 10.5),
                 (21.7, 7.2), (20.6, 5.0), (18.6, 3.4), (16.4, 2.75), (14.6, 2.6)]
INNER_PROFILE = [(14.0, 27.5), (14.0, 16.0), (14.0, 5.0), (14.1, 3.6), (14.6, 2.6)]
Z_TIP_BASE, Z_ARCH, Z_SHOULDER, Z_MID, Z_BOTTOM = 27.5, 24.6, 20.6, 12.0, 2.6
Z_POST_TOP, Z_RING_HI, Z_RING_LO, POST_W = 25.3, 9.0, 5.0, 1.6
TIP_LEAN_DEG, TIP_LEN, PADDLE_H = 12.0, 3.4, 2.2
LEAF_R, LEAF_ZMIN, Z_NADIR_NAT = 9.5, -6.58, 26.3
LEAF_ATTACH = [(4, -1.20), (8, -1.55), (12, -2.46), (16, -3.27), (20, -3.97), (24, -4.40), (28, -4.95), (32, -5.25), (36, -5.50),
               (40, -5.75), (44, -5.94), (48, -6.10), (52, -6.20), (56, -6.29), (60, -6.32)]
Z_NAT_TO_PKG, ROT_DEG = 16.7943, 90.0      # natural -> package: angle + 90 deg, z_pkg = 16.7943 - z_nat (same mapping as 02b)


def catmull_rom(pts, n_per=60):
    pts = np.asarray(pts, float); P = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]]); out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]; t = np.linspace(0, 1, n_per, endpoint=False)[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(pts[-1:]); return np.vstack(out)


def smoothstep(t):
    return t * t * (3 - 2 * t)


def to_pkg(P, rot_deg=ROT_DEG, z0=Z_NAT_TO_PKG):
    """Natural generator frame -> package frame (rotation about z, axial flip)."""
    th = np.radians(rot_deg); R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    Q = P @ R.T; Q[:, 2] = z0 - Q[:, 2]; return Q


def profiles():
    """Outer and inner surfaces of revolution (natural frame), z-monotone. Returns (outer, inner, max correction in mm)."""
    out = []
    for ctrl in (OUTER_PROFILE, INNER_PROFILE):
        r, z = catmull_rom(ctrl).T; r, z, dz = fd.monotone_z(r, z); out.append((fd.Profile(r, z), dz))
    return out[0][0], out[1][0], max(out[0][1], out[1][1])


def chart_struts(outer, inner):
    """Generator strut centre lines in chart parameters: list of (surface 0 outer | 1 inner, theta deg array, s array, W) + tip angles."""
    step = 360.0 / N; tip_ang = [step / 2 + step * k for k in range(N)]; anc_ang = [step * k for k in range(N)]; out = []
    s_fit = LEAF_R / (R_IN - STRUT_T / 2)

    def strut(surf, a0, z0, a1, z1, alpha=0.9, n=40, ease=None, W=STRUT_W):
        S = outer if surf == 0 else inner
        s0, s1 = S.s_of_z(z0), S.s_of_z(z1); t = np.linspace(0, 1, n)
        f = (alpha * smoothstep(t) + (1 - alpha) * t) if ease is None else ease(t)
        out.append((surf, a0 + (a1 - a0) * f, s0 + (s1 - s0) * t, W))

    arch = lambda t: 1 - (1 - t) ** 2.6
    for k in range(N):                                                       # outer frame: Lambda struts, ogive arches, band cells
        t, a0, a1 = tip_ang[k], anc_ang[k], anc_ang[k] + step
        for a in (a0, a1):
            strut(0, t, Z_TIP_BASE, a, Z_SHOULDER)
            strut(0, t, Z_ARCH, a, Z_SHOULDER, ease=arch)
            strut(0, a, Z_SHOULDER, t, Z_MID)
            strut(0, t, Z_MID, a, Z_BOTTOM)
    com = [0.0, 120.0, 240.0]
    d_com = lambda a: np.min(np.abs((np.atleast_1d(a) - np.array(com)[:, None] + 180) % 360 - 180), axis=0)
    tab_d, tab_z = np.array([p[0] for p in LEAF_ATTACH], float), np.array([p[1] for p in LEAF_ATTACH], float)
    rail_z = lambda a: Z_NADIR_NAT - (np.interp(d_com(a), tab_d, tab_z) - LEAF_ZMIN) / s_fit
    for c in com:                                                            # scalloped suture rail (inner cylinder)
        a = np.linspace(c + 4, c + 116, 120); out.append((1, a, inner.s_of_z(rail_z(a % 360)), STRUT_W))
    for k in range(N):                                                       # inner valve cylinder
        t = tip_ang[k]
        for a in (anc_ang[k], anc_ang[k] + step):
            zv = Z_POST_TOP if d_com(a % 360)[0] < 1 else float(rail_z(a % 360)[0])
            strut(1, t, Z_TIP_BASE, a, zv, alpha=0.6)
        strut(1, t, float(rail_z(t)[0]), t, Z_RING_HI, n=16)
        strut(1, t, Z_RING_HI, anc_ang[k], Z_RING_LO, alpha=0.6)
        strut(1, t, Z_RING_HI, anc_ang[k] + step, Z_RING_LO, alpha=0.6)
        strut(1, anc_ang[k], Z_RING_LO, anc_ang[k], Z_BOTTOM, n=16)
    for c in com:                                                            # commissural posts (suture windows ignored)
        zs = np.linspace(Z_RING_LO, Z_POST_TOP, 20); out.append((1, np.full(20, c), inner.s_of_z(zs), POST_W))
        for e in (c + 4, c - 4):
            z = float(rail_z(np.array([e % 360]))[0]); out.append((1, np.linspace(e, c, 6), np.full(6, float(inner.s_of_z(z))), STRUT_W))
    return out, tip_ang


def surface_grid(prof, offset, hs, nth, s_lo, s_hi, every=1):
    """Profile offset along its outward normal by `offset` (mm): structured quads (outward normals), package coordinates."""
    ns = int(np.ceil((s_hi - s_lo) / hs)); ns += ns % 2
    ss = np.linspace(s_lo, s_hi, ns + 1)[::every]; th = (np.arange(nth) * 360.0 / nth)[::every]
    T, S = np.meshgrid(th, ss); P, Nn = prof.eval(T, S)
    X = to_pkg((P + offset * Nn).reshape(-1, 3)); n_s, n_t = len(ss), len(th)
    Q = np.array([(i * n_t + j, i * n_t + (j + 1) % n_t, (i + 1) * n_t + (j + 1) % n_t, (i + 1) * n_t + j) for i in range(n_s - 1) for j in range(n_t)])
    c = X[Q].mean(1); nq = np.cross(X[Q[:, 1]] - X[Q[:, 0]], X[Q[:, 3]] - X[Q[:, 0]])
    if np.mean(np.einsum("ij,ij->i", nq[:, :2], c[:, :2])) < 0:
        Q = Q[:, ::-1]
    return X, Q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=cfg.DEVICE); ap.add_argument("--hs", type=float, default=cfg.SKIRT_H)
    a = ap.parse_args(); out = cfg.DEVICES / a.out; out.mkdir(parents=True, exist_ok=True)
    outer, inner, dz = profiles(); struts, tip_ang = chart_struts(outer, inner)
    # ---- pattern.npz (natural frame; 02d maps to the package frame with rot_deg / z_nat_to_pkg)
    fd.save_pattern(out / "pattern.npz", outer, inner, struts, T=STRUT_T, tip_ang=np.array(tip_ang), tip_lean_deg=TIP_LEAN_DEG,
                    tip_len=TIP_LEN + PADDLE_H, tip_W=STRUT_W * 1.15, tip_r0=R_IN, tip_z0=Z_TIP_BASE, rot_deg=ROT_DEG, z_nat_to_pkg=Z_NAT_TO_PKG, N=N)
    # ---- skirt (fabric mid-surface) + room skin (fabric outer face + leaflet room): SYNTHETIC = generator outer-skirt surface
    t = cfg.PET["t"]; s_lo = float(outer.s[np.argmin(np.abs(outer.z - (Z_TIP_BASE - 0.4)))]); s_hi = float(outer.s[-1])
    nth = 18 * int(round(2 * np.pi * 22.0 / a.hs / 18)); face = -(STRUT_T / 2 + 0.02)            # fabric outer face = strut inner face
    Xs, Qs = surface_grid(outer, face - t / 2, a.hs, nth, s_lo, s_hi)
    Xk, Qk = surface_grid(outer, face + cfg.NATIVE_LEAFLET_THICKNESS, a.hs, nth, s_lo, s_hi, every=2)
    np.savez(out / "skirt.npz", Xs=Xs, Qs=Qs, Xk=Xk, Qk=Qk, hs=a.hs, t=t)
    # ---- QC (pre-simulation): skirt quads, skin radius, centre lines inside the package strut STL
    e1, e2 = (np.linalg.norm(Xs[Qs[:, k]] - Xs[Qs[:, 0]], axis=1) for k in (1, 3))
    cosang = np.abs(np.einsum("ij,ij->i", Xs[Qs[:, 1]] - Xs[Qs[:, 0]], Xs[Qs[:, 3]] - Xs[Qs[:, 0]])) / (e1 * e2)
    C = [to_pkg((outer if sf == 0 else inner).eval(th, s)[0]) for sf, th, s, _ in struts]; Cc = np.vstack(C)
    rd = vtk.vtkSTLReader(); rd.SetFileName(str(cfg.EVOQUE / "EVOQUE_frame_real44_mm.stl")); rd.Update()
    dist = vtk.vtkImplicitPolyDataDistance(); dist.SetInput(rd.GetOutput()); inside = np.mean([dist.EvaluateFunction(p) < 0 for p in Cc])
    qc = dict(struts=len(struts), outer=sum(s == 0 for s, *_ in struts), inner=sum(s == 1 for s, *_ in struts), tips=N, profile_z_correction_mm=dz,
              centreline_inside_stl_pct=100 * float(inside), skirt_elements=len(Qs), skirt_aspect_max=float(np.max(np.maximum(e1, e2) / np.minimum(e1, e2))),
              skirt_skew_max_deg=float(np.degrees(np.arcsin(min(1.0, cosang.max())))), skin_nodes=len(Xk), skin_r_max_mm=float(np.hypot(Xk[:, 0], Xk[:, 1]).max()),
              frame_r_max_mm=float(outer.r.max() + STRUT_T / 2))
    qc["verdict"] = "PASS" if qc["centreline_inside_stl_pct"] > 95 and dz < 0.05 and qc["skirt_aspect_max"] < 2 else "CHECK"
    meta = dict(name=a.out, source="generator (appearance model, generar_frame_evoque.py rebuilt)", size_mm=44, symmetry_N=N, strut_W=STRUT_W, strut_T=STRUT_T,
                synthetic=["skirt = generator outer-skirt surface", "room skin = fabric + 1 mm"], measured=[], anchors="02b compliant beams + rigid balls (per heart)", qc=qc)
    (out / "device.json").write_text(json.dumps(meta, indent=1))
    lines, o = [], 0
    for c in C:
        lines += [(o + i, o + i + 1) for i in range(len(c) - 1)]; o += len(c)
    fd.write_vtp(out / "pattern_centrelines.vtp", Cc, lines); fd.write_vtp(out / "skirt.vtp", Xs, Qs); fd.write_vtp(out / "skin.vtp", Xk, Qk)
    print(f"{a.out}: {qc['struts']} struts ({qc['outer']} outer / {qc['inner']} inner) + {N} tips, profile correction {dz:.3f} mm, "
          f"{qc['centreline_inside_stl_pct']:.1f} % of centre-line points inside the package STL; skirt {len(Qs)} S4R (aspect max {qc['skirt_aspect_max']:.2f}), "
          f"skin {len(Xk)} nodes (r max {qc['skin_r_max_mm']:.2f}) -> {qc['verdict']}  ({out})")


if __name__ == "__main__":
    main()