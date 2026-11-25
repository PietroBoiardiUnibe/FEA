"""
Full EVOQUE-like device for one heart = polymeric cuff contact surface (expander) + nine compliant under bending loading conditions J anchors on ventricular sides
modelled as beam centre lines and ball tips.
Geometry is decided in this script while script 03 translates it into Abaqus-readable keywords.

Sizing: from IFU table directly, ISOTROPIC scale s = D/44 for every part (cuff, anchors, balls).

Landing: anchor-referenced with a rule calibrated on synthetic mean heart from SSM. Device lands along TV axis located and sits at the most atrial position
for which every ventricular anchor surface vertex lies >=c= cfg.LANDING_CLEARANCE mm VENTRICULAR of the local planned annulus height z_ring(theta) fitted earlier:
z_shift = min_v [ z_dev(v) - z_ring(theta_v) ] - c  

Anchors:  centre lines resampled at ~cfg.ANCHOR_ELEM*s (B31 beams), ball centre = last node. Built in the DEPLOYED, stress-free shape and started
          shifted radially INWARD by ONE common shift (smallest that puts every arm/ball point >= cfg.START_CLEAR mm inside the blood pool);
          FEA moves the root back out via a prescribed radial translation displacement while the arm bends freely against wall and cuff.

Cuff: crimped radially about the axis. Start crimp = the plan crimp (R0 / widest radius), reduced if needed so that the crimped
          cuff starts >= cfg.CUFF_START_GAP mm INSIDE every anchor arm (anchors cannot pass through the skirt: anchor-cuff contact) and inside the blood pool.


Writes    runs/<id>/device_full/cuff.vtp (expander format: crimped points, quads, 'final', 'displacement'), anchors_beam.npz, device_full.json
Usage     python 02b_device_full.py --id SYN_0001 [--clearance 1.75]          (shapeworks env: trimesh + pymeshlab)
"""
import argparse
import json
import importlib.util
from pathlib import Path
import numpy as np
import pymeshlab
import trimesh
import vtk
from fea_common import axis_plan, contour_centre, ring_about
import fea_config as cfg

#anchor geometry measured on the package for the device
N_ANCHOR=9
ANCHOR_DEG0, ANCHOR_STEP=10.0,40.0
ARM_D,BALL_D=2.4,4.8 # arm tube and ball diameters (mm, 44 model)
Z_NAT_TO_PKG = 16.7943 
ANCHOR_PATH_NAT=[(14.6, 2.6), (16.0, 1.6), (18.5, 1.25), (21.0, 1.25), (23.2, 1.9), (24.4, 3.6), (24.7, 6.0), (24.7, 10.0), (24.7, 13.6)]  # (r, z)
NTHETA=120 # cuff angular resolution

def load_module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

# anchor geometry
def anchor_path(theta_deg):
    """J-arm centre line (root on the ventricular cap-> out under the cuff-> back toward the atrium) + ball centre 2 mm beyond the arm end.
     The measured control polyline is smoothed and resampled at equal arc length"""
    P = np.array([(r, Z_NAT_TO_PKG - z) for r, z in ANCHOR_PATH_NAT], float)
    Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]]); sm = []
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]; t = np.linspace(0, 1, 14, endpoint=False)[:, None]
        sm.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    P = np.vstack(sm + [P[-1:]])
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))]); ss = np.linspace(0, s[-1], 90)
    rz = np.stack([np.interp(ss, s, P[:, 0]), np.interp(ss, s, P[:, 1])], 1)
    er = np.array([np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg)), 0.0])
    path = rz[:, 0:1] * er + rz[:, 1:2] * np.array([0, 0, 1.0])
    ball_c = path[-1] + (path[-1] - path[-6]) / np.linalg.norm(path[-1] - path[-6]) * 2.0
    return np.vstack([path, ball_c])

def tube(path, radius, nseg=24):
    """Closed tube along a polyline (parallel-transport frames, triangle-fan end caps)."""
    P = np.asarray(path, float); T = np.gradient(P, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True)
    ref = np.array([0, 0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0, 0])
    n = np.cross(T[0], ref); n /= np.linalg.norm(n); frames = []
    for t in T:
        n = n - np.dot(n, t) * t; n /= np.linalg.norm(n); frames.append((n.copy(), np.cross(t, n)))
    a = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
    V = np.concatenate([p + radius * (np.cos(a)[:, None] * f[0] + np.sin(a)[:, None] * f[1]) for p, f in zip(P, frames)])
    F = []
    for i in range(len(P) - 1):
        for j in range(nseg):
            j2 = (j + 1) % nseg; A, B, C, D = i * nseg + j, i * nseg + j2, (i + 1) * nseg + j2, (i + 1) * nseg + j
            F += [[A, B, C], [A, C, D]]
    c0, c1 = len(V), len(V) + 1; V = np.vstack([V, P[0], P[-1]]); e = (len(P) - 1) * nseg
    for j in range(nseg):
        j2 = (j + 1) % nseg; F += [[c0, j2, j], [c1, e + j, e + j2]]
    M = trimesh.Trimesh(V, np.array(F), process=True); trimesh.repair.fix_normals(M)
    return M

def to_trimesh(ms):
    cm=ms.current_mesh(); M=trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=True); trimesh.repair.fix_normals(M)
    return M

def anchor_surface(theta_deg,s):
    """closed anchor body as arm tube U ball at angle theta, scaled by s, re-meshed isotropically to  ~cfg.ANCHOR_ELEM*s.
    Built at ANCHOR_DEG0 and rotated into place.Used for the LANDING (every surface vertex counts) and the start-position checks; the FEA itself uses the beam centre lines.
    """
    path=anchor_path(ANCHOR_DEG0)
    ball=trimesh.creation.icosphere(subdivisions=3, radius=BALL_D/2); ball.apply_translation(path[-1])
    ms=pymeshlab.MeshSet()

    for m in (tube(path, ARM_D/2),ball):
        ms.add_mesh(pymeshlab.Mesh(np.asarray(m.vertices,float),np.asarray(m.faces,np.int32)))
    ms.generate_boolean_union(first_mesh=0, second_mesh=1)
    A=to_trimesh(ms)
    A.apply_transform(trimesh.transformations.rotation_matrix(np.radians(theta_deg-ANCHOR_DEG0),[0 , 0 , 1]))
    A.apply_scale(s)
    ms=pymeshlab.MeshSet(); ms.add_mesh(pymeshlab.Mesh(np.asarray(A.vertices,float),np.asarray(A.faces,np.int32)))
    edge=cfg.ANCHOR_ELEM*s
    ms.meshing_isotropic_explicit_remeshing(iterations=6, targetlen=pymeshlab.PureValue(edge) if hasattr(pymeshlab, "PureValue") else pymeshlab.AbsoluteValue(edge))
    A = to_trimesh(ms)
    assert A.is_watertight, f"anchor at {theta_deg} deg not watertight after remeshing"
    return A


def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id",default="SYN_0001")
    ap.add_argument("--clearance", type=float,default=cfg.LANDING_CLEARANCE,help="c (mm): worst anchor vertex below the local planned annulus")
    ap.add_argument("--axis", default=cfg.AXIS_RULE, choices=("centroid","perimeter","mec"), help="cuff axis rule (default: clinical CT convention)")
    a=ap.parse_args()
    d02=load_module("02_device")
    out=cfg.RUNS/a.id/"device_full"; out.mkdir(parents=True, exist_ok=True)
    plan=json.loads((cfg.PLANS/ f"{a.id}_plan.json").read_text())
    off=contour_centre(plan,a.axis); plan=axis_plan(plan,off); fr=plan["frame"] # the CUFF axis: placement, anchors, crimp about it
    print(f"cuff axis rule '{a.axis}': {np.linalg.norm(off):.2f} mm from the 07b centre at {np.degrees(np.arctan2(off[1], off[0])) % 360:.0f} deg")
    c, n, e1, e2 = (np.array(fr[k]) for k in ("centre", "normal", "e1", "e2"))
    assert np.dot(np.cross(e1, e2), n) > 0, "annulus frame must be right-handed"

    #sizing from device real IFU and the skirt contact surface in DEVICE coordinates
    m=d02.measure(cfg.EVOQUE)
    size,note=d02.choose_size(plan["d_perim_mm"], plan["d_max_mm"], cfg.SIZING_PHASE)
    print(f"{a.id}: d_perim {plan['d_perim_mm']:.1f} mm, d_max {plan['d_max_mm']:.1f} mm ({cfg.SIZING_PHASE}) -> EVOQUE {size} mm: {note}")
    if size is None:
        raise SystemExit(f"FLAG {a.id}: no IFU size ({note}) -> not deployed")
    s=size/44.0
    prof,z_mid,_=d02.expander_profile(m,size,cfg.NATIVE_LEAFLET_THICKNESS)
    th=np.linspace(0,2*np.pi, NTHETA, endpoint=False); npf=len(prof)
    cuff = np.stack([prof[:, 0][None, :] * np.cos(th)[:, None], prof[:, 0][None, :] * np.sin(th)[:, None],
                     np.broadcast_to(prof[:, 1], (NTHETA, npf))], -1).reshape(-1, 3)
    quads = np.array([[j * npf + i, ((j + 1) % NTHETA) * npf + i, ((j + 1) % NTHETA) * npf + i + 1, j * npf + i + 1]
                      for j in range(NTHETA) for i in range(npf - 1)])
    angles=[ANCHOR_DEG0+ANCHOR_STEP*k for k in range(N_ANCHOR)]
    anchors=[anchor_surface(ang,s) for ang in angles]
    #landing anchor referenced
    ring_th, ring_z=ring_about(plan, off)  # annulus height against the angle about the CUFF axis
    zring=lambda t: np.interp(np.mod(t,2*np.pi), ring_th, ring_z, period=2*np.pi)
    AV=np.vstack([A.vertices for A in anchors])
    z_shift = float(np.min(AV[:, 2] - zring(np.arctan2(AV[:, 1], AV[:, 0]))) - a.clearance)
    planned = [float(np.min(A.vertices[:, 2] - z_shift - zring(np.arctan2(A.vertices[:, 1], A.vertices[:, 0])))) for A in anchors]
    print(f"landing: c = {a.clearance:g} mm -> z_shift {z_shift:+.2f} mm; {z_mid - z_shift:+.2f} mm VENTRICULAR of the widest-band (pilot) landing")
    print("planned clearance of each anchor below the local annulus (mm): " + ", ".join(f"{v:.2f}" for v in planned))

    def place(P,f=1.0):
        #device to heart; f =radial crimp about the axis, unchanged
        P=np.asarray(P,float)*np.array([f,f,1.0])
        return c+np.outer(P[:,0],e1)+np.outer(P[:,1],e2)+np.outer(P[:,2]-z_shift,n)
    r_=vtk.vtkXMLPolyDataReader(); r_.SetFileName(str(cfg.HEARTS/f"{a.id}_RH.vtp")); r_.Update()
    dist = vtk.vtkImplicitPolyDataDistance(); dist.SetInput(r_.GetOutput())   # signed distance to the SSM wall (< 0 = blood pool)
    sd=lambda P: np.array([dist.EvaluateFunction(p) for p in P])
    radial = lambda P: np.linalg.norm((P - c) - np.outer((P - c) @ n, n), axis=1)
    # COMPLIANT ANCHORS: beam centre lines (final = deployed, stress-free), ONE common inward start shift
    arm_r, ball_r = ARM_D / 2 * s, BALL_D / 2 * s
    ball = trimesh.creation.icosphere(subdivisions=2, radius=ball_r)
    C, B, ER, ET, shifts = [], [], [], [], []
    for ang in angles:
        P = anchor_path(ang) * s
        arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        ne = max(int(round(arc[-1] / (cfg.ANCHOR_ELEM * s))), 4); ss = np.linspace(0, arc[-1], ne + 1)
        Cd = np.stack([np.interp(ss, arc, P[:, i]) for i in range(3)], 1)       # resampled centre line, last node = ball centre
        er_d = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang)), 0.0])
        er, et = er_d[0] * e1 + er_d[1] * e2, -er_d[1] * e1 + er_d[0] * e2      # radial / tangential unit vectors in the heart frame
        Cw = place(Cd); Bw = place(np.asarray(ball.vertices) + Cd[-1])
        tp = np.vstack([Cw[:-1] + d * arm_r for d in (er, -er, et, -et, n, -n)] + [Bw])   # arm surface probes + ball surface
        sh = 0.0
        while sd(tp - sh * er).max() > -cfg.START_CLEAR:
            sh += 0.25
            assert sh < 30, f"anchor at {ang:g} deg: no inward shift clears the wall"
        C.append(Cw); B.append(Bw); ER.append(er); ET.append(et); shifts.append(sh)
    shift = max(shifts)
    print(f"anchors: {len(C[0]) - 1} B31 elements each ({cfg.ANCHOR_ELEM * s:.2f} mm), start shift per anchor "
          + ", ".join(f"{v:.2f}" for v in shifts) + f" -> common {shift:.2f} mm inward")

    # ARM vs ROOT elements: an element is ARM (contact with the cuff) when its midpoint lies outside the final cuff radius at its height
    Xf = place(cuff); zc, rc = (Xf - c) @ n, radial(Xf)
    def r_cuff(z):                                                       # final cuff radius at height z (0 beyond its ends)
        sel = np.abs(zc - z) < 0.6
        return rc[sel].max() if sel.any() else 0.0
    # CUFF-CONTACT elements (anchors may not cross the skirt): ARM elements running ALONGSIDE the skirt, at least arm_r + 0.5 mm atrial of the cuff's
    # ventricular rim; where the arm wraps under the rim it is one structure with the frame (contact there = artificial clash with the prescribed cuff)
    z_rim = zc.max() - arm_r - 0.5
    arm_mask, cuff_mask, r_start = [], [], []
    for Cw, er in zip(C, ER):
        Cs = Cw - shift * er; mk, ck = [], []
        for j in range(len(Cw) - 1):
            mid = 0.5 * (Cw[j] + Cw[j + 1]); zm = (mid - c) @ n; rm = radial(mid[None])[0]
            arm = rm > r_cuff(zm) + 0.2; mk.append(arm); ck.append(arm and r_cuff(zm) > 0 and zm < z_rim)
            if arm and r_cuff(zm) > 0:                                   # the expanding cuff could overtake the arm here
                r_start += list(radial(Cs[j:j + 2]) - arm_r)
        arm_mask.append(mk); cuff_mask.append(ck)
    arm_mask, cuff_mask = np.array(arm_mask, bool), np.array(cuff_mask, bool)

    #cuff start crimp : plan crimp, reduced so the crimped skirt starts >= cuff_start_GAP inside every rm and inside the blood pool
    crimp_plan=plan["R0_mm"]/prof [:,0].max()
    crimp_arms=(min(r_start)-cfg.CUFF_START_GAP)/rc.max()
    crimp = min(crimp_plan, crimp_arms)
    while sd(place(cuff, crimp)).max() > 0:
        crimp *= 0.97
        assert crimp > 0.2, "no crimp factor puts the cuff inside the blood pool"
    Xc=place(cuff, crimp)
    print(f"cuff start crimp {crimp:.3f} (plan {crimp_plan:.3f}, inside the anchor arms {crimp_arms:.3f}); start clearance {-sd(Xc).max():.2f} mm; "
          f"radial move {np.linalg.norm(Xf - Xc, axis=1).min():.2f}..{np.linalg.norm(Xf - Xc, axis=1).max():.2f} mm")
    print(f"ARM beam elements {int(arm_mask.sum())} of {arm_mask.size} (the rest = root inside the ventricular cap); "
          f"anchor-cuff contact on {int(cuff_mask.sum())} (alongside the skirt, >= {arm_r + 0.5:.1f} mm from the ventricular rim)")
    for k, (Cw, Bw) in enumerate(zip(C, B)):
        print(f"   anchor {k} ({angles[k]:5.1f} deg): planned clearance {planned[k]:.2f} mm; final arm+ball vs undeformed wall: deepest "
              f"{max(sd(Cw[:-1]).max() + arm_r, sd(Bw).max()):+.2f} mm")
    print(f"final cuff vs undeformed wall: {int((sd(Xf) > 0).sum())} of {len(Xf)} nodes in tissue, deepest {sd(Xf).max():+.2f} mm")

    #write
    d02.write_expander(out/"cuff.vtp", Xc, quads, Xf)
    np.savez(out / "anchors_beam.npz", n=N_ANCHOR, elem=cfg.ANCHOR_ELEM * s, arm_r=arm_r, ball_r=ball_r, ball_faces=np.asarray(ball.faces),
             shift=shift, arm_mask=arm_mask, cuff_mask=cuff_mask, **{f"C{k}": C[k] for k in range(N_ANCHOR)}, **{f"B{k}": B[k] for k in range(N_ANCHOR)},
             **{f"er{k}": ER[k] for k in range(N_ANCHOR)}, **{f"et{k}": ET[k] for k in range(N_ANCHOR)})
    T = np.eye(4); T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = e1, e2, n, c - z_shift * n          # device -> heart (final)
    info = dict(id=a.id, size_mm=size, note=note, scale=s, landing="anchor-referenced", clearance_mm=a.clearance, z_shift_device_mm=z_shift,
                ventricular_of_widest_band_mm=z_mid - z_shift, anchor_planned_clearance_mm=planned, anchor_angles_deg=angles,
                oversizing_pct=100 * (size / plan["d_perim_mm"] - 1), expander_r_widest_mm=float(prof[:, 0].max()), sizing_phase=cfg.SIZING_PHASE,
                axis_rule=a.axis, axis_offset_e1e2_mm=off.tolist(), crimp_cuff=crimp, crimp_cuff_plan=crimp_plan, anchor_start_shift_mm=shift, arm_r_mm=arm_r, ball_r_mm=ball_r,
                n_cuff_nodes=len(Xc), n_beam_elements=int(arm_mask.size), device_to_world=T.tolist())
    (out / "device_full.json").write_text(json.dumps(info, indent=1))
    print("wrote", out)

if __name__=="__main__":
    main()


