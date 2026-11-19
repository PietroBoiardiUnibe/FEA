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
    ap.add_argument
