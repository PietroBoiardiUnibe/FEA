"""
fea_device.py --shared DEVICE geometry helpers Used by 02c (replica generator), 02d (hex mesher), 02e (microCT, later), 03c, 05f-16f.
DEVICE PACKAGE = one folder cfg.DEVICES / <name> (heart independent, package coordinates: device axis = z, flow +z), written by 02c or 02e:
  pattern.npz   strut centre lines in CHART coordinates of two surfaces of revolution (0 = outer frame, 1 = inner valve cylinder):
                prof<k>_r/_z (profile polylines), st_surf, st_W, st_off, st_theta (deg), st_s (mm, arc length), T (strut thickness),
                tip_ang (deg), tip_lean_deg, tip_len, tip_W  (atrial tips + paddles = straight swept members)
  skirt.npz     Xs, Qs (S4R skirt grid on the fabric mid-surface), Xk, Qk (1 mm room skin = fabric outer face + NATIVE_LEAFLET_THICKNESS)
  device.json   source, size_mm, symmetry N, thickness, radii, what is SYNTHETIC (skirt, anchor balls) vs MEASURED, QC verdicts
  frame_hex.npz 02d: hex mesh (X, H, part 0 outer / 1 inner / 2 tips, layer, sj, node sets for ties, skin drivers, skirt tie, paddles)
"""
import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtkIdTypeArray, numpy_to_vtk

HEX_FACES = {"S1": [0, 1, 2, 3], "S2": [4, 5, 6, 7], "S3": [0, 1, 5, 4], "S4": [1, 2, 6, 5], "S5": [2, 3, 7, 6], "S6": [3, 0, 4, 7]}   # Abaqus C3D8 faces

class Profile:
    """Surface of revolution from an (r, z) polyline with DECREASING z. s = arc length; eval(theta_deg, s) -> points, outward unit normals."""
    def __init__(self, r, z):
        self.r, self.z = np.asarray(r, float), np.asarray(z, float)
        self.s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(self.r), np.diff(self.z)))])
        dr, dz = np.gradient(self.r, self.s), np.gradient(self.z, self.s); n = np.stack([-dz, dr], 1)
        self.n = n / np.linalg.norm(n, axis=1, keepdims=True)

    def s_of_z(self, z):
        return np.interp(-np.asarray(z, float), -self.z, self.s)

    def r_of_s(self, s):
        return np.interp(s, self.s, self.r)

    def eval(self, th_deg, s):
        th = np.radians(th_deg); r, z = np.interp(s, self.s, self.r), np.interp(s, self.s, self.z)
        nr, nz = np.interp(s, self.s, self.n[:, 0]), np.interp(s, self.s, self.n[:, 1])
        return np.stack([r * np.cos(th), r * np.sin(th), z], -1), np.stack([nr * np.cos(th), nr * np.sin(th), nz], -1)


def monotone_z(r,z):
    """Force z strictly decreasing along the profile. Returns r, z, max change."""
    z2=np.minimum.accumulate(z)-np.arange(len(z))*1e-9
    return r,z2,float(np.max(np.abs(z2-z)))

def load_pattern(path):
    """pattern.npz -> (profiles [outer, inner], struts [(surf, theta_deg, s, W)], P dict)."""
    P = dict(np.load(path)); prof = [Profile(P["prof0_r"], P["prof0_z"]), Profile(P["prof1_r"], P["prof1_z"])]; o = P["st_off"]
    struts = [(int(P["st_surf"][i]), P["st_theta"][o[i]:o[i + 1]], P["st_s"][o[i]:o[i + 1]], float(P["st_W"][i])) for i in range(len(P["st_W"]))]
    return prof, struts, P

def save_pattern(path, prof_out, prof_in, struts, **extra):
    """struts = list of (surf 0|1, theta_deg array, s array, W) -> flat arrays (+ profiles and any extra arrays/scalars)."""
    off = np.concatenate([[0], np.cumsum([len(t) for _, t, _, _ in struts])])
    np.savez(path, prof0_r=prof_out.r, prof0_z=prof_out.z, prof1_r=prof_in.r, prof1_z=prof_in.z,
             st_surf=np.array([s for s, *_ in struts], int), st_W=np.array([w for *_, w in struts], float), st_off=off,
             st_theta=np.concatenate([t for _, t, _, _ in struts]), st_s=np.concatenate([s for _, _, s, _ in struts]), **extra)

def hex_scaled_jacobian(X, H):
    """Minimum corner scaled Jacobian of each 8-node hex (1 = perfect cube corner, <= 0 = inverted)."""
    J = []
    for c, (a, b, d) in enumerate([(1, 3, 4), (2, 0, 5), (3, 1, 6), (0, 2, 7), (7, 5, 0), (4, 6, 1), (5, 7, 2), (6, 4, 3)]):
        e1, e2, e3 = X[H[:, a]] - X[H[:, c]], X[H[:, b]] - X[H[:, c]], X[H[:, d]] - X[H[:, c]]
        J.append(np.einsum("ij,ij->i", np.cross(e1, e2), e3) / (np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) * np.linalg.norm(e3, axis=1)))
    return np.min(np.array(J), axis=0)

def hex_edges(X, H):
    """Edge lengths (n, 12) of 8-node hexes."""
    E = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])
    return np.linalg.norm(X[H[:, E[:, 0]]] - X[H[:, E[:, 1]]], axis=2)


def faces_in(H, labels, nodeset):
    """{Abaqus face id: element labels} of the hex faces whose 4 nodes all lie in nodeset (surface definitions for ties / contact)."""
    m = np.zeros(max(int(H.max()), int(np.max(nodeset))) + 1, bool); m[nodeset] = True; out = {}
    for lab, cols in HEX_FACES.items():
        sel = np.flatnonzero(m[H[:, cols]].all(1))
        if len(sel):
            out[lab] = labels[sel]
    return out

def write_vtp(path, X,cells,cell_arrays=None,point_arrays=None):
    """Lines (2 node cells) or polygons (3/4-node cells) -> .vtp with optional cell / point arrays (ParaView)."""
    pd=vtk.vtkPolyData();pts=vtk.vtkPoints(); pts.SetData(numpy_to_vtk(np.ascontiguousarray(X,float), deep=True)); pd.SetPoints(pts)
    ca=vtk.vtkCellArray()
    for c in cells:
        ca.InsertNextCell(len(c),[int(v) for v in c])
    (pd.SetLines if len(cells[0])==2 else pd.SetPolys)(ca)
    for data, arrays in ((pd.GetCellData(), cell_arrays), (pd.GetPointData(), point_arrays)):
        for k, v in (arrays or {}).items():
            a=numpy_to_vtk(np.ascontiguousarray(v), deep=True); a.SetName(k); data.AddArray(a)
    w=vtk.vtkXMLPolyDataWriter(); w.SetFileName(str(path)); w.SetInputData(pd); w.Write()

def write_hex_vtu(path,X,H,cell_arrays=None):
    """8-node hexes -> .vtu (VTK_HEXAHEDRON, same node order as Abaqus C3D8R) with cell arrays (part, quality, fields)."""
    ug=vtk.vtkUnstructuredGrid(); pts=vtk.vtkPoints(); pts.SetData(numpy_to_vtk(np.ascontiguousarray(X,float), deep=True)); ug.SetPoints(pts)
    ca=vtk.vtkCellArray(); conn=np.c_[np.full(len(H),8),H].ravel()
    ca.SetCells(len(H),numpy_to_vtkIdTypeArray(np.ascontiguousarray(conn,np.int64),deep=True)); ug.SetCells(vtk.VTK_HEXAHEDRON,ca)
    for k,v in  (cell_arrays or {}).items():
        a=numpy_to_vtk(np.ascontiguousarray(v),deep=True); a.SetName(k); ug.GetCellData().AddArray(a)
    w = vtk.vtkXMLUnstructuredGridWriter(); w.SetFileName(str(path)); w.SetInputData(ug); w.Write()


def to_pkg(P, rot_deg, z0):
    """Natural device frame -> package frame: rotation by rot_deg about z, then z_pkg = z0 - z (axial flip)."""
    th = np.radians(rot_deg); R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    Q = np.asarray(P, float) @ R.T; Q[:, 2] = z0 - Q[:, 2]; return Q

