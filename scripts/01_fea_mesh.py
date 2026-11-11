"""
Coarse FEA wall unstructured tet mesh for the pilot synthetic heart

Why: the 07a wall of SYN_0001 has 584k quadratic nodes, above the licence cap (250k). For the FEA ONLY we decimate the inner (SSM) surface,
grow the wall around it with the 07a recipe, and store a map that puts the FEA displacement back on EVERY vertex of the original SSM surface.

Adjustments will be made to refinement or coarsening strategy of the mesh when transitioning to surface 
blood pool triangulated meshes to shells that will undergo FEA simulations

Usage: python 01_fea_mesh.py --id SYN_0001 --level L3
Writes (local data side): runs/<id>/mesh/<id>_wall_<level>.vtu and <id>_wall_<level>_map.npz
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pymeshlab
import vtk
from scipy.spatial import cKDTree 
from vtkmodules.util.numpy_support import vtk_to_numpy
import fea_config as cfg

def load_module(name, folder):
    """07a_grow_wall.py starts with a digit, so an import statement cannot name it:loading it from its file path instead"""
    spec=importlib.util.spec_from_file_location(name,folder/f"{name}.py")
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def read_polydata(path):
    r=vtk.vtkXMLPolyDataReader(); r.SetFileName(str(path)); r.Update()
    return r.GetOutput()

def barycentric(P,F,Q):
    """For every query point Q: the nearest triangle of the surface (P, F) and the barycentric weights of Q projected onto it.
    A point in a triangle (a, b, c) is u*a + v*b + w*c with u + v + w = 1; if all weights are >= 0 the point is inside."""
    centres=P[F].mean(1)
    _,cand=cKDTree(centres).query(Q,k=8)# 8 candidate triangles per point (nearest centroids)
    best_t = np.zeros(len(Q), int); best_w = np.zeros((len(Q), 3)); best_d = np.full(len(Q), np.inf)

    for j in range(cand.shape[1]):
        t=cand[:,j]
        a,b,c=P[F[t,0]],P[F[t,1]],P[F[t,2]]
        v0, v1, v2 = b - a, c - a, Q - a
        d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
        d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
        den = d00 * d11 - d01 ** 2 + 1e-30
        v = (d11 * d20 - d01 * d21) / den
        w = (d00 * d21 - d01 * d20) / den
        u = 1 - v - w
        W = np.clip(np.stack([u, v, w], 1), 0, None)            # outside the triangle: clip to its edge / corner
        W /= W.sum(1, keepdims=True)
        proj = W[:, 0:1] * a + W[:, 1:2] * b + W[:, 2:3] * c
        d = np.linalg.norm(Q - proj, axis=1)
        better = d < best_d
        best_t[better], best_w[better], best_d[better] = t[better], W[better], d[better]
    return best_t, best_w, best_d

    
def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id",default="SYN_0001")
    ap.add_argument("--level", default="L3", choices=sorted(cfg.INNER_FACES))
    a=ap.parse_args()

    sys.path.append(str(cfg.SHAPE_SCRIPTS))   # 07a imports mesh_io from its own folder
    g=load_module("07a_grow_wall",cfg.SHAPE_SCRIPTS)
    heart=read_polydata(cfg.HEARTS/f"{a.id}_RH.vtp")
    P_in,F_in=g.arrays(heart)
    labels={k: vtk_to_numpy(heart.GetPointData().GetArray(k)) for k in ("chamber","tv_annulus","patch")}
    print(f"{a.id}: {len(P_in):,} SSM vertices, {len(F_in):,} faces; labels " +
          ", ".join(f"{k} {sorted(set(map(int, v)))}" for k, v in labels.items()))
    n_faces, outer_edge=cfg.INNER_FACES[a.level], cfg.OUTER_EDGE[a.level]
    t0=time.time()

    #decimate inner ssm generated surface to the target number of faces to run mesh convergence study and make computational bearable
    ms=pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=P_in,face_matrix=F_in))
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=n_faces, preservenormal=True,planarquadric=True,qualitythr=0.5)
    m=ms.current_mesh()
    P,F=m.vertex_matrix().astype(float),m.face_matrix()

    #outer surface= inner surface offset by outward wall growth, remeshed to outer edge length
    mesh_io = load_module("mesh_io", cfg.SHAPE_SCRIPTS)
    polydata_from_arrays = mesh_io.polydata_from_arrays
    offset_P,offset_F=g.offset_surface(polydata_from_arrays(P,F),cfg.WALL_THICKNESS,cfg.OFFSET_SPACING, cfg.OFFSET_SIGMA)
    P_out,F_out=g.remesh(offset_P,offset_F,outer_edge)

    # tetrahedral mesh between the two surfaces: every inner vertex must come oyt as a mesh node
    X,tets,quality=g.volume_mesh((P,F),(P_out,F_out))
    tree = cKDTree(X)
    miss, ssm_node = tree.query(P)
    assert miss.max() < 1e-4, f"inner vertices are not mesh nodes (worst {miss.max():.2e} mm)"

    print(f"{a.level}: inner {len(P):,} vertices / {len(F):,} faces; outer {len(P_out):,} vertices; {len(tets):,} tets on {len(X):,} corner nodes; "
          f"min tet quality {quality.min():.3f}; {time.time() - t0:.0f} s")

    #labels= only nodes on the inner surface are SSM vertices and they take labels of the nearest original SSM vertex
    nearest_original=cKDTree(P_in).query(P)[1]
    surface=np.zeros(len(X),np.int8) # 0 = inside the wall, 1 = inner (SSM) surface, 2 = outer surface
    surface[tree.query(P_out)[1]]=2
    surface[ssm_node]=1
    assert (surface==2).sum() >=0.9*len(P_out) and (surface==1).sum()== len(P), "surface labels inconsistent"
    point_data={"surface":surface}
    ssm_id=np.full(len(X),-1,np.int32)
    ssm_id[ssm_node]=np.arange(len(P))  # index of the coarse inner vertex that this node is
    point_data["ssm_id"]=ssm_id

    for k,v in labels.items():
        arr=np.full(len(X),-1,np.int32) # -1 = not an inner node, so no label
        arr[ssm_node]=v[nearest_original]
        point_data[k]=arr

    # 5) the map: every ORIGINAL SSM vertex -> a triangle of the coarse inner surface + barycentric weights
    tri,bary,dist=barycentric(P,F,P_in)
    out_dir=cfg.RUNS/a.id/"mesh"
    out_dir.mkdir(parents=True,exist_ok=True)
    name=f"{a.id}_wall_{a.level}"
    g.write_vtu(out_dir/ f"{name}.vtu",X,tets,point_data,{"minSICN":quality})
    np.savez(out_dir / f"{name}_map.npz", tri=tri, bary=bary, dist=dist, coarse_nodes=ssm_node, faces=F)
    print(f"{name}: SSM -> coarse distance p99 {np.percentile(dist, 99):.3f} max {dist.max():.3f} mm; "
          f"nodes under the vessel caps (patch 10/11/12): {int(np.isin(point_data['patch'][ssm_node], (10, 11, 12)).sum())}; written to {out_dir}")



if __name__=="__main__":
    main()

