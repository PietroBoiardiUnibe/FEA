"""
L1: a thick tube of soft tissue expanded by a cylinder, written as an Abaqus/explicit input file.
Same keywords and workflow as the real deployment simulation, tiny mesh, analytic check for code verification purposes.
Units: mm, tonne, s, N, MPa
"""
import numpy as np
import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))  # find fea_config.py in the repo
FEA = import_module("fea_config").FEA
OUT=FEA/"lessons"/"L1_tube"
OUT.mkdir(parents=True, exist_ok=True)

#geometry
RI,RO,L=10.0,14.0,12.0 #tube inner radius outer radius and length
NT,NR,NZ=48,4,6  #elements around, through the wall, along the axis

# tube nodes and hexahedra
th=np.linspace(0,2*np.pi,NT, endpoint=False)
rr=np.linspace(RI,RO,NR+1)
zz=np.linspace(0,L,NZ+1)
nid=lambda i,j,k: 1+i +(NR+1)*(j%NT)+(NR+1)*NT*k # node number of (radial i, angular j, axial k)

nodes = [(nid(i, j, k), rr[i] * np.cos(th[j]), rr[i] * np.sin(th[j]), zz[k])
         for k in range(NZ + 1) for j in range(NT) for i in range(NR + 1)]
elems = []

for k in range(NZ):
    for j in range(NT):
        for i in range(NR):
            elems.append([nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                          nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)])

inner_elems = [e + 1 for e in range(len(elems)) if (e % NR) == 0]          # first radial layer: its face S6 lies on the inner radius
ends = [n[0] for n in nodes if abs(n[3]) < 1e-9 or abs(n[3] - L) < 1e-9]    # nodes on the two end faces

if __name__ == "__main__":
    print(len(nodes), "tube nodes,", len(elems), "hexahedra,", len(inner_elems), "inner-face elements,", len(ends), "end nodes")
    print("outputs ->", OUT)
    

