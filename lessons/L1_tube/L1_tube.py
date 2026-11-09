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
cfg = import_module("fea_config")
FEA = cfg.FEA
OUT = FEA / "lessons" / "L1_tube"
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

# expander: a thin cylinder pf surface elements inside the tube
R0,R1=8.0,13.0 #from start to end of expansion
ZC0,ZC1,NZC=-2.0,L+2.0,8 # it reaches 2 mm beyond both tube ends
C0=10000 

zc=np.linspace(ZC0,ZC1,NZC+1)
cid=lambda j,k: C0+1+(j%NT) + NT*k
cnodes=[(cid(j, k), R0 * np.cos(th[j]), R0 * np.sin(th[j]), zc[k]) for k in range(NZC + 1) for j in range(NT)]
cquads=[[cid(j, k), cid(j + 1, k), cid(j + 1, k + 1), cid(j, k + 1)] for k in range(NZC) for j in range(NT)]
radial_move=R1-R0 
# every expander node moves radially by this much (mm)
E,nu,rho=cfg.E_MPA,cfg.NU,cfg.DENSITY
mu=E/(2*(1+nu)) #shear modulus in MPa
K=E/(3*(1-2*nu)) #bulk modulus in MPa
C10,D1=mu/2.0,2.0/K  #Abaqus neo-hookean constants for the constitutive law mu0 = 2*C10, K0 = 2/D1

#explicit stability
c_wave=np.sqrt((K+4*mu/3)/rho) # dilatational wave speed (mm/s)
h_min=min(rr[1]-rr[0],RI*2*np.pi/NT, zz[1]-zz[0]) #smallest element edge (mm)
dt_est=h_min/c_wave  # stable time increment ~ h / c
t_total=cfg.T_RAMP + cfg.T_HOLD

if __name__ == "__main__":
    print(len(nodes), "tube nodes,", len(elems), "hexahedra,", len(inner_elems), "inner-face elements,", len(ends), "end nodes")
    print("outputs ->", OUT)
    print(f"expander: {len(cnodes)} nodes, {len(cquads)} quads, moves radially {radial_move:.1f} mm")
    print(f"neo-Hooke: mu {mu:.4f} MPa, K {K:.3f} MPa (K/mu = {K / mu:.1f}), C10 {C10:.4e}, D1 {D1:.4e}")
    print(f"wave speed {c_wave:.3e} mm/s, smallest edge {h_min:.2f} mm -> stable dt ~ {dt_est:.1e} s -> ~{t_total / dt_est:,.0f} increments for {t_total:g} s")


