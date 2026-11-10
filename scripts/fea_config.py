"""
Finite Element Analysis pipeline to complement SSM generation of right heart chambers prior to IBAMR FSI simulations.
Configuration: every path and technical parameter stored in ONE place.
Tags: MEASURED from data, RULE ( decision taken), ASSUMED (placeholder, needs sensitivity check), TBD
Units everywhere:
mm, tonne, s, N, MPa  (density in t/mm3, stiffness per area in N/mm3).
"""

from pathlib import Path

# paths
DATA=Path(r"C:\data\Inselspital_Dataset\rh100")
FEA=DATA/"deploy"/"FEA"
RUNS=FEA/"runs"
SCRIPTS=Path(r"C:\Users\boiar\OneDrive - Universitaet Bern\Master Thesis Pietro\Code\FEA\scripts")
SHAPE_SCRIPTS=Path(r"C:\Users\boiar\OneDrive - Universitaet Bern\Master Thesis Pietro\Code\ShapeAnalysis\scripts")
HEARTS=DATA/"synthetic"/"2026-11-05_pilot" / "meshes" / "vtp" #SSM generated ones
WALLS=DATA/"deploy"/"2026-11-05_pilot" / "wall"  # vtu files
PLANS=DATA/"deploy"/"2026-11-05_pilot" /"plan" # the json report for the pre-procedural planning 
EVOQUE=FEA/"inputs"/"EVOQUE_frame"
ABAQUS=Path(r"C:\SIMULIA\Commands\abaqus.bat")

MAX_NODES=250_000
CPUS=4

# FEA mesh (RULE)
INNER_FACES={"L4":6000,"L3":12000,"L2":26000} 
OUTER_EDGE={"L4":2.6,"L3":2.6,"L2":2.0} #outer surface edge length in mm
WALL_THICKNESS=4.0
OFFSET_SPACING=0.4 #mm voxel size of the distance field used to offset the surface
OFFSET_SIGMA=0.8 #mm gaussian smoothing of the distance field


# assumed constitutive laws for material property assignment
E_MPA,NU,DENSITY=0.1, 0.49, 1.0e-9 # Neo-Hookean simple hyperelasticity with nearly incompressible biological tissue having bulk modulus much higher than the shear one

# Boundary conditions definition
LID_DEPTH=4.6 # RULE: wall nodes within this depth under the vessel caps (patch 10/11/12) are fixed points during the simulation
# outer-wall elastic support = stand-in for pericardium and surrounding tissue (RULE, decided from the support sweep)
SUPPORT_MODE="normal_graded"   # normal-only springs: the tissue may still slide sideways, as in a lubricated pericardial sac
SUPPORT_K=0.4                  # N/mm3 (= MPa/mm); any value >= 0.4 gave the same result in the sweep (0.4 vs 1.0), the wall's own E/h = 0.025
SUPPORT_D_ON=20.0              # mm: no support within this distance of the TV ring (the expansion zone stays free)
SUPPORT_RAMP=15.0              # mm: smooth-step rise to full support (full beyond 35 mm)

# EXPLICIT SOLVER settings (RULE)
T_RAMP,T_HOLD=0.1,0.05 # s: smooth step expansion, then hold so the wall settles when the contact is defined
DT_TARGET=1.0e-5 # s: fixed mass scaling target (added mass must stay small)
EXPANDER_NODE_MASS=1.0e-6 #t: point mass per expander mode (the penalty contact modeling for master/slave surfaces needs enforcing a nodal mass value to avoid not physical penetration)
FRICTION=0.0 # frictionless contact not modeling the tangential pressure during contact mechanics, 0.1 as sensitivity check possible.

#device, Measured on the package for EVOQUE and  Rule for scaling
R_IN_NOMINAL=13.72  #leaflet attachment radius = lumen wall
R_MID_NOMINAL=21.73 # outer skirt radius at its widest band in the 44 mm model
ANCHOR_CLEARANCE=1.0 # TBD vs IB grid: minimum distance anchor to wall (mm)
# EVOQUE sizing (RULE from the IFU: Edwards eIFU DOC-0565239A; FDA P230013C has the same 44-52 rows).
# per size and cardiac phase: (min, max) perimeter-derived annulus diameter, max annulus length (mm), from CT
EVOQUE_SIZES={44: {"systole":(36.5,43.0,45.5),"diastole":(39.6,45.5,50.0)},
              48: {"systole": (40.0, 47.0, 49.5), "diastole": (43.2, 49.5, 54.0)},
              52: {"systole": (45.0, 51.0, 53.5), "diastole": (46.8, 53.5, 58.0)},
              56: {"systole": (48.0, 55.0, 57.5), "diastole": (50.4, 57.5, 62.0)}}
SIZING_PHASE="diastole" # to be confirmed
SMALL_ANNULUS_POLICY="exclude" # RULE : below the 44 mm range -> clinically ineligible, not simulated ('oversize' = 44 + flag)

NATIVE_LEAFLET_THICKNESS=1.0   # mm; STANDARDISED for IB compatibility (literature 0.3-0.7 mm modern, 0.8-1.1 mm older): a modelling choice, sensitivity later
NATIVE_LEAFLET_LENGTH_REF = 17.4   # mm; mean cadaveric height: anterior 20.3, posterior 15.0, septal 17.0
ANNULUS_D_REF=40.0    # mm; normal 3D-echo annulus diameter (4.0 +- 0.7 cm): leaflet length scales with d_perim / this
DEVICE_SCALE_RULE = "s = D / 44: the measured STL package is the 44 model, larger sizes scale it linearly"

# acceptance of a deployment (RULE)
MAX_KE_IE_PCT=5.0 # wall kinetic / internal energy at the end (must be Quasi-Static simulation)
MAX_ALLAE_PCT=5.0 #artificial energy / internal energy
MAX_RING_SD=0.5 #  mm, out of roundness of the deployed deformed configuration in the model
MIN_CONTACT_PCT=95.0 # share of the ring touching the device

#sanity checks: a typo in a constant must fail at import,not later in pipeline
assert R_MID_NOMINAL>R_IN_NOMINAL>0 , "device radii are inconsistent"
assert 0<NU<0.5 and E_MPA>0 and DENSITY >0, "material parameters out of range"
assert T_RAMP>0 and T_HOLD>=0 and DT_TARGET>0, "time settings out of range"
assert SUPPORT_MODE in ("none","normal","normal_graded"),"unknown support mode"
assert all(lo < hi <= mx for D in EVOQUE_SIZES for lo, hi, mx in EVOQUE_SIZES[D].values()), "sizing table inconsistent"
assert SIZING_PHASE in ("systole", "diastole") and 0 < NATIVE_LEAFLET_THICKNESS < 2, "sizing / leaflet settings out of range"


def check():
    "Print whether every input path exists indeed"
    for name in ("SHAPE_SCRIPTS", "HEARTS", "WALLS", "PLANS", "EVOQUE", "ABAQUS"):
        p=globals()[name]
        print(f"{name:14s}{'OK     ' if p.exists() else 'MISSING'} {p}")

if __name__=="__main__":
    check()

                
