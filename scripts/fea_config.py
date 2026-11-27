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
INNER_FACES={"L5":2600,"L4":6000,"L3":12000,"L2":26000,"L1":26000} 
OUTER_EDGE={"L5":6.5,"L4":4.4,"L3":2.6,"L2":2.0,"L1":2.0} #outer surface edge length in mm
ZONE_MM=15.0 #close to device where local refinement and sizing of mesh grid elements can be employed to get finest computational discretization of domain
REFINE_ZONE={"L1":1.5} # L1 = L2 + LOCAL refinement: surface edges / 1.5 inside the device zone (3D ratio ~1.33 there to be correct and consistent when analyzing the GCI
WALL_THICKNESS=4.0
OFFSET_SPACING=0.4 #mm voxel size of the distance field used to offset the surface
OFFSET_SIGMA=0.8 #mm gaussian smoothing of the distance field


# assumed constitutive laws for material property assignment
E_MPA,NU,DENSITY=0.1, 0.49, 1.0e-9 # Neo-Hookean simple hyperelasticity with nearly incompressible biological tissue having bulk modulus much higher than the shear one
ANNULUS_BAND_MM=5.0  # mm: tets whose centroid lies within this distance of the TV-annulus nodes = annular band (UQ input, assumption: no measured band width)
ANNULUS_BAND=_MM=5.0 # mm: tets whose centroid lies within the distance of the TV annulus node= annular band (UQ input, assumption : no measured band width)
#in reality due to TAPSE, there is longitudinal displacement of annulus.

#---Uncertainty Quantification for wall material MODEL FORMS (UQ).Values as published; units MPa, mm.
# Volumetric term: every source is (nearly) incompressible without a usable bulk value for our neo-Hookean-matrix forms -> D from nu = 0.49 on the
# small-strain shear modulus mu0 of each form (same compressibility as the V&V): K = 2 mu0 (1 + nu) / (3 (1 - 2 nu)), D = 2 / K.
NU_UQ=0.49

def _D(mu0,nu=NU_UQ):
    return 2.0/(2.0*mu0*(1+nu)/(3*(1-2*nu)))

_A_NORD=0.200e-3 #Nordsletten 2021 to cite, a_e = 0.200 kPa (Fig. 4 caption, p. 450): W = a/2 (I1bar - 3) -> a = mu0, C10 = a / 2
_C10_WANG= 0.0374e-3 * 15.3875       # Wang 2012 myocardium C10 * C01 (kPa -> MPa): small-strain match of the exponential matrix, mu0 = 2 C10
_C_KOL,_AL_KOL=4.98e-3, 32.0       # Kolawole 2025 Guccione: c at alpha = 32, MEAN of the 6 personalised subjects of Table 3 (8.7, 2.3, 8.3, 2.4, 3.7, 4.5 kPa)
_BF, _BT, _BFT = 0.55 * _AL_KOL, 0.25 * _AL_KOL, 0.20 * _AL_KOL   # fixed ratios r_f 0.55, r_t 0.25, r_ft 0.20 (p. 8) -> bf 17.6, bt 8.0, bft 6.4
MATERIALS = {
    "NH_nordsletten": dict(law="nh", C10=_A_NORD / 2, D=_D(_A_NORD),
                           src="Nordsletten et al. 2021 Acta Biomater 135:441 (human ventricular myocardium, Sommer 2015 data): isochoric neo-Hookean "
                               "baseline matrix a_e = 0.200 kPa (equilibrium/relaxed matrix of the viscoelastic model; anisotropy and stiffening are in "
                               "the viscous part, not modelled -> soft lower bound); paper incompressible, D from nu 0.49"),
    "HGO_wang": dict(law="hgo", C10=_C10_WANG, D=_D(2 * _C10_WANG), k1=6.0798e-3, k2=98.3666, kappa=0.144, theta_deg=6.78,
                     src="Wang, Sirois, Sun 2012 J Biomech 45:1965, myocardium (human LV, 79 y, equibiaxial), kPa per Mao 2018; Abaqus HGO = neo-Hookean "
                         "matrix C10 = C10_W * C01_W (small-strain match, matrix stiffening lost); fibres k1, k2, kappa (calibrated dispersion) as published; "
                         "2 families +-6.78 deg about circumferential; D from nu 0.49 (Wang D 0.0005 1/kPa with the NH matrix: K/mu ~3500, Abaqus warns)"),
    "FUNG_kolawole": dict(law="fung", c=_C_KOL, b1111=_BF, b2222=_BT, b3333=_BT, b1122=0.0, b1133=0.0, b2233=0.0,
                          b1212=_BFT / 2, b1313=_BFT / 2, b2323=_BT / 2, D=_D(_C_KOL * _BFT / 2),
                          src="Kolawole et al. 2025 Sci Rep 15:5556 (7 healthy humans, in vivo MRI + FE, LV): Guccione transversely isotropic "
                              "psi = c/2 (e^Q - 1), Q = bf Eff^2 + bt (Ess^2 + Enn^2 + 2 Esn^2) + 2 bft (Efs^2 + Efn^2), alpha = 32, ratios 0.55/0.25/0.20; "
                              "c = mean of the 6 personalised subjects (derived, Table 3); mapped EXACTLY to Abaqus FUNG-ORTHOTROPIC (Q = ... + 4 b1212 E12^2, "
                              "verified): b1111 bf, b2222 = b3333 bt, b2323 bt/2, b1212 = b1313 bft/2, couplings 0; axis 1 = fibre = circumferential (our rule)"),
    "FUNG_nemavhola_RV": dict(law="fung", c=2.826e-3, b1111=1.763, b2222=3.239, b3333=3.239, b1122=1.294, b1133=1.294, b2233=1.294,
                              b1212=1.207 / 4, b1313=1.207 / 4, b2323=1.207 / 4, D=_D(2.826e-3 * 1.207 / 4),
                              src="Nemavhola 2021 porcine RV biaxial (glutaraldehyde-treated, planar): 2D->3D assumptions (out-of-plane = cross-fibre, b5/b6 "
                                  "dropped, E12 tensor reading b1212 = b4/4 to verify) -> ALTERNATIVE only"),
}

# ---- ParaView views made for every run by run_batch (15b_paraview_view.py, run with ParaView's own Python)
PVPYTHON = Path(r"C:\Users\boiar\Downloads\ParaView-6.2.0-RC1-Windows-Python3.12-msvc2017-AMD64\ParaView-6.2.0-RC1-Windows-Python3.12-msvc2017-AMD64\bin\pvpython.exe")
PV_FIELDS = ("S_maxP_kPa", "LE_maxP_pct", "SED_kPa", "VM_kPa", "CPRESS_kPa")   # fields of 13 shown (range = device-zone p99, CPRESS p95)
PV_NORMAL = "y"                                                            # cut-away plane normal through the cuff centre (y: best SYN_0001 view)

# Boundary conditions definition
LID_DEPTH=4.6 # RULE: wall nodes within this depth under the vessel caps (patch 10/11/12) are fixed points during the simulation
# outer-wall elastic support = stand-in for pericardium and surrounding tissue (RULE, decided from the support sweep)
SUPPORT_MODE="normal_graded"   # normal-only springs: the tissue may still slide sideways, as in a lubricated pericardial sac
SUPPORT_K=0.4                  # N/mm3 (= MPa/mm); any value >= 0.4 gave the same result in the sweep (0.4 vs 1.0), the wall's own E/h = 0.025
SUPPORT_D_ON=20.0              # mm: no support within this distance of the TV ring (the expansion zone stays free)
SUPPORT_RAMP=15.0              # mm: smooth-step rise to full support (full beyond 35 mm)

# EXPLICIT SOLVER settings (RULE)
T_RAMP,T_HOLD=0.2,0.05 # s: smooth step expansion, then hold so the wall settles when the contact is defined (0.2 s quasi static on every mesh level, see the 0.1 vs 0.2 s ramp grid sweeps)
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

#full device (RULE, calibrated on SYN001 from SSM N=94)
LANDING_CLEARANCE=1.75 # c(mm) accounts for th emost intr-annular possible landing with the configured anatomy
# so that every anchor at /below deformed annulus (On L3 mesh, the sweep was on 1/1.75/2.5)
START_CLEAR=0.5 #mm: the started (shifted) anchors sit at least this far inside the blood pool
ANCHOR_ELEM=0.5 # mm: beam element length = anchor surface edge length
CUFF_START_GAP=0.3 #mm the crimped cuff starts at least this far inside every anchor arm (anchor cuff contact shall be modelled)
ANCHOR_E_EFF=319.8 # MPa : circular section r 1.2 s with the bending stiffness EI of a nitinol strut 0.5 x 1.0 mm of (E=50 GPa), so EI=521 N mm2
ANCHOR_NU=0.33 #Nitinol Poisson Ratio (far from incompressibility)
NITI_DENSITY=6.45e-9 # t/mm3 (6.45 g/cm3)
ANCHOR_ALPHA=50.0 # 1/s mass-proportional damping (NUMERICAL: zeta ~10 % at the ~40 Hz mass-scaled cantilever mode; dissipates 2e-4 of the external work)
CUFF_WALL_CONTACT="hard" #(hard mu 0: lift-off allowed, no crushed tet); "bonded"= hard + NO SEPARATION + ROUGH (Explicit general contact demands ROUGH with No Separation) | "hard" = hard + separation allowed (+ Coulomb --friction)
AXIS_RULE="centroid" # cuff axis through the centroid of the area enclosed by the TV annulus contour | "perimeter" | "mec" (07b centre)
# mimics clinical practice as in CT based pre-procedural planning of delivery and landing of the transcatheter device
assert CUFF_WALL_CONTACT in ("bonded","hard") and LANDING_CLEARANCE>=0, "full-device settings are out of range"
assert AXIS_RULE in ("centroid","perimeter","mec"), "unknown axis deployment rule"

# Force-controlled BC (crimping and self expansion) for better clinical plausibility and more realistic outcome for the domain to hand to CFD/FSI
# plate crimp + superelastic expansion with under expansion, SOLID frame
# Boxwell et al. 2026 CMPB 282:109404 (nitinol Table 2, 12 rigid plates, C3D8R struts, damping, mass-scaling target, energy gates),
# Bressloff et al. 2022 CVET 13:735 (PET skirt as S4R shell, 0.1 s crimp). Everything not listed here is inherited from the frozen stage 0.
DEVICES=FEA/"inputs"/"devices"  # one folder per device package (heart independent): pattern, hex frame, skirt + skin, anchors, QC, VTK views
DEVICE="replica_44"  # ACTIVE device: replica_44 (appearance-model generator, 02c) | evoque_48_ct (microCT reconstruction, 02e, later)
NITI_SE=dict(E_A=54440.0,nu_A=0.3,E_M=26910.0,nu_M=0.3,eps_L=0.047,sL_S=350.0,sL_E=375.0,sU_S=152.0,sU_E=120.0,sCL_S=520.0,epsV_L=0.047,
             T0=37.0,dsdT_L=6.527, dsdT_U=6.527)# MPa; Boxwell Table 2 (E_A > E_M: stable in Explicit)
  # analysis at T = T0 = 37 C -> the dsig/dT terms have no effect (stated)
NITI_ALPHA=35.0  # 1/s mass-proportional frame damping (Boxwell); divided by the wall-law time scale s (similarity)
PET=dict(E=50.0,nu=0.3,t=0.06, density=1.38e-9) # Bressloff 2022: PET skirt, linear elastic S4R shell, no damping, assumed simplification
SKIRT_H=0.5 #mm skirt S4R element size (shell) structured grid on the grid surface (swept), skirt attahced to outer frame inner surfaces
HEX_H=0.26#mm in plan quad size of the strut pattern (~3 elements across a 0.70 mm strut = in-plane crimp bending)
HEX_NL = 3                 # hex layers through the thickness: 3
HEX_JUNCTION_R=0.55 #mm junction disk radius in the struct chart (generator NODE_R)
HEX_SJ_MIN=0.02  # hexes with a corner scaled Jacobian <= this are DROPPED (degenerate junction slivers; count reported in the QC)
PLATES_N=12 #rigid R3D4 crimp plates as in paper referenced
PLATE_GAP=0.15 #mm start gap before ramp of crimp phase
PLATE_FACET=2.0 #mm plate facet size(a flat rigid plate is exact at any size)
PLATE_NODE_MASS=1.0e-6  #t: point mass + rotary inertia of each plate reference node (prescribed motion: value irrelevant, must be > 0 for hard penalisation based contact upon crimping)
CRIMP_LAMBDA_MAX=0.60  # RULE: radial crimp ratio (plates r -> lam r), calibrated per DEVICE (crimp is heart independent)
CRIMP_CLEAR=0.5 # mm: per HEART the crimped room skin must start >= this inside the wall + final anchors 
OPEN_LAMBDA=1.06  #plates re-open to this x their start radius (they leave the frame: purely force-controlled expansion)
T_CRIMP=0.1   # s: plate crimp duration at time scale 1 (Bressloff 0.1 s; Boxwell 0.3 s = s1 check run)
DT_FRAME=2.0e-6 #s frame and skirt VARIABLE mass scaling target  at time scale 1 (Boxwell's coarsest tested; effective dt ~1.67e-6 set by the wall crushed element cells during deployment at annular region, L min driving FEA)
MS_FREQUENCY=1000  # increments between mass-scaling updates (Boxwell's semi-automatic scheme)
SKIN_NODE_MASS=1.0e-9 # t (1 mg) per room-skin node: the skin follows the struts by *EQUATION
SELF_CONTACT_EXCL=1.2 #mm frame elements this close to tied seams / junction circles are excluded from self contact (overlap by construction)
FRAME_AE_GATE_PCT = (10.0, 15.0)  # frame ALLAE/IE at crimp end: <= 10 PASS, 10-15 FLAG, > 15 FAIL; plus: no AE growth in release + hold

FORCEBC_IDS=dict(F=7_000_000, S=8_000_000, K=8_200_000, C=8_300_000,M=8_400_000,RP=8_600_001)# node/element label offsets of the device
                           # (frame up to 1 M nodes: 3 layers = 125.5k; the 03 V&V model ends at ~6.01 M; 03c asserts no overlap)
assert NITI_SE["E_A"] > NITI_SE["E_M"] > 0 and 0 < NITI_SE["eps_L"] < 0.1, "nitinol parameters out of range (E_A < E_M is unstable in Explicit)"
assert 0.3 < CRIMP_LAMBDA_MAX < 1 < OPEN_LAMBDA and 0 < T_CRIMP and 0 < DT_FRAME < 1e-4, "crimp settings out of range"
assert HEX_NL in (2, 3) and 0.15 <= HEX_H <= 0.35 and 0 < HEX_SJ_MIN < 0.2, "hex frame mesh settings out of range"


# wall element output written by the 03 script also accounts ESEDEN, EVOL, S, LE, so strain and stresses): number interval of its own field request; 10= every 3rd displacement frame (30/10)
ELEM_OUT_INTERVAL=10
#clinical comparator for the sizing outputs for validation purposes of FEA setup and outcomes
CLINICAL=dict(source="Penta et al. 2025, JACC Cardiovasc Interv 18:1492-4 (EVOQUE, CT n = 16)", oversizing_mean_pct=8.5, oversizing_range_pct=(3.0,14.0),frame_node3_perimeter_mm=147.9, annulus_pre_perimeter_mm=149.7)

#ordered study outputs runs/<id>/compare/<subfolder>
COMPARE_DIRS = {"mesh": "01_mesh_convergence", "qc": "02_mesh_qc", "sens": "03_sensitivity",
                "deliverable": "05_direct_vs_mapped", "device": "06_device_trials","uqin":"07_uq_inputs","landing":"08_landing_depth",
                "crimp":"09_crimp_test","figures":"10_figures"}

def compare_dir(run,kind):
    """runs/<id>/compare/<ordered subfolder of this kind of study> (created if missing)"""
    d=run/"compare"/COMPARE_DIRS[kind]; d.mkdir(parents=True, exist_ok=True); return d


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

                
