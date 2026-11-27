"""
STAGE 1 --  Force-Controlled deployment input: Boxwell typology as seen in reference paper.
Plates crimp the device and superelastic nitinol ensures self expansion of solid frame.

Converts 03's full-device input (kept as <name>_03full.inp) into the force-BC input <name>.inp in the same run folder:
  FROZEN V&V PART (from 03, unchanged): wall L3 C3D10M + law + time scale s, graded normal supports, lids, anchors (B31 + skins + rigid balls,
    roots PRESCRIBED over the ramp = deployed first), contact property CPROP (hard, frictionless, default penalty), wall fixed mass scaling.
    The cuff expander is removed.
  DEVICE (cfg.DEVICES/<device>, package -> world by device_full.json 'device_to_world' = centroid axis + anchor-referenced landing):
    frame   C3D8R hexes (02d frame_hex.npz), Boxwell superelastic nitinol, T = T0 = 37 C, alpha NITI_ALPHA / s
    ties    chart seams (outer, inner), outer-frame end circles -> inner frame, tip bases -> frame (*TIE, ADJUST=NO)
    skirt   S4R PET shell (skirt.npz) tied to the inner faces of the outer struts (Bressloff)
    skin    1 mm room skin SFM3D4R (skirt.npz), each node = inverse-distance combination of its 4 nearest UNTIED outer-face strut nodes
            (*EQUATION): the struts carry all device-wall contact; the skin is removed after the FEA (room for the native leaflets)
    plates  PLATES_N rigid R3D4 plates at frame r max + PLATE_GAP, reference nodes moved radially r0 -> lam r0 -> OPEN_LAMBDA r0
    DRP     delivery surrogate: paddle ends -> distributing coupling -> DRP in the device frame (*TRANSFORM): U3 + rotations held, U1-U2 FREE
  CONTACT general, hard, frictionless, default penalty (CPROP).
    step 1: device self-contact (plates-outer frame, plates-tips, outer self, inner self, outer-inner, skirt-inner, skirt self; elements within
            SELF_CONTACT_EXCL of tied zones excluded) + anchors-wall (anchor skins, balls)
    step 2-3: + room skin-wall, anchor arm skins-skin, balls-skin
  STEPS (smooth steps in TOTAL time, every duration x s): 1 CRIMP_ANCHORS (plates crimp in T_CRIMP x s while the anchors deploy over the ramp),
    2 RELEASE (ramp: plates open, the frame self-expands against wall + anchors), 3 HOLD (03 hold)
  MASS SCALING: V&V part fixed (03 DT, step 1); frame + skirt VARIABLE to DT_FRAME x s every MS_FREQUENCY increments.
  OUTPUT: U, RF, CSTRESS, CDISP, CFORCE (nodal contact forces -> radial force); WALL ESEDEN/EVOL/S/LE; frame S, LE, ELASE; skirt S, LE;
    EMSF (element mass-scaling factor: added mass per part) on wall, frame, skirt; ELASE (artificial energy per element) on wall + frame;
    anchors S, LE, SF; energies per part (WALL, ANC_B, DFRAME, DSKIRT); DRP U1-3 (in-plane self-centring); plate RF (crimp force);
    contact force + area on the skin and on the wall (CFN, CAREA); everything every 10th of a step (energies 1/100).
CRIMP (calibrated per DEVICE, verified per HEART): the crimp is heart independent (air crimp a0 = in-heart crimp of s2 within the paddle-BC
difference), so lam = CRIMP_LAMBDA_MAX for every heart. If the device folder holds crimped_skin_lam<lam>.npz (the crimped room skin in package
coordinates, written by 07f from the first run of this device), the per-angle start clearance to the wall and the final anchors is checked here
(< CRIMP_CLEAR -> stop: this heart needs a deeper crimp); 07f always verifies the ACTUAL clearance at the end of step 1.
Usage: python 03c_forcebc_inp.py --id SYN_0001 --level L3 --tag cen_hard_ramp0.2_fbc [--device replica_44] [--mesh frame_hex] [--lam 0.60]
(after 03 --full with the same tag; run_batch does both)
"""
import argparse
import json
import shutil
import numpy as np
import vtk
from scipy.spatial import cKDTree 
from vtkmodules.util.numpy_support import vtk_to_numpy
import fea_config as cfg
import fea_device as fd

EXCESS_DTH,EXCESS_DZ=4.0, 0.75 # deg, mm angular/axial window of the per angle clearance check at start

def world(T,P):
    return np.asarray(P)@ T[:3,:3].T + T[:3,3]

def block(head,ids):
    ids=list(ids); return [head] + [", ".join(map(str, ids[i:i + 16])) for i in range(0, len(ids), 16)]


def frozen_vv(L,info):
    """03 full device input -> model lines without the expander, anchor root boundary lines, V&V fixed mass scaling DT"""
    i_exp=next(i for i, l in enumerate(L) if l.startswith("*NODE, NSET=EXP_N")); i_anc = next(i for i, l in enumerate(L) if l.startswith("*NODE,NSET=ANC0_N"))
    i_step=next(i for i,l in enumerate(L) if l.startswith("*STEP"))
    i_bc=next(i for i in range(i_step, len(L)) if L[i].startswith("*BOUNDARY, AMPLITUDE=RAMP"));  i_ct = next(i for i in range(i_bc, len(L)) if L[i].startswith("*CONTACT"))
    exp0 = str(info["expander_node0"] // 100000)
    anchor_bc = [l for l in L[i_bc + 1:i_ct] if l and not l.startswith(exp0)]
    assert anchor_bc and all(l.startswith("22") for l in anchor_bc), "unexpected boundary lines in the 03 step"
    model = L[:i_exp] + L[i_anc:i_step]
    k = next(i for i, l in enumerate(model) if l.replace(" ", "").startswith("*AMPLITUDE,NAME=RAMP"))
    model[k] = "*AMPLITUDE, NAME=RAMP, DEFINITION=SMOOTH STEP, TIME=TOTAL TIME"; model[k + 1] = f"0., 0., {float(info['time']['ramp_s']):g}, 1."
    dt_vv=float(next(l for l in L[i_step:i_bc] if l.startswith("*FIXED MASS SCALING")).split("DT=")[1].split(",")[0])
    return model, anchor_bc,dt_vv



