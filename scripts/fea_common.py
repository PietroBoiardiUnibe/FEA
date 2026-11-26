"""
Tiny helpers shared by several pipeline scripts (one implementation, one place).
The cuff axis: every placement (02,02b) and every measurement about the device (06 ring and seal line, 07 checks, 10 QoIs),
uses the same AXIS stored once in the device record (device.json/device_full.json) and copied by 03 into the run's info file (axis_rule, axis_offset_e1e2_mm).
"""
import re
import numpy as np

def contour_centre(plan,rule):
    """Centre of the TV annulus contour in the 07b plane (identified earlier), as (e1,e2) offsets in mm from the 07b frame centre.
    'centroid' = centroid of the AREA enclosed by the annular contour = the annulus centre used in clinical CT pre-procedural planning and sizing of device;
    'perimeter' = centroid of the contour curve;  'mec' = the 07b centre itself (minimum enclosing circle, no shift).
    """
    if rule=="mec":
        return np.zeros(2)

    th,r=np.array(plan["ring"]["theta"]), np.array(plan["ring"]["r"]); o=np.argsort(np.mod(th,2*np.pi)); th,r=th[o],r[o]
    X=np.stack([r*np.cos(th), r*np.sin(th)],1); Xn=np.roll(X,-1,0)
    if rule=="perimeter":
        seg=np.linalg.norm(Xn-X,axis=1)
        return (0.5*(X+Xn)*seg[:,None]).sum(0)/seg.sum()
    cr=X[:,0]*Xn[:,1]-X[:,1]*Xn[:,0]   # shoelace: area centroid = sum (x_i + x_i+1) cr_i / (3 sum cr_i)
    return ((X+Xn)*cr[:,None]).sum(0)/(3*cr.sum())

def axis_plan(plan,offset):
    """Copy of 07b plan whose frame centre sits on the CUFF axis (offset = (e1, e2) mm from the 07b centre; None = no shift)
    The ring samples (theta, r, z) stay those of 07b: use ring_about() for heights as a function of the angle about the cuff axis."""
    off=np.zeros(2) if offset is None else np.asarray(offset, float); fr=plan["frame"]
    c=np.array(fr["centre"]) + off[0]*np.array(fr["e1"]) + off[1]*np.array(fr["e2"])
    return dict(plan, frame=dict(fr,centre=c.tolist()))

def ring_about(plan,offset):
    """Annulus height z (along the 07b normal) against the angle about the cuff axis: (theta in [0, 2 pi), z), sorted by theta."""
    th,r,z=(np.array(plan["ring"][k]) for k in ("theta","r","z")); off=np.zeros(2) if offset is None else np.asarray(offset, float)
    X=np.stack([r* np.cos(th), r*np.sin(th)],1)-off
    t=np.mod(np.arctan2(X[:,1],X[:,0]),2*np.pi); o=np.argsort(t)
    return t[o],z[o]

def clean_ring(r, k=7, tol=1.0):
    """Ray-cast ring radii r(theta) (equally spaced, closed loop): a ray deviating more than tol mm from the circular median of its 2k+1
    neighbours is a ray-casting glitch (the ray slips past the wall through a gap of the coarse surface) -> replaced by that median.
    Returns (cleaned radii, number of rays replaced)."""
    r = np.asarray(r, float); pad = np.concatenate([r[-k:], r, r[:k]])
    med = np.array([np.median(pad[i:i + 2 * k + 1]) for i in range(len(r))]); bad = np.abs(r - med) > tol
    return np.where(bad, med, r), int(bad.sum())

def band_ring(P0, P1, plan, nsec=24):
    """Annulus ring from the TV-annulus band NODES of the FEA inner surface (Lagrangian: the same nodes before and after, no ray casting).
    Nodes are grouped in nsec angular sectors by their UNDEFORMED angle about the cuff axis (plan = axis_plan); per sector the median position
    is one ring point; perimeter = closed polygon through the sector points. Robust on every mesh level (ray casting on coarse surfaces is not).
    Returns perimeter before/after (mm), stretch (%), mean radius and SD after (mm, about the cuff axis), sectors used."""
    fr = plan["frame"]; c, n, e1, e2 = (np.array(fr[k]) for k in ("centre", "normal", "e1", "e2"))
    A0, A1 = [np.stack([(P - c) @ e1, (P - c) @ e2, (P - c) @ n], 1) for P in (P0, P1)]
    sec = np.minimum((np.mod(np.arctan2(A0[:, 1], A0[:, 0]), 2 * np.pi) / (2 * np.pi) * nsec).astype(int), nsec - 1)
    used = [s for s in range(nsec) if (sec == s).any()]
    R0, R1 = [np.array([np.median(A[sec == s], axis=0) for s in used]) for A in (A0, A1)]
    per = lambda R: float(np.linalg.norm(np.roll(R, -1, 0) - R, axis=1).sum())
    rad = np.hypot(R1[:, 0], R1[:, 1])
    return dict(perimeter_before=per(R0), perimeter_after=per(R1), stretch_pct=100 * (per(R1) / per(R0) - 1),
                mean_r=float(rad.mean()), sd_r=float(rad.std()), sectors=len(used))

def annulus_sizing(P0,P1,plan,d_device,nsec=24):
    """CLINICAL sizing of the TV annulus before and after deployment, measured with ONE tool on same annulus band nodes.
    per angular sector (undeformed angle about the cuff axis) the median node = one annulus point; perimeter = closed 3D polygon, area = area enclosed
    by the polygon projected on the annulus plane, perimeter-derived diameter Dp=p/pi, area derived diameter  Da = 2 sqrt(A / pi).
    Post-procedural CT reports the same pair: annular expansion and over-sizing of nominal device diameter. 
    In particular d_device against Dp / Da (perimeter-derived, area-derived diameter, and area oversizing = (d_device / Da)^2 - 1)"""
    fr=plan["frame"]; c,n,e1,e2=(np.array(fr[k]) for k in ("centre","normal","e1","e2"))
    A0,A1=[np.stack([(P - c) @ e1, (P - c) @ e2, (P - c) @ n], 1) for P in (P0,P1)]
    ang=np.mod(np.arctan2(A0[:, 1], A0[:, 0]),2*np.pi); sec = np.minimum((ang / (2 * np.pi) * nsec).astype(int), nsec - 1)
    used=[s for s in range(nsec) if (sec==s).any()]
    R0, R1 = [np.array([np.median(A[sec == s], axis=0) for s in used]) for A in (A0, A1)]

    def meas(R):
        per=float(np.linalg.norm(np.roll(R,-1,0)-R, axis=1).sum()); x,y=R[:,0], R[:,1]
        area = float(0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))
        return dict(perimeter_mm=per, area_mm2=area, Dp_mm=per / np.pi, Da_mm=2 * np.sqrt(area / np.pi))
    
    pre,post=meas(R0),meas(R1)
    over = lambda m: dict(perimeter_derived_pct=100 * (d_device / m["Dp_mm"] - 1), area_derived_pct=100 * (d_device / m["Da_mm"] - 1),
                          area_pct=100 * ((d_device / m["Da_mm"]) ** 2 - 1))
    r1 = np.hypot(R1[:, 0], R1[:, 1])
    return dict(pre=pre, post=post, perimeter_stretch_pct=100 * (post["perimeter_mm"] / pre["perimeter_mm"] - 1),
                area_increase_pct=100 * (post["area_mm2"] / pre["area_mm2"] - 1), oversizing_vs_pre=over(pre), oversizing_vs_post=over(post),
                d_device_mm=float(d_device), sectors=len(used), theta=np.array([np.median(ang[sec == s]) for s in used]),
                r0=np.hypot(R0[:, 0], R0[:, 1]), r1=r1, sd_r_after=float(r1.std()), mean_r_after=float(r1.mean()))

def solver_log(dep,name):
    """Fixed mass scaling as Abaqus applied it (.sta table): added mass (% of the model mass; scaled once at the step start, constant), stable
    increment at the start and its minimum (it falls below the target when elements shrink: smaller increments, NO extra mass), increments."""
    rows=[l.split() for l in (dep / f"{name}.sta").read_text(errors="ignore").splitlines() if re.match(r"\s+\d+\s+\d\.\d+E[+-]\d+\s+\d\.\d+E[+-]\d+\s+\d\d:\d\d:\d\d", l)]
    return dict(added_mass_pct=float(rows[-1][-1]), dt_start_s=float(rows[0][4]),dt_min_s=min(float(r[4]) for r in rows), increments=int(rows[-1][0]))

def mesh_U(s10,f,N):
    """U(L1) in QoI units from the L1-L2-L3 triplet script 10's gci3: GCI if converging, spread band otherwise)."""
    spread=1.25*max(abs(f[1]-f[0]),abs(f[2]-f[0]))
    if abs(f[0])<1e-9:
        return spread, "spread band (finest value 0)"
    g=s10.gci3(f,N)
    if g["type"]=="converged" and spread>0:
        return spread, "spread band (one step zero)"
    return g["uncertainty_pct"]/100*abs(f[0]), f"{g['method']}, {g['type']} R {g['R']:.2f}"
