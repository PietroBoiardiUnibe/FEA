"""
Pulls the results out of an Abaqus result file (.odb) into a plain .npz that numpy can handle cleanly.

Run with Abaqus' OWN Python (the .odb can only be opened by the Abaqus libraries):
    abaqus python 05_extract_odb.py <job>.odb <out>.npz
Saves: times (frames), labels (node ids, sorted), U (frames x nodes x 3 displacement), RF (reaction force, if written),
       hist_<name>: energy histories (model level) and hist_wall_<name> (the WALL element set alone). 
"""

import sys
import numpy as np
from odbAccess import openOdb

odb_path,out_path=sys.argv[1],sys.argv[2]
odb=openOdb(odb_path, readOnly=True)
step=odb.steps[list(odb.steps.keys())[-1]]
times, labels,U,RF=[],None,[],[]

for frame in step.frames:
    u=frame.fieldOutputs["U"]
    blocks=u.bulkDataBlocks # bulk access: the whole field as arrays, much faster than looping over values
    lab=np.concatenate([b.nodeLabels for b in blocks])
    dat=np.concatenate([b.data for b in blocks])
    order=np.argsort(lab)  # same node order in every frame
    if labels is None:
        labels=lab[order]
    U.append(dat[order].astype(np.float32)); times.append(frame.frameValue)
    if "RF" in frame.fieldOutputs:
        r=frame.fieldOutputs["RF"]
        rl=np.concatenate([b.nodeLabels for b in r.bulkDataBlocks]); rd = np.concatenate([b.data for b in r.bulkDataBlocks])
        RF.append(rd[np.argsort(rl)].astype(np.float32))

hist={}

for region_name, region in step.historyRegions.items():
    tag="wall_" if "WALL" in region_name.upper() else ""   # the energies of the WALL element set get a prefix, model-level ones keep the plain name
    for key, out in region.historyOutputs.items():
        if key in ("ALLKE", "ALLIE", "ALLAE", "ALLVD", "ALLWK", "ETOTAL"):
            hist[tag+key]=np.array(out.data,dtype=float)

result={"times":np.array(times),"labels":labels,"U":np.stack(U)}
if RF:
    result["RF"] = np.stack(RF)
for k, v in hist.items():
    result["hist_" + k] = v
np.savez(out_path, **result)
print("frames %d, nodes %d, histories %s" % (len(times), len(labels), sorted(hist)))
odb.close()


