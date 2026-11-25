"""
ParaView (pvpython) scene of ONE deployment run: deformed FEA wall coloured by a field of 13 (device zone), translucent cuff, anchors (15),
fixed colour range = the run's p99 in the device zone (maxima are mesh-dominated: shown, not used for the range). Saves the last frame as PNG
and the whole deployment as a PNG sequence (make a video/GIF from it), then can be opened interactively.
Run with ParaView's own Python:
    pvpython 15b_paraview_view.py <deploy folder> [--field S_maxP_kPa | LE_maxP_pct | SED_kPa | VM_kPa | CPRESS_kPa] [--show]
Writes  <deploy folder>/post/paraview/views/<field>_final.png and <field>_####.png
"""
import json
import sys
from pathlib import Path
from paraview.simple import *                                             # noqa: F401,F403 (ParaView scripting API)

dep = Path(sys.argv[1]); args = sys.argv[2:]
field = args[args.index("--field") + 1] if "--field" in args else "S_maxP_kPa"
pv = dep / "post" / "paraview"; views = pv / "views"; views.mkdir(exist_ok=True)
fj = next((dep / "post").glob("*_fields.json"), None)
hi = None
if fj:                                                                    # colour range: device-zone p99 of the last frame (point field: contact p95)
    fin = json.loads(fj.read_text())["final"]
    hi = fin["CPRESS"]["p95_kPa"] if field == "CPRESS_kPa" else fin["zone"][field]["p99"]

axis = args[args.index("--normal") + 1] if "--normal" in args else "x"       # cut-away plane normal (global axis) through the device centre
nrm = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[axis]
view = GetActiveViewOrCreate("RenderView"); view.ViewSize = [1600, 1000]
try:
    view.UseColorPaletteForBackground = 0                                 # ParaView >= 5.10: otherwise the palette overrides Background
except AttributeError:
    pass
view.Background = [1, 1, 1]; view.OrientationAxesVisibility = 0
cuff = PVDReader(FileName=str(pv / "expander.pvd"))
tend = cuff.TimestepValues[-1] if cuff.TimestepValues else 0.0
cuff.UpdatePipeline(tend); b = cuff.GetDataInformation().GetBounds()
centre = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]
wall = PVDReader(FileName=str(pv / "fields.pvd"))
cut = Clip(Input=wall); cut.ClipType = "Plane"; cut.ClipType.Origin = centre; cut.ClipType.Normal = nrm   # half of the heart removed: device visible
d = Show(cut, view); d.SetRepresentationType("Surface")
assoc = "POINTS" if field == "CPRESS_kPa" else "CELLS"
ColorBy(d, (assoc, field)); lut = GetColorTransferFunction(field)
for preset in ("Viridis (matplotlib)", "Viridis", "Cool to Warm"):        # preset names differ between ParaView versions
    try:
        lut.ApplyPreset(preset, True); break
    except RuntimeError:
        pass
if hi:
    lut.RescaleTransferFunction(0.0 if field != "S_minP_kPa" else -hi, hi)
lut.AutomaticRescaleRangeMode = "Never"
bar = GetScalarBar(lut, view); bar.Title = f"{field} (range = device-zone p99)"; bar.TitleColor = [0, 0, 0]; bar.LabelColor = [0, 0, 0]
d.SetScalarBarVisibility(view, True)
c = Show(cuff, view); c.DiffuseColor = [0.16, 0.47, 0.84]; c.Opacity = 0.35; c.ColorArrayName = ["POINTS", ""]
if (pv / "anchors.pvd").exists():
    anc = PVDReader(FileName=str(pv / "anchors.pvd"))
    an = Show(anc, view); an.ColorArrayName = ["POINTS", ""]; an.DiffuseColor = [0.82, 0.23, 0.23]
scene = GetAnimationScene(); scene.UpdateAnimationUsingDataTimeSteps()
view.CameraFocalPoint = centre                                            # look at the cut face from the removed side
view.CameraPosition = [centre[i] + 250 * nrm[i] for i in range(3)]       # Clip keeps the negative side: look from the positive one
view.CameraViewUp = [0, 0, 1] if axis != "z" else [0, 1, 0]
view.ResetCamera(); view.CameraParallelScale = 45
scene.AnimationTime = scene.EndTime
SaveScreenshot(str(views / f"{field}_final.png"), view, ImageResolution=[1600, 1000])
SaveAnimation(str(views / f"{field}.png"), view, ImageResolution=[1600, 1000], FrameWindow=[0, len(scene.TimeKeeper.TimestepValues) - 1])
print(f"wrote {views / (field + '_final.png')} and the frame sequence {field}_####.png")
if "--show" in args:
    Interact(view)