"""
Load a PA reconstruction (from TRHexArray3D_gpu.py) into 3D Slicer.

Launched automatically by TRHexArray3D_gpu.py --slicer, i.e.:
    Slicer.exe --python-script load_recon_into_slicer.py <path to img_slicer.npz>

Or by hand, in Slicer's Python Console:
    p = r"D:\kennedy\PMUT\TR\TR_Recon_Package_V1\load_recon_into_slicer.py"
    exec(open(p).read(), {"__file__": p})

Units are mm (Slicer's default), so no unit change is needed.
"""
import os
import sys
import numpy as np
import slicer

# ---------------------------------------------------------------------------
# Which reconstruction to load: path given on the command line, else
# img_slicer.npz next to this script
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(globals().get("__file__", os.getcwd())))
args = [a for a in sys.argv[1:] if a.lower().endswith(".npz")]
filePath = args[0] if args else os.path.join(script_dir, "img_slicer.npz")

CLIP_NEGATIVE = True     # hide negative (artifactual) values, like clim([0 ...]) in MATLAB
SHOW_FROM     = 0.5      # show only voxels above this fraction of peak (0.5 = -6 dB)
USE_VOLUME_RENDERING = True   # the fuzzy cloud rendering; the surface is usually clearer

d       = np.load(filePath)
Img     = d["volume"].astype(np.float32)   # (z, x, y), as produced by the reconstruction
origin  = d["origin"]                      # (x0, y0, z0) mm
spacing = d["spacing"]                     # (dx, dy, dz) mm
name = d["name"].item() if "name" in d.files else os.path.splitext(os.path.basename(filePath))[0]
d.close()

if CLIP_NEGATIVE:
    Img = np.clip(Img, 0, None)

# Slicer's updateVolumeFromArray wants (k, j, i) = (z, y, x)
vol = np.ascontiguousarray(np.transpose(Img, (0, 2, 1)))
peak = float(vol.max())

volumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name)
slicer.util.updateVolumeFromArray(volumeNode, vol)
volumeNode.SetSpacing(*map(float, spacing))
volumeNode.SetOrigin(*map(float, origin))
volumeNode.CreateDefaultDisplayNodes()

# --- slice views: default grey colormap, -6 dB window, smooth interpolation ---
disp = volumeNode.GetDisplayNode()
disp.SetAndObserveColorNodeID("vtkMRMLColorTableNodeGrey")
disp.SetAutoWindowLevel(False)
disp.SetWindowLevelMinMax(SHOW_FROM * peak, peak)
disp.SetInterpolate(True)
slicer.util.setSliceViewerLayers(background=volumeNode, fit=True)

# --- 3D volume rendering: transparent below SHOW_FROM*peak, opaque at the peak ---
vrLogic = slicer.modules.volumerendering.logic()
vrDisp  = vrLogic.CreateDefaultVolumeRenderingNodes(volumeNode)
prop    = vrDisp.GetVolumePropertyNode().GetVolumeProperty()

opacity = prop.GetScalarOpacity()
opacity.RemoveAllPoints()
opacity.AddPoint(float(vol.min()), 0.0)
opacity.AddPoint(SHOW_FROM * peak, 0.0)
opacity.AddPoint(peak, 0.8)

color = prop.GetRGBTransferFunction()
color.RemoveAllPoints()
color.AddRGBPoint(SHOW_FROM * peak, 0.5, 0.0, 0.0)   # dark red
color.AddRGBPoint(0.75 * peak,      1.0, 0.4, 0.0)   # orange
color.AddRGBPoint(peak,             1.0, 1.0, 0.6)   # pale yellow

prop.SetInterpolationTypeToLinear()
vrDisp.SetVisibility(USE_VOLUME_RENDERING)

slicer.app.layoutManager().threeDWidget(0).threeDView().resetFocalPoint()
print(f"Loaded {filePath}: shape (z,y,x) {vol.shape}, peak {peak:.3g}")