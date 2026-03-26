import slicer
import numpy as np

filePath = "/Users/jillian/Desktop/PMUT/volume.npy" # Adjust as needed
volume = np.load(filePath)

volume_slicer = np.transpose(volume, (2, 1, 0)) # Slicer expects (x,y,z)

# vtkMRMLScalarVolumeNode type = standard 3D image with each voxel having a scalar value (intensity)
volumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode") 

# Take NumPy array and copy into slicer volume node
slicer.util.updateVolumeFromArray(volumeNode, volume_slicer)

dx = 1 / 1000
dy = 1 / 1000
dz = 37*2/1000 # Take into account depth sampling

volumeNode.SetSpacing(dx, dy, dz)
slicer.util.setSliceViewerLayers(background=volumeNode)

# Get the Volume Rendering module logic
vrLogic = slicer.modules.volumerendering.logic()
# Get or create a display node for your volume
vrDisplayNode = vrLogic.CreateDefaultVolumeRenderingNodes(volumeNode)
# Turn volume rendering on
vrDisplayNode.SetVisibility(True)



""" In slcer, open Python Console and run: exec(open("/FILEPATH/load_volume_into_slicer.py").read())

Make sure unit µm is selected unless you divide dx, dy, dz by 1000
Turn on nearest neighbor in Volume Properties -> Advanced -> Interpolation
"""