"""
3D Photoacoustic Image Reconstruction - Time Reversal Algorithm
Translated from MATLAB to Python
"""

import numpy as np
import scipy.io as sio
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import time
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

from tr_cpp import time_reversal_reconstruction
#from visualize_volume import show_volume
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))



# =============================================================================
# INPUTS
# =============================================================================

# Load PA signal dataset
data = sio.loadmat('Processed/d3_PMUT_61_channels_400.mat')
data_all = data['data_all']  # adjust key name if needed

# Use rows 491–900 (MATLAB 1-indexed → Python 0-indexed: 490:900)
PASignal = data_all[490:900, :].astype(float)

# Zero out noisy channels (MATLAB col 52,34 → Python col 51,33)
PASignal[:, 51] = 0
PASignal[:, 33] = 0

# Remove pMUT reverb from laser pulse (first 79 samples)
PASignal[:79, :] = 0

# =============================================================================
# RECONSTRUCTION PARAMETERS
# =============================================================================

# Load PMUT element center coordinates (X, Y) in mm
coord = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'coordsHexagonal0p5mm.txt'))

co  = 1.5    # speed of sound, mm/us (water at room temp)
dt  = 0.02   # time step, microseconds (50 MHz sampling rate)
dxo = 0.05   # voxel size x, mm
dyo = 0.05   # voxel size y, mm
dzo = 0.05   # voxel size z, mm

zmin = 0.0   # axial bounds, mm
zmax = 13.0
xmax = 7.0   # lateral bounds, mm
ymax = 7.0

xo = np.arange(-xmax, xmax + dxo, dxo)
yo = np.arange(-ymax, ymax + dyo, dyo)
zo = np.arange(zmin,  zmax + dzo, dzo)

#Img = np.zeros((len(zo), len(xo), len(yo)), dtype=np.float64)

ThetaMax = np.pi / 6  # maximum half-angle

# =============================================================================
# TIME REVERSAL RECONSTRUCTION — parallelized over sensors
# =============================================================================


print("Starting parallel reconstruction...")
t_start = time.time()

Img = time_reversal_reconstruction(
    PASignal, coord, zo, xo, yo,
    co, dt, dxo, dyo, dzo,
    zmin, xmax, ymax, ThetaMax
)

print(f"Reconstruction complete in {time.time() - t_start:.1f} seconds.")


# =============================================================================
# IMAGE DISPLAY
# =============================================================================

# --- 2D slice at X = 0 ---
x0_idx = 131

ImgSlice = np.squeeze(Img[:, x0_idx, :])

X0Max  = np.max(np.abs(Img[:, x0_idx, :]))
ImgMax = np.max(np.abs(Img))

fig, ax = plt.subplots(figsize=(6, 8))
im = ax.imshow(
    ImgSlice,
    extent=[yo[0], yo[-1], zo[0], zo[-1]],
    origin='lower',
    aspect='equal',
    cmap='hot',
    vmin=0.5 * X0Max,
    vmax=X0Max
)
ax.set_xlabel('y [mm]')
ax.set_ylabel('z [mm]')
ax.set_title('PA Image — X=0 slice')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('pa_slice_x0.png', dpi=150)
plt.show()

# =============================================================================
# 3D VOLUME RENDERING
# =============================================================================

# Using existing imaging in PyVista
"""# Transpose from (zo, xo, yo) → (xo, yo, zo) to match show_volume's axis order
volume = np.transpose(Img, (1, 2, 0)).astype(np.float32)
# Clip to top signal percentiles to handle backprojection dynamic range
# Mirror MATLAB's clim([0.5*X0Max X0Max]) — only show top half of brightness
volume = np.clip(volume, ImgMax / 2, ImgMax)
volume = (volume - ImgMax / 2) / (ImgMax / 2)
sensor_points = np.column_stack([
    coord[:, 0] * 0.001,
    coord[:, 1] * 0.001,
    np.zeros(len(coord))
])

show_volume(
    volume,
    dx=dxo,
    dy=dyo,
    dz=dzo,
    x_min=xo[0],
    y_min=yo[0],
    scale=1.0,
    extra_actors=[
        {
            'points':  sensor_points,
            'color':   'cyan',
            'label':   'Sensors',
            'size':    6,
            'spheres': True
        }
    ]
)
"""
# Isosurface image - looks similar to current MATLAB version
Img1 = np.transpose(Img, (1, 2, 0))
ImgMax = np.max(np.abs(Img))

fig3d = plt.figure(figsize=(8, 8))
ax3d  = fig3d.add_subplot(111, projection='3d')

try:
    verts, faces, _, _ = measure.marching_cubes(Img1, level=ImgMax / 2, spacing=(dxo, dyo, dzo))
    verts[:, 0] += xo[0]
    verts[:, 1] += yo[0]
    verts[:, 2] += zo[0]

    mesh = Poly3DCollection(verts[faces], alpha=0.4, edgecolor='none')
    mesh.set_facecolor('steelblue')
    ax3d.add_collection3d(mesh)
    ax3d.set_xlim(xo[0], xo[-1])
    ax3d.set_ylim(yo[0], yo[-1])
    ax3d.set_zlim(zo[0], zo[-1])
except Exception as e:
    print(f"Isosurface could not be generated: {e}")

ax3d.scatter(coord[:, 0] * 0.001, coord[:, 1] * 0.001,
             zs=0, zdir='z', c='red', s=10, label='Sensors')
ax3d.set_xlabel('x [mm]')
ax3d.set_ylabel('y [mm]')
ax3d.set_zlabel('z [mm]')
ax3d.set_title('3D PA Isosurface')
ax3d.legend()
plt.tight_layout()
plt.savefig('pa_isosurface_3d.png', dpi=150)
plt.show()