""" Written with Claude - needs to be heavily edited / checked"""

"""
3D Photoacoustic Image Reconstruction - Time Reversal
Backends:  gpu (CUDA via CuPy, default if available)  |  cpu (C++/OpenMP via tr_cpp)

Usage (run from this folder):
    python3 TRHexArray3D_gpu.py              # auto: GPU if CuPy + NVIDIA GPU found, else CPU
    python3 TRHexArray3D_gpu.py --backend gpu
    python3 TRHexArray3D_gpu.py --backend cpu
    python3 TRHexArray3D_gpu.py --compare    # run both, print agreement metrics

Over SSH there is no display, so figures are saved as PNGs instead of shown.
"""

import os
import sys
import time
import argparse
import numpy as np
import scipy.io as sio

# --- headless-safe plotting (SSH sessions have no display) ---
import matplotlib
if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--backend', choices=['auto', 'gpu', 'cpu'], default='auto')
parser.add_argument('--compare', action='store_true', help='run GPU and CPU, report agreement')
parser.add_argument('--data', default='Processed/d3_PMUT_61_channels_400.mat')
args = parser.parse_args()

# =============================================================================
# INPUTS
# =============================================================================
data = sio.loadmat(args.data)
data_all = data['data_all']

PASignal = data_all[490:900, :].astype(float)   # MATLAB rows 491-900
PASignal[:, 51] = 0                              # noisy channels (MATLAB 52, 34)
PASignal[:, 33] = 0
PASignal[:79, :] = 0                             # laser-pulse reverb

coord = np.loadtxt('coordsHexagonal0p5mm.txt')   # PMUT centers (file units; x0.001 -> mm)

# =============================================================================
# PARAMETERS
# =============================================================================
co  = 1.5     # speed of sound, mm/us
dt  = 0.02    # time step, us (50 MHz)
dxo = dyo = dzo = 0.05   # voxel size, mm
zmin, zmax = 0.0, 13.0
xmax, ymax = 7.0, 7.0

xo = np.arange(-xmax, xmax + dxo, dxo)
yo = np.arange(-ymax, ymax + dyo, dyo)
zo = np.arange(zmin,  zmax + dzo, dzo)
ThetaMax = np.pi / 6

# =============================================================================
# BACKENDS
# =============================================================================
def gpu_available():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False

def run_gpu():
    import cupy as cp
    from tr_gpu import tr_gpu
    name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    print(f"[GPU] {name}")
    tr_gpu(PASignal, coord, zo, xo, yo, co, dt, ThetaMax)       # warm-up (kernel compile)
    cp.cuda.Device().synchronize()
    t0 = time.time()
    img = tr_gpu(PASignal, coord, zo, xo, yo, co, dt, ThetaMax)
    cp.cuda.Device().synchronize()
    print(f"[GPU] reconstruction: {(time.time() - t0) * 1000:.1f} ms (incl. host<->device copies)")
    return img

def run_cpu():
    from tr_cpp import time_reversal_reconstruction
    print(f"[CPU] OpenMP threads: "
          f"{__import__('tr_cpp').num_threads if __import__('tr_cpp').openmp_enabled else 1}")
    t0 = time.time()
    img = time_reversal_reconstruction(PASignal, coord, zo, xo, yo,
                                       co, dt, dxo, dyo, dzo,
                                       zmin, xmax, ymax, ThetaMax)
    print(f"[CPU] reconstruction: {time.time() - t0:.2f} s")
    return img

if args.compare:
    Img_gpu = run_gpu()
    Img_cpu = run_cpu()
    a, b = np.abs(Img_cpu).ravel(), np.abs(Img_gpu).ravel()
    print(f"correlation of |image| (CPU vs GPU): {np.corrcoef(a, b)[0, 1]:.4f}")
    for name, im in (('CPU', Img_cpu), ('GPU', Img_gpu)):
        k = np.unravel_index(np.argmax(np.abs(im)), im.shape)
        print(f"peak {name}: z={zo[k[0]]:.2f} x={xo[k[1]]:.2f} y={yo[k[2]]:.2f} mm")
    Img = Img_gpu
else:
    backend = args.backend
    if backend == 'auto':
        backend = 'gpu' if gpu_available() else 'cpu'
        print(f"Backend: {backend}")
    Img = run_gpu() if backend == 'gpu' else run_cpu()

np.save('img_reconstruction.npy', Img)

# =============================================================================
# 2D SLICE (X = 0)
# =============================================================================
x0_idx = np.argmin(np.abs(xo))
ImgSlice = Img[:, x0_idx, :]
X0Max = np.max(np.abs(ImgSlice))
ImgMax = np.max(np.abs(Img))

fig, ax = plt.subplots(figsize=(6, 8))
im = ax.imshow(ImgSlice, extent=[yo[0], yo[-1], zo[0], zo[-1]],
               origin='lower', aspect='equal', cmap='hot',
               vmin=0.5 * X0Max, vmax=X0Max)
ax.set_xlabel('y [mm]'); ax.set_ylabel('z [mm]'); ax.set_title('PA Image — X=0 slice')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('pa_slice_x0.png', dpi=150)

# =============================================================================
# 3D ISOSURFACE
# =============================================================================
Img1 = np.transpose(Img, (1, 2, 0))
fig3d = plt.figure(figsize=(8, 8))
ax3d = fig3d.add_subplot(111, projection='3d')
try:
    verts, faces, _, _ = measure.marching_cubes(Img1, level=ImgMax / 2, spacing=(dxo, dyo, dzo))
    verts[:, 0] += xo[0]; verts[:, 1] += yo[0]; verts[:, 2] += zo[0]
    mesh = Poly3DCollection(verts[faces], alpha=0.4, edgecolor='none')
    mesh.set_facecolor('steelblue')
    ax3d.add_collection3d(mesh)
    ax3d.set_xlim(xo[0], xo[-1]); ax3d.set_ylim(yo[0], yo[-1]); ax3d.set_zlim(zo[0], zo[-1])
except Exception as e:
    print(f"Isosurface could not be generated: {e}")

ax3d.scatter(coord[:, 0] * 0.001, coord[:, 1] * 0.001, zs=0, zdir='z',
             c='red', s=10, label='Sensors')
ax3d.set_xlabel('x [mm]'); ax3d.set_ylabel('y [mm]'); ax3d.set_zlabel('z [mm]')
ax3d.set_title('3D PA Isosurface'); ax3d.legend()
plt.tight_layout()
plt.savefig('pa_isosurface_3d.png', dpi=150)

if matplotlib.get_backend().lower() != 'agg':
    plt.show()
else:
    print("Headless session: saved pa_slice_x0.png, pa_isosurface_3d.png, img_reconstruction.npy")
