""" Written with help of Claude """

"""
3D Photoacoustic Image Reconstruction - Time Reversal
Backends:  gpu (CUDA via CuPy, default if available)  |  cpu (C++/OpenMP via tr_cpp)

Usage:
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
import warnings
warnings.filterwarnings('ignore', message='CUDA path could not be detected')

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
parser.add_argument('--data', default='../Processed/d3_PMUT_61_channels_400.mat')
parser.add_argument('--no-slicer', action='store_true', help="don't open the result in 3D Slicer")
parser.add_argument('--t0-row', type=int, default=None,
                    help='set time zero by hand (0-based row); default = detect the laser-artifact onset')
args = parser.parse_args()

# =============================================================================
# PARAMETERS
# =============================================================================
co  = 1.5    # speed of sound, mm/us (water at room temp)
dt  = 0.016  # time step, microseconds (62.5 MHz sampling rate)
dxo = 0.05   # voxel size x, mm
dyo = 0.05   # voxel size y, mm
dzo = 0.025   # voxel size z, mm

zmin = 0.0   # axial bounds, mm
zmax = 13.0
xmax = 7.0   # lateral bounds, mm
ymax = 7.0

xo = np.arange(-xmax, xmax + dxo, dxo)
yo = np.arange(-ymax, ymax + dyo, dyo)
zo = np.arange(zmin,  zmax + dzo, dzo)
ThetaMax = np.pi / 6

# =============================================================================
# INPUTS
# =============================================================================
data = sio.loadmat(args.data)
data_all = data['data_filt']

# ---- time zero: first sample where the laser artifact rises above the background noise ----
raw_t0 = data['data_raw'].astype(float)
base   = np.median(raw_t0[:50], axis=0)
level  = np.abs(raw_t0 - base).mean(axis=1)
quiet  = level[:50]
thr    = quiet.mean() + 5 * quiet.std()
above  = level > thr                     # must exist before the next line
RUN    = 3                               # must stay above for 3 samples in a row
T0_ROW = next(i for i in range(len(level) - RUN) if above[i:i + RUN].all())

if args.t0_row is not None:
    T0_ROW = args.t0_row                 # manual override
elif T0_ROW < 50:
    print("WARNING: time zero detected inside the quiet window, check the data")
print(f"time zero (laser artifact onset): row {T0_ROW}")

BLANK_MM = 0.8                                               # blank the artifact: ~first 0.8 mm
n_blank  = int(np.ceil(BLANK_MM / (co * dt)))
n_keep   = int(np.ceil(zmax / (co * dt))) + 2                # enough samples to cover the whole grid depth

PASignal = data_all[T0_ROW:T0_ROW + n_keep, :].astype(float)
PASignal[:n_blank, :] = 0

coord = np.loadtxt('coordsHex61_2mm.txt')   # PMUT centers (file units; x0.001 -> mm)

# =============================================================================
# BACKENDS
# =============================================================================
def gpu_available():
    try:
        import cupy as cp # CuPy must be installed
        return cp.cuda.runtime.getDeviceCount() > 0 # asks CUDA driver how many GPUs it sees
    except Exception:
        return False

# Run time reversal reconstruction on GPU
def run_gpu():
    import cupy as cp
    from tr_gpu import tr_gpu
    name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    print(f"[GPU] {name}")
    tr_gpu(PASignal, coord, zo, xo, yo, co, dt, ThetaMax)       # warm-up (kernel compile)
    cp.cuda.Device().synchronize() # Synchronize so time is accurate
    t0 = time.time()
    img = tr_gpu(PASignal, coord, zo, xo, yo, co, dt, ThetaMax)
    cp.cuda.Device().synchronize()
    print(f"[GPU] reconstruction: {(time.time() - t0) * 1000:.1f} ms (incl. host<->device copies)")
    return img

# Run time reversal reconstruction on CPU
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

# =============================================================================
# COMPARE / BENCHMARK  (used when --compare is given)
# =============================================================================
GPU_RUNS = 20             # timed GPU runs (after one untimed warm-up)
CPU_RUNS = 20             # timed CPU runs (after one untimed warm-up)



def time_runs(fn, n, sync=None):
    """One untimed warm-up, then n timed runs. Returns (last result, array of seconds)."""
    fn()
    if sync:
        sync()
    times, out = [], None
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn()
        if sync:
            sync()
        times.append(time.perf_counter() - t0)
    return out, np.array(times)


def compare_volumes(cpu, gpu):
    """Agreement metrics. Each volume is divided by its own peak first, so overall scale is ignored."""
    a = cpu.astype(np.float64)
    b = gpu.astype(np.float64)
    a /= np.abs(a).max()
    b /= np.abs(b).max()
    mask = (np.abs(a) > 0.1) | (np.abs(b) > 0.1)    # voxels where either image has real signal
    d = a - b
    pos = lambda k: np.array([zo[k[0]], xo[k[1]], yo[k[2]]])
    ka = np.unravel_index(np.argmax(np.abs(a)), a.shape)
    kb = np.unravel_index(np.argmax(np.abs(b)), b.shape)
    res = {
        'corr_all':        np.corrcoef(a.ravel(), b.ravel())[0, 1],
        'corr_signal':     np.corrcoef(a[mask], b[mask])[0, 1],
        'nrms_all':        np.linalg.norm(d) / np.linalg.norm(a),
        'nrms_signal':     np.linalg.norm(d[mask]) / np.linalg.norm(a[mask]),
        'max_diff':        np.abs(d).max(),
        'frac_within_1pct': np.mean(np.abs(d) < 0.01),
        'n_signal_voxels': int(mask.sum()),
        'peak_cpu_mm':     pos(ka),
        'peak_gpu_mm':     pos(kb),
        'peak_offset_mm':  np.linalg.norm(pos(ka) - pos(kb)),
    }
    return res, a, b, d


if args.compare:
    import cupy as cp
    import tr_cpp
    from tr_gpu import tr_gpu
    from tr_cpp import time_reversal_reconstruction

    gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    cpu_threads = tr_cpp.num_threads if tr_cpp.openmp_enabled else 1
    print(f"GPU: {gpu_name}   |   CPU OpenMP threads: {cpu_threads}")
    print(f"Grid: {len(zo)} x {len(xo)} x {len(yo)} voxels (dx={dxo}, dy={dyo}, dz={dzo} mm)")

    print(f"Running GPU ({GPU_RUNS} timed runs + warm-up)...")
    Img_gpu, t_gpu = time_runs(
        lambda: tr_gpu(PASignal, coord, zo, xo, yo, co, dt, ThetaMax),
        GPU_RUNS, cp.cuda.Device().synchronize)

    print(f"Running CPU ({CPU_RUNS} timed runs + warm-up)...")
    Img_cpu, t_cpu = time_runs(
        lambda: time_reversal_reconstruction(PASignal, coord, zo, xo, yo,
                                             co, dt, dxo, dyo, dzo,
                                             zmin, xmax, ymax, ThetaMax),
        CPU_RUNS)

    # ---- speed ----
    g_med, c_med = np.median(t_gpu), np.median(t_cpu)
    print("\n================ SPEED ================")
    print(f"GPU  median {g_med*1000:9.1f} ms   (min {t_gpu.min()*1000:.1f}, max {t_gpu.max()*1000:.1f})   {1/g_med:7.1f} frames/s   [incl. host<->device copies]")
    print(f"CPU  median {c_med*1000:9.1f} ms   (min {t_cpu.min()*1000:.1f}, max {t_cpu.max()*1000:.1f})   {1/c_med:7.1f} frames/s")
    print(f"Speedup (CPU median / GPU median): {c_med / g_med:.1f}x")

    # ---- agreement ----
    res, a_n, b_n, d_n = compare_volumes(Img_cpu, Img_gpu)
    print("\n============== AGREEMENT ==============")
    print(f"Correlation, whole volume:           {res['corr_all']:.4f}")
    print(f"Correlation, signal voxels only:     {res['corr_signal']:.4f}   ({res['n_signal_voxels']} voxels above 10% of peak)")
    print(f"Normalized RMS diff, whole volume:   {res['nrms_all']:.4f}")
    print(f"Normalized RMS diff, signal voxels:  {res['nrms_signal']:.4f}")
    print(f"Max |difference| (fraction of peak): {res['max_diff']:.4f}")
    print(f"Voxels within 1% of peak:            {100*res['frac_within_1pct']:.2f}%")
    print(f"Peak CPU (z,x,y) mm: {np.round(res['peak_cpu_mm'], 3)}   Peak GPU: {np.round(res['peak_gpu_mm'], 3)}")
    print(f"Peak position offset:                {res['peak_offset_mm']:.3f} mm")

    # ---- difference picture (X = 0 slice, shared color scale) ----
    ix = np.argmin(np.abs(xo))
    fig, axs = plt.subplots(1, 3, figsize=(14, 7))
    ext = [yo[0], yo[-1], zo[0], zo[-1]]
    for ax, im_, title, cm, lim in ((axs[0], a_n[:, ix, :], 'CPU (normalized)', 'hot', (0, 1)),
                                    (axs[1], b_n[:, ix, :], 'GPU (normalized)', 'hot', (0, 1)),
                                    (axs[2], d_n[:, ix, :], 'CPU - GPU', 'coolwarm', (-0.2, 0.2))):
        h = ax.imshow(im_, extent=ext, origin='lower', aspect='equal', cmap=cm, vmin=lim[0], vmax=lim[1])
        ax.set_title(title); ax.set_xlabel('y [mm]'); ax.set_ylabel('z [mm]')
        plt.colorbar(h, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig('compare_x0.png', dpi=150)

    # ---- save volumes and append a row to a results table ----
    np.save('img_cpu.npy', Img_cpu)
    np.save('img_gpu.npy', Img_gpu)
    import csv, datetime
    csv_path = 'benchmark_results.csv'
    new_file = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(['date', 'dataset', 'voxel_mm', 'gpu', 'cpu_threads', 'gpu_runs', 'cpu_runs',
                        'gpu_median_ms', 'cpu_median_ms', 'speedup', 'corr_signal', 'nrms_signal',
                        'peak_offset_mm'])
        w.writerow([datetime.datetime.now().isoformat(timespec='seconds'), os.path.basename(args.data), f"{dxo}x{dyo}x{dzo}",
                    gpu_name, cpu_threads, GPU_RUNS, CPU_RUNS, round(g_med*1000, 2), round(c_med*1000, 2),
                    round(c_med / g_med, 2), round(res['corr_signal'], 5), round(res['nrms_signal'], 5),
                    round(res['peak_offset_mm'], 4)])
    print(f"\nSaved: compare_x0.png, img_cpu.npy, img_gpu.npy, and a row in {csv_path}\n")
    Img = Img_gpu
else:
    backend = args.backend
    if backend == 'auto':
        backend = 'gpu' if gpu_available() else 'cpu'
        print(f"Backend: {backend}")
    Img = run_gpu() if backend == 'gpu' else run_cpu()

np.save('img_reconstruction.npy', Img)


"""# =============================================================================
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

if matplotlib.get_backend().lower() != 'agg': # for headless space setup
    plt.show()
else:
    print("Headless session: saved pa_slice_x0.png, pa_isosurface_3d.png, img_reconstruction.npy")
"""
# =============================================================================
# 3D Slicer
# =============================================================================
# Format for 3D slicer
from scipy.signal import hilbert
Env = np.abs(hilbert(Img, axis=0)).astype(np.float32)        # envelope along depth (z)
np.savez('img_slicer_tmp.npz', volume=Env,
         origin=[xo[0], yo[0], zo[0]], spacing=[dxo, dyo, dzo],
         name=os.path.splitext(os.path.basename(args.data))[0])
os.replace('img_slicer_tmp.npz', 'img_slicer.npz') # Stop slicer from reading file before written

def find_slicer():
    """Find 3D Slicer on Mac or Windows. Set a SLICER_EXE environment variable to override."""
    import glob, shutil
    env = os.environ.get('SLICER_EXE')
    if env and os.path.exists(env):
        return env
    cands = []
    if sys.platform == 'darwin':
        cands += ['/Applications/Slicer.app/Contents/MacOS/Slicer',
                  os.path.expanduser('~/Applications/Slicer.app/Contents/MacOS/Slicer')]
    elif os.name == 'nt':
        for base in (os.environ.get('LOCALAPPDATA', ''), os.environ.get('ProgramFiles', '')):
            cands += glob.glob(os.path.join(base, 'slicer.org', '*', 'Slicer.exe'))
            cands += glob.glob(os.path.join(base, '*Slicer*', 'Slicer.exe'))
    found = shutil.which('Slicer')
    if found:
        cands.append(found)
    cands = [c for c in cands if c and os.path.exists(c)]
    return max(cands, key=os.path.getmtime) if cands else None   # newest install if several


if not args.no_slicer:
    beat = os.path.join(HERE, '.slicer_watching')
    if os.path.exists(beat) and time.time() - os.path.getmtime(beat) < 5:
        print("Slicer is already open - it will reload the new result automatically.")
    else:
        SLICER_EXE = find_slicer()
        if SLICER_EXE is None:
            print("Couldn't find 3D Slicer - install it, or set the SLICER_EXE environment variable.")
        else:
            import subprocess
            extra = (dict(creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
                     if os.name == 'nt' else dict(start_new_session=True))
            subprocess.Popen([SLICER_EXE, '--python-script', os.path.join(HERE, 'slicer_watch.py')],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **extra)
            print(f"Opening 3D Slicer ({SLICER_EXE})...")