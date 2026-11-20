"""
Look inside a .mat dataset before reconstructing it.

    python inspect_data.py path\\to\\file.mat
    python inspect_data.py path\\to\\file.mat --var data_raw --dt 0.02 --co 1.5

Prints every numeric variable, then for the chosen one (default: data_all if present, otherwise
the first 2D array with 61 columns) it reports per-channel statistics and saves inspect_data.png:
  top    - all channels as an image (rows = time sample index, columns = channel)
  bottom - mean |signal| across channels vs row index, to show where the laser artifact and echoes are.
"""
import sys
import argparse
import numpy as np
import scipy.io as sio
import matplotlib
if sys.platform.startswith('linux'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('file')
ap.add_argument('--var', default=None, help='variable name (auto-detected if omitted)')
ap.add_argument('--dt', type=float, default=0.02, help='sample period in microseconds (default 0.02 = 50 MHz)')
ap.add_argument('--co', type=float, default=1.5, help='speed of sound, mm/us (default 1.5)')
args = ap.parse_args()

try:
    mat = sio.loadmat(args.file)
except NotImplementedError:
    sys.exit("This looks like a MATLAB -v7.3 file, which scipy can't read. "
             "Ask for it to be saved in the default format, or use h5py to open it.")

print(f"File: {args.file}\n\nVariables:")
numeric = {}
for k, v in mat.items():
    if k.startswith('__'):
        continue
    if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
        numeric[k] = v
        print(f"  {k:12s} shape={v.shape}  dtype={v.dtype}  min={v.min():.4g}  max={v.max():.4g}")
    else:
        print(f"  {k:12s} (non-numeric: {type(v).__name__})")

name = args.var
if name is None:
    name = 'data_all' if 'data_all' in numeric else next(
        (k for k, v in numeric.items() if v.ndim == 2 and 61 in v.shape), None)
if name is None or name not in numeric:
    sys.exit("\nCould not pick a variable automatically. Re-run with --var NAME.")
X = numeric[name].astype(float)
print(f"\nUsing '{name}'  shape={X.shape}")
if X.ndim != 2:
    sys.exit("Expected a 2D array (time samples x channels).")
if X.shape[1] != 61:
    if X.shape[0] == 61:
        print("NOTE: it has 61 ROWS - the script expects time down the rows and 61 channels across the columns, so transpose it (X.T).")
    else:
        print(f"WARNING: {X.shape[1]} columns, but the script and coordinates file expect 61 channels.")
    sys.exit(0)

nT, nC = X.shape
print(f"{nT} samples per channel -> {nT * args.dt:.2f} us -> {nT * args.dt * args.co:.1f} mm of range "
      f"(assuming dt={args.dt} us, c={args.co} mm/us)")

mean = X.mean(axis=0); std = X.std(axis=0)
print(f"\nPer-channel mean: median {np.median(mean):.4g}, range [{mean.min():.4g}, {mean.max():.4g}]")
if np.abs(np.median(mean)) > 0.05 * np.median(std[std > 0] if (std > 0).any() else [1]):
    print("  -> The signal sits on a non-zero baseline (DC offset). Zeroing the first samples would create a step;"
          " subtract each channel's mean before reconstructing.")
print(f"Per-channel std:  median {np.median(std):.4g}")
dead = np.where(std < 1e-6 * max(std.max(), 1e-12))[0]
loud = np.where(std > 5 * np.median(std))[0]
print(f"Flat/dead channels (0-indexed): {dead.tolist() if len(dead) else 'none'}")
print(f"Unusually noisy channels (>5x median std, 0-indexed): {loud.tolist() if len(loud) else 'none'}")
print("  (the script currently zeroes 0-indexed channels 33 and 51)")

env = np.abs(X - mean).mean(axis=1)
peak_row = int(np.argmax(env))
print(f"\nStrongest mean |signal| is at row {peak_row} (0-indexed). Use the plot to find the row where the laser "
      f"artifact/ringing BEGINS - the reconstruction treats the first row of its slice as time zero.")

fig, axs = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={'height_ratios': [3, 1.5]}, sharex=False)
lim = np.percentile(np.abs(X - mean), 99)
axs[0].imshow((X - mean).T, aspect='auto', cmap='seismic', vmin=-lim, vmax=lim, origin='lower',
              extent=[0, nT, 0, nC])
axs[0].set_xlabel('row index (time sample, 0-indexed)'); axs[0].set_ylabel('channel')
axs[0].set_title(f"{name}: all channels (baseline-subtracted, clipped at 99th percentile)")
axs[1].plot(env); axs[1].set_xlim(0, nT)
axs[1].set_xlabel('row index (time sample, 0-indexed)'); axs[1].set_ylabel('mean |signal|')
axs[1].set_title('Look for: laser artifact / ringing near the start, then echoes. Note the row where the artifact begins.')
plt.tight_layout(); plt.savefig('inspect_data.png', dpi=130)
print("\nSaved inspect_data.png")
if matplotlib.get_backend().lower() != 'agg':
    plt.show()