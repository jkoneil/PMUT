"""
Measure resolution two ways on the last reconstruction, so results can be compared
with the previous team's numbers.

  1) MATLAB method: abs() of the raw reconstruction, -6 dB width through the brightest voxel
     (find first/last sample >= 0.5 of the profile max, exactly like the MATLAB script)
  2) Envelope method: same -6 dB width, but on the envelope (Hilbert along depth)

Run right after a reconstruction (reads img_reconstruction.npy and img_slicer.npz):
    python compare_resolution.py
"""
import numpy as np

d = np.load("img_slicer.npz")
env = d["volume"].astype(np.float64)                 # envelope, (z, x, y)
x0, y0, z0 = map(float, d["origin"])
dx, dy, dz = map(float, d["spacing"])
name = d["name"].item() if "name" in d.files else "(name not stored)"
d.close()
raw = np.load("img_reconstruction.npy").astype(np.float64)   # raw reconstruction, (z, x, y)

nz, nx, ny = raw.shape
zo = z0 + dz * np.arange(nz)
xo = x0 + dx * np.arange(nx)
yo = y0 + dy * np.arange(ny)


def width_other_way(profile, axis):
    """MATLAB: normalize, find all samples >= 0.5, width = last - first."""
    p = profile / profile.max()
    idx = np.where(p >= 0.5)[0]
    return axis[idx[-1]] - axis[idx[0]]


print(f"dataset: {name}\n")
print(f"{'method':32s} {'peak z (mm)':>11s} {'axial':>8s} {'lateral Y':>10s} {'lateral X':>10s}")
for label, vol in (("abs(raw)  [MATLAB method]", np.abs(raw)), ("envelope  [Hilbert along z]", env)):
    iz, ix, iy = np.unravel_index(np.argmax(vol), vol.shape)
    ax_w = width_other_way(vol[:, ix, iy], zo)
    ly_w = width_other_way(vol[iz, ix, :], yo)
    lx_w = width_other_way(vol[iz, :, iy], xo)
    print(f"{label:32s} {zo[iz]:11.3f} {ax_w:8.3f} {ly_w:10.3f} {lx_w:10.3f}")
print("\n(all widths in mm, -6 dB = 50% of the profile maximum)")