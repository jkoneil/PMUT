"""
Measure the ring around the bead in a reconstruction, instead of judging it by eye.

Run after a reconstruction (reads img_slicer.npz, the envelope volume):
    python ring_probe.py
    python ring_probe.py img_slicer.npz

For each depth near the bead it prints the brightness at the center and the
radius + brightness of the strongest ring (radially averaged), all relative to
the bead's peak.
"""
import sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "img_slicer.npz"
d = np.load(path)
env = d["volume"].astype(np.float64)              # (z, x, y), envelope
x0, y0, z0 = map(float, d["origin"])
dx, dy, dz = map(float, d["spacing"])
d.close()

nz, nx, ny = env.shape
zo = z0 + dz * np.arange(nz)
iz, ix, iy = np.unravel_index(np.argmax(env), env.shape)
peak = env[iz, ix, iy]
env /= peak

# radial distance (mm) from the bead's lateral position, for one slice
X = (np.arange(nx) - ix)[:, None] * dx
Y = (np.arange(ny) - iy)[None, :] * dy
rho = np.hypot(X, Y)
bins = np.arange(0, 4.0, dx)

print(f"{path}: bead peak at z = {zo[iz]:.2f} mm, "
      f"x = {x0 + ix*dx:.2f}, y = {y0 + iy*dy:.2f} mm")
print(f"bead half-max radius at its own depth: ", end="")
prof0 = np.array([env[iz][(rho >= b) & (rho < b + dx)].mean() for b in bins])
print(f"{bins[np.argmax(prof0 < 0.5)]:.2f} mm\n")

print(" depth vs bead   center   ring radius   ring brightness")
for k in range(max(iz - 30, 0), min(iz + 11, nz)):        # 1.5 mm above to 0.5 mm below
    prof = np.array([env[k][(rho >= b) & (rho < b + dx)].mean() for b in bins])
    # ring = strongest radial bump beyond where the profile first dips
    dips = np.where((prof[1:-1] < prof[:-2]) & (prof[1:-1] <= prof[2:]))[0]
    if len(dips):
        start = dips[0] + 1
        j = start + int(np.argmax(prof[start:]))
        ring = f"{bins[j]:6.2f} mm      {prof[j]:.2f}"
    else:
        ring = "     -            -"
    print(f"   {zo[k] - zo[iz]:+.2f} mm      {prof[0]:.2f}     {ring}")