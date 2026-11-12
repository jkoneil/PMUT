"""
GPU time-reversal reconstruction

Same physics as tr_reconstruction.cpp, but with the loops inverted:
  CPU (scatter): for sensor -> for time -> for shell angle -> add into voxel
  GPU (gather):  one thread per voxel -> for sensor -> look up signal at t = R/(co*dt)

No two threads write to the same voxel, so no atomics are needed.
"""

import os
import numpy as np

KERNEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tr_kernel.cu")
_module = None


def _get_kernel():
    """Compile tr_kernel.cu once (CuPy/NVRTC) and cache the module."""
    global _module
    if _module is None:
        import cupy as cp
        with open(KERNEL_PATH, "r") as f:
            _module = cp.RawModule(code=f.read())
    return _module.get_function("tr_gather")


def tr_gpu(PASignal, coord, zo, xo, yo, co, dt, ThetaMax, block=128):
    """
    PASignal : (nT, nSens) array (already cropped / channels zeroed / reverb removed)
    coord    : (nSens, 2) sensor positions in the file's units (x0.001 -> mm, as in the C++ code)
    zo,xo,yo : 1D coordinate arrays (mm), exactly as built in TRHexArray3D_para.py
    Returns  : (nz, nx, ny) float32 numpy array
    """
    import cupy as cp

    nT, nSens = PASignal.shape
    nz, nx, ny = len(zo), len(xo), len(yo)

    # derivative along time, laid out [sensor][time] so each sensor's trace is contiguous
    sig = cp.asarray(PASignal, dtype=cp.float32)
    Receive = cp.ascontiguousarray(cp.diff(sig, axis=0).T)          # (nSens, nT-1)

    sx = cp.asarray(coord[:, 0] * 0.001, dtype=cp.float32)
    sy = cp.asarray(coord[:, 1] * 0.001, dtype=cp.float32)

    img = cp.empty((nz, nx, ny), dtype=cp.float32)

    kernel = _get_kernel()
    grid = ((ny + block - 1) // block, nx, nz)

    kernel(grid, (block,), (
        Receive, sx, sy, img,
        np.int32(nSens), np.int32(nT - 1), np.int32(nz), np.int32(nx), np.int32(ny),
        np.float32(xo[0]), np.float32(yo[0]), np.float32(zo[0]),
        np.float32(xo[1] - xo[0]), np.float32(yo[1] - yo[0]), np.float32(zo[1] - zo[0]),
        np.float32(1.0 / (co * dt)), np.float32(np.tan(ThetaMax)),
    ))
    return cp.asnumpy(img)


def tr_gather_numpy(PASignal, coord, zo, xo, yo, co, dt, ThetaMax):
    """CPU reference of the exact same gather math (slow; use a small grid)."""
    nT, nSens = PASignal.shape
    Receive = np.diff(PASignal.astype(np.float32), axis=0).T         # (nSens, nT-1)
    nTm1 = nT - 1

    Z, X, Y = np.meshgrid(zo, xo, yo, indexing='ij')                 # (nz, nx, ny)
    Z = Z.astype(np.float32); X = X.astype(np.float32); Y = Y.astype(np.float32)
    rhoMax2 = (np.maximum(Z, 0) * np.tan(ThetaMax)) ** 2
    img = np.zeros(Z.shape, dtype=np.float32)

    for s in range(nSens):
        ddx = X - coord[s, 0] * 0.001
        ddy = Y - coord[s, 1] * 0.001
        rho2 = ddx ** 2 + ddy ** 2
        R = np.sqrt(rho2 + Z ** 2)
        tf = R / (co * dt) - 1.0
        t0 = np.floor(tf).astype(np.int64)
        valid = (Z > 0) & (rho2 <= rhoMax2) & (t0 >= 0) & (t0 < nTm1 - 1)
        t0c = np.clip(t0, 0, nTm1 - 2)
        w = (tf - t0).astype(np.float32)
        val = (1 - w) * Receive[s][t0c] + w * Receive[s][t0c + 1]
        img += np.where(valid, R * val, 0).astype(np.float32)
    return img
