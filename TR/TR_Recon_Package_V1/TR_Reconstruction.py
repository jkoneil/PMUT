"""
Time Reversal Reconstruction for 3D Photoacoustic Imaging
"""

import numpy as np
from joblib import Parallel, delayed


def reconstruct_sensor(XY, coord, PASignal, zo, xo, yo,
                       co, dt, dxo, dyo, dzo, zmin, xmax, ymax, ThetaMax):
    """
    Backproject the PA signal from a single sensor into a 3D image volume.
    Returns this sensor's contribution to the full volume.
    """
    Xsens = coord[XY, 0] * 0.001
    Ysens = coord[XY, 1] * 0.001

    Receive = np.diff(PASignal[:, XY])

    NumPoints = np.zeros((len(zo), len(xo), len(yo)), dtype=np.float32)
    ImgTemp   = np.zeros((len(zo), len(xo), len(yo)), dtype=np.float64)

    for t in range(len(Receive)):
        l = (t + 1) * dt * co

        if l == 0:
            continue

        dTheta = np.pi / (l * np.pi / dxo)
        dphi   = dTheta

        phi_vals   = np.arange(0, 2 * np.pi, dphi)
        theta_vals = np.arange(-ThetaMax, ThetaMax + dTheta, dTheta)

        PHI, THETA = np.meshgrid(phi_vals, theta_vals)
        PHI   = PHI.ravel()
        THETA = THETA.ravel()

        mask  = (l * np.cos(THETA) - zmin) >= 0
        PHI   = PHI[mask]
        THETA = THETA[mask]

        xind = np.round((l * np.sin(THETA) * np.cos(PHI) + Xsens + xmax) / dxo).astype(int)
        yind = np.round((l * np.sin(THETA) * np.sin(PHI) + Ysens + ymax) / dyo).astype(int)
        zind = np.round((l * np.cos(THETA) - zmin)                        / dzo).astype(int)

        valid = (
            (xind >= 0) & (xind < len(xo)) &
            (yind >= 0) & (yind < len(yo)) &
            (zind >= 0) & (zind < len(zo))
        )
        xind = xind[valid]
        yind = yind[valid]
        zind = zind[valid]

        val = l * Receive[t]
        np.add.at(ImgTemp,   (zind, xind, yind), val)
        np.add.at(NumPoints, (zind, xind, yind), 1)

    return ImgTemp / (NumPoints + 1)


def time_reversal_reconstruction(PASignal, coord, zo, xo, yo,
                                  co, dt, dxo, dyo, dzo,
                                  zmin, xmax, ymax, ThetaMax,
                                  n_jobs=-1):
    """
    Run the full time reversal reconstruction over all sensors in parallel.

    Returns
    -------
    Img : np.ndarray, shape (len(zo), len(xo), len(yo))
    """
    Img = np.zeros((len(zo), len(xo), len(yo)), dtype=np.float64)

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(reconstruct_sensor)(
            XY, coord, PASignal, zo, xo, yo,
            co, dt, dxo, dyo, dzo, zmin, xmax, ymax, ThetaMax
        )
        for XY in range(len(coord))
    )

    for r in results:
        Img += r

    return Img