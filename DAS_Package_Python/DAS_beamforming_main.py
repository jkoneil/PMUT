import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from DAS import DAS

if __name__ == "__main__":
    data = sio.loadmat('simulated_sensor_data_double_pitch_18MHz.mat')

    sensor_data = data['sensor_data']
    fs = data['fs'].squeeze().item()
    pitch_mm = data['pitch_mm'].squeeze().item()

    channel_data = sensor_data.T
    rf = channel_data

    no_ele = rf.shape[1]  # number of transducer elements

    # Physical parameters
    channel_spacing = pitch_mm  # mm
    speed_sound = 1540  # m/s

    sample_spacing = (1 / fs * speed_sound) * 1000  # mm per sample

    # Axes
    x_axis = np.arange(1, rf.shape[1] + 1) * channel_spacing
    y_axis = np.arange(1, rf.shape[0] + 1) * sample_spacing

    # Run beamforming
    bf_data = DAS(rf, no_ele, fs, channel_spacing, speed_sound)

    # Plot before beamforming
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(
        channel_data,
        aspect='auto',
        cmap='gray',
        extent=[x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]]
    )

    plt.xlabel("lateral [mm]")
    plt.ylabel("axial [mm]")
    plt.title("Before beamforming")
    plt.colorbar()

    bf_data_env = np.abs(hilbert(bf_data, axis=0))
    bf_data_norm = bf_data_env / np.max(bf_data_env) # Normalize
    bf_data_db = 20 * np.log10(bf_data_norm + 1e-12) # Convert to dB

    # Plot after beamforming
    plt.subplot(1, 2, 2)
    plt.imshow(
        bf_data_db,
        aspect='auto',
        cmap='gray',
        vmin=-40,
        vmax=0,
        extent=[x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]]
    )

    plt.xlabel("lateral [mm]")
    plt.ylabel("axial [mm]")
    plt.title("After beamforming")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

