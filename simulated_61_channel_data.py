# This script is written to simulate 61 channel data collected by PMUT
# array, outputting multiplexed records (1024 x 16 x 4)
# Copied from MATLAB version into Python

import numpy as np
from scipy.io import savemat

# Initiate a 2D matrix
# vertical axis: depth
# horizontal axis: channel

num_samples = 1024
num_channels = 61
all_data = np.zeros((num_samples, num_channels))

# PMUT information
f_c = 10e6  # [Hz]
f_s = 4 * f_c  # 200% Nyquist
s_o_s = 1480e3  # [mm/s]
sample_spacing = (1 / f_s) * s_o_s  # [mm]

# Simulate point targets at 0.5, 1, 2, 5, 10 mm
target_depths_mm = np.array([0.5, 1, 2, 5, 10])  # [mm]
target_depths_pixel = np.round(target_depths_mm / sample_spacing).astype(int)
int_curve_hor = np.array([1, 0.7, 0.3])  # for each layer from the center
int_curve_ver = np.array([0.1, 0.3, 0.7, 0.9, 1, 0.9, 0.7, 0.3, 0.1])

# Index array of selected elements
# MATLAB indices → convert to 0-based Python indices
element_array = np.array([
    14, 15, 16, 21, 22, 23, 24, 29, 30, 31,
    32, 33, 38, 39, 40, 41, 46, 47, 48
]) - 1
third_layer = np.array([14, 15, 16, 21, 24, 29, 33, 38, 41, 46, 47, 48]) - 1
second_layer = np.array([22, 23, 30, 32, 39, 40]) - 1
first_layer = np.array([31]) - 1

# Render in the vertical direction
for target_depth in target_depths_pixel:
    neighbor_size = (len(int_curve_ver) - 1) // 2
    target_neighbor = np.arange(
        target_depth - neighbor_size,
        target_depth + neighbor_size + 1
    )

    # Boundary protection
    target_neighbor = target_neighbor[
        (target_neighbor >= 0) & (target_neighbor < num_samples)
    ]

    for ch in element_array:
        all_data[target_neighbor, ch] = 1
        all_data[target_neighbor, ch] *= int_curve_ver[:len(target_neighbor)]

# Render in the horizontal direction
all_data[:, first_layer] *= int_curve_hor[0]
all_data[:, second_layer] *= int_curve_hor[1]
all_data[:, third_layer] *= int_curve_hor[2]

# Convert full 61-channel data into multiplexed form (1024 x 16 x 4)
multiplexed_data = np.zeros((num_samples, 16, 4))

for rec in range(16):
    for sec in range(4):
        ch = sec * 16 + rec
        if ch < 61:
            multiplexed_data[:, rec, sec] = all_data[:, ch]

# Save data
np.savez("data_visualization.npz", multiplexed_data=multiplexed_data)
