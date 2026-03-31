import numpy as np
import plotly.graph_objects as go
from scipy.io import loadmat
from scipy.interpolate import Rbf
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import combine_multiplexed
import hex61_pitch80

def quick_3D_visualization():

    # Load simulated multiplexed data
    data = np.load("data_visualization.npz")
    multiplexed_data = data["multiplexed_data"]
    # Load geometry
    coords = hex61_pitch80.coords
    # Combine multiplexed records -> full channel data (1024 × 61)
    full_data = combine_multiplexed.combine_multiplexed(multiplexed_data)

    # Extract coordinates
    x = coords[:, 0]
    y = coords[:, 1]

    # Layer index definitions
    layer_1 = np.array([1,2,3,4,5,6,11,12,18,19,26,27,35,
                        36,43,44,50,51,56,57,58,59,60,61]) - 1

    layer_2 = np.array([7,8,9,10,13,17,20,25,28,34,37,42,
                        45,49,52,53,54,55]) - 1

    layer_3 = np.array([14,15,16,21,24,29,33,38,41,46,47,48]) - 1

    layer_4 = np.array([22,23,30,32,39,40]) - 1

    layer_5 = np.array([31]) - 1

    # Simulate signal intensity
    intensity = np.zeros(61)

    for i in layer_1:
        intensity[i] = 0.6

    for i in layer_2:
        intensity[i] = 0.7

    for i in layer_3:
        intensity[i] = 0.8

    for i in layer_4:
        intensity[i] = 0.9

    for i in layer_5:
        intensity[i] = 1.0

    # Create grid (MATLAB: -400:1:400)
    dx = 5
    dy = 5

    grid_x_vals = np.arange(-400, 401, dx)
    grid_y_vals = np.arange(-400, 401, dy)

    grid_x, grid_y = np.meshgrid(grid_x_vals, grid_y_vals)

    interpolated_intensity = griddata(
        (x, y),
        intensity,
        (grid_x, grid_y),
        method='cubic'
    )

    # Remove NaNs
    interpolated_intensity = np.nan_to_num(interpolated_intensity)


    # Plot
    plt.figure(figsize=(6,6))

    axis_vals = np.arange(-400, 401, 1)

    plt.imshow(
        interpolated_intensity,
        origin='lower',
        extent=[-400, 400, -400, 400],
        cmap='hot',
        aspect='equal',
        interpolation='none',
        vmin=0.6,
        vmax=1 
    )

    plt.colorbar(label="Intensity")
    plt.title("Interpolated Intensity Image")

    plt.xlabel("x")
    plt.ylabel("y")

    plt.grid(False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    quick_3D_visualization() 