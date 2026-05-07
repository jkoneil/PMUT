import numpy as np
from scipy.interpolate import griddata
import combine_multiplexed
import hex61_pitch80

from visualize_volume import show_volume

def build_interpolated_volume():

    # Load data
    data = np.load("data_visualization.npz")
    multiplexed_data = data["multiplexed_data"]
    # Load geometry
    coords = hex61_pitch80.coords
    # Combine multiplexed records -> full channel data (1024 × 61)
    full_data = combine_multiplexed.combine_multiplexed(multiplexed_data)

    # Extract coordinates
    x = coords[:, 0]
    y = coords[:, 1]

    # Grid setup
    dx = 5
    dy = 5

    max_extent = max(np.abs(coords[:, 0]).max(), np.abs(coords[:, 1]).max())
    grid_x_vals = np.arange(-max_extent, max_extent + dx, dx)
    grid_y_vals = np.arange(-max_extent, max_extent + dy, dy)
    grid_x, grid_y = np.meshgrid(grid_x_vals, grid_y_vals)

    # Volume construction
    dz_sample = 37  # µm per sample
    depth_step = 1 # Depth subsampling (important for performance)
    dz = dz_sample * depth_step

    num_slices = len(range(0, full_data.shape[0], depth_step))
    volume = np.zeros((grid_x.shape[0], grid_x.shape[1], num_slices), dtype=np.float32)

    for i, t in enumerate(range(0, full_data.shape[0], depth_step)):

        intensity = full_data[t, :]
        # fix later and use DAS
        interp_img = griddata(
            (x, y),
            intensity,
            (grid_x, grid_y),
            method='cubic'
        )

        # Fallback if cubic fails
        if np.isnan(interp_img).any():
            interp_img = griddata(
                (x, y),
                intensity,
                (grid_x, grid_y),
                method='linear'
            )

        interp_img = np.nan_to_num(interp_img)
        volume[:, :, i] = interp_img

    # Pad volume to reach 40 mm in Z
    z_max_data = dz * num_slices  # actual max Z in mm
    desired_z_max = 40000  # target round number for axis

    extra_slices = int(np.ceil((desired_z_max - z_max_data) / dz))
    if extra_slices > 0:
        volume = np.pad(volume,
                        ((0, 0), (0, 0), (0, extra_slices)),
                        mode='constant', constant_values=0)
    return volume, dx, dy, dz, grid_x_vals.min(), grid_y_vals.min()


if __name__ == "__main__":
    volume, dx, dy, dz, x_min, y_min = build_interpolated_volume()
    show_volume(volume, dx, dy, dz, x_min, y_min)