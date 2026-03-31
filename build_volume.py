import numpy as np
from scipy.interpolate import griddata
import pyvista as pv
import combine_multiplexed
import hex61_pitch80

def build_volume():

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
    grid_x_vals = np.linspace(-max_extent, max_extent, num=int(2*max_extent/dx)+1)
    grid_y_vals = np.linspace(-max_extent, max_extent, num=int(2*max_extent/dy)+1)
    grid_x, grid_y = np.meshgrid(grid_x_vals, grid_y_vals)

    # Volume construction
    dz_sample = 37  # µm per sample
    depth_step = 1 # Depth subsampling (important for performance)
    dz = dz_sample * depth_step

    num_slices = len(range(0, full_data.shape[0], depth_step))
    volume = np.zeros((grid_x.shape[0], grid_x.shape[1], num_slices), dtype=np.float32)

    for i, t in enumerate(range(0, full_data.shape[0], depth_step)):

        intensity = full_data[t, :]

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

    # Export
    np.save("volume.npy", volume)

    # Save origin info for Slicer
    origin_x = -dx * (grid_x.shape[1]-1)/2
    origin_y = -dy * (grid_y.shape[0]-1)/2
    origin_z = 0
    np.savez("volume_origin.npz", origin=(origin_x, origin_y, origin_z), spacing=(dx, dy, dz))


if __name__ == "__main__":
    build_volume() 
