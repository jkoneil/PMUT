import numpy as np
from scipy.interpolate import griddata
import pyvista as pv
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

    # Create grid
    grid_x_vals = np.arange(np.min(x), np.max(x) + 1, 1)
    grid_y_vals = np.arange(np.min(y), np.max(y) + 1, 1)

    grid_x, grid_y = np.meshgrid(grid_x_vals, grid_y_vals)

    # Build volumetric stack

    volume_slices = []

    # Depth subsampling (important for performance)
    depth_step = 4
    for t in range(0, full_data.shape[0], depth_step):

        intensity = full_data[t, :]
        # Interpolate discrete sensor measurements onto spatial grid
        interp_img = griddata(
            (x, y),
            intensity,
            (grid_x, grid_y),
            method='cubic'
        )
        # Remove NaNs
        interp_img = np.nan_to_num(interp_img)

        volume_slices.append(interp_img)

    # Stack slices into 3D volume array
    volume = np.stack(volume_slices, axis=2)
    
    # Create PyVista grid

    grid = pv.ImageData()

    # Set volume dimensions
    grid.dimensions = volume.shape
    
    # Define physical origin of the dataset
    grid.origin = (grid_x_vals.min(), grid_y_vals.min(), 0)

    # Spatial resolution parameters
    dx = grid_x_vals[1] - grid_x_vals[0]
    dy = grid_y_vals[1] - grid_y_vals[0]
    z_spacing = 10 # thickness

    grid.spacing = (dx, dy, z_spacing)


    grid.point_data["intensity"] = volume.flatten(order="F")

    # Render volume
    plotter = pv.Plotter()
    plotter.add_volume(
        grid,
        scalars="intensity",
        cmap="hot",
        opacity="sigmoid",
        shade=True
    )

    plotter.add_axes()
    plotter.show()


if __name__ == "__main__":
    quick_3D_visualization() 