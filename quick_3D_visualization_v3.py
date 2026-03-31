import numpy as np
from scipy.interpolate import griddata
import pyvista as pv
import combine_multiplexed
import hex61_pitch80

def quick_3D_visualization():

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

    # PyVista grid
    scale = 1e-3 # put in mm
    grid = pv.ImageData()
    grid.dimensions = volume.shape # Set volume dimensions
    
    # Define physical origin of the dataset
    grid.origin = ( 
        grid_x_vals.min() * scale, 
        grid_y_vals.min() * scale, 
        0
    )

    grid.spacing = (
        dx * scale, 
        dy * scale, 
        dz * scale
    )
    grid.point_data["intensity"] = volume.flatten(order="F")

    # Visualization
    plotter = pv.Plotter()
    plotter.add_volume(
        grid,
        scalars="intensity",
        cmap="hot",
        opacity="sigmoid",
        shade=True
    )

    # Z axis ticks
    z_max = dz * scale * (volume.shape[2] - 1)
    tick_spacing = 5
    n_zlabels = int(np.ceil(z_max / tick_spacing)) + 1  # include bottom tick at 0

    plotter.show_bounds(
        grid=True,
        location='outer',
        xtitle='x (mm)',
        ytitle='y (mm)',
        ztitle='z (mm)',
        ticks='outside',
        fmt="%.1f",
        n_zlabels=n_zlabels
    )

    plotter.add_axes(line_width=2)
    plotter.show()


if __name__ == "__main__":
    quick_3D_visualization() 
