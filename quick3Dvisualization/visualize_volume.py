import numpy as np
import pyvista as pv


def show_volume(volume, dx, dy, dz, x_min, y_min, scale=1e-3, extra_actors=None):
    """
    Generic 3D volume visualization using PyVista.

    Parameters:
        volume : 3D numpy array
        dx, dy, dz : voxel spacing (same units as coords, typically µm)
        x_min, y_min : grid origin (same units as coords)
        scale : unit conversion (default µm → mm)
        extra_actors : list of dict, optional
        Additional point clouds or meshes to overlay. Each dict should have:
            - 'points'  : np.ndarray, shape (N, 3), in display units (post-scale)
            - 'color'   : str, e.g. 'cyan'
            - 'label'   : str, legend label
            - 'size'    : float, point size (default: 6)
            - 'spheres' : bool, render as spheres (default: True)
    """

    grid = pv.ImageData()
    grid.dimensions = volume.shape # Set volume dimensions

    # Define physical origin of the dataset
    grid.origin = (
        x_min * scale,
        y_min * scale,
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

    # Overlay any extra point clouds / actors
    if extra_actors:
        for actor in extra_actors:
            plotter.add_points(
                actor['points'],
                color=actor.get('color', 'white'),
                point_size=actor.get('size', 6),
                render_points_as_spheres=actor.get('spheres', True),
                label=actor.get('label', '')
            )
        plotter.add_legend()

    # Z-axis ticks
    z_max = dz * scale * (volume.shape[2] - 1)
    tick_spacing = 5
    n_zlabels = int(np.ceil(z_max / tick_spacing)) + 1

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