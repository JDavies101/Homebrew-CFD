# viewport: domain box, relaxation layers, walls and part surfaces in cells, redrawn from the case file after every edit
import numpy as np
import pyvista
from pyvistaqt import QtInteractor
from src.geometry.airfoil import naca_four_digit, place_section
from src.geometry.mesh import read_stl

minimum_cells_across = 10 # fewer cells than this across a part's smallest in-plane size resolves it poorly
wall_colors = {"static": "#8a8a8a", "moving": "#3d7fd1"}
layer_color = "#e0a030"

# camera views: (direction from the domain center to the camera, view-up); x streamwise, y up, z span
named_views = {
    "Side (+z)": ((0, 0, 1), (0, 1, 0)),
    "Side (-z)": ((0, 0, -1), (0, 1, 0)),
    "Top (+y)": ((0, 1, 0), (-1, 0, 0)),
    "Bottom (-y)": ((0, -1, 0), (1, 0, 0)),
    "Front (-x, from inlet)": ((-1, 0, 0), (0, 1, 0)),
    "Rear (+x, from outlet)": ((1, 0, 0), (0, 1, 0)),
    "Isometric": ((1, 1, 1), (0, 1, 0)),
    "Dimetric": ((1, 1, 0.5), (0, 1, 0)),
    "Trimetric": ((1, 0.7, 0.4), (0, 1, 0)),
}
corner_views = {f"Corner ({'+' if x > 0 else '-'}x {'+' if y > 0 else '-'}y {'+' if z > 0 else '-'}z)": ((x, y, z), (0, 1, 0))
                for x in (1, -1) for y in (1, -1) for z in (1, -1)}
default_view = "Side (+z)"

def part_surface(spec, domain, stl_cache=None):
    """
    Display surface of one geometry spec in grid cells, without voxelizing.

    Returns a pyvista.PolyData; raises OSError / ValueError if an STL cannot be read.
    """

    if spec.kind == "naca":
        polygon_x, polygon_y = naca_four_digit(spec.section)
        placed_x, placed_y = place_section(polygon_x, polygon_y, spec.chord, spec.angle_degrees, spec.leading_edge[0], spec.leading_edge[1])
        points = np.column_stack([placed_x, placed_y, np.zeros(len(placed_x))])
        outline = pyvista.PolyData(points, faces=np.concatenate([[len(points)], np.arange(len(points))]))

        return outline.triangulate().extrude((0.0, 0.0, float(domain.nz)), capping=True)

    # stl: read once per path, then scale and offset like build_part does
    if stl_cache is not None and spec.path in stl_cache:
        triangles = stl_cache[spec.path]
    else:
        triangles = read_stl(spec.path)
        if stl_cache is not None:
            stl_cache[spec.path] = triangles
    placed = triangles * spec.cells_per_unit + np.asarray(spec.offset, np.float64)
    vertices = placed.reshape(-1, 3)
    faces = np.column_stack([np.full(len(placed), 3), np.arange(len(vertices)).reshape(-1, 3)]).ravel()

    return pyvista.PolyData(vertices, faces)

def part_report(name, surface, domain):
    """
    Size of a part in cells, with warnings for coarse resolution or geometry outside the domain.

    Returns one line of text.
    """

    x_min, x_max, y_min, y_max, z_min, z_max = surface.bounds
    extent = (x_max - x_min, y_max - y_min, z_max - z_min)
    line = f"{name}: {extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f} cells"
    warnings = []
    if min(extent[0], extent[1]) < minimum_cells_across:
        warnings.append(f"under {minimum_cells_across} cells across (check STL units / scale)")
    if x_min < 0 or y_min < 0 or z_min < 0 or x_max > domain.nx or y_max > domain.ny or z_max > domain.nz:
        warnings.append("extends outside the domain")
    if warnings:
        line += " <span style='color:#d33'>" + "; ".join(warnings) + "</span>"

    return line

class CaseViewport:
    """
    PyVista view of one case file, owned by the main window.
    """

    def __init__(self, parent):
        """
        Create the interactor widget and the STL cache.
        """

        self.plotter = QtInteractor(parent)
        self.plotter.set_background("#2b2f36", top="#4a5562")
        self.stl_cache = {}

    def draw(self, case_file, reset_camera=False):
        """
        Redraw domain, layers, walls and parts; keeps the camera unless reset_camera.

        Returns one report line per part.
        """

        domain = case_file.domain
        nx = domain.nx
        ny = domain.ny
        nz = domain.nz
        plotter = self.plotter
        plotter.clear()

        # domain box
        plotter.add_mesh(pyvista.Box(bounds=(0, nx, 0, ny, 0, nz)).outline(), color="white", line_width=1)
        pyvista.Box(bounds=(-0.5, nx - 0.5, -0.5, ny - 0.5, -0.5, nz - 0.5)).outline()

        # relaxation layers, translucent
        layers = []
        if domain.relax_width_x > 0:
            layers += [(0, domain.relax_width_x, 0, ny, 0, nz), (nx - domain.relax_width_x, nx, 0, ny, 0, nz)]
        if domain.relax_width_z > 0:
            layers += [(0, nx, 0, ny, 0, domain.relax_width_z), (0, nx, 0, ny, nz - domain.relax_width_z, nz)]
        for bounds in layers:
            plotter.add_mesh(pyvista.Box(bounds=bounds), color=layer_color, opacity=0.12)

        # floor and ceiling as the solid rows they are: cells j = 0 and j = ny - 1 (halfway walls at 0.5 and ny - 1.5)
        if domain.y_boundary == "walls":
            for bounds, kind in (((-0.5, nx - 0.5, -0.5, 0.5, -0.5, nz - 0.5), domain.floor),
                                 ((-0.5, nx - 0.5, ny - 1.5, ny - 0.5, -0.5, nz - 0.5), domain.ceiling)):
                plotter.add_mesh(pyvista.Box(bounds=bounds), color=wall_colors[kind], opacity=0.6)

        # parts
        reports = []
        for spec in case_file.geometry:
            try:
                surface = part_surface(spec, domain, self.stl_cache)
            except (OSError, ValueError) as error:
                reports.append(f"{spec.name}: <span style='color:#d33'>cannot read geometry ({error})</span>")
                continue
            plotter.add_mesh(surface, color="#d8d8d8", smooth_shading=True)
            reports.append(part_report(spec.name, surface, domain))

        plotter.add_axes()
        if reset_camera:
            self.set_view(default_view)

        return reports

    def close(self):
        """
        Release the VTK render window before Qt destroys the widget (avoids errors on exit).
        """

        self.plotter.close()

    def set_view(self, name):
        """
        Look at the whole case from a named direction, fitted to the window.
        """

        direction, view_up = {**named_views, **corner_views}[name]
        self.plotter.view_vector(direction, viewup=view_up)
        self.plotter.reset_camera()

    def fit(self):
        """
        Zoom to fit everything, keeping the current direction.
        """

        self.plotter.reset_camera()

    def set_orthographic(self, enabled):
        """
        Parallel projection (true lengths, no perspective) or perspective.
        """

        if enabled:
            self.plotter.enable_parallel_projection()
        else:
            self.plotter.disable_parallel_projection()