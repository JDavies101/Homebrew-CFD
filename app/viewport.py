# viewport: domain box, relaxation layers, walls and part surfaces in cells, redrawn from the case file after every edit
import numpy as np
import pyvista
from pyvistaqt import QtInteractor
from src.geometry.airfoil import naca_four_digit, place_section
from src.geometry.mesh import read_stl, inspect_mesh
from src.geometry.preview import wall_points
from src.run.estimate import case_device_bytes, run_seconds, format_duration
from app import theme

minimum_cells_across = 10 # fewer cells than this across a part's smallest in-plane size resolves it poorly
wall_colors = {"static": theme.disabled, "moving": theme.accent}
layer_color = theme.amber

memory_fraction = 0.9 # leave room for the driver, the display and the boundary-link lists

def estimate_lines(case_file, throughput_mlups, measured, device_total_gb):
    """
    Memory and run-time estimate, the memory line in red when the fields will not fit the GPU.

    Returns a list of lines.
    """

    memory_gb = case_device_bytes(case_file) / 1e9
    memory_line = f"memory = {memory_gb:.2f} GB on the GPU"
    if device_total_gb is not None:
        memory_line += f" of {device_total_gb:.1f} GB"
        if memory_gb > memory_fraction * device_total_gb:
            memory_line = f"<span style='color:{theme.error}'>{memory_line}: will not fit (coarser resolution or smaller domain)</span>"

    source = "measured on this machine" if measured else "typical GPU, refined after the first run"
    time_line = f"run time ≈ {format_duration(run_seconds(case_file, throughput_mlups))} at {throughput_mlups:,.0f} MLUPS ({source})"

    return [memory_line, time_line]

def preview_report(preview):
    """
    Per part: solid cells, boundary links and q range, in red when a part has no solid cells, a Bouzidi part
    has no links (its wall would be silently off), or a q falls outside (0, 1].

    Returns a list of lines.
    """

    lines = []
    for index, name in enumerate(preview["names"]):
        part_number = index + 1
        solid_cells = int(np.count_nonzero(preview["part_id"] == part_number))
        part_fractions = preview["fractions"][preview["link_part"] == part_number]
        problems = []
        if solid_cells == 0:
            problems.append("no solid cells (below the resolution or outside the domain)")

        if not preview["bouzidi"][index]:
            line = f"{name}: {solid_cells:,} solid cells, staircase walls (q = 1/2)"
        elif len(part_fractions) == 0:
            line = f"{name}: {solid_cells:,} solid cells"
            problems.append("no boundary links: its Bouzidi wall would be off")
        else:
            line = (f"{name}: {solid_cells:,} solid cells, {len(part_fractions):,} boundary links, "
                    f"q {part_fractions.min():.3f} to {part_fractions.max():.3f}")
            outside = int(np.count_nonzero((part_fractions <= 0.0) | (part_fractions > 1.0)))
            if outside:
                problems.append(f"{outside:,} links with q outside (0, 1]")

        if problems:
            line += f" <span style='color:{theme.error}'>" + "; ".join(problems) + "</span>"
        lines.append(line)

    return lines

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

def read_part_mesh(path, stl_cache=None):
    """
    Read an STL and inspect it in its own units, once per path when a cache is given.

    Returns (triangles, inspection).
    """

    if stl_cache is not None and path in stl_cache:
        return stl_cache[path]
    
    triangles = read_stl(path)
    entry = (triangles, inspect_mesh(triangles, 1.0))
    if stl_cache is not None:
        stl_cache[path] = entry

    return entry

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
    triangles, _ = read_part_mesh(spec.path, stl_cache)
    placed = triangles * spec.cells_per_unit + np.asarray(spec.offset, np.float64)
    vertices = placed.reshape(-1, 3)
    faces = np.column_stack([np.full(len(placed), 3), np.arange(len(vertices)).reshape(-1, 3)]).ravel()

    return pyvista.PolyData(vertices, faces)

def part_report(name, surface, domain, inspection=None, cells_per_unit=1.0):
    """
    Size of a part in cells, with warnings for coarse resolution or geometry outside the domain
    or a STL that is not watertight.

    Returns one line of text.
    """

    x_min, x_max, y_min, y_max, z_min, z_max = surface.bounds
    extent = (x_max - x_min, y_max - y_min, z_max - z_min)
    line = f"{name}: {extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f} cells"
    warnings = []
    if inspection is not None:
        size = inspection["size"]
        line += f" ({size[0]:.4g} x {size[1]:.4g} x {size[2]:.4g} units at {cells_per_unit:.4g} cells/unit)"
        if not inspection["watertight"]:
            warnings.append(f"not watertight ({inspection['open_edges']} open, {inspection['flipped_edges']} flipped, "
                            f"{inspection['nonmanifold_edges']} non-manifold edges)")
    if min(extent[0], extent[1]) < minimum_cells_across:
        warnings.append(f"under {minimum_cells_across} cells across (check STL units / scale)")
    
    if x_min < 0 or y_min < 0 or z_min < 0 or x_max > domain.nx or y_max > domain.ny or z_max > domain.nz:
        warnings.append("extends outside the domain")
    
    if warnings:
        line += f" <span style='color:{theme.error}'>" + "; ".join(warnings) + "</span>"

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
        self.plotter.set_background(theme.viewport_background)
        self.stl_cache = {}

    def draw(self, case_file, reset_camera=False, part_opacity=1.0):
        """
        Redraw domain, layers, walls and parts; keeps the camera unless reset_camera; part_opacity below 1 lets a
        voxel preview show through the true surfaces.

        Returns one report line per part.
        """

        domain = case_file.domain
        nx = domain.nx
        ny = domain.ny
        nz = domain.nz
        plotter = self.plotter
        plotter.clear()

        # domain box
        plotter.add_mesh(pyvista.Box(bounds=(0, nx, 0, ny, 0, nz)).outline(), color=theme.viewport_outline, line_width=1)

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
                inspection = read_part_mesh(spec.path, self.stl_cache)[1] if spec.kind == "stl" else None
            except (OSError, ValueError) as error:
                reports.append(f"{spec.name}: <span style='color:{theme.error}'>cannot read geometry ({error})</span>")
                continue
            plotter.add_mesh(surface, color=theme.viewport_part, smooth_shading=True, opacity=part_opacity)
            reports.append(part_report(spec.name, surface, domain, inspection, spec.cells_per_unit))

        plotter.add_axes()
        if reset_camera:
            self.set_view(default_view)

        return reports

    def show_preview(self, preview):
        """
        Overlay the voxelized parts (solid cells as cubes) and the Bouzidi wall points coloured by q.
        """

        part_id = preview["part_id"]

        # cell (i, j, k) spans node +- 1/2, like the wall rows in draw
        grid = pyvista.ImageData(dimensions=np.array(part_id.shape) + 1, origin=(-0.5, -0.5, -0.5))
        grid.cell_data["part"] = part_id.ravel(order="F")
        solid = grid.threshold(0.5, scalars="part")
        if solid.n_cells > 0:
            self.plotter.add_mesh(solid.extract_surface(), color=theme.viewport_part, show_edges=True, name="preview_solid")

        # wall points x + q c_q, coloured by q
        if len(preview["fractions"]) > 0:
            points = pyvista.PolyData(wall_points(preview["nodes"], preview["directions"], preview["fractions"]))
            points.point_data["q"] = preview["fractions"]
            self.plotter.add_mesh(points, scalars="q", clim=(0.0, 1.0), cmap="viridis", point_size=4,
                                  render_points_as_spheres=True, name="preview_links")

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