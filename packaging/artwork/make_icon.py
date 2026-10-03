# icon artwork (python make_icon.py [small]): potential flow (Hess-Smith panels + Kutta) around an inverted cambered wing, rendered with glow
import sys
import numpy as np
from matplotlib.path import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.ndimage import distance_transform_edt, gaussian_filter

# variants: full icon, small icon for 16-32 px (bigger wing, five bold lines, no grid), splash screen with wordmark
SMALL = "small" in sys.argv
SPLASH = "splash" in sys.argv
SUPER = 2
SIZE = 1024 * SUPER
WIDTH, HEIGHT = (1200 * SUPER, 675 * SUPER) if SPLASH else (SIZE, SIZE)
CHORD_PX = (820 if SMALL else 470 if SPLASH else 660) * SUPER
ORIGIN_PX = np.array([95.0, 650.0] if SMALL else [90.0, 360.0] if SPLASH else [150.0, 620.0]) * SUPER
# visible flow region in chord units
X_MIN = -ORIGIN_PX[0] / CHORD_PX - 0.05
X_MAX = (WIDTH - ORIGIN_PX[0]) / CHORD_PX + 0.05
Y_MAX = ORIGIN_PX[1] / CHORD_PX + 0.05
Y_MIN = -(HEIGHT - ORIGIN_PX[1]) / CHORD_PX - 0.05
ANGLE_DEG = 14.0
NAVY = np.array([15, 28, 44], float)
GRID = (40, 62, 90)
STOPS = np.array([[30, 120, 255], [125, 60, 240], [255, 85, 30], [255, 170, 0]], float)


def airfoil(camber=0.08, camber_pos=0.4, thickness=0.15, n=120):
    """Inverted NACA 4-digit section rotated trailing edge up, counter-clockwise from the trailing edge.

    Returns (n_points, 2) in chord units, y up.
    """

    beta = np.linspace(0.0, np.pi, n)
    x = (1 - np.cos(beta)) / 2
    half = 5 * thickness * (0.2969 * np.sqrt(x) - 0.126 * x - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    front = x < camber_pos
    yc = np.where(front, camber / camber_pos ** 2 * (2 * camber_pos * x - x * x),
                  camber / (1 - camber_pos) ** 2 * (1 - 2 * camber_pos + 2 * camber_pos * x - x * x))
    slope = np.where(front, 2 * camber / camber_pos ** 2 * (camber_pos - x),
                     2 * camber / (1 - camber_pos) ** 2 * (camber_pos - x))
    theta = np.arctan(slope)
    xu, yu = x - half * np.sin(theta), yc + half * np.cos(theta)
    xl, yl = x + half * np.sin(theta), yc - half * np.cos(theta)
    # inverted: mirror y, so the old lower surface is now on top
    top = np.stack([xl, -yl], 1)[::-1]
    bottom = np.stack([xu, -yu], 1)[1:]
    points = np.concatenate([top, bottom])
    angle = np.radians(ANGLE_DEG)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    return points @ rotation.T


def panel_velocity(points, starts, ends):
    """Unit source and unit vortex velocities induced at points by each panel.

    Returns (source_u, source_v, vortex_u, vortex_v), each (n_points, n_panels).
    """

    tangent = ends - starts
    length = np.hypot(tangent[:, 0], tangent[:, 1])
    cos_phi = tangent[:, 0] / length
    sin_phi = tangent[:, 1] / length
    dx = points[:, None, 0] - starts[None, :, 0]
    dy = points[:, None, 1] - starts[None, :, 1]
    local_x = dx * cos_phi + dy * sin_phi
    local_y = -dx * sin_phi + dy * cos_phi
    log_ratio = np.log((local_x * local_x + local_y * local_y) / ((local_x - length) ** 2 + local_y * local_y)) / (4 * np.pi)
    angle_span = (np.arctan2(local_y, local_x - length) - np.arctan2(local_y, local_x)) / (2 * np.pi)
    source_u = log_ratio * cos_phi - angle_span * sin_phi
    source_v = log_ratio * sin_phi + angle_span * cos_phi
    vortex_u = angle_span * cos_phi + log_ratio * sin_phi
    vortex_v = angle_span * sin_phi - log_ratio * cos_phi

    return source_u, source_v, vortex_u, vortex_v


def solve_flow(nodes):
    """Hess-Smith source strengths and vortex strength with the Kutta condition, unit freestream along +x.

    Returns a velocity(points) function.
    """

    starts = nodes[:-1]
    ends = nodes[1:]
    tangent = ends - starts
    tangent /= np.hypot(tangent[:, 0], tangent[:, 1])[:, None]
    normal = np.stack([tangent[:, 1], -tangent[:, 0]], 1)
    control = (starts + ends) / 2 + 1e-7 * normal
    source_u, source_v, vortex_u, vortex_v = panel_velocity(control, starts, ends)
    count = len(starts)
    matrix = np.zeros((count + 1, count + 1))
    rhs = np.zeros(count + 1)
    matrix[:count, :count] = source_u * normal[:, 0:1] + source_v * normal[:, 1:2]
    matrix[:count, count] = (vortex_u * normal[:, 0:1] + vortex_v * normal[:, 1:2]).sum(1)
    rhs[:count] = -normal[:, 0]
    for i in (0, count - 1):
        matrix[count, :count] += source_u[i] * tangent[i, 0] + source_v[i] * tangent[i, 1]
        matrix[count, count] += (vortex_u[i] * tangent[i, 0] + vortex_v[i] * tangent[i, 1]).sum()
        rhs[count] -= tangent[i, 0]
    solution = np.linalg.solve(matrix, rhs)
    sources = solution[:count]
    vortex = solution[count]

    def velocity(points):
        su, sv, vu, vv = panel_velocity(points, starts, ends)
        u = 1 + su @ sources + vortex * vu.sum(1)
        v = sv @ sources + vortex * vv.sum(1)

        return np.stack([u, v], 1)

    return velocity


def to_pixels(points):
    """Chord units (y up) to supersampled pixels (y down)."""

    return np.stack([ORIGIN_PX[0] + points[:, 0] * CHORD_PX, ORIGIN_PX[1] - points[:, 1] * CHORD_PX], 1)


def trace(velocity, seeds, inside, step=0.004, max_steps=1200):
    """RK4 streamlines along the normalized velocity from each seed.

    Returns a list of (n, 2) arrays in chord units.
    """

    def direction(points):
        field = velocity(points)

        return field / np.hypot(field[:, 0], field[:, 1])[:, None]

    current = seeds.copy()
    paths = [current.copy()]
    for _ in range(max_steps):
        k1 = direction(current)
        k2 = direction(current + 0.5 * step * k1)
        k3 = direction(current + 0.5 * step * k2)
        k4 = direction(current + step * k3)
        current = current + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        paths.append(current.copy())
    paths = np.stack(paths, 1)
    lines = []
    for path in paths:
        keep = (path[:, 0] < X_MAX) & (path[:, 1] < Y_MAX) & (path[:, 1] > Y_MIN) & ~inside(path)
        stop = np.argmin(keep) if not keep.all() else len(keep)
        lines.append(path[:stop])

    return lines


def colour(points, wake_y):
    """Blue upstream, warm stops downstream, warmest near the wake centreline."""

    downstream = np.clip(points[:, 0] - 0.25, 0, 1)
    near_wake = np.exp(-((points[:, 1] - wake_y(points[:, 0])) / 0.55) ** 2)
    position = downstream * (0.35 + 0.65 * near_wake) * (len(STOPS) - 1)
    index = np.minimum(position.astype(int), len(STOPS) - 2)
    fraction = (position - index)[:, None]

    ramp = STOPS[index] * (1 - fraction) + STOPS[index + 1] * fraction
    # splash: same ramp as the icon, then the warm lines fade into the navy further downstream
    if SPLASH:
        ramp = ramp * np.exp(-np.clip(points[:, 0] - 1.4, 0, None) / 0.6)[:, None]

    return ramp


def draw_lines(lines, colours, widths):
    """Coloured polylines on a black RGB canvas, segment by segment, one width per line."""

    canvas = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    pen = ImageDraw.Draw(canvas)
    for line, line_colours, width in zip(lines, colours, widths):
        pixels = to_pixels(line)
        for k in range(len(pixels) - 1):
            pen.line([tuple(pixels[k]), tuple(pixels[k + 1])], fill=tuple(int(c) for c in line_colours[k]), width=width)
            pen.ellipse([pixels[k][0] - width / 2, pixels[k][1] - width / 2, pixels[k][0] + width / 2, pixels[k][1] + width / 2],
                        fill=tuple(int(c) for c in line_colours[k]))

    return np.asarray(canvas, float)


def main():

    nodes = airfoil()
    velocity = solve_flow(nodes)
    outline = Path(nodes)

    def inside(points):

        return outline.contains_points(points, radius=-0.004)

    # stagnation streamline seed by bisection (passes over or under the wing at mid-chord)
    mid_x = 0.5
    mid_y = np.interp(mid_x, nodes[len(nodes) // 2:, 0], nodes[len(nodes) // 2:, 1])
    low, high = -0.4, 0.4
    for _ in range(25):
        middle = (low + high) / 2
        line = trace(velocity, np.array([[X_MIN, middle]]), inside, max_steps=400)[0]
        passes_over = line[np.argmin(np.abs(line[:, 0] - mid_x)), 1] > mid_y and line[:, 0].max() > mid_x
        if passes_over:
            high = middle
        else:
            low = middle
    # seeds half a spacing either side of the stagnation streamline, the two nearest pushed further out
    spacing = 0.145
    offsets = np.arange(-7, 5) + 0.5
    offsets[offsets == -0.5] = -0.9
    offsets[offsets > 0] -= 0.25
    if SMALL:
        spacing = 0.27
        offsets = np.array([-1.7, -0.85, 0.5, 1.2, 1.9])
    if SPLASH:
        offsets = np.arange(np.floor((Y_MIN - middle) / spacing), np.ceil((Y_MAX - middle) / spacing)) + 0.5
        offsets[offsets == -0.5] = -0.9
        offsets[offsets > 0] -= 0.25
    seeds_y = middle + spacing * offsets
    seeds = np.stack([np.full_like(seeds_y, X_MIN), seeds_y], 1)
    lines = [line for line in trace(velocity, seeds, inside) if len(line) > 10]
    trailing_edge = nodes[0]
    wake = trace(velocity, np.array([trailing_edge + [0.01, 0.0]]), inside)[0]

    def wake_y(x):

        return np.interp(x, wake[:, 0], wake[:, 1], left=trailing_edge[1], right=wake[-1, 1])

    # lines near the wake are the brightest and boldest, outer lines recede
    strengths = []
    for line in lines:
        end = line[np.argmin(np.abs(line[:, 0] - 1.2))]
        strengths.append(np.exp(-((end[1] - wake_y(end[0])) / 0.4) ** 2))
    colours = [colour(line, wake_y) * min(1.0, (0.8 if SMALL else 0.35) + 0.65 * strength) for line, strength in zip(lines, strengths)]
    widths = [int(((16 + 6 * strength) if SMALL else (5 + 6 * strength)) * SUPER) for strength in strengths]

    # background with grid
    background = Image.new("RGB", (WIDTH, HEIGHT), tuple(int(c) for c in NAVY))
    grid_pen = ImageDraw.Draw(background)
    for k in range(0, max(WIDTH, HEIGHT), (128 if SMALL else 64) * SUPER):
        grid_pen.line([(k, 0), (k, HEIGHT)], fill=GRID, width=(6 if SMALL else 1) * SUPER)
        grid_pen.line([(0, k), (WIDTH, k)], fill=GRID, width=(6 if SMALL else 1) * SUPER)
    image = np.asarray(background, float)

    # wake band, wide glow, line glow, line core (additive)
    wake_colour = np.array([colour(wake, wake_y)[k] for k in range(len(wake))])
    band = draw_lines([wake], [wake_colour], [(150 if SMALL else 80) * SUPER])
    band_glow = np.asarray(Image.fromarray(band.astype(np.uint8)).filter(ImageFilter.GaussianBlur((70 if SMALL else 45) * SUPER)), float)
    image += (1.6 if SMALL else 1.2) * band_glow
    lines_layer = draw_lines(lines, colours, widths)
    line_glow = np.asarray(Image.fromarray(lines_layer.astype(np.uint8)).filter(ImageFilter.GaussianBlur(10 * SUPER)), float)
    image += 0.8 * line_glow
    image += 0.35 * np.asarray(Image.fromarray(lines_layer.astype(np.uint8)).filter(ImageFilter.GaussianBlur(3 * SUPER)), float)
    covered = lines_layer.max(2, keepdims=True) > 0
    image = np.where(covered, lines_layer * 0.9 + 0.1 * lines_layer.max(2, keepdims=True), image)
    image = np.clip(image, 0, 255)
    # environment light for the wing reflections: the surrounding glow, blurred wide
    environment = np.asarray(Image.fromarray(np.clip(band_glow + 2 * line_glow, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(28 * SUPER)), float)

    # wing: soft shadow, then silver shading light on top to darker underneath, light rim
    wing_pixels = [tuple(p) for p in to_pixels(nodes)]
    wing_mask = Image.new("L", (WIDTH, HEIGHT), 0)
    ImageDraw.Draw(wing_mask).polygon(wing_pixels, fill=255)
    shadow = np.asarray(wing_mask.filter(ImageFilter.GaussianBlur(14 * SUPER)), float)[..., None] / 255
    shadow = np.roll(shadow, (10 * SUPER, 4 * SUPER), (0, 1))
    image = image * (1 - 0.55 * shadow)
    # rounded height field from the distance to the outline, then surface normals
    inside_wing = np.asarray(wing_mask) > 127
    distance = distance_transform_edt(inside_wing)
    radius = distance.max()
    height = np.sqrt(np.clip(distance * (2 * radius - distance), 0, None))
    height = gaussian_filter(height, 6 * SUPER)
    slope_y, slope_x = np.gradient(height)
    normal = np.stack([-slope_x, -slope_y, np.ones_like(height)], -1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    # key light from the upper left, Blinn specular toward the viewer
    light = np.array([-0.35, -0.75, 0.55])
    light /= np.linalg.norm(light)
    halfway = light + np.array([0.0, 0.0, 1.0])
    halfway /= np.linalg.norm(halfway)
    diffuse = np.clip(normal @ light, 0, 1)[..., None]
    specular = np.clip(normal @ halfway, 0, 1)[..., None] ** 60
    # reflections: surfaces facing down mirror the flow underneath, surfaces facing back mirror the wake
    facing_down = np.clip(normal[..., 1], 0, 1)[..., None]
    tint = environment * (environment / (environment.max(2, keepdims=True) + 1)) ** 0.5
    base = np.array([168, 174, 186], float)
    rounded = base * (0.3 + 0.8 * diffuse) + 0.9 * tint * (0.35 + 0.65 * facing_down) + 255 * 0.85 * specular
    # flat look (v7): light on top to darker underneath across the chord, flow tint, thin streak
    angle = np.radians(ANGLE_DEG)
    rows, columns = np.mgrid[0:HEIGHT, 0:WIDTH].astype(float)
    across = (rows - ORIGIN_PX[1]) * np.cos(angle) + (columns - ORIGIN_PX[0]) * np.sin(angle)
    # shade measured from the local top surface to the local bottom surface, so it follows the camber
    along = (columns - ORIGIN_PX[0]) * np.cos(angle) - (rows - ORIGIN_PX[1]) * np.sin(angle)
    outline_pixels = to_pixels(nodes)
    outline_along = (outline_pixels[:, 0] - ORIGIN_PX[0]) * np.cos(angle) - (outline_pixels[:, 1] - ORIGIN_PX[1]) * np.sin(angle)
    outline_across = (outline_pixels[:, 1] - ORIGIN_PX[1]) * np.cos(angle) + (outline_pixels[:, 0] - ORIGIN_PX[0]) * np.sin(angle)
    leading = np.argmin(outline_along)
    top_along, top_across = outline_along[:leading + 1][::-1], outline_across[:leading + 1][::-1]
    bottom_along, bottom_across = outline_along[leading:], outline_across[leading:]
    local_top = np.interp(along, top_along, top_across)
    local_bottom = np.interp(along, bottom_along, bottom_across)
    shade = np.clip((across - local_top) / np.maximum(local_bottom - local_top, 1.0), 0, 1)[..., None]
    flat = np.array([196, 201, 210], float) * (1 - shade) + np.array([84, 90, 106], float) * shade
    flat = flat * (1 - 0.25 * np.clip(environment.max(2, keepdims=True) / 255, 0, 1)) + 0.75 * tint
    flat = flat + 45 * np.exp(-((shade - 0.18) / 0.08) ** 2)
    # mostly flat, a touch of the rounded lighting
    silver = 0.85 * flat + 0.15 * rounded
    mask = np.asarray(wing_mask, float)[..., None] / 255
    image = np.clip(image * (1 - mask) + silver * mask, 0, 255)

    if SPLASH:
        save_splash(image, "Open-source wind tunnel for your desktop", "splash.png")
        Image.open("splash.png").save("../../app/splash.png")

        return
    # rounded square with transparent corners
    alpha = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(alpha).rounded_rectangle([24 * SUPER, 24 * SUPER, SIZE - 24 * SUPER, SIZE - 24 * SUPER], radius=190 * SUPER, fill=255)
    result = Image.fromarray(image.astype(np.uint8)).convert("RGBA")
    result.putalpha(alpha)
    result = result.resize((1024, 1024), Image.LANCZOS)
    result.save("icon_small.png" if SMALL else "icon_full.png")
    # once both renders exist: 16-32 px from the small variant, 48-256 px from the full one
    try:
        full = Image.open("icon_full.png").convert("RGBA")
        small = Image.open("icon_small.png").convert("RGBA")
    except FileNotFoundError:
        return
    frames = {size: (small if size <= 32 else full).resize((size, size), Image.LANCZOS) for size in (16, 24, 32, 48, 64, 128, 256)}
    frames[256].save("../icon.ico", format="ICO", sizes=[(size, size) for size in frames], append_images=[frames[size] for size in (16, 24, 32, 48, 64, 128)])
    frames[256].save("../../app/icon.png")

def saira(size, weight):
    """Saira semi-expanded italic at the given weight."""

    font = ImageFont.truetype("../fonts/Saira-Italic[wdth,wght].ttf", size)
    values = {b"Width": 112.5, b"Weight": weight}
    font.set_variation_by_axes([values[axis["name"]] for axis in font.get_variation_axes()])

    return font


def save_splash(image, tagline, name):
    """Darken behind the wordmark, set Homebrew (silver) CFD (amber) in Saira, round the corners, save splash.png."""

    size = 112 * SUPER
    homebrew_font = saira(size, 600)
    cfd_font = saira(size, 800)
    gap = 26 * SUPER
    homebrew_width = homebrew_font.getlength("Homebrew")
    total = homebrew_width + gap + cfd_font.getlength("CFD")
    right = WIDTH - 70 * SUPER
    baseline = HEIGHT - 120 * SUPER
    left = right - total
    # soft navy scrim so the lines stay behind the text
    scrim = Image.new("L", (WIDTH, HEIGHT), 0)
    ImageDraw.Draw(scrim).rounded_rectangle([left - 40 * SUPER, baseline - size * 1.05, right + 40 * SUPER, baseline + size * 0.75], radius=60 * SUPER, fill=255)
    scrim = np.asarray(scrim.filter(ImageFilter.GaussianBlur(40 * SUPER)), float)[..., None] / 255
    image = image * (1 - 0.9 * scrim) + NAVY * 0.9 * scrim
    result = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).convert("RGBA")
    pen = ImageDraw.Draw(result)
    pen.text((left, baseline), "Homebrew", font=homebrew_font, fill=(214, 217, 223, 255), anchor="ls")
    pen.text((left + homebrew_width + gap, baseline), "CFD", font=cfd_font, fill=(233, 156, 0, 255), anchor="ls")
    alpha = Image.new("L", (WIDTH, HEIGHT), 0)
    ImageDraw.Draw(alpha).rounded_rectangle([0, 0, WIDTH - 1, HEIGHT - 1], radius=24 * SUPER, fill=255)
    result.putalpha(alpha)
    tagline_font = saira(30 * SUPER, 400)
    pen.text((right, baseline + 50 * SUPER), tagline, font=tagline_font, fill=(150, 160, 178, 255), anchor="rs")
    result.resize((WIDTH // SUPER, HEIGHT // SUPER), Image.LANCZOS).save(name)


main()
