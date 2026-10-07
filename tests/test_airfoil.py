# airfoil geometry: NACA thickness and area, placement, signed distance
import numpy as np
from src.geometry.airfoil import naca_four_digit, place_section, extruded_section_sdf, naca_leading_edge_radius

# test 1: NACA 0012 max thickness is 12% of chord at ~30% chord
def test_naca_0012_thickness():

    x, y = naca_four_digit("0012", 400)
    thickest = np.argmax(y)

    assert abs(2 * y.max() - 0.12) < 1e-3
    assert abs(x[thickest] - 0.30) < 0.02

# test 2: section area ~ 0.685 t c^2 (standard 4-digit result), shoelace formula
def test_naca_0012_area():

    x, y = naca_four_digit("0012", 400)
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    assert abs(area / (0.685 * 0.12) - 1) < 0.01

# test 3: placement gives the requested chord length, lowest point and leading edge
def test_place_section():

    x, y = naca_four_digit("4412", 200)
    placed_x, placed_y = place_section(x, y, chord=80.0, angle_degrees=4.0, leading_edge_x=50.0, lowest_y=12.5)
    leading_edge = 199  # upper surface is TE -> LE, so its last point (point_count - 1) is the (0, 0) leading edge

    assert np.isclose(placed_y.min(), 12.5)
    assert np.isclose(placed_x[leading_edge], 50.0)
    assert abs(np.hypot(placed_x[0] - placed_x[leading_edge], placed_y[0] - placed_y[leading_edge]) - 80.0) < 1e-6

# test 4: signed distance: negative inside, exact distance above the thickest point, independent of z
def test_extruded_section_sdf():

    x, y = naca_four_digit("0012", 400)
    phi = extruded_section_sdf(100.0 * x, 100.0 * y)
    thickest = np.argmax(y)
    top_x, top_y = 100.0 * x[thickest], 100.0 * y[thickest]

    assert phi(50.0, 0.0, 0.0) < 0.0
    assert phi(-10.0, 0.0, 0.0) > 0.0
    assert abs(phi(top_x, top_y + 5.0, 0.0) - 5.0) < 1e-6
    assert np.isclose(phi(top_x, top_y + 5.0, 3.0), phi(top_x, top_y + 5.0, 0.0))

# test 5: leading-edge radius 1.1019 t^2 matches the generated section's nose, y^2 / (2 x) at the first point
def test_leading_edge_radius_matches_section():

    polygon_x, polygon_y = naca_four_digit("0012", point_count=2000)
    nose = np.argmin(np.where(polygon_x > 0.0, polygon_x, np.inf))
    radius_from_points = polygon_y[nose] * polygon_y[nose] / (2.0 * polygon_x[nose])

    assert np.isclose(naca_leading_edge_radius("0012"), 0.015867, atol=1e-6)
    assert abs(radius_from_points / naca_leading_edge_radius("0012") - 1.0) < 0.02