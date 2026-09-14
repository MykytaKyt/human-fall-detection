import numpy as np
import pytest

from urfd_cascade.geometry import (
    convex_hull,
    dist_point_seg,
    margin_of_stability,
    point_in_poly,
    polygon_area,
)


def test_convex_hull_square():
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]])  # interior point
    hull = convex_hull(pts)
    assert len(hull) == 4  # interior point excluded
    assert set(map(tuple, hull)) == {(0, 0), (1, 0), (1, 1), (0, 1)}


def test_convex_hull_degenerate():
    assert len(convex_hull(np.array([[0, 0]]))) == 1
    assert len(convex_hull(np.array([[0, 0], [1, 1]]))) == 2


def test_point_in_poly_inside_and_outside():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    assert point_in_poly((0.5, 0.5), square) is True
    assert point_in_poly((2.0, 2.0), square) is False


def test_point_in_poly_too_few_vertices():
    assert point_in_poly((0.0, 0.0), np.array([[0, 0], [1, 1]])) is False


def test_dist_point_seg_endpoints_and_midpoint():
    a, b = (0, 0), (10, 0)
    assert dist_point_seg((0, 0), a, b) == pytest.approx(0.0)
    assert dist_point_seg((5, 5), a, b) == pytest.approx(5.0)
    assert dist_point_seg((-5, 0), a, b) == pytest.approx(5.0)  # clamped to endpoint


def test_margin_of_stability_sign():
    # BoS = unit square; CoM projection inside -> positive MoS
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    assert margin_of_stability((0.5, 0.5), square) > 0
    # CoM projection outside -> negative MoS (instability, Sec. 3.1 of paper)
    assert margin_of_stability((2.0, 0.5), square) < 0


def test_margin_of_stability_degenerate_polygon():
    assert margin_of_stability((0.5, 0.5), np.array([])) == 0.0


def test_polygon_area_unit_square():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    assert polygon_area(square) == pytest.approx(1.0)


def test_polygon_area_degenerate():
    assert polygon_area(np.array([[0, 0], [1, 1]])) == 0.0
