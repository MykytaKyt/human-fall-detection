"""Pure geometric primitives used by the DDL layer: convex hull, point-in-
polygon test, point-to-segment distance, and the signed Margin of Stability.

These functions have no external dependencies beyond numpy and are unit
tested in isolation (see tests/test_geometry.py) since they encode the
core physical logic of the inverted-pendulum model described in the paper.
"""
from __future__ import annotations

import numpy as np


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone-chain convex hull.

    Args:
        points: (n, 2) array of 2D points.

    Returns:
        (m, 2) array of hull vertices in counter-clockwise order
        (m <= n). Returns the input unchanged if it has <= 2 points.
    """
    pts = sorted(map(tuple, points))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def point_in_poly(pt: tuple[float, float], poly: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test (poly given as ordered vertices)."""
    x, y = pt
    n = len(poly)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def dist_point_seg(p, a, b) -> float:
    """Euclidean distance from point p to segment [a, b]."""
    p, a, b = np.asarray(p, dtype=float), np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ab = b - a
    t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def margin_of_stability(com_proj: tuple[float, float], poly: np.ndarray) -> float:
    """Signed Margin of Stability (MoS), Eq. (mos) in the paper.

    Positive when the CoM projection lies inside the base of support
    (poly); negative when outside. Magnitude is the distance to the
    nearest polygon edge.
    """
    if len(poly) < 3:
        return 0.0
    d = min(
        dist_point_seg(com_proj, poly[i], poly[(i + 1) % len(poly)])
        for i in range(len(poly))
    )
    sign = 1.0 if point_in_poly(com_proj, poly) else -1.0
    return sign * d


def polygon_area(poly: np.ndarray) -> float:
    """Shoelace formula for polygon area; 0 for degenerate polygons."""
    if len(poly) < 3:
        return 0.0
    n = len(poly)
    s = sum(
        poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]
        for i in range(n)
    )
    return 0.5 * abs(s)
