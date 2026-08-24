"""
Core Geometry Utilities.

Ray-tracing and geometric intersection routines shared by Module 5
(matrix construction) and Module 6 (visual validation). Neither module
depends on the other's implementation — both consume this layer
independently, which is what makes the M6 dashboard a genuine
independent audit of the M5 matrix rather than a duplicate of its
internals.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from matplotlib.path import Path

from .equilibrium import MagneticEquilibrium


@dataclass
class RayTraceResult:
    """Sampled physical parameters along a single LOS, restricted to
    the segment that lies inside the LCFS and has a valid psi_N."""

    s: np.ndarray    # Arc length parameter (m)
    X: np.ndarray    # X coordinates, equatorial plane
    Y: np.ndarray    # Y coordinates, equatorial plane
    R: np.ndarray    # R coordinates (axisymmetric projection)
    Z: np.ndarray    # Z coordinates (axisymmetric projection)
    psi: np.ndarray  # Valid psi_N values along the sampled path


def calculate_ray_limits(
    origin: np.ndarray, direction: np.ndarray, r_max: float
) -> Optional[Tuple[float, float]]:
    """Exact intersection of a ray with the bounding cylinder r_max.

    Returns (s_start, s_end) or None if there is no forward intersection.
    """
    P_dot_v = origin[0] * direction[0] + origin[1] * direction[1]
    P_sq = origin[0] ** 2 + origin[1] ** 2
    discriminant = P_dot_v**2 - (P_sq - r_max**2)

    if discriminant <= 0:
        return None

    sqrt_disc = np.sqrt(discriminant)
    s_start = max(0.0, -P_dot_v - sqrt_disc)
    s_end = max(0.0, -P_dot_v + sqrt_disc)

    if s_start >= s_end:
        return None

    return s_start, s_end


def generate_lcfs_mask(R_points: np.ndarray, Z_points: np.ndarray, eq: MagneticEquilibrium) -> np.ndarray:
    """Boolean mask for coordinates inside the physical LCFS polygon."""
    lcfs_points = np.column_stack((eq.R_boundary, eq.Z_boundary))
    lcfs_path = Path(lcfs_points)
    points_rz = np.column_stack((R_points, Z_points))
    return lcfs_path.contains_points(points_rz)


def trace_single_los(
    origin: np.ndarray,
    direction: np.ndarray,
    limits: Tuple[float, float],
    eq: MagneticEquilibrium,
    ds: float,
) -> Optional[RayTraceResult]:
    """Sample a single LOS within precomputed limits and restrict it to
    the segment inside the LCFS with a valid interpolated psi_N.

    ``limits`` must come from :func:`calculate_ray_limits` — callers
    own that computation so it is not duplicated here.
    """
    s_start, s_end = limits
    s_array = np.arange(s_start, s_end, ds)

    if len(s_array) == 0:
        return None

    X_path = origin[0] + s_array * direction[0]
    Y_path = origin[1] + s_array * direction[1]
    R_path = np.hypot(X_path, Y_path)
    Z_path = np.zeros_like(R_path)

    inside_lcfs_mask = generate_lcfs_mask(R_path, Z_path, eq)
    if np.sum(inside_lcfs_mask) == 0:
        return None

    s_valid = s_array[inside_lcfs_mask]
    X_valid = X_path[inside_lcfs_mask]
    Y_valid = Y_path[inside_lcfs_mask]
    R_valid = R_path[inside_lcfs_mask]
    Z_valid = Z_path[inside_lcfs_mask]

    psi_valid = eq.psi_interp((R_valid, Z_valid))
    valid_psi_mask = (~np.isnan(psi_valid)) & (psi_valid >= 0.0)

    if np.sum(valid_psi_mask) == 0:
        return None

    return RayTraceResult(
        s=s_valid[valid_psi_mask],
        X=X_valid[valid_psi_mask],
        Y=Y_valid[valid_psi_mask],
        R=R_valid[valid_psi_mask],
        Z=Z_valid[valid_psi_mask],
        psi=psi_valid[valid_psi_mask],
    )


def calculate_equatorial_lcfs_limits(eq: MagneticEquilibrium) -> Tuple[float, float]:
    """Exact inner/outer radii (R_in, R_out) where the LCFS polygon
    intersects the equatorial plane (Z = 0), including the closing
    segment between the last and first boundary points."""
    R_closed = np.append(eq.R_boundary, eq.R_boundary[0])
    Z_closed = np.append(eq.Z_boundary, eq.Z_boundary[0])

    r_crossings = []

    exact_zeros = np.where(Z_closed == 0)[0]
    for idx in exact_zeros:
        r_crossings.append(R_closed[idx])

    z_signs = np.sign(Z_closed)
    crossings = np.where(z_signs[:-1] != z_signs[1:])[0]

    for i in crossings:
        r1, z1 = R_closed[i], Z_closed[i]
        r2, z2 = R_closed[i + 1], Z_closed[i + 1]
        if z1 != z2:
            r_zero = r1 - z1 * (r2 - r1) / (z2 - z1)
            r_crossings.append(r_zero)

    if not r_crossings:
        return np.min(eq.R_boundary), np.max(eq.R_boundary)

    return np.min(r_crossings), np.max(r_crossings)
