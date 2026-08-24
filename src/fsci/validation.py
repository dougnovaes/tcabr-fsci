"""
Validation utilities for the inversion matrix G.

Two distinct checks, deliberately not conflated:

- :func:`run_regression_tests` — physical/geometric invariants
  (positivity, length conservation, psi_N ~ 0 at the magnetic axis,
  core-vs-edge monotonicity anchored on the physical impact parameter
  ``d``). These are invariant to plasma shape, so they survive future
  GEQDSK files that look nothing like TCABR's current equilibrium.

- :func:`run_independent_allocation_validation` — compares the length
  allocated in G against a strict geometric count based solely on the
  LCFS polygon, bypassing the psi_N interpolator. This is independent
  of the interpolator and of cell classification, but it still shares
  ``calculate_ray_limits``/``generate_lcfs_mask`` with the matrix
  builder, so it is *not* an independent check of the ray-tracing
  geometry itself — only of whether the interpolation/classification
  stage silently drops points the geometry considers valid.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .equilibrium import MagneticEquilibrium
from .optics import OpticalGeometry
from .inversion_matrix import InversionMatrix
from .geometry_core import calculate_ray_limits, generate_lcfs_mask


@dataclass
class RegressionReport:
    passed: bool
    messages: List[str] = field(default_factory=list)


def run_regression_tests(
    optics: OpticalGeometry,
    eq: MagneticEquilibrium,
    inversion_matrix: InversionMatrix,
    expected_shape: tuple = None,
) -> RegressionReport:
    """Run invariant-based regression checks. Raises AssertionError on failure."""
    G = inversion_matrix.data
    messages = []

    assert np.all(G >= 0), "Matrix G contains negative path lengths."

    row_sums = np.sum(G, axis=1)
    assert np.allclose(row_sums, inversion_matrix.length_in_plasma, atol=1e-12), (
        "Length conservation broken: matrix row sums diverge from metadata."
    )

    psi_axis_eval = float(eq.psi_interp((eq.R_axis, eq.Z_axis)).item())
    assert psi_axis_eval < 0.02, f"psi_N at magnetic axis is {psi_axis_eval:.5f} (expected near 0.0)."

    if len(optics.sightlines) > 1 and np.any(inversion_matrix.length_in_plasma > 0):
        impact_radii = np.array([sl.d for sl in optics.sightlines])
        idx_core = int(np.argmin(np.abs(impact_radii - eq.R_axis)))
        idx_edge = int(np.argmax(impact_radii))
        core_len = inversion_matrix.length_in_plasma[idx_core]
        edge_len = inversion_matrix.length_in_plasma[idx_edge]
        assert core_len >= edge_len, (
            f"Monotonicity broken: edge channel {idx_edge + 1} ({edge_len:.3f} m) "
            f"exceeds core-aligned channel {idx_core + 1} ({core_len:.3f} m)."
        )

    if expected_shape is not None and G.shape != expected_shape:
        messages.append(f"Matrix shape drift: expected {expected_shape}, got {G.shape}.")

    return RegressionReport(passed=True, messages=messages)


@dataclass
class AllocationValidationReport:
    passed: bool
    max_divergence: float
    geometric_lengths: np.ndarray
    matrix_lengths: np.ndarray


def run_independent_allocation_validation(
    optics: OpticalGeometry,
    eq: MagneticEquilibrium,
    inversion_matrix: InversionMatrix,
) -> AllocationValidationReport:
    """Compare G's allocated length against a pure LCFS-polygon count."""
    n_channels = len(optics.sightlines)
    ds = inversion_matrix.ds
    r_max = np.max(eq.R_boundary)

    geometric_lengths = np.zeros(n_channels)
    matrix_lengths = inversion_matrix.length_in_plasma

    for j, sl in enumerate(optics.sightlines):
        limits = calculate_ray_limits(sl.origin, sl.direction, r_max)
        if limits is None:
            continue
        s_start, s_end = limits
        s_array = np.arange(s_start, s_end, ds)
        if len(s_array) == 0:
            continue

        X_path = sl.origin[0] + s_array * sl.direction[0]
        Y_path = sl.origin[1] + s_array * sl.direction[1]
        R_path = np.hypot(X_path, Y_path)
        Z_path = np.zeros_like(R_path)

        inside_lcfs_mask = generate_lcfs_mask(R_path, Z_path, eq)
        geometric_lengths[j] = np.sum(inside_lcfs_mask) * ds

    divergences = np.abs(geometric_lengths - matrix_lengths)
    max_divergence = float(np.max(divergences))

    return AllocationValidationReport(
        passed=bool(np.isclose(max_divergence, 0.0, atol=1e-12)),
        max_divergence=max_divergence,
        geometric_lengths=geometric_lengths,
        matrix_lengths=matrix_lengths,
    )
