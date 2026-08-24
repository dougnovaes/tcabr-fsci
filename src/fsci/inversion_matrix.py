"""
Module 5 — Continuous ray tracing and the geometric inversion matrix G.

G_ji = integral over LOS_j of H_i(psi_N) ds

No onion-peeling, no tangency assumption, no triangular matrix: each
channel's line of sight is traced continuously through the equatorial
plane, mapped onto the real magnetic equilibrium via psi_N(R, 0), and
accumulated into whichever flux-surface cell (Module 3) each sampled
point belongs to.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from .equilibrium import MagneticEquilibrium
from .flux_grid import FluxGrid
from .optics import OpticalGeometry
from .geometry_core import calculate_ray_limits, trace_single_los


@dataclass
class InversionMatrix:
    """The geometric transfer matrix G and its integration metadata.

    ``length_in_plasma`` is defined as ``sum(G[j, :])`` by
    construction (not an independent geometric count) — this
    guarantees exact conservation between the matrix and the metadata
    and eliminates silent floating-point drift, at the cost of the
    metric no longer being a purely geometric measurement. See
    ``tests/test_inversion_matrix.py`` for the independent geometric
    cross-check against the LCFS mask alone.
    """

    data: np.ndarray              # G, shape (n_channels, n_shells)
    ds: float                     # Spatial integration step (m)
    samples_per_los: np.ndarray   # Valid integration samples per chord
    length_in_plasma: np.ndarray  # Length effectively allocated in G (m)
    length_out_plasma: np.ndarray  # Length outside the allocation but inside bounds (m)


def build_geometric_matrix(
    optics: OpticalGeometry,
    grid: FluxGrid,
    eq: MagneticEquilibrium,
    ds_custom: float = None,
) -> InversionMatrix:
    """Build G using high-resolution continuous ray tracing.

    The integration step defaults to half the equilibrium mesh
    resolution (``0.5 * min(dR, dZ)``), which guarantees no flux-grid
    cell is skipped during sampling; pass ``ds_custom`` to override.
    """
    n_channels = len(optics.sightlines)
    n_shells = grid.n_shells
    G = np.zeros((n_channels, n_shells))

    dR = np.abs(eq.R[1] - eq.R[0])
    dZ = np.abs(eq.Z[1] - eq.Z[0])
    ds = ds_custom if ds_custom is not None else 0.5 * min(dR, dZ)

    samples_per_los = np.zeros(n_channels, dtype=int)
    length_in = np.zeros(n_channels)
    length_out = np.zeros(n_channels)

    r_max = np.max(eq.R_boundary)

    for j, sl in enumerate(optics.sightlines):
        limits = calculate_ray_limits(sl.origin, sl.direction, r_max)
        if limits is None:
            warnings.warn(f"LOS {sl.id} does not intersect the machine bounding geometry.")
            continue

        s_start, s_end = limits
        s_array_len = len(np.arange(s_start, s_end, ds))

        ray_result = trace_single_los(sl.origin, sl.direction, limits, eq, ds)
        if ray_result is None:
            warnings.warn(f"LOS {sl.id} does not intersect the plasma volume (LCFS).")
            length_out[j] = s_array_len * ds
            continue

        cells = grid.get_cell_indices(ray_result.psi)
        counts = np.bincount(cells, minlength=n_shells)
        G[j, :] = counts * ds

        assigned_length = np.sum(G[j, :])
        length_in[j] = assigned_length
        length_out[j] = (s_array_len * ds) - assigned_length
        samples_per_los[j] = len(ray_result.s)

    return InversionMatrix(
        data=G,
        ds=ds,
        samples_per_los=samples_per_los,
        length_in_plasma=length_in,
        length_out_plasma=length_out,
    )
