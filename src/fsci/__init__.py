"""
FSCI — Flux-Surface Constrained Inversion
==========================================

A pipeline for reconstructing plasma emissivity profiles from
multichannel spectrometer line-of-sight measurements, constrained by
the real magnetic equilibrium (GEQDSK) rather than assumed circular
symmetry (classic onion-peeling / Pearce method).

Architecture
------------
- ``fsci.profiles``          — M1: physical profiles vs. psi_N
- ``fsci.equilibrium``       — M2: GEQDSK magnetic equilibrium
- ``fsci.flux_grid``         — M3: flux-surface discretisation
- ``fsci.optics``            — M4: optical geometry of the LOS array
- ``fsci.geometry_core``     — shared ray-tracing / geometry utilities
- ``fsci.inversion_matrix``  — M5: geometric transfer matrix G
- ``fsci.visualization``     — M6: validation dashboard
- ``fsci.solvers``           — M7: inversion (NNLS / TSVD / Tikhonov)

Physics axiom
-------------
All lines of sight are assumed strictly equatorial (Z = 0) and
perpendicular to their nominal measurement radii, which licenses an
analytic reduction of the 3D diagnostic geometry to the 2D equatorial
plane (X, Y). This is a design assumption of the TCABR diagnostic, not
something derived or verified by the code — see ``fsci.optics.Sightline``.
"""

from .equilibrium import MagneticEquilibrium, load_equilibrium
from .flux_grid import FluxGrid, build_flux_grid
from .optics import OpticalGeometry, Sightline, build_optical_geometry
from .geometry_core import (
    RayTraceResult,
    calculate_ray_limits,
    generate_lcfs_mask,
    trace_single_los,
    calculate_equatorial_lcfs_limits,
)
from .inversion_matrix import InversionMatrix, build_geometric_matrix
from .solvers import (
    InversionResult,
    InverseSolver,
    NNLSSolver,
    TSVDSolver,
    TikhonovSolver,
)
from .validation import run_regression_tests, run_independent_allocation_validation
from .synthetic import build_synthetic_equilibrium, build_synthetic_optics
from .visualization import plot_inversion_validation, launch_interactive_dashboard
from . import profiles, equilibrium, flux_grid, optics, geometry_core, inversion_matrix, visualization, solvers, validation, synthetic

__version__ = "0.1.0"

__all__ = [
    "MagneticEquilibrium",
    "load_equilibrium",
    "FluxGrid",
    "build_flux_grid",
    "OpticalGeometry",
    "Sightline",
    "build_optical_geometry",
    "RayTraceResult",
    "calculate_ray_limits",
    "generate_lcfs_mask",
    "trace_single_los",
    "calculate_equatorial_lcfs_limits",
    "InversionMatrix",
    "build_geometric_matrix",
    "InversionResult",
    "InverseSolver",
    "NNLSSolver",
    "TSVDSolver",
    "TikhonovSolver",
    "run_regression_tests",
    "run_independent_allocation_validation",
    "build_synthetic_equilibrium",
    "build_synthetic_optics",
    "plot_inversion_validation",
    "launch_interactive_dashboard",
]
