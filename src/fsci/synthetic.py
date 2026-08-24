"""
Synthetic equilibrium and optics builders.

These construct a simple circular ``MagneticEquilibrium`` and a
matching ``OpticalGeometry`` directly in memory, without reading a
GEQDSK file. Two uses:

1. The test suite exercises the geometry / matrix / solver logic
   against this fixture, independent of any real TCABR data file.
2. Anyone cloning this repository without access to TCABR's actual
   g-file can still run the full pipeline end to end.

Do not expect the numeric values produced here (e.g. channel path
lengths) to match TCABR's real equilibrium — this is a deliberately
simple circular case for testing and demonstration only.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .equilibrium import MagneticEquilibrium
from .optics import OpticalGeometry, build_optical_geometry


def build_synthetic_equilibrium(
    R0: float = 0.6, Z0: float = 0.0, a_minor: float = 0.2, n_mesh: int = 61
) -> MagneticEquilibrium:
    """A simple circular, concentric equilibrium: psi_N = ((R-R0)^2 + (Z-Z0)^2) / a_minor^2."""
    R = np.linspace(R0 - 1.5 * a_minor, R0 + 1.5 * a_minor, n_mesh)
    Z = np.linspace(Z0 - 1.5 * a_minor, Z0 + 1.5 * a_minor, n_mesh)
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")  # shape (len(R), len(Z))

    psiN = ((RR - R0) ** 2 + (ZZ - Z0) ** 2) / a_minor**2
    psiN = np.clip(psiN, 0.0, 1.1)
    psi = psiN.copy()  # psi_axis = 0.0, psi_lcfs = 1.0 by construction

    theta = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    R_boundary = R0 + a_minor * np.cos(theta)
    Z_boundary = Z0 + a_minor * np.sin(theta)

    theta_lim = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    R_limiter = R0 + (a_minor + 0.03) * np.cos(theta_lim)
    Z_limiter = Z0 + (a_minor + 0.03) * np.sin(theta_lim)

    psi_interp = RegularGridInterpolator((R, Z), psiN, bounds_error=False, fill_value=None)

    return MagneticEquilibrium(
        R=R, Z=Z, psi=psi, psiN=psiN, psi_axis=0.0, psi_lcfs=1.0,
        R_axis=R0, Z_axis=Z0, R_geo=R0, Z_geo=Z0, shafranov_shift=0.0,
        R_boundary=R_boundary, Z_boundary=Z_boundary,
        R_limiter=R_limiter, Z_limiter=Z_limiter, psi_interp=psi_interp,
    )


def build_synthetic_optics(
    eq: MagneticEquilibrium, xc: float = 0.3, yc: float = -1.0, n_channels: int = 8
) -> OpticalGeometry:
    """A small LOS fan aimed roughly across the synthetic plasma's core-to-edge range."""
    span = 0.9 * (np.max(eq.R_boundary) - eq.R_axis)
    hardware_params = np.linspace(eq.R_axis - span, eq.R_axis + span, n_channels)
    return build_optical_geometry(xc, yc, hardware_params)
