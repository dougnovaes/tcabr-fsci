"""
Module 2 — Magnetic Equilibrium (GEQDSK).

Reads a GEQDSK g-file and produces a normalised representation of the
magnetic equilibrium: (R, Z) mesh, poloidal flux psi, normalised flux
psi_N, magnetic axis, LCFS, limiter, and a continuous interpolator for
psi_N(R, Z).

Architectural invariant (do not change without re-reading the project
history): psi / psi_N are stored natively as returned by freeqdsk, with
shape (len(R), len(Z)). The ``.T`` transpose is used *exclusively*
inside plotting calls (Matplotlib's ``contour``/``pcolormesh`` expect
the (Ny, Nx) convention) and never touches the stored arrays or the
interpolator. An earlier attempt to "unify" this by transposing at
read time silently broke the pipeline on TCABR's square (257x257)
equilibrium mesh, because ``shape == (len(R), len(Z))`` is vacuously
true whenever nR == nZ. See project history for the full post-mortem.
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

try:
    import freeqdsk.geqdsk
except ImportError as exc:  # pragma: no cover - environment guidance only
    raise ImportError(
        "freeqdsk is required to read GEQDSK files. Install it with "
        "`pip install freeqdsk`."
    ) from exc


@dataclass
class MagneticEquilibrium:
    """Container for a magnetic equilibrium read from a GEQDSK file."""

    R: np.ndarray
    Z: np.ndarray
    psi: np.ndarray
    psiN: np.ndarray
    psi_axis: float
    psi_lcfs: float
    R_axis: float
    Z_axis: float
    R_geo: float
    Z_geo: float
    shafranov_shift: float
    R_boundary: np.ndarray
    Z_boundary: np.ndarray
    R_limiter: np.ndarray
    Z_limiter: np.ndarray
    psi_interp: RegularGridInterpolator


def load_equilibrium(filename: str) -> MagneticEquilibrium:
    """Read a GEQDSK g-file and build a :class:`MagneticEquilibrium`."""
    with open(filename, "r") as f:
        eq = freeqdsk.geqdsk.read(f)

    # R is index 0 (nw), Z is index 1 (nh) — native freeqdsk convention.
    R = np.linspace(eq["rleft"], eq["rleft"] + eq["rdim"], eq["nw"])
    Z = np.linspace(eq["zmid"] - eq["zdim"] / 2, eq["zmid"] + eq["zdim"] / 2, eq["nh"])

    psi = eq["psi"]
    psi_axis = eq["simagx"]
    psi_lcfs = eq["sibdry"]

    psiN = (psi - psi_axis) / (psi_lcfs - psi_axis)
    psiN = np.clip(psiN, 0.0, 1.1)

    R_axis, Z_axis = eq["rmaxis"], eq["zmaxis"]
    R_boundary, Z_boundary = np.asarray(eq["rbdry"]), np.asarray(eq["zbdry"])
    R_limiter, Z_limiter = np.asarray(eq["rlim"]), np.asarray(eq["zlim"])

    R_geo = 0.5 * (R_boundary.min() + R_boundary.max())
    Z_geo = 0.5 * (Z_boundary.min() + Z_boundary.max())
    shafranov_shift = R_axis - R_geo

    # Interpolator consumes psiN natively as (R, Z) — no transpose here.
    psi_interp = RegularGridInterpolator((R, Z), psiN, bounds_error=False, fill_value=None)

    return MagneticEquilibrium(
        R=R, Z=Z, psi=psi, psiN=psiN, psi_axis=psi_axis, psi_lcfs=psi_lcfs,
        R_axis=R_axis, Z_axis=Z_axis, R_geo=R_geo, Z_geo=Z_geo,
        shafranov_shift=shafranov_shift, R_boundary=R_boundary, Z_boundary=Z_boundary,
        R_limiter=R_limiter, Z_limiter=Z_limiter, psi_interp=psi_interp,
    )


def plot_equilibrium(eq: MagneticEquilibrium, xlim=None, ylim=None):
    """Plot psi_N contours, LCFS, limiter, magnetic axis and geometric centre.

    ``xlim``/``ylim`` default to None (Matplotlib auto-scales); pass
    e.g. ``xlim=(0.4, 0.85), ylim=(-0.3, 0.3)`` to reproduce the TCABR
    zoomed view used during development.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    levels = np.linspace(0, 1, 21)

    # .T is the Presentation Layer adapter for Matplotlib's (Ny, Nx)
    # requirement — it must never leak into MagneticEquilibrium.psiN.
    c = ax.contour(eq.R, eq.Z, eq.psiN.T, levels=levels, cmap="viridis")

    ax.plot(eq.R_boundary, eq.Z_boundary, color="red", linewidth=2, label="LCFS")
    ax.plot(eq.R_limiter, eq.Z_limiter, color="black", linewidth=2, label="Limiter")
    ax.plot(eq.R_axis, eq.Z_axis, "+", color="blue", markersize=14, mew=3, label="Magnetic axis")
    ax.plot(eq.R_geo, eq.Z_geo, "+", color="black", markersize=14, mew=3, label="Geometric centre")

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Magnetic Equilibrium")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.colorbar(c, label=r"$\psi_N$")

    return fig, ax
