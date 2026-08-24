"""
Module 3 — Flux-surface discretisation.

Builds the tomographic/inversion cells as intervals of normalised
poloidal flux psi_N, independent of any line-of-sight geometry.
``get_cell_indices`` is the single source of truth for mapping a
continuous psi_N value to a discrete cell index — Modules 5 and 6
must always resolve cells through this method rather than
re-implementing the search.
"""

from dataclasses import dataclass

import numpy as np

from .equilibrium import MagneticEquilibrium


@dataclass
class FluxGrid:
    """Discretisation of the plasma into shells of constant psi_N."""

    n_shells: int
    psi_edges: np.ndarray
    psi_centres: np.ndarray

    def get_cell_id(self, r: float, z: float, eq: MagneticEquilibrium) -> int:
        """Classify a single (R, Z) coordinate into a shell ID (-1 if outside)."""
        psi_val = eq.psi_interp((r, z))
        if np.isnan(psi_val) or psi_val > 1.0 or psi_val < 0.0:
            return -1
        return int(np.searchsorted(self.psi_edges, psi_val) - 1)

    def get_cell_indices(self, psi_array: np.ndarray) -> np.ndarray:
        """Vectorised classification of psi_N values into cell indices.

        Sole authority for psi_N -> cell mapping in the pipeline;
        Modules 5 and 6 both delegate to this method.
        """
        cells = np.searchsorted(self.psi_edges, psi_array) - 1
        return np.clip(cells, 0, self.n_shells - 1)


def build_flux_grid(n_shells: int = 20) -> FluxGrid:
    """Build a uniform FluxGrid over psi_N in [0, 1]."""
    psi_edges = np.linspace(0.0, 1.0, n_shells + 1)
    psi_centres = 0.5 * (psi_edges[:-1] + psi_edges[1:])
    return FluxGrid(n_shells=n_shells, psi_edges=psi_edges, psi_centres=psi_centres)


def plot_flux_grid(eq: MagneticEquilibrium, grid: FluxGrid, xlim=None, ylim=None):
    """Overlay the discrete psi_N shells on the continuous equilibrium field."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))

    pc = ax.pcolormesh(
        eq.R, eq.Z, eq.psiN.T, cmap="viridis_r", shading="auto", alpha=0.9, vmin=0.0, vmax=1.0
    )
    ax.contour(eq.R, eq.Z, eq.psiN.T, levels=grid.psi_edges, colors="white", linewidths=0.8, alpha=0.5)

    ax.plot(eq.R_boundary, eq.Z_boundary, "r-", linewidth=2, label="LCFS")
    ax.plot(eq.R_limiter, eq.Z_limiter, "k-", label="Limiter")
    ax.scatter(eq.R_axis, eq.Z_axis, marker="+", color="cyan", s=150, zorder=5, label="Axis")

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(f"Flux Grid: {grid.n_shells} shells")
    ax.legend()
    plt.colorbar(pc, label=r"$\psi_N$")

    return fig, ax
