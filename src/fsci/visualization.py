"""
Module 6 — Visual validation of the inversion matrix G.

Audits the G matrix produced by Module 5 by re-tracing one selected
channel and overlaying it on the equilibrium and on G itself. This
module never reconstructs the plasma and never imports Module 5's
internals — it consumes the same Core Geometry Utilities that Module 5
does, independently, so that it is a genuine second opinion rather
than a restatement of Module 5's own bookkeeping.

The (R, Z) panel is deliberately labelled "Radial Sampling Diagnostic",
not a line-of-sight trajectory: because all LOS are equatorial (see
``fsci.optics.Sightline``), what is plotted there is the sequence of
psi_N(R, 0) samples visited by the chord, not a physical path through
the poloidal cross-section.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

from .equilibrium import MagneticEquilibrium
from .flux_grid import FluxGrid
from .optics import OpticalGeometry
from .geometry_core import calculate_ray_limits, trace_single_los, calculate_equatorial_lcfs_limits
from .inversion_matrix import InversionMatrix


def plot_inversion_validation(
    optics: OpticalGeometry,
    grid: FluxGrid,
    eq: MagneticEquilibrium,
    inversion_matrix: InversionMatrix,
    selected_channel: int = 1,
    bg_image_path: str = None,
):
    """Render the 4-panel validation dashboard for one selected channel."""
    if not (1 <= selected_channel <= len(optics.sightlines)):
        raise ValueError(f"Selected channel {selected_channel} is out of bounds.")

    j = selected_channel - 1
    sl = optics.sightlines[j]
    G = inversion_matrix.data
    ds = inversion_matrix.ds

    r_max = np.max(eq.R_boundary)
    trace_limits = calculate_ray_limits(sl.origin, sl.direction, r_max)
    ray_result = trace_single_los(sl.origin, sl.direction, trace_limits, eq, ds) if trace_limits else None

    fig, axs = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(f"Module 6: Geometric Validation - Channel {selected_channel}", fontsize=16)

    cmap = plt.get_cmap("turbo", grid.n_shells)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, grid.n_shells + 0.5, 1), cmap.N)

    # Panel 1 — full G heatmap
    ax1 = axs[0, 0]
    im = ax1.imshow(G, aspect="auto", cmap="plasma", origin="upper")
    ax1.set_title("Inversion Matrix G")
    ax1.set_xlabel("Cell Index (Core -> Edge)")
    ax1.set_ylabel("Channel Index")
    fig.colorbar(im, ax=ax1, label="Path Length \u0394s (m)")
    ax1.axhline(y=j, color="white", linestyle="--", linewidth=1.5)

    # Panel 2 — length per cell, selected channel
    ax2 = axs[0, 1]
    ax2.bar(range(grid.n_shells), G[j, :], color=cmap(range(grid.n_shells)), edgecolor="black")
    ax2.set_title(f"Path Length per Cell (Channel {selected_channel})")
    ax2.set_xlabel("Cell Index")
    ax2.set_ylabel("Length (m)")
    ax2.set_xticks(range(0, grid.n_shells, max(1, grid.n_shells // 10)))
    ax2.set_ylim(0, max(np.max(G) * 1.1, 1e-6))

    # Panel 3 — radial sampling diagnostic (NOT a physical LOS trajectory)
    ax3 = axs[1, 0]
    ax3.pcolormesh(eq.R, eq.Z, eq.psiN.T, cmap="viridis_r", shading="auto", alpha=0.2, vmin=0, vmax=1)
    ax3.contour(eq.R, eq.Z, eq.psiN.T, levels=grid.psi_edges, cmap="viridis_r", alpha=0.4)
    ax3.plot(eq.R_boundary, eq.Z_boundary, "r-", linewidth=2, label="LCFS")
    ax3.plot(eq.R_limiter, eq.Z_limiter, "k-", linewidth=1.5, label="Limiter")

    if ray_result is not None and len(ray_result.R) > 0:
        cells = grid.get_cell_indices(ray_result.psi)
        points = np.array([ray_result.R, ray_result.Z]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=3)
        lc.set_array(cells[:-1])
        ax3.add_collection(lc)

    ax3.set_aspect("equal")
    ax3.set_title("Radial Sampling Diagnostic (R, Z)")
    ax3.set_xlabel("R (m)")
    ax3.set_ylabel("Z (m)")
    ax3.legend(loc="upper right", fontsize="small")

    # Panel 4 — equatorial projection (the physically real LOS plane)
    ax4 = axs[1, 1]

    if bg_image_path and os.path.exists(bg_image_path):
        import matplotlib.image as mpimg
        img = mpimg.imread(bg_image_path)
        ax4.imshow(img, extent=[-1.0, 1.0, -1.0, 1.0], alpha=0.7, zorder=0)
    else:
        ax4.set_facecolor("#5a606b")

    R_in, R_out = calculate_equatorial_lcfs_limits(eq)
    ax4.add_patch(plt.Circle((0, 0), R_in, color="red", fill=False, ls="--", alpha=0.7, zorder=1))
    ax4.add_patch(plt.Circle((0, 0), R_out, color="red", fill=False, ls="-", lw=2, label="Equatorial radial extent (Z=0)", zorder=1))

    for s_global in optics.sightlines:
        lims = calculate_ray_limits(s_global.origin, s_global.direction, R_out)
        s_g_end = lims[1] if lims is not None else 1.5
        p_g_end = s_global.get_point(s_g_end)
        ax4.plot([s_global.origin[0], p_g_end[0]], [s_global.origin[1], p_g_end[1]], color="white", lw=0.2, alpha=0.3, zorder=2)

    vis_limits = calculate_ray_limits(sl.origin, sl.direction, R_out)
    s_end_vis = vis_limits[1] if vis_limits is not None else 1.5
    p_end = sl.get_point(s_end_vis)
    ax4.plot([sl.origin[0], p_end[0]], [sl.origin[1], p_end[1]], color="yellow", ls=":", lw=1.2, zorder=3, label=f"LOS {selected_channel}")

    if ray_result is not None and len(ray_result.X) > 0:
        cells = grid.get_cell_indices(ray_result.psi)
        points_xy = np.array([ray_result.X, ray_result.Y]).T.reshape(-1, 1, 2)
        segments_xy = np.concatenate([points_xy[:-1], points_xy[1:]], axis=1)
        lc_xy = LineCollection(segments_xy, cmap=cmap, norm=norm, linewidth=2.5, zorder=4)
        lc_xy.set_array(cells[:-1])
        ax4.add_collection(lc_xy)

    ax4.scatter(*sl.origin, color="white", edgecolors="black", s=80, zorder=5, label="Collector")
    ax4.set_xlim(-1.0, 1.0)
    ax4.set_ylim(-1.0, 1.0)
    ax4.set_aspect("equal")
    ax4.set_title("Equatorial Projection (X, Y)")
    ax4.set_xlabel("X (m)")
    ax4.set_ylabel("Y (m)")
    ax4.legend(loc="upper right", fontsize="small")

    fig.tight_layout()
    return fig, axs


def launch_interactive_dashboard(optics, grid, eq, inversion_matrix, bg_image: str = None):
    """Jupyter-only: wrap :func:`plot_inversion_validation` in an ipywidgets slider.

    Requires ``ipywidgets`` and a running IPython/Jupyter kernel;
    imported lazily so the rest of the package has no hard dependency
    on it.
    """
    from ipywidgets import interact, IntSlider, fixed

    n_channels = len(optics.sightlines)
    channel_slider = IntSlider(min=1, max=n_channels, step=1, value=1, description="Channel:", continuous_update=False)
    interact(
        plot_inversion_validation,
        optics=fixed(optics), grid=fixed(grid), eq=fixed(eq),
        inversion_matrix=fixed(inversion_matrix),
        selected_channel=channel_slider, bg_image_path=fixed(bg_image),
    )
