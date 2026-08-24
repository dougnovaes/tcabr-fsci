"""
Module 4 — Optical geometry.

Defines the diagnostic's lines of sight (LOS) in the equatorial plane
(X, Y), from the hardware impact parameters alone — independent of any
magnetic equilibrium.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

# TCABR reference geometry (defaults for plotting only; override for
# other machines or other diagnostic layouts).
TCABR_R0 = 0.615      # Major radius (m)
TCABR_A_MINOR = 0.17  # Minor radius (m)


@dataclass
class Sightline:
    """A single diagnostic line of sight.

    Physics axiom
    -------------
    This project assumes that all sightlines are strictly equatorial
    (Z = 0) and perpendicular to the nominal radii of observation.
    Consequently, the 3D diagnostic geometry is analytically reduced
    to a 2D plane (X, Y). This is a *design assumption* of the TCABR
    diagnostic, not something derived or verified by this code.
    """

    id: int
    d: float               # Impact parameter (min. distance to the machine axis)
    angle: float            # Geometric orientation (radians)
    origin: np.ndarray       # [XC, YC]
    direction: np.ndarray    # Unit vector [ux, uy]
    tangent_point: np.ndarray  # Geometric tangency to radius d (hardware validation only)

    def get_point(self, s: float) -> np.ndarray:
        """Parametric line equation: P(s) = origin + s * direction."""
        return self.origin + s * self.direction


@dataclass
class OpticalGeometry:
    """Container for the full diagnostic assembly.

    Relies on the 3D -> 2D equatorial reduction axiom documented on
    :class:`Sightline`.
    """

    origin: np.ndarray
    sightlines: List[Sightline]


def build_optical_geometry(xc: float, yc: float, hardware_params: np.ndarray) -> OpticalGeometry:
    """Construct the LOS objects from pre-defined hardware impact parameters.

    Parameters
    ----------
    xc, yc : float
        Collector position in the equatorial plane.
    hardware_params : np.ndarray
        Impact parameter ``d`` for each channel, ordered arbitrarily
        (no assumption is made elsewhere in the pipeline about
        core-to-edge ordering; where that matters — e.g. regression
        tests — the physical impact parameter itself is used, not the
        array index).
    """
    origin = np.array([xc, yc])
    dist_to_axis = np.hypot(xc, yc)
    phi = np.arctan2(yc, xc)

    sightlines = []
    for i, d in enumerate(hardware_params):
        alpha = phi - np.arccos(d / dist_to_axis)
        direction = np.array([np.sin(alpha), -np.cos(alpha)])

        s_tangent = np.sqrt(max(0.0, dist_to_axis**2 - d**2))
        tangent_point = origin + s_tangent * direction

        sightlines.append(
            Sightline(
                id=i + 1, d=d, angle=alpha, origin=origin,
                direction=direction, tangent_point=tangent_point,
            )
        )

    return OpticalGeometry(origin=origin, sightlines=sightlines)


def plot_optical_geometry(
    optics: OpticalGeometry,
    selected_id: int = 1,
    background_img=None,
    r0: float = TCABR_R0,
    a_minor: float = TCABR_A_MINOR,
):
    """Plot the LOS fan in the equatorial plane, highlighting one channel."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_length = 1.8

    if background_img is not None:
        ax.imshow(background_img, extent=[-1, 1, -1, 1], alpha=0.7)

    ax.add_patch(plt.Circle((0, 0), r0, color="black", fill=False, ls=":", alpha=0.5, label=f"R0 ({r0}m)"))
    ax.add_patch(plt.Circle((0, 0), r0 - a_minor, color="gray", fill=False, ls="--", alpha=0.3))
    ax.add_patch(plt.Circle((0, 0), r0 + a_minor, color="gray", fill=False, ls="--", alpha=0.3, label=f"a_minor ({a_minor}m)"))

    for sl in optics.sightlines:
        is_selected = sl.id == selected_id
        color, alpha, lw = ("yellow", 1.0, 2.0) if is_selected else ("white", 0.3, 0.5)

        p_start = sl.origin
        p_end = sl.get_point(plot_length)
        ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], color=color, alpha=alpha, lw=lw)

        if is_selected:
            ax.scatter(*sl.tangent_point, color="yellow", s=50, edgecolors="black", zorder=10, label=f"Tangent d={sl.d:.3f}m")
            ax.plot([0, sl.tangent_point[0]], [0, sl.tangent_point[1]], "y:", lw=1.2)

    ax.scatter(*optics.origin, color="white", edgecolors="black", s=100, zorder=15, label="Collector (XC, YC)")

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 0.2)
    ax.set_aspect("equal")
    ax.set_title("Hardware-Validated Optical Geometry")
    ax.legend(loc="lower left", fontsize=9)

    return fig, ax
