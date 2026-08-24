"""
Module 1 — Plasma profiles.

Loads and organises physical profiles (densities, temperatures,
current, pressure, ExB rotation, Zeff, fz) as a function of the
normalised poloidal flux psi_N. Purely organisational — no geometry,
no inversion logic.

Expects whitespace-delimited two-column files (psi_N, value) named
``profile_<quantity>`` inside ``data_dir``, as produced by the Plasma
Scenario Design (PSD) tool.
"""

import os

import numpy as np

PROFILE_GROUPS = {
    "Densities": {"left": ["profile_ne", "profile_ni"], "right": ["profile_nfe", "profile_nc"]},
    "Temperatures": {"left": ["profile_te", "profile_ti"], "right": []},
    "Electromagnetism/Pressure": {"left": ["profile_jpar", "profile_p"], "right": []},
    "Rotation (ExB)": {"left": ["profile_omega_exb"], "right": []},
}


def load_profile(data_dir: str, filename: str) -> np.ndarray:
    """Load a single two-column (psi_N, value) profile file."""
    return np.loadtxt(os.path.join(data_dir, filename))


def load_constant(data_dir: str, filename: str):
    """Load a profile file that stores a single constant value (Zeff, fz)."""
    try:
        return load_profile(data_dir, filename)[0, 1]
    except (FileNotFoundError, OSError):
        return None


def plot_profiles(data_dir: str, groups: dict = None):
    """Plot the grouped profiles in a 2x2 panel, as a function of psi_N."""
    import matplotlib.pyplot as plt

    groups = groups or PROFILE_GROUPS
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    axs = axs.flatten()

    for i, (title, mapping) in enumerate(groups.items()):
        ax_left = axs[i]
        for filename in mapping["left"]:
            try:
                data = load_profile(data_dir, filename)
                ax_left.plot(data[:, 0], data[:, 1], label=filename, linewidth=2)
            except OSError as e:
                print(f"Error loading {filename}: {e}")

        if mapping["right"]:
            ax_right = ax_left.twinx()
            for filename in mapping["right"]:
                try:
                    data = load_profile(data_dir, filename)
                    ax_right.plot(data[:, 0], data[:, 1], label=f"{filename} (RHS)", linewidth=2, linestyle="--")
                except OSError as e:
                    print(f"Error loading {filename}: {e}")
            ax_right.legend(loc="upper right", fontsize="small")

        ax_left.set_title(title)
        ax_left.set_xlabel(r"$\psi_N$ (Normalised Poloidal Flux)")
        ax_left.set_xlim(0, 1)
        ax_left.grid(True, alpha=0.3)
        ax_left.legend(loc="upper left", fontsize="small")

    fig.tight_layout()
    return fig, axs
