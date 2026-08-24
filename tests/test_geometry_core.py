import numpy as np

from fsci.geometry_core import calculate_equatorial_lcfs_limits, generate_lcfs_mask


def test_equatorial_lcfs_limits(synthetic_equilibrium):
    eq = synthetic_equilibrium
    r_in, r_out = calculate_equatorial_lcfs_limits(eq)
    assert np.isclose(r_in, eq.R_axis - 0.2, atol=1e-3)
    assert np.isclose(r_out, eq.R_axis + 0.2, atol=1e-3)


def test_lcfs_mask_contains_axis(synthetic_equilibrium):
    eq = synthetic_equilibrium
    mask = generate_lcfs_mask(np.array([eq.R_axis]), np.array([eq.Z_axis]), eq)
    assert mask[0]


def test_lcfs_mask_excludes_far_outside_point(synthetic_equilibrium):
    eq = synthetic_equilibrium
    mask = generate_lcfs_mask(np.array([eq.R.max() + 1.0]), np.array([eq.Z_axis]), eq)
    assert not mask[0]


def test_closure_segment_is_checked():
    """Regression test for the LCFS-closure bug: the segment between the
    last and first boundary points must be included in the Z=0 crossing
    search, not silently skipped."""
    from fsci.equilibrium import MagneticEquilibrium
    from scipy.interpolate import RegularGridInterpolator

    # A boundary that only crosses Z=0 across the wrap-around segment.
    R_boundary = np.array([0.5, 0.6, 0.7])
    Z_boundary = np.array([0.1, 0.2, -0.1])  # last->first crosses Z=0
    R = np.linspace(0.4, 0.8, 5)
    Z = np.linspace(-0.3, 0.3, 5)
    psiN = np.ones((5, 5)) * 0.5
    interp = RegularGridInterpolator((R, Z), psiN, bounds_error=False, fill_value=None)

    eq = MagneticEquilibrium(
        R=R, Z=Z, psi=psiN, psiN=psiN, psi_axis=0.0, psi_lcfs=1.0,
        R_axis=0.6, Z_axis=0.0, R_geo=0.6, Z_geo=0.0, shafranov_shift=0.0,
        R_boundary=R_boundary, Z_boundary=Z_boundary,
        R_limiter=R_boundary, Z_limiter=Z_boundary, psi_interp=interp,
    )

    r_in, r_out = calculate_equatorial_lcfs_limits(eq)
    assert r_in < r_out  # a crossing was actually found via the closing segment
