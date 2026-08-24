def test_psiN_shape_matches_R_Z_convention(synthetic_equilibrium):
    eq = synthetic_equilibrium
    assert eq.psiN.shape == (len(eq.R), len(eq.Z))


def test_psiN_near_zero_at_axis(synthetic_equilibrium):
    eq = synthetic_equilibrium
    val = float(eq.psi_interp((eq.R_axis, eq.Z_axis)).item())
    assert val < 0.02


def test_psiN_near_one_at_lcfs(synthetic_equilibrium):
    eq = synthetic_equilibrium
    val = float(eq.psi_interp((eq.R_boundary[0], eq.Z_boundary[0])).item())
    assert abs(val - 1.0) < 0.05
