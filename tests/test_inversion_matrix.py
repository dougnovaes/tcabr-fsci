from fsci.inversion_matrix import build_geometric_matrix
from fsci.validation import run_regression_tests, run_independent_allocation_validation


def test_matrix_shape(synthetic_optics, synthetic_grid, synthetic_equilibrium):
    inv = build_geometric_matrix(synthetic_optics, synthetic_grid, synthetic_equilibrium)
    assert inv.data.shape == (len(synthetic_optics.sightlines), synthetic_grid.n_shells)


def test_at_least_one_channel_intersects_plasma(synthetic_optics, synthetic_grid, synthetic_equilibrium):
    inv = build_geometric_matrix(synthetic_optics, synthetic_grid, synthetic_equilibrium)
    assert inv.length_in_plasma.max() > 0


def test_regression_invariants(synthetic_optics, synthetic_grid, synthetic_equilibrium):
    inv = build_geometric_matrix(synthetic_optics, synthetic_grid, synthetic_equilibrium)
    report = run_regression_tests(synthetic_optics, synthetic_equilibrium, inv)
    assert report.passed


def test_independent_allocation_validation(synthetic_optics, synthetic_grid, synthetic_equilibrium):
    inv = build_geometric_matrix(synthetic_optics, synthetic_grid, synthetic_equilibrium)
    report = run_independent_allocation_validation(synthetic_optics, synthetic_equilibrium, inv)
    assert report.passed
