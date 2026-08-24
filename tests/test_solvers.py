import numpy as np

from fsci.inversion_matrix import build_geometric_matrix
from fsci.solvers import NNLSSolver, TSVDSolver, TikhonovSolver


def _fake_signals(G):
    rng = np.random.default_rng(0)
    true_eps = np.linspace(1.0, 0.1, G.shape[1])
    return G @ true_eps + rng.normal(scale=0.01, size=G.shape[0])


def test_nnls_solver_runs(synthetic_optics, synthetic_grid, synthetic_equilibrium):
    inv = build_geometric_matrix(synthetic_optics, synthetic_grid, synthetic_equilibrium)
    signals = _fake_signals(inv.data)
    result = NNLSSolver().solve(inv.data, signals)
    assert result.emissivity.shape == (synthetic_grid.n_shells,)
    assert np.all(result.emissivity >= 0)


def test_tsvd_solver_runs(synthetic_optics, synthetic_grid, synthetic_equilibrium):
    inv = build_geometric_matrix(synthetic_optics, synthetic_grid, synthetic_equilibrium)
    signals = _fake_signals(inv.data)
    result = TSVDSolver(truncation_index=3).solve(inv.data, signals)
    assert result.emissivity.shape == (synthetic_grid.n_shells,)


def test_tikhonov_solver_runs(synthetic_optics, synthetic_grid, synthetic_equilibrium):
    inv = build_geometric_matrix(synthetic_optics, synthetic_grid, synthetic_equilibrium)
    signals = _fake_signals(inv.data)
    result = TikhonovSolver(lambda_param=0.1).solve(inv.data, signals)
    assert result.emissivity.shape == (synthetic_grid.n_shells,)
    assert np.all(result.emissivity >= 0)


def test_tikhonov_invalid_order_raises(synthetic_optics, synthetic_grid, synthetic_equilibrium):
    inv = build_geometric_matrix(synthetic_optics, synthetic_grid, synthetic_equilibrium)
    signals = _fake_signals(inv.data)
    solver = TikhonovSolver(lambda_param=0.1, operator_order=5)
    try:
        solver.solve(inv.data, signals)
        assert False, "expected ValueError for invalid operator_order"
    except ValueError:
        pass
