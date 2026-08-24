import numpy as np

from fsci.flux_grid import build_flux_grid


def test_edges_span_zero_to_one():
    grid = build_flux_grid(n_shells=10)
    assert grid.psi_edges[0] == 0.0
    assert grid.psi_edges[-1] == 1.0
    assert len(grid.psi_edges) == 11


def test_get_cell_indices_bounds():
    grid = build_flux_grid(n_shells=10)
    idx = grid.get_cell_indices(np.array([-0.1, 0.0, 0.55, 1.0, 1.5]))
    assert np.all(idx >= 0)
    assert np.all(idx <= 9)


def test_get_cell_id_outside_plasma(synthetic_equilibrium):
    grid = build_flux_grid(n_shells=10)
    eq = synthetic_equilibrium
    far_outside = grid.get_cell_id(eq.R.max() + 1.0, eq.Z_axis, eq)
    assert far_outside == -1
