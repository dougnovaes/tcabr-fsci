import pytest

from fsci.synthetic import build_synthetic_equilibrium, build_synthetic_optics
from fsci.flux_grid import build_flux_grid


@pytest.fixture
def synthetic_equilibrium():
    return build_synthetic_equilibrium()


@pytest.fixture
def synthetic_optics(synthetic_equilibrium):
    return build_synthetic_optics(synthetic_equilibrium, n_channels=8)


@pytest.fixture
def synthetic_grid():
    return build_flux_grid(n_shells=10)
