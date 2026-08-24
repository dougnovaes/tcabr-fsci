"""
Module 7 — Inversion.

Solves I = G * epsilon for the flux-surface emissivity profile
epsilon(psi_N), given the geometric matrix G from Module 5 and a
vector of measured channel intensities I. Because G tends to be
ill-conditioned even when overdetermined (neighbouring channels
traverse strongly correlated regions of plasma), plain least squares
is not offered as a solver here — only regularised / constrained
methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.linalg import svd
from scipy.optimize import nnls


@dataclass
class InversionResult:
    """Output of an inversion solver."""

    emissivity: np.ndarray            # epsilon(psi_N)
    reconstructed_signals: np.ndarray  # I_rec = G @ epsilon
    residuals: np.ndarray              # I_measured - I_rec
    chi_square: float                  # Goodness-of-fit metric
    method_name: str


class InverseSolver(ABC):
    """Abstract base class for all inversion methods."""

    @abstractmethod
    def solve(self, G: np.ndarray, signals: np.ndarray, errors: np.ndarray = None) -> InversionResult:
        """Execute the inversion algorithm."""
        raise NotImplementedError

    def _calculate_metrics(self, G, signals, emissivity, errors):
        i_rec = G @ emissivity
        residuals = signals - i_rec
        if errors is not None:
            chi_square = np.sum((residuals / errors) ** 2) / len(signals)
        else:
            chi_square = np.sum(residuals**2) / len(signals)
        return i_rec, residuals, chi_square


class NNLSSolver(InverseSolver):
    """Non-Negative Least Squares inversion."""

    def solve(self, G, signals, errors=None) -> InversionResult:
        if errors is not None:
            W = np.diag(1.0 / errors)
            Gw, sw = W @ G, W @ signals
        else:
            Gw, sw = G, signals

        emissivity, _ = nnls(Gw, sw)
        i_rec, residuals, chi_square = self._calculate_metrics(G, signals, emissivity, errors)
        return InversionResult(emissivity, i_rec, residuals, chi_square, "NNLS")


class TSVDSolver(InverseSolver):
    """Truncated Singular Value Decomposition."""

    def __init__(self, truncation_index: int):
        self.k = truncation_index

    def solve(self, G, signals, errors=None) -> InversionResult:
        U, S, Vt = svd(G, full_matrices=False)

        S_inv = np.zeros_like(S)
        S_inv[: self.k] = 1.0 / S[: self.k]

        G_pinv = Vt.T @ np.diag(S_inv) @ U.T
        emissivity = np.clip(G_pinv @ signals, 0, None)

        i_rec, residuals, chi_square = self._calculate_metrics(G, signals, emissivity, errors)
        return InversionResult(emissivity, i_rec, residuals, chi_square, f"TSVD (k={self.k})")


class TikhonovSolver(InverseSolver):
    """Tikhonov regularisation (0th/1st/2nd-order smoothness operator)."""

    def __init__(self, lambda_param: float, operator_order: int = 2):
        self.lambda_param = lambda_param
        self.order = operator_order

    def _build_operator(self, n: int) -> np.ndarray:
        L = np.zeros((n, n))
        if self.order == 0:
            np.fill_diagonal(L, 1.0)
        elif self.order == 1:
            for i in range(n - 1):
                L[i, i] = -1.0
                L[i, i + 1] = 1.0
        elif self.order == 2:
            for i in range(1, n - 1):
                L[i, i - 1] = 1.0
                L[i, i] = -2.0
                L[i, i + 1] = 1.0
            L[0, 0] = -2.0
            L[0, 1] = 1.0
            L[-1, -2] = 1.0
            L[-1, -1] = -2.0
        else:
            raise ValueError("operator_order must be 0, 1, or 2")
        return L

    def solve(self, G, signals, errors=None) -> InversionResult:
        n_shells = G.shape[1]
        L = self._build_operator(n_shells)

        if errors is not None:
            W = np.diag(1.0 / errors)
            Gw, sw = W @ G, W @ signals
        else:
            Gw, sw = G, signals

        G_aug = np.vstack((Gw, self.lambda_param * L))
        s_aug = np.concatenate((sw, np.zeros(n_shells)))

        emissivity, _ = nnls(G_aug, s_aug)
        i_rec, residuals, chi_square = self._calculate_metrics(G, signals, emissivity, errors)
        return InversionResult(emissivity, i_rec, residuals, chi_square, f"Tikhonov (\u03bb={self.lambda_param})")
