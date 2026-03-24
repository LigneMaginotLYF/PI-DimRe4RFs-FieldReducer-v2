"""
utils.py
========
Shared mathematical helpers:
  - 2D DCT-II basis construction and caching
  - Matérn spectral variance
  - Metrics (R², RMSE, relative L²)
  - Normalisation helpers
  - Plot helpers
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.fft import dctn

# ---------------------------------------------------------------------------
# DCT basis
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def compute_dct_basis(
    n_nodes_x: int,
    n_nodes_z: int,
    n_terms: int,
) -> np.ndarray:
    """Compute the first *n_terms* columns of the 2-D DCT-II basis.

    Modes are ordered by ascending frequency magnitude  ‖ω_k‖₂  (lowest-
    frequency, smoothest modes first).  Each column is L2-normalised.

    The returned matrix has shape ``(n_nodes_x * n_nodes_z, n_terms)``.
    When *n_terms* is 0 the function returns an empty matrix with shape
    ``(n_nodes_x * n_nodes_z, 0)`` — callers that handle homogeneous
    fields should call with n_terms = 0 and treat the field as a scalar.
    """
    if n_terms == 0:
        return np.empty((n_nodes_x * n_nodes_z, 0), dtype=np.float64)

    n_total = n_nodes_x * n_nodes_z
    # Frequency magnitudes for all mode combinations (ix, iz)
    ix_grid, iz_grid = np.meshgrid(
        np.arange(n_nodes_x), np.arange(n_nodes_z), indexing="ij"
    )
    freq_magnitudes = np.sqrt(
        (ix_grid / n_nodes_x) ** 2 + (iz_grid / n_nodes_z) ** 2
    ).ravel()
    # Sort modes by frequency magnitude (low → high)
    sorted_indices = np.argsort(freq_magnitudes, kind="stable")

    # Build the selected mode matrix
    n_modes = min(n_terms, n_total)
    basis = np.zeros((n_total, n_modes), dtype=np.float64)

    for col, flat_idx in enumerate(sorted_indices[:n_modes]):
        ix = flat_idx // n_nodes_z
        iz = flat_idx % n_nodes_z
        # DCT-II impulse image for mode (ix, iz)
        impulse = np.zeros((n_nodes_x, n_nodes_z), dtype=np.float64)
        impulse[ix, iz] = 1.0
        # IDCT (type 3 = inverse DCT-II)
        mode = np.real(dctn(impulse, type=3, norm="ortho"))
        col_vec = mode.ravel()
        norm = np.linalg.norm(col_vec)
        if norm > 0:
            col_vec = col_vec / norm
        basis[:, col] = col_vec

    return basis


# ---------------------------------------------------------------------------
# Matérn spectral variance
# ---------------------------------------------------------------------------

def matern_spectral_variance(
    n_nodes_x: int,
    n_nodes_z: int,
    n_terms: int,
    nu: float = 1.5,
    length_scale: float = 0.3,
) -> np.ndarray:
    """Matérn-shaped spectral variance for the first *n_terms* DCT modes.

    For a 2-D field the spectral density of a Matérn(ν, ℓ) kernel is

        S(ω) ∝ (2ν/ℓ² + ‖ω‖²)^{-(ν + 1)}

    Returns a 1-D array of length *n_terms* (or 0 if n_terms == 0).
    """
    if n_terms == 0:
        return np.empty(0, dtype=np.float64)

    ix_grid, iz_grid = np.meshgrid(
        np.arange(n_nodes_x), np.arange(n_nodes_z), indexing="ij"
    )
    freq_magnitudes = np.sqrt(
        (ix_grid / n_nodes_x) ** 2 + (iz_grid / n_nodes_z) ** 2
    ).ravel()
    sorted_indices = np.argsort(freq_magnitudes, kind="stable")

    n_modes = min(n_terms, n_nodes_x * n_nodes_z)
    selected_freqs = freq_magnitudes[sorted_indices[:n_modes]]

    alpha = 2.0 * nu / (length_scale ** 2)
    variances = (alpha + selected_freqs ** 2) ** (-(nu + 1.0))
    # Normalise so the sum equals 1 (relative importance)
    total = variances.sum()
    if total > 0:
        variances = variances / total
    return variances


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return float(1.0 - ss_res / ss_tot)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def relative_l2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative L² error: ‖y_pred - y_true‖₂ / ‖y_true‖₂."""
    denom = np.linalg.norm(y_true)
    if denom < 1e-12:
        return float(np.linalg.norm(y_pred - y_true))
    return float(np.linalg.norm(y_pred - y_true) / denom)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return a dict with R², RMSE, relative_L2, and roughness.

    Roughness is computed as the mean absolute second-order finite difference
    of the prediction (lower = smoother settlement profile).
    """
    if y_true.ndim == 1:
        y_true = y_true[np.newaxis, :]
        y_pred = y_pred[np.newaxis, :]
    metrics = {
        "R2": r2_score(y_true.ravel(), y_pred.ravel()),
        "RMSE": rmse(y_true.ravel(), y_pred.ravel()),
        "relative_L2": relative_l2(y_true.ravel(), y_pred.ravel()),
        "roughness": float(np.mean(np.abs(np.diff(y_pred, n=2, axis=-1)))),
    }
    # Per-sample relative L2
    per_sample = [
        relative_l2(y_true[i], y_pred[i]) for i in range(len(y_true))
    ]
    metrics["per_sample_errors"] = per_sample
    return metrics


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class StandardScaler:
    """Simple mean/std normaliser for numpy arrays."""

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-12] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.std_ + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# Output reconstruction helpers (used in surrogate / reducer)
# ---------------------------------------------------------------------------

def reconstruct_from_dct(
    coeffs: np.ndarray,
    n_nodes_x: int,
    n_nodes_z: int,
    axis: int = -1,
) -> np.ndarray:
    """Reconstruct spatial field from DCT coefficients (inverse DCT-II / type-3).

    Parameters
    ----------
    coeffs : (..., n_modes) array
    n_nodes_x, n_nodes_z : spatial resolution
    axis : axis along which the modes are stored (default -1)

    Returns
    -------
    field : (..., n_nodes_x) array  (surface settlement uses first axis only)
    """
    # Pad to full resolution along the mode axis
    n_modes = coeffs.shape[axis]
    n_total = n_nodes_x  # for 1D surface settlement
    if n_modes < n_total:
        pad_width = [(0, 0)] * coeffs.ndim
        pad_width[axis] = (0, n_total - n_modes)
        coeffs = np.pad(coeffs, pad_width)
    from scipy.fft import idct
    return np.real(idct(coeffs, type=2, norm="ortho", axis=axis))


def reconstruct_from_poly(
    poly_coeffs: np.ndarray,
    n_nodes_x: int,
) -> np.ndarray:
    """Evaluate polynomial fit at *n_nodes_x* equi-spaced nodes in [0,1].

    Parameters
    ----------
    poly_coeffs : (..., degree+1) array  (NumPy poly convention: highest power first)
    """
    x = np.linspace(0.0, 1.0, n_nodes_x)
    if poly_coeffs.ndim == 1:
        return np.polyval(poly_coeffs, x)
    return np.stack([np.polyval(c, x) for c in poly_coeffs.reshape(-1, poly_coeffs.shape[-1])]).reshape(
        poly_coeffs.shape[:-1] + (n_nodes_x,)
    )


def reconstruct_from_bspline(
    ctrl_pts: np.ndarray,
    n_nodes_x: int,
) -> np.ndarray:
    """Evaluate uniform cubic B-spline at *n_nodes_x* equi-spaced points.

    Parameters
    ----------
    ctrl_pts : (..., n_ctrl) array
    """
    from scipy.interpolate import make_interp_spline
    x_out = np.linspace(0.0, 1.0, n_nodes_x)

    def _eval_single(cp: np.ndarray) -> np.ndarray:
        n_ctrl = len(cp)
        x_ctrl = np.linspace(0.0, 1.0, n_ctrl)
        k = min(3, n_ctrl - 1)
        spl = make_interp_spline(x_ctrl, cp, k=k)
        return spl(x_out)

    if ctrl_pts.ndim == 1:
        return _eval_single(ctrl_pts)
    flat = ctrl_pts.reshape(-1, ctrl_pts.shape[-1])
    return np.stack([_eval_single(row) for row in flat]).reshape(
        ctrl_pts.shape[:-1] + (n_nodes_x,)
    )


# ---------------------------------------------------------------------------
# Plotting helpers (lazy import matplotlib)
# ---------------------------------------------------------------------------

def plot_settlement_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_nodes_x: int,
    save_path: str | None = None,
    title: str = "Settlement comparison",
) -> None:
    """Plot true vs predicted settlement profiles for a subset of samples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.linspace(0.0, 1.0, n_nodes_x)
    n_show = min(5, len(y_true))
    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 3), sharey=True)
    if n_show == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(x, y_true[i], "k-", label="True")
        ax.plot(x, y_pred[i], "r--", label="Pred")
        ax.set_xlabel("x")
        ax.set_title(f"Sample {i}")
        if i == 0:
            ax.set_ylabel("Settlement")
    axes[0].legend()
    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=100)
    plt.close(fig)


def plot_field_2d(
    field: np.ndarray,
    n_nodes_x: int,
    n_nodes_z: int,
    title: str = "Field",
    save_path: str | None = None,
) -> None:
    """Render a 2-D material field as an image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    im = ax.imshow(
        field.reshape(n_nodes_x, n_nodes_z).T,
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("x-node")
    ax.set_ylabel("z-node")
    plt.tight_layout()
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=100)
    plt.close(fig)
