"""
surrogate_models.py
===================
Surrogate model implementations for Phase 2.

Supported model types
---------------------
- :class:`NNSurrogate`   — residual neural network
- :class:`PCESurrogate`  — polynomial chaos expansion (Legendre basis)

Both support four output representations:
- "direct"   — predict settlement at all n_nodes_x surface nodes
- "dct"      — predict first n_output_modes DCT coefficients, then reconstruct
- "poly"     — predict polynomial coefficients (degree = n_output_modes - 1)
- "bspline"  — predict B-spline control points, then evaluate spline

Both models expose a common interface:
    fit(X, Y)
    predict(X) -> np.ndarray   shape (n_samples, n_nodes_x)
    save(path)
    load(path)
"""

from __future__ import annotations

import json
import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .utils import (
    reconstruct_from_dct,
    reconstruct_from_poly,
    reconstruct_from_bspline,
)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseSurrogate(ABC):
    """Common interface for all surrogate models."""

    def __init__(
        self,
        input_dim: int,
        n_nodes_x: int,
        output_repr: str = "direct",
        n_output_modes: int = 10,
    ) -> None:
        self.input_dim = input_dim
        self.n_nodes_x = n_nodes_x
        self.output_repr = output_repr
        self.n_output_modes = n_output_modes

        if output_repr == "direct":
            self.output_dim = n_nodes_x
        else:
            self.output_dim = n_output_modes

    def _reconstruct_output(self, raw: np.ndarray) -> np.ndarray:
        """Convert raw model output to settlement profiles."""
        if self.output_repr == "direct":
            return raw
        elif self.output_repr == "dct":
            return reconstruct_from_dct(raw, self.n_nodes_x, 1)
        elif self.output_repr == "poly":
            return reconstruct_from_poly(raw, self.n_nodes_x)
        elif self.output_repr == "bspline":
            return reconstruct_from_bspline(raw, self.n_nodes_x)
        else:
            raise ValueError(f"Unknown output_repr: {self.output_repr}")

    def _prepare_targets(self, Y: np.ndarray) -> np.ndarray:
        """Convert settlement profiles to training targets in output representation."""
        if self.output_repr == "direct":
            return Y
        elif self.output_repr == "dct":
            from scipy.fft import dct
            return dct(Y, type=2, norm="ortho", axis=-1)[:, : self.n_output_modes]
        elif self.output_repr == "poly":
            # Fit polynomial to each profile
            x = np.linspace(0.0, 1.0, self.n_nodes_x)
            degree = self.n_output_modes - 1
            return np.array([np.polyfit(x, y, degree) for y in Y])
        elif self.output_repr == "bspline":
            # Fit uniform B-spline control points
            from scipy.interpolate import make_lsq_spline
            x = np.linspace(0.0, 1.0, self.n_nodes_x)
            n_ctrl = self.n_output_modes
            k = min(3, n_ctrl - 1)
            n_inner = n_ctrl - k - 1
            if n_inner <= 0:
                knots = np.concatenate(
                    [[0.0] * (k + 1), [1.0] * (k + 1)]
                )
            else:
                inner = np.linspace(0.0, 1.0, n_inner + 2)[1:-1]
                knots = np.concatenate([[0.0] * (k + 1), inner, [1.0] * (k + 1)])
            ctrl_pts = np.zeros((len(Y), n_ctrl))
            for i, y in enumerate(Y):
                try:
                    spl = make_lsq_spline(x, y, knots, k=k)
                    ctrl_pts[i] = spl.c
                except Exception:
                    ctrl_pts[i] = np.interp(
                        np.linspace(0, 1, n_ctrl), x, y
                    )
            return ctrl_pts
        else:
            raise ValueError(f"Unknown output_repr: {self.output_repr}")

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> "BaseSurrogate":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseSurrogate":
        ...


# ---------------------------------------------------------------------------
# Neural Network Surrogate
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    """Single residual block: Linear → ReLU → Linear + skip."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc2(self.act(self.fc1(x))) + x)


class _ResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
        for i in range(len(hidden_dims) - 1):
            if hidden_dims[i] == hidden_dims[i + 1]:
                layers.append(_ResBlock(hidden_dims[i]))
            else:
                layers.extend([nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()])
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NNSurrogate(BaseSurrogate):
    """Residual neural network surrogate.

    Parameters
    ----------
    input_dim : int
    n_nodes_x : int
    output_repr : str
    n_output_modes : int
    hidden_dims : list of int
    epochs : int
    lr : float
    batch_size : int
    patience : int  — early stopping patience
    """

    def __init__(
        self,
        input_dim: int,
        n_nodes_x: int,
        output_repr: str = "direct",
        n_output_modes: int = 10,
        hidden_dims: Sequence[int] = (128, 128, 64),
        epochs: int = 500,
        lr: float = 1e-3,
        batch_size: int = 32,
        patience: int = 50,
    ) -> None:
        super().__init__(input_dim, n_nodes_x, output_repr, n_output_modes)
        self.hidden_dims = list(hidden_dims)
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience

        self._model: Optional[_ResNet] = None
        self._X_mean: Optional[np.ndarray] = None
        self._X_std: Optional[np.ndarray] = None
        self._Y_mean: Optional[np.ndarray] = None
        self._Y_std: Optional[np.ndarray] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "NNSurrogate":
        T = self._prepare_targets(Y)

        # Normalise inputs
        self._X_mean = X.mean(axis=0)
        self._X_std = X.std(axis=0)
        self._X_std[self._X_std < 1e-12] = 1.0
        Xn = (X - self._X_mean) / self._X_std

        # Normalise targets
        self._Y_mean = T.mean(axis=0)
        self._Y_std = T.std(axis=0)
        self._Y_std[self._Y_std < 1e-12] = 1.0
        Tn = (T - self._Y_mean) / self._Y_std

        # Build model
        self._model = _ResNet(
            self.input_dim, self.output_dim, self.hidden_dims
        ).to(self.device)
        optimiser = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        # Dataset split for early stopping
        n = len(Xn)
        n_val = max(1, int(0.1 * n))
        idx = np.random.permutation(n)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        X_t = torch.tensor(Xn[train_idx], dtype=torch.float32, device=self.device)
        Y_t = torch.tensor(Tn[train_idx], dtype=torch.float32, device=self.device)
        X_v = torch.tensor(Xn[val_idx], dtype=torch.float32, device=self.device)
        Y_v = torch.tensor(Tn[val_idx], dtype=torch.float32, device=self.device)

        dataset = TensorDataset(X_t, Y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self._model.train()
        for epoch in range(self.epochs):
            for Xb, Yb in loader:
                optimiser.zero_grad()
                loss = loss_fn(self._model(Xb), Yb)
                loss.backward()
                optimiser.step()

            self._model.eval()
            with torch.no_grad():
                val_loss = loss_fn(self._model(X_v), Y_v).item()
            self._model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        Xn = (X - self._X_mean) / self._X_std
        xt = torch.tensor(Xn, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            raw_n = self._model(xt).cpu().numpy()
        raw = raw_n * self._Y_std + self._Y_mean
        return self._reconstruct_output(raw)

    def save(self, path: str | Path) -> None:
        """Save the surrogate model.

        If *path* ends with ``.pt`` the entire model (weights + metadata +
        normalisation arrays) is saved as a single ``torch.save`` bundle so
        that the file can be identified by its dimension-stamped name
        (e.g. ``surrogate_nn_dim15.pt``).  Otherwise the legacy directory
        format is used for backward compatibility.
        """
        path = Path(path)
        meta = {
            "input_dim": self.input_dim,
            "n_nodes_x": self.n_nodes_x,
            "output_repr": self.output_repr,
            "n_output_modes": self.n_output_modes,
            "hidden_dims": self.hidden_dims,
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "patience": self.patience,
        }
        if path.suffix == ".pt":
            # Single-file bundle (dimension-stamped format)
            path.parent.mkdir(parents=True, exist_ok=True)
            bundle = {
                "meta": meta,
                "state_dict": self._model.state_dict(),
                "X_mean": self._X_mean,
                "X_std": self._X_std,
                "Y_mean": self._Y_mean,
                "Y_std": self._Y_std,
            }
            torch.save(bundle, path)
        else:
            # Legacy directory format
            path.mkdir(parents=True, exist_ok=True)
            torch.save(self._model.state_dict(), path / "model.pt")
            with open(path / "meta.json", "w", encoding='utf-8') as f:
                json.dump(meta, f)
            np.save(path / "X_mean.npy", self._X_mean)
            np.save(path / "X_std.npy", self._X_std)
            np.save(path / "Y_mean.npy", self._Y_mean)
            np.save(path / "Y_std.npy", self._Y_std)

    @classmethod
    def load(cls, path: str | Path) -> "NNSurrogate":
        path = Path(path)
        if path.suffix == ".pt":
            # Single-file bundle (dimension-stamped format)
            bundle = torch.load(path, map_location="cpu", weights_only=False)
            meta = bundle["meta"]
            obj = cls(
                input_dim=meta["input_dim"],
                n_nodes_x=meta["n_nodes_x"],
                output_repr=meta["output_repr"],
                n_output_modes=meta["n_output_modes"],
                hidden_dims=meta["hidden_dims"],
                epochs=meta["epochs"],
                lr=meta["lr"],
                batch_size=meta["batch_size"],
                patience=meta["patience"],
            )
            obj._model = _ResNet(
                obj.input_dim, obj.output_dim, obj.hidden_dims
            ).to(obj.device)
            obj._model.load_state_dict(bundle["state_dict"])
            obj._model.eval()
            obj._X_mean = bundle["X_mean"]
            obj._X_std = bundle["X_std"]
            obj._Y_mean = bundle["Y_mean"]
            obj._Y_std = bundle["Y_std"]
            return obj
        else:
            # Legacy directory format
            with open(path / "meta.json", encoding='utf-8') as f:
                meta = json.load(f)
            obj = cls(
                input_dim=meta["input_dim"],
                n_nodes_x=meta["n_nodes_x"],
                output_repr=meta["output_repr"],
                n_output_modes=meta["n_output_modes"],
                hidden_dims=meta["hidden_dims"],
                epochs=meta["epochs"],
                lr=meta["lr"],
                batch_size=meta["batch_size"],
                patience=meta["patience"],
            )
            obj._model = _ResNet(
                obj.input_dim, obj.output_dim, obj.hidden_dims
            ).to(obj.device)
            state = torch.load(path / "model.pt", map_location=obj.device, weights_only=True)
            obj._model.load_state_dict(state)
            obj._model.eval()
            obj._X_mean = np.load(path / "X_mean.npy")
            obj._X_std = np.load(path / "X_std.npy")
            obj._Y_mean = np.load(path / "Y_mean.npy")
            obj._Y_std = np.load(path / "Y_std.npy")
            return obj


# ---------------------------------------------------------------------------
# PCE Surrogate
# ---------------------------------------------------------------------------

class PCESurrogate(BaseSurrogate):
    """Polynomial Chaos Expansion (Legendre basis) surrogate.

    Constructs a multi-variate polynomial basis up to total degree *degree*
    and fits ordinary least squares.

    Parameters
    ----------
    input_dim : int
    n_nodes_x : int
    output_repr : str
    n_output_modes : int
    degree : int  — maximum total polynomial degree
    """

    def __init__(
        self,
        input_dim: int,
        n_nodes_x: int,
        output_repr: str = "direct",
        n_output_modes: int = 10,
        degree: int = 3,
    ) -> None:
        super().__init__(input_dim, n_nodes_x, output_repr, n_output_modes)
        self.degree = degree
        self._coeffs: Optional[np.ndarray] = None
        self._X_mean: Optional[np.ndarray] = None
        self._X_std: Optional[np.ndarray] = None

    def _build_vandermonde(self, X: np.ndarray) -> np.ndarray:
        """Build multi-variate Legendre Vandermonde matrix.

        Only include terms with total degree ≤ self.degree to keep the
        basis tractable.  For high-dimensional inputs this still produces
        many terms; a sparse index set is used.
        """
        from itertools import combinations_with_replacement
        n, d = X.shape
        # Generate multi-indices (i_1, ..., i_d) with sum ≤ degree
        indices = [
            combo
            for r in range(self.degree + 1)
            for combo in combinations_with_replacement(range(d), r)
        ]
        # Convert to multi-index format
        multi_idx = []
        for combo in indices:
            idx = [0] * d
            for j in combo:
                idx[j] += 1
            multi_idx.append(idx)

        from numpy.polynomial.legendre import legval
        n_basis = len(multi_idx)
        V = np.ones((n, n_basis))
        for col, idx in enumerate(multi_idx):
            for dim, power in enumerate(idx):
                if power > 0:
                    coeffs_1d = [0.0] * (power + 1)
                    coeffs_1d[power] = 1.0
                    V[:, col] *= legval(X[:, dim], coeffs_1d)
        return V

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "PCESurrogate":
        T = self._prepare_targets(Y)

        # Normalise to [-1, 1] for Legendre polynomials
        self._X_mean = X.mean(axis=0)
        self._X_std = X.std(axis=0)
        self._X_std[self._X_std < 1e-12] = 1.0
        Xn = (X - self._X_mean) / self._X_std

        V = self._build_vandermonde(Xn)
        # Least-squares fit
        self._coeffs, _, _, _ = np.linalg.lstsq(V, T, rcond=None)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._coeffs is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        Xn = (X - self._X_mean) / self._X_std
        V = self._build_vandermonde(Xn)
        raw = V @ self._coeffs
        return self._reconstruct_output(raw)

    def save(self, path: str | Path) -> None:
        """Save the PCE surrogate.

        If *path* ends with ``.pkl`` the data is saved directly to that file
        (dimension-stamped format, e.g. ``surrogate_pce_dim15.pkl``).
        Otherwise the legacy directory format is used (saves ``pce.pkl``
        inside the given directory).
        """
        path = Path(path)
        data = {
            "coeffs": self._coeffs,
            "X_mean": self._X_mean,
            "X_std": self._X_std,
            "meta": {
                "input_dim": self.input_dim,
                "n_nodes_x": self.n_nodes_x,
                "output_repr": self.output_repr,
                "n_output_modes": self.n_output_modes,
                "degree": self.degree,
            },
        }
        if path.suffix == ".pkl":
            # Single-file (dimension-stamped) format
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(data, f)
        else:
            # Legacy directory format
            path.mkdir(parents=True, exist_ok=True)
            with open(path / "pce.pkl", "wb") as f:
                pickle.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> "PCESurrogate":
        path = Path(path)
        if path.suffix == ".pkl":
            # Single-file (dimension-stamped) format
            pkl_path = path
        else:
            # Legacy directory format
            pkl_path = path / "pce.pkl"
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        meta = data["meta"]
        obj = cls(
            input_dim=meta["input_dim"],
            n_nodes_x=meta["n_nodes_x"],
            output_repr=meta["output_repr"],
            n_output_modes=meta["n_output_modes"],
            degree=meta["degree"],
        )
        obj._coeffs = data["coeffs"]
        obj._X_mean = data["X_mean"]
        obj._X_std = data["X_std"]
        return obj


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_surrogate(
    surrogate_type: str,
    input_dim: int,
    n_nodes_x: int,
    output_repr: str = "direct",
    n_output_modes: int = 10,
    nn_cfg: Optional[Dict[str, Any]] = None,
    pce_cfg: Optional[Dict[str, Any]] = None,
) -> BaseSurrogate:
    """Construct a surrogate model from a type string."""
    nn_cfg = nn_cfg or {}
    pce_cfg = pce_cfg or {}
    if surrogate_type == "nn":
        return NNSurrogate(
            input_dim=input_dim,
            n_nodes_x=n_nodes_x,
            output_repr=output_repr,
            n_output_modes=n_output_modes,
            hidden_dims=nn_cfg.get("hidden_dims", [128, 128, 64]),
            epochs=nn_cfg.get("epochs", 500),
            lr=nn_cfg.get("lr", 1e-3),
            batch_size=nn_cfg.get("batch_size", 32),
            patience=nn_cfg.get("patience", 50),
        )
    elif surrogate_type == "pce":
        return PCESurrogate(
            input_dim=input_dim,
            n_nodes_x=n_nodes_x,
            output_repr=output_repr,
            n_output_modes=n_output_modes,
            degree=pce_cfg.get("degree", 3),
        )
    else:
        raise ValueError(f"Unknown surrogate_type: '{surrogate_type}'")


def load_surrogate(path: str | Path) -> BaseSurrogate:
    """Auto-detect and load a saved surrogate model.

    Handles three path forms:
    1. A single ``.pt`` file  → :class:`NNSurrogate` (dimension-stamped format)
    2. A single ``.pkl`` file → :class:`PCESurrogate` (dimension-stamped format)
    3. A directory           → legacy format (looks for ``model.pt`` or ``pce.pkl`` inside)
    """
    path = Path(path)
    if path.suffix == ".pt":
        return NNSurrogate.load(path)
    elif path.suffix == ".pkl":
        return PCESurrogate.load(path)
    elif path.is_dir():
        if (path / "model.pt").exists():
            return NNSurrogate.load(path)
        elif (path / "pce.pkl").exists():
            return PCESurrogate.load(path)
        else:
            raise FileNotFoundError(
                f"No surrogate model found in directory {path}. "
                "Expected 'model.pt' (NN) or 'pce.pkl' (PCE)."
            )
    else:
        raise FileNotFoundError(
            f"Surrogate path not found: {path}. "
            "Expected a '.pt' file, '.pkl' file, or a directory."
        )
