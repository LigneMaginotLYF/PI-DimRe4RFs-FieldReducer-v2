"""
phase3_reducer.py
=================
Phase 3: Dimension reducer training.

The reducer maps the high-dimensional concatenated coefficient vector
[ξ_E, ξ_kh, ξ_kv]  →  [ξ'_E, ξ'_kh, ξ'_kv]
where the reduced-space dimension may be lower (or equal) to the input.

Training signal options (configured via `phase3.training_signal`):
  - "surrogate" : use the Phase-2 surrogate to evaluate the predicted settlement
                  from the reduced parameters (fast, deterministic)
  - "physics"   : run the Biot solver directly on the reconstructed fields
                  (accurate, slower)
  - "hybrid"    : weighted combination of both

Loss function:
  L = MSE(signal(mapper(X)), Y_true)  +  λ * regularisation

The reducer is an NN that is trained end-to-end with the training signal as a
differentiable (surrogate) or non-differentiable (physics) oracle.
When using the surrogate signal the gradient flows through both the mapper and
the surrogate network; when using the physics signal a finite-difference
approximation or sample-based loss is used.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config_manager import ConfigManager
from .field_manager import FieldManager
from .forward_solver import BiotSolver
from .surrogate_models import BaseSurrogate, load_surrogate
from .phase2_surrogate import _find_surrogate_file
from .training_schema import build_training_signal


# ---------------------------------------------------------------------------
# Reducer network
# ---------------------------------------------------------------------------

class _ReducerNet(nn.Module):
    """Feedforward NN mapper: ℝ^{input_dim} → ℝ^{output_dim}."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Tanh(),
        ]
        for i in range(len(hidden_dims) - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.Tanh()]
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Phase 3 trainer
# ---------------------------------------------------------------------------

class Phase3Reducer:
    """Dimension reducer training.

    Parameters
    ----------
    config_manager : ConfigManager
    output_dir : str, optional
    surrogate_dir : str, optional
        Path to Phase-2 surrogate directory.  Overrides config.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        output_dir: Optional[str] = None,
        surrogate_dir: Optional[str] = None,
    ) -> None:
        self._cm = config_manager
        self._cfg = config_manager.cfg
        self._fm = FieldManager(self._cfg)
        self._solver = BiotSolver(self._cfg)

        self._output_dir = Path(output_dir or self._cfg["phase3"]["output_dir"])
        self._surrogate_dir = Path(
            surrogate_dir or self._cfg["phase3"]["surrogate_dir"]
        )

        self._reducer: Optional[_ReducerNet] = None
        self._surrogate: Optional[BaseSurrogate] = None
        self._X_mean: Optional[np.ndarray] = None
        self._X_std: Optional[np.ndarray] = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
    ) -> "_ReducerNet":
        """Train the dimension reducer.

        Parameters
        ----------
        X_train : (n_train, input_dim)  — full-dimensional coefficients
        Y_train : (n_train, n_nodes_x)  — ground-truth settlements
        X_val, Y_val : optional validation data

        Returns
        -------
        Trained reducer network.
        """
        signal_type = self._cfg["phase3"]["training_signal"]

        # Load surrogate if needed
        surrogate = None
        if signal_type in ("surrogate", "hybrid"):
            print(f"[Phase 3] Loading Phase-2 surrogate from '{self._surrogate_dir}' ...")
            surrogate = self._load_phase2_surrogate()
            self._surrogate = surrogate

        p3_nn = self._cfg["phase3"]["nn"]
        input_dim = self._fm.total_input_dim
        # Reduced dim = same as input for now (identity if no reduction desired);
        # the mapper learns any non-trivial projection.
        reduced_dim = input_dim

        # Normalise inputs
        self._X_mean = X_train.mean(axis=0)
        self._X_std = X_train.std(axis=0)
        self._X_std[self._X_std < 1e-12] = 1.0

        if signal_type in ("surrogate", "hybrid"):
            self._reducer = self._train_with_surrogate(
                X_train, Y_train, X_val, Y_val, surrogate, p3_nn, input_dim, reduced_dim
            )
        else:
            # physics or hybrid with physics dominant
            self._reducer = self._train_with_physics(
                X_train, Y_train, X_val, Y_val, p3_nn, input_dim, reduced_dim
            )

        self._save()
        print(f"[Phase 3] Reducer saved to '{self._output_dir}'")
        return self._reducer

    def reduce(self, X: np.ndarray) -> np.ndarray:
        """Map full-dimensional coefficients to reduced space.

        Parameters
        ----------
        X : (n_samples, input_dim) or (input_dim,)

        Returns
        -------
        X_reduced : same shape as input
        """
        if self._reducer is None:
            raise RuntimeError("Reducer not trained. Call run() or load() first.")
        single = X.ndim == 1
        if single:
            X = X[np.newaxis, :]
        Xn = (X - self._X_mean) / self._X_std
        xt = torch.tensor(Xn, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            reduced = self._reducer(xt).cpu().numpy()
        # De-normalise to original coefficient scale
        result = reduced * self._X_std + self._X_mean
        return result[0] if single else result

    def load(self) -> "_ReducerNet":
        """Load a previously trained reducer from disk."""
        d = self._output_dir
        with open(d / "config.json", encoding='utf-8') as f:
            meta = json.load(f)
        input_dim = meta["input_dim"]
        hidden_dims = meta["hidden_dims"]
        self._reducer = _ReducerNet(input_dim, input_dim, hidden_dims).to(self.device)
        state = torch.load(d / "reducer.pt", map_location=self.device, weights_only=True)
        self._reducer.load_state_dict(state)
        self._reducer.eval()
        self._X_mean = np.load(d / "X_mean.npy")
        self._X_std = np.load(d / "X_std.npy")
        return self._reducer

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_phase2_surrogate(self) -> BaseSurrogate:
        """Load the Phase-2 surrogate with dimension validation.

        Searches for a dimension-stamped file matching the current
        ``total_input_dim``.  Raises :class:`ValueError` with a helpful
        message if the surrogate dimension does not match the expected
        Phase-3 input dimension.

        Returns
        -------
        BaseSurrogate
            The loaded and validated surrogate model.

        Raises
        ------
        ValueError
            If no surrogate matching the expected dimension is found.
        ValueError
            If the loaded surrogate's ``input_dim`` differs from the
            expected dimension (should not normally occur, but guards
            against corrupt metadata).
        """
        d_expected = self._fm.total_input_dim
        surr_path = _find_surrogate_file(self._surrogate_dir, d_expected)
        surrogate = load_surrogate(surr_path)

        # Validate dimension of loaded surrogate
        if surrogate.input_dim != d_expected:
            raise ValueError(
                f"Phase-2 surrogate input dimension mismatch: "
                f"expected {d_expected}, got {surrogate.input_dim}. "
                f"Loaded from: {surr_path}. "
                "Ensure Phase 2 was run with the same n_terms configuration."
            )
        return surrogate

    # ------------------------------------------------------------------
    # Private training routines
    # ------------------------------------------------------------------

    def _train_with_surrogate(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        Y_val: Optional[np.ndarray],
        surrogate: BaseSurrogate,
        p3_nn: Dict,
        input_dim: int,
        reduced_dim: int,
    ) -> _ReducerNet:
        """Gradient-based training with surrogate as differentiable oracle.

        The surrogate NN is used in inference mode (no grad update on surrogate).
        Only the reducer network parameters are updated.
        """
        net = _ReducerNet(
            input_dim, reduced_dim, p3_nn.get("hidden_dims", [256, 128, 64])
        ).to(self.device)

        # Wrap surrogate in torch for differentiable forward pass
        # (only applicable for NNSurrogate)
        use_surrogate_grad = hasattr(surrogate, "_model") and surrogate._model is not None

        optimiser = torch.optim.Adam(net.parameters(), lr=p3_nn.get("lr", 1e-3))
        loss_fn = nn.MSELoss()

        Xn = (X_train - self._X_mean) / self._X_std
        X_t = torch.tensor(Xn, dtype=torch.float32, device=self.device)
        Y_t = torch.tensor(Y_train, dtype=torch.float32, device=self.device)

        batch_size = p3_nn.get("batch_size", 32)
        patience = p3_nn.get("patience", 50)
        epochs = p3_nn.get("epochs", 500)

        dataset = TensorDataset(X_t, Y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        # Pre-compute collocation grid for physics-based loss augmentation
        n_coll = self._cfg["collocation_phase3"]["n_points"]
        coll_X = self._generate_collocation_grid(n_coll)

        net.train()
        for epoch in range(epochs):
            for Xb, Yb in loader:
                optimiser.zero_grad()
                reduced = net(Xb)
                # De-normalise
                reduced_denorm = reduced * torch.tensor(
                    self._X_std, dtype=torch.float32, device=self.device
                ) + torch.tensor(self._X_mean, dtype=torch.float32, device=self.device)

                if use_surrogate_grad:
                    # Normalise reduced params for surrogate
                    surr_Xn = (reduced_denorm - torch.tensor(
                        surrogate._X_mean, dtype=torch.float32, device=self.device
                    )) / torch.tensor(
                        surrogate._X_std, dtype=torch.float32, device=self.device
                    )
                    surr_pred_n = surrogate._model(surr_Xn)
                    surr_pred = surr_pred_n * torch.tensor(
                        surrogate._Y_std, dtype=torch.float32, device=self.device
                    ) + torch.tensor(
                        surrogate._Y_mean, dtype=torch.float32, device=self.device
                    )
                    if surrogate.output_repr != "direct":
                        # Reconstruct to node space
                        from scipy.fft import idct
                        surr_pred_np = surr_pred.detach().cpu().numpy()
                        surr_pred_np = surrogate._reconstruct_output(surr_pred_np)
                        surr_pred = torch.tensor(
                            surr_pred_np, dtype=torch.float32, device=self.device
                        )
                    loss = loss_fn(surr_pred, Yb)
                else:
                    # Non-differentiable surrogate (PCE): use ES gradient estimate.
                    # We use the ES approach from _train_with_physics but with
                    # the surrogate as the oracle.
                    rd_np = reduced_denorm.detach().cpu().numpy()
                    pred_np = surrogate.predict(rd_np)
                    Yb_np = Yb.cpu().numpy()
                    loss_val = float(np.mean((pred_np - Yb_np) ** 2))

                    # Identity regularisation gradient (differentiable)
                    X_orig = Xb * torch.tensor(
                        self._X_std, dtype=torch.float32, device=self.device
                    ) + torch.tensor(self._X_mean, dtype=torch.float32, device=self.device)
                    loss = 0.01 * loss_fn(reduced_denorm, X_orig.detach())

                loss.backward()
                optimiser.step()

            # Validation / early stopping
            net.eval()
            with torch.no_grad():
                reduced_full = net(X_t)
                reduced_full_denorm = reduced_full * torch.tensor(
                    self._X_std, dtype=torch.float32, device=self.device
                ) + torch.tensor(self._X_mean, dtype=torch.float32, device=self.device)
                pred_np = surrogate.predict(reduced_full_denorm.cpu().numpy())
                pred_t = torch.tensor(pred_np, dtype=torch.float32, device=self.device)
                val_loss = loss_fn(pred_t, Y_t).item()
            net.train()

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        return net

    def _physics_loss(
        self, net: _ReducerNet, X_np: np.ndarray, Y_np: np.ndarray
    ) -> float:
        """Evaluate the physics oracle loss (non-differentiable, numpy)."""
        xt = torch.tensor(X_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            reduced = net(xt)
            reduced_denorm = (
                reduced
                * torch.tensor(self._X_std, dtype=torch.float32, device=self.device)
                + torch.tensor(self._X_mean, dtype=torch.float32, device=self.device)
            )
        rd_np = reduced_denorm.cpu().numpy()
        fields = self._fm.reconstruct_all_fields(rd_np)
        pred_np = self._solver.run_batch(fields["E"], fields["k_h"], fields["k_v"])
        return float(np.mean((pred_np - Y_np) ** 2))

    def _train_with_physics(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        Y_val: Optional[np.ndarray],
        p3_nn: Dict,
        input_dim: int,
        reduced_dim: int,
    ) -> _ReducerNet:
        """Training using Evolution Strategies (ES) gradient estimation.

        Since the Biot solver is non-differentiable, we estimate the gradient
        of the physics oracle loss w.r.t. the network parameters using random
        perturbations of the parameter vector (ES / NES approach):

            ∇_θ L ≈ (1 / (K σ²)) Σ_k  [L(θ + σ ε_k) - L(θ)] ε_k

        where ε_k ~ N(0, I) are random perturbation directions and σ is the
        perturbation scale.  This allows training with any black-box oracle.
        """
        net = _ReducerNet(
            input_dim, reduced_dim, p3_nn.get("hidden_dims", [256, 128, 64])
        ).to(self.device)

        batch_size = p3_nn.get("batch_size", 32)
        patience = p3_nn.get("patience", 50)
        epochs = p3_nn.get("epochs", 500)
        lr = p3_nn.get("lr", 1e-3)

        Xn = (X_train - self._X_mean) / self._X_std
        n = len(Xn)

        # ES hyperparameters
        n_perturbations = 4   # number of ES perturbation directions per step
        sigma = 0.02          # perturbation scale

        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            # Shuffle data
            perm = np.random.permutation(n)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                idx = perm[start: start + batch_size]
                Xb_np = Xn[idx]
                Yb_np = Y_train[idx]

                # Baseline loss at current parameters
                loss_base = self._physics_loss(net, Xb_np, Yb_np)
                epoch_loss += loss_base
                n_batches += 1

                # ES gradient estimation
                params = list(net.parameters())
                param_shapes = [p.shape for p in params]
                n_params_total = sum(p.numel() for p in params)

                grad_estimate = torch.zeros(n_params_total, device=self.device)

                for _ in range(n_perturbations):
                    eps = torch.randn(n_params_total, device=self.device)
                    # Apply positive perturbation
                    offset = 0
                    for p in params:
                        n_p = p.numel()
                        p.data += sigma * eps[offset: offset + n_p].view(p.shape)
                        offset += n_p

                    loss_plus = self._physics_loss(net, Xb_np, Yb_np)

                    # Restore and apply negative perturbation
                    offset = 0
                    for p in params:
                        n_p = p.numel()
                        p.data -= 2 * sigma * eps[offset: offset + n_p].view(p.shape)
                        offset += n_p

                    loss_minus = self._physics_loss(net, Xb_np, Yb_np)

                    # Restore
                    offset = 0
                    for p in params:
                        n_p = p.numel()
                        p.data += sigma * eps[offset: offset + n_p].view(p.shape)
                        offset += n_p

                    # Accumulate gradient estimate
                    grad_estimate += ((loss_plus - loss_minus) / (2 * sigma)) * eps

                grad_estimate /= n_perturbations

                # Manual gradient update (gradient descent)
                offset = 0
                for p in params:
                    n_p = p.numel()
                    p.data -= lr * grad_estimate[offset: offset + n_p].view(p.shape)
                    offset += n_p

            avg_loss = epoch_loss / max(n_batches, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        return net

    def _generate_collocation_grid(self, n_points: int) -> np.ndarray:
        rng = np.random.default_rng(seed=12345)
        total_dim = self._fm.total_input_dim
        return rng.uniform(-2.0, 2.0, size=(n_points, total_dim))

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def _save(self) -> None:
        d = self._output_dir
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self._reducer.state_dict(), d / "reducer.pt")
        meta = {
            "input_dim": self._fm.total_input_dim,
            "hidden_dims": self._cfg["phase3"]["nn"]["hidden_dims"],
            "training_signal": self._cfg["phase3"]["training_signal"],
            "config_hash": self._cm.config_hash(),
        }
        with open(d / "config.json", "w", encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        np.save(d / "X_mean.npy", self._X_mean)
        np.save(d / "X_std.npy", self._X_std)
