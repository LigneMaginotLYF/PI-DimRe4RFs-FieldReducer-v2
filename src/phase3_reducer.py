"""
phase3_reducer.py
=================
Phase 3: Dimension reducer from FULL to REDUCED parameter space.

Maps full-dimensional concatenated coefficients → reduced-dimensional
concatenated coefficients that can be used as Phase-2 surrogate input.

  P3 : ℝ^{full_dim} → ℝ^{reduced_dim}

where ``full_dim``    = sum(max(n_terms, 1) for each field in full_fields)
      ``reduced_dim`` = sum(max(n_terms, 1) for each field in reduced_fields)

The reduced_fields MUST match phase2.reduced_fields so that the Phase-2
surrogate can evaluate the output of P3.

Training signal options (``phase3.training_signal``):
  - "surrogate" : L = ||Y_full - P2(P3(ξ_full))||  (fast, differentiable)
  - "physics"   : L = ||Y_full - P(reconstruct(P3(ξ_full)))||  (Biot, slow)

Evaluation is ALWAYS physics-driven (Biot solver), regardless of training mode.
"""

from __future__ import annotations

import json
from datetime import datetime
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
from .phase3_data_generator import Phase3DataGenerator


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
    """Dimension reducer: FULL parameters → REDUCED parameters.

    Parameters
    ----------
    config_manager : ConfigManager
    output_dir : str, optional
    surrogate_dir : str, optional
        Path to Phase-2 surrogate directory.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        output_dir: Optional[str] = None,
        surrogate_dir: Optional[str] = None,
    ) -> None:
        self._cm = config_manager
        self._cfg = config_manager.cfg
        p3 = self._cfg["phase3"]

        # Full-space FieldManager (input to reducer)
        full_fields_cfg = p3.get("full_fields")
        self._fm_full = FieldManager(self._cfg, fields_override=full_fields_cfg)

        # Reduced-space FieldManager (output of reducer; must match phase2)
        reduced_fields_cfg = p3.get("reduced_fields")
        self._fm_reduced = FieldManager(self._cfg, fields_override=reduced_fields_cfg)

        self._solver = BiotSolver(self._cfg)
        self._output_dir = Path(output_dir or p3["output_dir"])
        self._surrogate_dir = Path(surrogate_dir or p3["surrogate_dir"])

        self._reducer: Optional[_ReducerNet] = None
        self._surrogate: Optional[BaseSurrogate] = None

        # Normalisation statistics (full-space input)
        self._X_mean: Optional[np.ndarray] = None
        self._X_std: Optional[np.ndarray] = None
        # Normalisation statistics (reduced-space output target)
        self._Xr_mean: Optional[np.ndarray] = None
        self._Xr_std: Optional[np.ndarray] = None

        # Data generator for full space
        self._data_gen = Phase3DataGenerator(config_manager, output_dir=str(self._output_dir))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def full_dim(self) -> int:
        return self._fm_full.total_input_dim

    @property
    def reduced_dim(self) -> int:
        return self._fm_reduced.total_input_dim

    @property
    def field_manager_full(self) -> FieldManager:
        return self._fm_full

    @property
    def field_manager_reduced(self) -> FieldManager:
        return self._fm_reduced

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        X_train: Optional[np.ndarray] = None,
        Y_train: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
    ) -> "_ReducerNet":
        """Train the dimension reducer.

        Generates a fresh dataset in the FULL parameter space if
        X_train/Y_train are not provided.

        Parameters
        ----------
        X_train : (n_train, full_dim) — full-dim coefficients (optional)
        Y_train : (n_train, n_nodes_x) — ground-truth settlements (optional)
        X_val, Y_val : optional separate validation data

        Returns
        -------
        Trained reducer network.
        """
        p3 = self._cfg["phase3"]
        signal_type = p3.get("training_signal", "surrogate")

        # --- 1. Generate / use data in FULL parameter space ---
        if X_train is None or Y_train is None:
            print("[Phase 3] Generating training data in full parameter space ...")
            X_full, Y_full = self._data_gen.generate()
        else:
            print(f"[Phase 3] Using provided training data: {X_train.shape}")
            X_full, Y_full = X_train, Y_train

        # --- 2. Validate input dimension ---
        if X_full.shape[1] != self.full_dim:
            raise ValueError(
                f"Phase 3 expects full_dim={self.full_dim} but got "
                f"X.shape[1]={X_full.shape[1]}. "
                "Check that phase3.full_fields matches the provided data."
            )

        # --- 3. Train/val split ---
        eval_cfg = p3.get("evaluation", {})
        test_fraction = float(eval_cfg.get("test_fraction", 0.2))
        n_total = len(X_full)
        n_test = max(1, int(n_total * test_fraction))
        idx = np.random.default_rng(seed=2345).permutation(n_total)
        X_test_full = X_full[idx[:n_test]]
        Y_test_full = Y_full[idx[:n_test]]
        X_tr = X_full[idx[n_test:]]
        Y_tr = Y_full[idx[n_test:]]

        # --- 4. Compute normalisation stats ---
        self._X_mean = X_tr.mean(axis=0)
        self._X_std = X_tr.std(axis=0)
        self._X_std[self._X_std < 1e-12] = 1.0

        # Use a sample of reduced-space params to compute Xr normalization
        # (sample from reduced space directly)
        from .phase2_data_generator import Phase2DataGenerator
        p2_gen = Phase2DataGenerator(self._cm)
        Xr_sample, _ = p2_gen.generate()
        self._Xr_mean = Xr_sample.mean(axis=0)
        self._Xr_std = Xr_sample.std(axis=0)
        self._Xr_std[self._Xr_std < 1e-12] = 1.0

        p3_nn = p3["nn"]
        print(
            f"[Phase 3] Training reducer "
            f"(full_dim={self.full_dim} → reduced_dim={self.reduced_dim}, "
            f"signal={signal_type}) ..."
        )

        # --- 5. Load Phase-2 surrogate if needed ---
        surrogate = None
        if signal_type == "surrogate":
            print(f"[Phase 3] Loading Phase-2 surrogate from '{self._surrogate_dir}' ...")
            surrogate = self._load_phase2_surrogate()
            self._surrogate = surrogate

        # --- 6. Train ---
        if signal_type == "surrogate" and surrogate is not None:
            self._reducer = self._train_with_surrogate(
                X_tr, Y_tr, X_test_full, Y_test_full, surrogate, p3_nn
            )
        else:
            self._reducer = self._train_with_physics(
                X_tr, Y_tr, X_test_full, Y_test_full, p3_nn
            )

        # --- 7. Save ---
        self._save(X_test_full, Y_test_full)
        print(f"[Phase 3] Reducer saved to '{self._output_dir}'")
        return self._reducer

    def reduce(self, X: np.ndarray) -> np.ndarray:
        """Map full-dimensional coefficients to reduced-dimensional space.

        Parameters
        ----------
        X : (n_samples, full_dim) or (full_dim,)

        Returns
        -------
        X_reduced : (n_samples, reduced_dim) or (reduced_dim,)
        """
        if self._reducer is None:
            raise RuntimeError("Reducer not trained. Call run() or load() first.")
        single = X.ndim == 1
        if single:
            X = X[np.newaxis, :]

        Xn = (X - self._X_mean) / self._X_std
        xt = torch.tensor(Xn, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            reduced_n = self._reducer(xt).cpu().numpy()

        # De-normalise to reduced coefficient scale
        result = reduced_n * self._Xr_std + self._Xr_mean
        return result[0] if single else result

    def load(self) -> "_ReducerNet":
        """Load a previously trained reducer from disk."""
        d = self._output_dir
        with open(d / "config.json", encoding="utf-8") as f:
            meta = json.load(f)
        full_dim = meta["full_dim"]
        reduced_dim = meta["reduced_dim"]
        hidden_dims = meta["hidden_dims"]
        self._reducer = _ReducerNet(full_dim, reduced_dim, hidden_dims).to(self.device)
        state = torch.load(d / "reducer.pt", map_location=self.device, weights_only=True)
        self._reducer.load_state_dict(state)
        self._reducer.eval()
        self._X_mean = np.load(d / "X_mean.npy")
        self._X_std = np.load(d / "X_std.npy")
        self._Xr_mean = np.load(d / "Xr_mean.npy")
        self._Xr_std = np.load(d / "Xr_std.npy")
        return self._reducer

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_phase2_surrogate(self) -> BaseSurrogate:
        """Load the Phase-2 surrogate with dimension validation.

        The surrogate must accept the REDUCED dimension as input.
        """
        d_expected = self.reduced_dim
        surr_path = _find_surrogate_file(self._surrogate_dir, d_expected)
        surrogate = load_surrogate(surr_path)

        if surrogate.input_dim != d_expected:
            raise ValueError(
                f"Phase-2 surrogate input dimension mismatch: "
                f"expected {d_expected} (reduced_dim), got {surrogate.input_dim}. "
                f"Loaded from: {surr_path}. "
                "Ensure Phase 2 was run with the same reduced_fields configuration."
            )
        return surrogate

    def _train_with_surrogate(
        self,
        X_tr: np.ndarray,
        Y_tr: np.ndarray,
        X_val: Optional[np.ndarray],
        Y_val: Optional[np.ndarray],
        surrogate: BaseSurrogate,
        p3_nn: Dict,
    ) -> _ReducerNet:
        """Gradient-based training: loss = ||Y_full - P2(P3(ξ_full))||.

        Phase-2 surrogate is frozen; only reducer parameters are updated.
        """
        net = _ReducerNet(
            self.full_dim, self.reduced_dim,
            p3_nn.get("hidden_dims", [256, 128, 64])
        ).to(self.device)

        use_surrogate_grad = hasattr(surrogate, "_model") and surrogate._model is not None

        optimiser = torch.optim.Adam(net.parameters(), lr=p3_nn.get("lr", 1e-3))
        loss_fn = nn.MSELoss()

        # Normalise full-space inputs
        Xn = (X_tr - self._X_mean) / self._X_std
        X_t = torch.tensor(Xn, dtype=torch.float32, device=self.device)
        Y_t = torch.tensor(Y_tr, dtype=torch.float32, device=self.device)

        batch_size = p3_nn.get("batch_size", 32)
        patience = p3_nn.get("patience", 50)
        epochs = p3_nn.get("epochs", 300)

        dataset = TensorDataset(X_t, Y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        # Surrogate normalisation tensors
        if use_surrogate_grad:
            surr_X_mean = torch.tensor(surrogate._X_mean, dtype=torch.float32, device=self.device)
            surr_X_std = torch.tensor(surrogate._X_std, dtype=torch.float32, device=self.device)
            surr_Y_mean = torch.tensor(surrogate._Y_mean, dtype=torch.float32, device=self.device)
            surr_Y_std = torch.tensor(surrogate._Y_std, dtype=torch.float32, device=self.device)
            Xr_mean_t = torch.tensor(self._Xr_mean, dtype=torch.float32, device=self.device)
            Xr_std_t = torch.tensor(self._Xr_std, dtype=torch.float32, device=self.device)

        net.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for Xb, Yb in loader:
                optimiser.zero_grad()
                # Predict reduced coefficients (normalised)
                reduced_n = net(Xb)
                # De-normalise to reduced coefficient scale
                reduced_denorm = reduced_n * Xr_std_t + Xr_mean_t

                if use_surrogate_grad:
                    # Normalize for surrogate
                    surr_Xn = (reduced_denorm - surr_X_mean) / surr_X_std
                    surr_pred_n = surrogate._model(surr_Xn)
                    surr_pred = surr_pred_n * surr_Y_std + surr_Y_mean
                    if surrogate.output_repr != "direct":
                        surr_pred_np = surr_pred.detach().cpu().numpy()
                        surr_pred_np = surrogate._reconstruct_output(surr_pred_np)
                        surr_pred = torch.tensor(
                            surr_pred_np, dtype=torch.float32, device=self.device
                        )
                    loss = loss_fn(surr_pred, Yb)
                else:
                    # Non-differentiable: use identity regularization
                    loss = 0.01 * loss_fn(reduced_denorm,
                                          torch.zeros_like(reduced_denorm))

                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            # Validation
            net.eval()
            with torch.no_grad():
                red_full_n = net(X_t)
                red_full = red_full_n * Xr_std_t + Xr_mean_t
                pred_np = surrogate.predict(red_full.cpu().numpy())
                pred_t = torch.tensor(pred_np, dtype=torch.float32, device=self.device)
                val_loss = loss_fn(pred_t, Y_t).item()
            net.train()

            if (epoch + 1) % 20 == 0:
                print(
                    f"[Phase 3] Epoch {epoch+1:4d}/{epochs} | "
                    f"train_loss={avg_loss:.4e} | val_loss={val_loss:.4e}"
                )

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[Phase 3] Early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        return net

    def _train_with_physics(
        self,
        X_tr: np.ndarray,
        Y_tr: np.ndarray,
        X_val: Optional[np.ndarray],
        Y_val: Optional[np.ndarray],
        p3_nn: Dict,
    ) -> _ReducerNet:
        """Physics-based training: loss = ||Y_full - P(reduced_fields)||.

        Non-differentiable; uses Evolution Strategies gradient estimation.
        """
        net = _ReducerNet(
            self.full_dim, self.reduced_dim,
            p3_nn.get("hidden_dims", [256, 128, 64])
        ).to(self.device)

        epochs = p3_nn.get("epochs", 300)
        batch_size = p3_nn.get("batch_size", 32)
        patience = p3_nn.get("patience", 50)
        lr = p3_nn.get("lr", 1e-3)
        sigma = 0.02
        n_es_samples = 20

        optimiser = torch.optim.Adam(net.parameters(), lr=lr)

        Xn = (X_tr - self._X_mean) / self._X_std
        Xr_mean_t = torch.tensor(self._Xr_mean, dtype=torch.float32, device=self.device)
        Xr_std_t = torch.tensor(self._Xr_std, dtype=torch.float32, device=self.device)

        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            # Batch selection
            n_train = len(X_tr)
            batch_idx = np.random.choice(n_train, size=min(batch_size, n_train), replace=False)
            Xb_np = Xn[batch_idx]
            Yb_np = Y_tr[batch_idx]

            Xb_t = torch.tensor(Xb_np, dtype=torch.float32, device=self.device)

            # ES gradient estimation
            theta = list(net.parameters())
            theta_flat = torch.cat([p.data.view(-1) for p in theta])
            n_params = len(theta_flat)

            perturbations = torch.randn(n_es_samples, n_params, device=self.device)
            losses = np.zeros(n_es_samples)

            for k in range(n_es_samples):
                # Perturb
                offset = 0
                for p in theta:
                    sz = p.numel()
                    p.data.add_(sigma * perturbations[k, offset:offset + sz].view(p.shape))
                    offset += sz

                with torch.no_grad():
                    red_n = net(Xb_t)
                    red_denorm = (red_n * Xr_std_t + Xr_mean_t).cpu().numpy()

                fields = self._fm_reduced.reconstruct_all_fields(red_denorm)
                pred = self._solver.run_batch(fields["E"], fields["k_h"], fields["k_v"])
                losses[k] = float(np.mean((pred - Yb_np) ** 2))

                # Restore
                offset = 0
                for p in theta:
                    sz = p.numel()
                    p.data.sub_(sigma * perturbations[k, offset:offset + sz].view(p.shape))
                    offset += sz

            # ES gradient
            losses_t = torch.tensor(losses, dtype=torch.float32, device=self.device)
            losses_norm = (losses_t - losses_t.mean()) / (losses_t.std() + 1e-8)
            grad_flat = (losses_norm.unsqueeze(1) * perturbations).mean(0) / sigma

            optimiser.zero_grad()
            offset = 0
            for p in theta:
                sz = p.numel()
                p.grad = grad_flat[offset:offset + sz].view(p.shape).clone()
                offset += sz
            optimiser.step()

            avg_loss = float(losses.mean())

            # Full validation
            net.eval()
            with torch.no_grad():
                Xn_val_t = torch.tensor(Xn[:min(50, len(Xn))], dtype=torch.float32, device=self.device)
                red_n = net(Xn_val_t)
                red_denorm = (red_n * Xr_std_t + Xr_mean_t).cpu().numpy()
            fields = self._fm_reduced.reconstruct_all_fields(red_denorm)
            val_pred = self._solver.run_batch(fields["E"], fields["k_h"], fields["k_v"])
            val_loss = float(np.mean((val_pred - Y_tr[:min(50, len(Y_tr))]) ** 2))
            net.train()

            if (epoch + 1) % 20 == 0:
                print(
                    f"[Phase 3] Epoch {epoch+1:4d}/{epochs} | "
                    f"train_loss={avg_loss:.4e} | val_loss={val_loss:.4e} (physics)"
                )

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[Phase 3] Early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        return net

    def _save(self, X_test: np.ndarray, Y_test: np.ndarray) -> None:
        d = self._output_dir
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "phase3_X_test_full.npy", X_test)
        np.save(d / "phase3_Y_test_full.npy", Y_test)

        p3 = self._cfg["phase3"]
        p3_nn = p3["nn"]
        hidden_dims = p3_nn.get("hidden_dims", [256, 128, 64])

        torch.save(self._reducer.state_dict(), d / "reducer.pt")
        np.save(d / "X_mean.npy", self._X_mean)
        np.save(d / "X_std.npy", self._X_std)
        np.save(d / "Xr_mean.npy", self._Xr_mean)
        np.save(d / "Xr_std.npy", self._Xr_std)

        meta = {
            "full_dim": self.full_dim,
            "reduced_dim": self.reduced_dim,
            "hidden_dims": hidden_dims,
            "training_signal": p3.get("training_signal", "surrogate"),
            "surrogate_dir": str(self._surrogate_dir),
            "config_hash": self._cm.config_hash(),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        with open(d / "config.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _generate_collocation_grid(self, n_points: int) -> np.ndarray:
        """Generate a grid in the full parameter space for legacy compatibility."""
        rng = np.random.default_rng(seed=999)
        total_dim = self.full_dim
        return rng.uniform(-2.0, 2.0, size=(n_points, total_dim))
