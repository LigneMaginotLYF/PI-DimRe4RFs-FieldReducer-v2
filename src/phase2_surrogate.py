"""
phase2_surrogate.py
===================
Phase 2: Surrogate model in the REDUCED parameter space.

Maps reduced parameters → settlement profile.
Trained independently from Phase 3 with optional hybrid physics
regularization (periodic Biot check every N epochs).

Architecture
------------
- Input: reduced-space parameter vector ξ' = [ξ'_E, ξ'_kh, ξ'_kv]
  (dimension = sum of n_terms per reduced field, each min 1)
- Output: settlement profile Y at n_nodes_x surface points

Training signal options (``phase2.training_signal``):
  - "data"    : supervised MSE on collocation dataset only
  - "physics" : periodic Biot check every ``physics_check_interval`` epochs
  - "hybrid"  : supervised MSE + weighted physics check

Dimension-stamped filenames
---------------------------
The surrogate is saved as::

    surrogate_{type}_dim{d_reduced}.pt   (NN)
    surrogate_{type}_dim{d_reduced}.pkl  (PCE)

where ``d_reduced`` is the reduced input dimension.
"""

from __future__ import annotations

import glob as _glob
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config_manager import ConfigManager
from .field_manager import FieldManager
from .forward_solver import BiotSolver
from .phase2_data_generator import Phase2DataGenerator
from .surrogate_models import BaseSurrogate, build_surrogate, load_surrogate


class Phase2Surrogate:
    """Phase-2 surrogate training in the reduced parameter space.

    Parameters
    ----------
    config_manager : ConfigManager
    output_dir : str, optional
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        output_dir: Optional[str] = None,
    ) -> None:
        self._cm = config_manager
        self._cfg = config_manager.cfg
        p2 = self._cfg["phase2"]

        # Use the REDUCED parameter space FieldManager
        reduced_fields_cfg = p2.get("reduced_fields")
        self._fm = FieldManager(self._cfg, fields_override=reduced_fields_cfg)
        self._solver = BiotSolver(self._cfg)
        self._output_dir = Path(output_dir or p2["output_dir"])
        self._surrogate: Optional[BaseSurrogate] = None

        # Data generator for reduced space
        self._data_gen = Phase2DataGenerator(config_manager, output_dir=str(self._output_dir))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        X_train: Optional[np.ndarray] = None,
        Y_train: Optional[np.ndarray] = None,
    ) -> BaseSurrogate:
        """Run the full Phase-2 pipeline.

        Generates training data in the reduced parameter space,
        trains the surrogate, and saves all artefacts.

        Parameters
        ----------
        X_train, Y_train : Optional external data (rarely needed).

        Returns
        -------
        Trained surrogate model.
        """
        p2 = self._cfg["phase2"]
        signal_type = p2.get("training_signal", "data")

        # --- 1. Generate data in REDUCED parameter space ---
        print("[Phase 2] Generating training data in reduced parameter space ...")
        X, Y = self._data_gen.generate()

        # --- 2. Optionally augment with external data ---
        if X_train is not None and Y_train is not None:
            if X_train.shape[1] == X.shape[1]:
                print(f"[Phase 2] Augmenting with {len(X_train)} external samples")
                X = np.concatenate([X, X_train], axis=0)
                Y = np.concatenate([Y, Y_train], axis=0)
            else:
                print(
                    f"[Phase 2] Warning: external data dim {X_train.shape[1]} != "
                    f"reduced dim {X.shape[1]} — skipping augmentation"
                )

        # --- 3. Train/test split ---
        eval_cfg = p2.get("evaluation", {})
        test_fraction = float(eval_cfg.get("test_fraction", 0.2))
        n_total = len(X)
        n_test = max(1, int(n_total * test_fraction))
        idx = np.random.default_rng(seed=1234).permutation(n_total)
        X_test, Y_test = X[idx[:n_test]], Y[idx[:n_test]]
        X_tr, Y_tr = X[idx[n_test:]], Y[idx[n_test:]]

        # --- 4. Build and train surrogate ---
        print(
            f"[Phase 2] Training {p2['surrogate_type']} surrogate "
            f"(input_dim={self._fm.total_input_dim}, "
            f"signal={signal_type}) ..."
        )
        self._surrogate = self._build_and_fit_surrogate(X_tr, Y_tr, signal_type)

        # --- 5. Save ---
        self._save(X, Y, X_test, Y_test)
        print(f"[Phase 2] Surrogate saved to '{self._output_dir}'")
        return self._surrogate

    def load_surrogate(self) -> BaseSurrogate:
        """Load a previously trained Phase-2 surrogate."""
        d_reduced = self._fm.total_input_dim
        surr_path = _find_surrogate_file(self._output_dir, d_reduced)
        self._surrogate = load_surrogate(surr_path)
        return self._surrogate

    @property
    def surrogate(self) -> Optional[BaseSurrogate]:
        return self._surrogate

    @property
    def reduced_dim(self) -> int:
        """Dimension of the reduced parameter space."""
        return self._fm.total_input_dim

    @property
    def field_manager(self) -> FieldManager:
        """FieldManager for the reduced parameter space."""
        return self._fm

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_and_fit_surrogate(
        self,
        X_tr: np.ndarray,
        Y_tr: np.ndarray,
        signal_type: str,
    ) -> BaseSurrogate:
        p2 = self._cfg["phase2"]
        nn_cfg = p2.get("nn", {})

        surrogate = build_surrogate(
            surrogate_type=p2["surrogate_type"],
            input_dim=self._fm.total_input_dim,
            n_nodes_x=self._fm.n_nodes_x,
            output_repr=p2["output_repr"],
            n_output_modes=p2["n_output_modes"],
            nn_cfg=nn_cfg,
            pce_cfg=p2.get("pce", {}),
        )

        if signal_type in ("physics", "hybrid"):
            surrogate = self._fit_with_physics_checks(surrogate, X_tr, Y_tr, p2, nn_cfg)
        else:
            surrogate.fit(X_tr, Y_tr)

        return surrogate

    def _fit_with_physics_checks(
        self,
        surrogate: BaseSurrogate,
        X_tr: np.ndarray,
        Y_tr: np.ndarray,
        p2: Dict,
        nn_cfg: Dict,
    ) -> BaseSurrogate:
        """Training loop with periodic Biot physics check (NN only)."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        if not hasattr(surrogate, "_model"):
            surrogate.fit(X_tr, Y_tr)
            return surrogate

        # Pre-fit initializes normalization parameters
        surrogate.fit(X_tr, Y_tr)

        physics_weight = float(p2.get("hybrid_alpha", 0.1))
        check_interval = int(p2.get("physics_check_interval", 10))
        epochs = int(nn_cfg.get("epochs", 200))
        lr = float(nn_cfg.get("lr", 1e-3))
        batch_size = int(nn_cfg.get("batch_size", 32))
        patience = int(nn_cfg.get("patience", 50))
        signal_type = p2.get("training_signal", "data")

        device = next(surrogate._model.parameters()).device

        Xn = (X_tr - surrogate._X_mean) / surrogate._X_std
        Yn = (Y_tr - surrogate._Y_mean) / surrogate._Y_std

        X_t = torch.tensor(Xn, dtype=torch.float32, device=device)
        Y_t = torch.tensor(Yn, dtype=torch.float32, device=device)

        dataset = TensorDataset(X_t, Y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimiser = torch.optim.Adam(surrogate._model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        surrogate._model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for Xb, Yb in loader:
                optimiser.zero_grad()
                pred = surrogate._model(Xb)
                loss = loss_fn(pred, Yb)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            # Physics check every check_interval epochs
            if signal_type in ("physics", "hybrid") and (epoch + 1) % check_interval == 0:
                x_new, y_new = self._data_gen.generate_single()
                surrogate._model.eval()
                with torch.no_grad():
                    xn_new = (x_new - surrogate._X_mean) / surrogate._X_std
                    xt_new = torch.tensor(xn_new, dtype=torch.float32, device=device)
                    pred_n = surrogate._model(xt_new).cpu().numpy()
                    pred_new = pred_n * surrogate._Y_std + surrogate._Y_mean

                physics_mse = float(np.mean((pred_new - y_new) ** 2))
                print(
                    f"[Phase 2] Epoch {epoch+1:4d}/{epochs} | "
                    f"data_loss={avg_loss:.4e} | "
                    f"physics_check_MSE={physics_mse:.4e}"
                )
                surrogate._model.train()

                if signal_type == "hybrid":
                    xn_new_t = torch.tensor(
                        (x_new - surrogate._X_mean) / surrogate._X_std,
                        dtype=torch.float32, device=device
                    )
                    yn_new_t = torch.tensor(
                        (y_new - surrogate._Y_mean) / surrogate._Y_std,
                        dtype=torch.float32, device=device
                    )
                    optimiser.zero_grad()
                    pred_phys = surrogate._model(xn_new_t)
                    phys_loss = physics_weight * loss_fn(pred_phys, yn_new_t)
                    phys_loss.backward()
                    optimiser.step()
            elif (epoch + 1) % 20 == 0:
                print(
                    f"[Phase 2] Epoch {epoch+1:4d}/{epochs} | "
                    f"data_loss={avg_loss:.4e}"
                )

            # Validation / early stopping
            surrogate._model.eval()
            with torch.no_grad():
                val_pred = surrogate._model(X_t)
                val_loss = loss_fn(val_pred, Y_t).item()
            surrogate._model.train()

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in surrogate._model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[Phase 2] Early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            surrogate._model.load_state_dict(best_state)
        surrogate._model.eval()
        return surrogate

    def _save(
        self,
        X_all: np.ndarray,
        Y_all: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
    ) -> None:
        d = self._output_dir
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "phase2_X_test.npy", X_test)
        np.save(d / "phase2_Y_test.npy", Y_test)

        p2 = self._cfg["phase2"]
        surrogate_type = p2["surrogate_type"]
        d_reduced = self._fm.total_input_dim
        ext = "pt" if surrogate_type == "nn" else "pkl"
        surr_filename = f"surrogate_{surrogate_type}_dim{d_reduced}.{ext}"
        self._surrogate.save(d / surr_filename)

        meta = {
            "input_dim": d_reduced,
            "reduced_dim": d_reduced,
            "n_nodes_x": self._fm.n_nodes_x,
            "surrogate_type": surrogate_type,
            "surrogate_filename": surr_filename,
            "output_repr": p2["output_repr"],
            "n_output_modes": p2["n_output_modes"],
            "training_signal": p2.get("training_signal", "data"),
            "config_hash": self._cm.config_hash(),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        with open(d / "config.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Module-level helper (used by Phase 3 for dimension-validated loading)
# ---------------------------------------------------------------------------

def _find_surrogate_file(surr_dir: Path, d_expected: int) -> Path:
    """Find a dimension-stamped surrogate file in *surr_dir*.

    Parameters
    ----------
    surr_dir : Path
    d_expected : int
        Expected input dimension.

    Returns
    -------
    Path to the surrogate file.

    Raises
    ------
    ValueError if no match found.
    """
    surr_dir = Path(surr_dir)
    matches = []
    for ext in ("pt", "pkl"):
        pattern = str(surr_dir / f"surrogate_*_dim{d_expected}.{ext}")
        matches.extend(_glob.glob(pattern))
    if matches:
        return Path(matches[0])

    legacy = surr_dir / "surrogate"
    if legacy.exists():
        return legacy

    if surr_dir.exists():
        available = [
            p.name for p in surr_dir.iterdir()
            if p.suffix in (".pt", ".pkl") and p.name.startswith("surrogate_")
        ]
    else:
        available = []
    raise ValueError(
        f"No Phase-2 surrogate found with input dimension {d_expected} "
        f"in '{surr_dir}'. "
        f"Expected file pattern: surrogate_*_dim{d_expected}.(pt|pkl). "
        f"Available surrogate files: {available}. "
        "Re-run Phase 2 with the current configuration to generate a "
        "matching surrogate."
    )
