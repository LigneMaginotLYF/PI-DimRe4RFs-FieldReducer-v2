"""
phase1_dataset.py
=================
Phase 1: Dataset generation with train/validation split.

Generates random field realisations for E, k_h, k_v using FieldManager, runs
the Biot solver on each realisation, and saves:
  - X_train / X_val  : coefficient arrays (n_samples, total_input_dim)
  - Y_train / Y_val  : settlement profiles (n_samples, n_nodes_x)
  - dataset_metadata.json : reproducibility info, shapes, seeds, solver config

All random seeds are fixed per field for full reproducibility.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config_manager import ConfigManager
from .field_manager import FieldManager
from .forward_solver import BiotSolver


class Phase1DatasetGenerator:
    """Generate and persist the Phase-1 training dataset.

    Parameters
    ----------
    config_manager : ConfigManager
    output_dir : str, optional
        Override the output directory from config.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        output_dir: Optional[str] = None,
    ) -> None:
        self._cm = config_manager
        cfg = config_manager.cfg
        self._cfg = cfg

        self._fm = FieldManager(cfg)
        self._solver = BiotSolver(cfg)

        self._output_dir = Path(output_dir or cfg["phase1"]["output_dir"])
        self._n_samples: int = cfg["phase1"]["n_samples"]
        self._val_fraction: float = cfg["phase1"]["val_fraction"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Generate the dataset and save to disk.

        Returns
        -------
        dict with paths to all saved artefacts.
        """
        print(f"[Phase 1] Generating {self._n_samples} samples ...")
        X, fields, xi_dict = self._fm.generate_dataset(self._n_samples)

        print("[Phase 1] Running Biot solver ...")
        Y = self._solver.run_batch(
            fields["E"], fields["k_h"], fields["k_v"]
        )

        # Train / validation split
        X_train, X_val, Y_train, Y_val = self._split(X, Y)

        # Save
        self._output_dir.mkdir(parents=True, exist_ok=True)
        paths = self._save(X_train, X_val, Y_train, Y_val)

        # Metadata
        meta = self._build_metadata(X_train, Y_train, X_val, Y_val)
        meta_path = self._output_dir / "dataset_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        paths["metadata"] = str(meta_path)

        print(f"[Phase 1] Done. Saved to '{self._output_dir}'")
        return paths

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load previously generated dataset from disk.

        Returns
        -------
        X_train, Y_train, X_val, Y_val
        """
        d = self._output_dir
        return (
            np.load(d / "X_train.npy"),
            np.load(d / "Y_train.npy"),
            np.load(d / "X_val.npy"),
            np.load(d / "Y_val.npy"),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split(
        self, X: np.ndarray, Y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(X)
        n_val = max(1, int(n * self._val_fraction))
        rng = np.random.default_rng(seed=0)
        perm = rng.permutation(n)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        return X[train_idx], X[val_idx], Y[train_idx], Y[val_idx]

    def _save(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        Y_train: np.ndarray,
        Y_val: np.ndarray,
    ) -> Dict[str, str]:
        d = self._output_dir
        np.save(d / "X_train.npy", X_train)
        np.save(d / "X_val.npy", X_val)
        np.save(d / "Y_train.npy", Y_train)
        np.save(d / "Y_val.npy", Y_val)
        return {
            "X_train": str(d / "X_train.npy"),
            "X_val": str(d / "X_val.npy"),
            "Y_train": str(d / "Y_train.npy"),
            "Y_val": str(d / "Y_val.npy"),
        }

    def _build_metadata(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
    ) -> Dict[str, Any]:
        fm = self._fm
        return {
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_samples_total": self._n_samples,
            "input_dim": fm.total_input_dim,
            "n_nodes_x": fm.n_nodes_x,
            "n_nodes_z": fm.n_nodes_z,
            "field_dims": {
                name: fm.field_configs[name].effective_dim
                for name in FieldManager.FIELD_NAMES
            },
            "seeds": {
                name: fm.field_configs[name].seed
                for name in FieldManager.FIELD_NAMES
            },
            "solver_type": self._cfg["solver"]["type"],
            "solver_mode": self._cfg["solver"]["mode"],
            "X_shape_train": list(X_train.shape),
            "Y_shape_train": list(Y_train.shape),
            "config_hash": self._cm.config_hash(),
        }
