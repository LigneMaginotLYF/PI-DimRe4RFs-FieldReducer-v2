"""
phase3_data_generator.py
========================
Phase 3: Dataset generation in the FULL parameter space.

Samples N₃ random points in the full parameter space defined by
``phase3.full_fields`` and runs the Biot solver on each to obtain the
corresponding settlement profiles (used as ground truth during training
and evaluation).

The full parameter space is INDEPENDENT of Phase 2 sampling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config_manager import ConfigManager
from .field_manager import FieldManager
from .forward_solver import BiotSolver


class Phase3DataGenerator:
    """Generate Phase-3 training data in the full parameter space.

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
        p3 = self._cfg["phase3"]

        # FieldManager for FULL parameter space
        full_fields_cfg = p3.get("full_fields")
        self._fm_full = FieldManager(self._cfg, fields_override=full_fields_cfg)
        self._solver = BiotSolver(self._cfg)
        self._output_dir = Path(output_dir or p3["output_dir"])
        self._n_samples: int = int(p3.get("n_training_samples", 500))

    @property
    def field_manager_full(self) -> FieldManager:
        return self._fm_full

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Sample full-space parameters and run Biot solver.

        Returns
        -------
        X_full : (n_samples, full_dim)
            Concatenated full-space coefficient vectors [ξ_E, ξ_kh, ξ_kv].
        Y_full : (n_samples, n_nodes_x)
            Settlement profiles from Biot solver applied to FULL fields.
        """
        full_dim = self._fm_full.total_input_dim
        print(
            f"[Phase 3 DataGen] Generating {self._n_samples} samples "
            f"in full space (dim={full_dim}) ..."
        )
        X_full, fields, _ = self._fm_full.generate_dataset(self._n_samples)
        print(f"[Phase 3 DataGen] Running Biot solver on {self._n_samples} samples ...")
        Y_full = self._solver.run_batch(fields["E"], fields["k_h"], fields["k_v"])
        print(f"[Phase 3 DataGen] Done. X_full: {X_full.shape}, Y_full: {Y_full.shape}")
        return X_full, Y_full

    def generate_single(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single unseen sample in the full parameter space.

        Used for hybrid physics checks during Phase-3 training.

        Returns
        -------
        X : (1, full_dim)
        Y : (1, n_nodes_x)
        """
        X_full, fields, _ = self._fm_full.generate_dataset(1)
        Y_full = self._solver.run_batch(fields["E"], fields["k_h"], fields["k_v"])
        return X_full, Y_full

    def save(self, X_full: np.ndarray, Y_full: np.ndarray) -> Dict[str, str]:
        """Persist dataset to disk."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        xp = self._output_dir / "phase3_X_full.npy"
        yp = self._output_dir / "phase3_Y_full.npy"
        np.save(xp, X_full)
        np.save(yp, Y_full)
        meta = {
            "n_samples": len(X_full),
            "full_dim": int(self._fm_full.total_input_dim),
            "n_nodes_x": self._fm_full.n_nodes_x,
        }
        with open(self._output_dir / "phase3_data_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return {"X_full": str(xp), "Y_full": str(yp)}

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load previously generated dataset."""
        X_full = np.load(self._output_dir / "phase3_X_full.npy")
        Y_full = np.load(self._output_dir / "phase3_Y_full.npy")
        return X_full, Y_full
