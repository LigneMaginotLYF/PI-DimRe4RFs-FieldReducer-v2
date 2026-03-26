"""
phase2_data_generator.py
========================
Phase 2: Dataset generation in the REDUCED parameter space.

Samples N₂ random points in the reduced parameter space defined by
``phase2.reduced_fields`` and runs the Biot solver on each to obtain
the corresponding settlement profiles.

The reduced parameter space is INDEPENDENT of Phase 1 and Phase 3.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config_manager import ConfigManager
from .field_manager import FieldManager
from .forward_solver import BiotSolver


class Phase2DataGenerator:
    """Generate Phase-2 training data in the reduced parameter space.

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

        # FieldManager for REDUCED parameter space
        reduced_fields_cfg = p2.get("reduced_fields")
        self._fm = FieldManager(self._cfg, fields_override=reduced_fields_cfg)
        self._solver = BiotSolver(self._cfg)
        self._output_dir = Path(output_dir or p2["output_dir"])
        self._n_samples: int = int(p2.get("n_training_samples", 200))

    @property
    def field_manager(self) -> FieldManager:
        return self._fm

    def generate(self, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Sample reduced-space parameters and run Biot solver.

        Returns
        -------
        X : (n_samples, reduced_dim)
            Concatenated reduced coefficient vectors [ξ'_E, ξ'_kh, ξ'_kv].
        Y : (n_samples, n_nodes_x)
            Corresponding settlement profiles from Biot solver.
        """
        print(
            f"[Phase 2 DataGen] Generating {self._n_samples} samples "
            f"in reduced space (dim={self._fm.total_input_dim}) ..."
        )
        X, fields, _ = self._fm.generate_dataset(self._n_samples)
        print(f"[Phase 2 DataGen] Running Biot solver on {self._n_samples} samples ...")
        Y = self._solver.run_batch(fields["E"], fields["k_h"], fields["k_v"])
        print(f"[Phase 2 DataGen] Done. X: {X.shape}, Y: {Y.shape}")
        return X, Y

    def generate_single(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single unseen sample for physics regularization.

        Returns
        -------
        x : (1, reduced_dim)
        y : (1, n_nodes_x)
        """
        X, fields, _ = self._fm.generate_dataset(1)
        Y = self._solver.run_batch(fields["E"], fields["k_h"], fields["k_v"])
        return X, Y

    def save(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, str]:
        """Persist dataset to disk."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        xp = self._output_dir / "phase2_X.npy"
        yp = self._output_dir / "phase2_Y.npy"
        np.save(xp, X)
        np.save(yp, Y)
        meta = {
            "n_samples": len(X),
            "reduced_dim": int(self._fm.total_input_dim),
            "n_nodes_x": self._fm.n_nodes_x,
        }
        with open(self._output_dir / "phase2_data_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return {"X": str(xp), "Y": str(yp)}

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load previously generated dataset."""
        X = np.load(self._output_dir / "phase2_X.npy")
        Y = np.load(self._output_dir / "phase2_Y.npy")
        return X, Y
