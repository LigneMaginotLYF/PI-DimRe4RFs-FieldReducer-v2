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
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config_manager import ConfigManager
from .field_manager import FieldManager
from .forward_solver import BiotSolver

logger = logging.getLogger(__name__)

# Maximum retries per sample before skipping
_MAX_RETRIES = 3
# Checkpoint interval (save every N samples)
_CHECKPOINT_INTERVAL = 50


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
        self._n_samples: int = int(cfg["phase1"]["n_samples"])
        self._val_fraction: float = float(cfg["phase1"]["val_fraction"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, resume_from_checkpoint: bool = False) -> Dict[str, Any]:
        """Generate the dataset and save to disk.

        Parameters
        ----------
        resume_from_checkpoint : bool
            If True, attempt to resume from a previously saved checkpoint.

        Returns
        -------
        dict with paths to all saved artefacts.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self._output_dir / "phase1_checkpoint.npz"

        X_list: List[np.ndarray] = []
        Y_list: List[np.ndarray] = []
        start_idx = 0

        if resume_from_checkpoint and checkpoint_path.exists():
            logger.info("[Phase 1] Loading checkpoint from %s", checkpoint_path)
            data = np.load(checkpoint_path, allow_pickle=False)
            X_list = list(data["X"])
            Y_list = list(data["Y"])
            start_idx = len(X_list)
            logger.info("[Phase 1] Resuming from sample %d/%d", start_idx, self._n_samples)

        logger.info("[Phase 1] Generating %d samples ...", self._n_samples)
        failed_samples: List[int] = []

        for i in range(start_idx, self._n_samples):
            success = False
            last_error: Optional[Exception] = None

            for retry in range(_MAX_RETRIES):
                try:
                    xi_dict, fields = self._generate_one_sample()

                    # Run solver
                    Y_i = self._solver.run(
                        fields["E"], fields["k_h"], fields["k_v"]
                    )

                    # Validate solver output
                    Y_i = np.asarray(Y_i, dtype=np.float64)
                    if np.any(np.isnan(Y_i)) or np.any(np.isinf(Y_i)):
                        raise ValueError(
                            f"Solver output for sample {i} contains NaN/Inf"
                        )

                    # Concatenate field coefficients into a single vector
                    xi_concat = np.concatenate(
                        [xi_dict[name].flatten() for name in FieldManager.FIELD_NAMES]
                    ).astype(np.float64)

                    X_list.append(xi_concat)
                    Y_list.append(Y_i)
                    success = True
                    break

                except Exception as e:
                    last_error = e
                    logger.warning(
                        "[Phase 1] Sample %d retry %d/%d failed: %s",
                        i + 1, retry + 1, _MAX_RETRIES, e,
                    )

            if not success:
                logger.error(
                    "[Phase 1] Sample %d failed after %d retries: %s",
                    i + 1, _MAX_RETRIES, last_error,
                )
                failed_samples.append(i)
                continue

            # Periodic progress log and checkpoint
            if (i + 1) % _CHECKPOINT_INTERVAL == 0:
                logger.info(
                    "[Phase 1] Generated %d/%d samples", i + 1, self._n_samples
                )
                np.savez_compressed(
                    checkpoint_path,
                    X=np.array(X_list, dtype=np.float64),
                    Y=np.array(Y_list, dtype=np.float64),
                )

        if failed_samples:
            logger.warning("[Phase 1] Failed sample indices: %s", failed_samples)

        if len(X_list) == 0:
            raise RuntimeError("[Phase 1] No samples were successfully generated.")

        X = np.array(X_list, dtype=np.float64)
        Y = np.array(Y_list, dtype=np.float64)

        if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
            raise ValueError("[Phase 1] Generated dataset contains NaN values.")

        logger.info(
            "[Phase 1] Dataset ready: X %s, Y %s", X.shape, Y.shape
        )

        # Train / validation split
        X_train, X_val, Y_train, Y_val = self._split(X, Y)

        # Save
        paths = self._save(X_train, X_val, Y_train, Y_val)

        # Metadata
        meta = self._build_metadata(X_train, Y_train, X_val, Y_val)
        meta_path = self._output_dir / "dataset_metadata.json"
        with open(meta_path, "w", encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        paths["metadata"] = str(meta_path)

        # Remove checkpoint on successful completion
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        logger.info("[Phase 1] Done. Saved to '%s'", self._output_dir)
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

    def _generate_one_sample(
        self,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Generate coefficients and physical fields for one sample.

        Returns
        -------
        xi_dict : per-field coefficient arrays
        fields  : per-field physical field arrays (n_nodes,)
        """
        xi_dict: Dict[str, np.ndarray] = {}
        fields: Dict[str, np.ndarray] = {}

        for name in FieldManager.FIELD_NAMES:
            xi = self._fm.sample_coefficients(1, name)[0]  # (effective_dim,)
            xi = np.asarray(xi, dtype=np.float64)
            field = self._fm.reconstruct_field(xi, name)   # (n_nodes,)
            field = np.asarray(field, dtype=np.float64)

            if np.any(np.isnan(field)) or np.any(np.isinf(field)):
                raise ValueError(
                    f"Field '{name}' contains NaN/Inf after reconstruction."
                )

            xi_dict[name] = xi
            fields[name] = field

        return xi_dict, fields

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
