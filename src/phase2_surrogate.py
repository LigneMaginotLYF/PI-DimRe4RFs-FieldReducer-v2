"""
phase2_surrogate.py
===================
Phase 2: Reduced-Space Lookup Table (LUT) and surrogate training.

Can be run independently of Phase 3.  Saves the trained surrogate plus LUT
artefacts so Phase 3 can reload them.

Steps
-----
1. Generate a grid of reduced parameters (collocation points).
2. Run the Biot solver on all grid points to build the LUT.
3. Optionally augment training with Phase-1 data.
4. Fit an NN or PCE surrogate mapping reduced params → settlement.
5. Save model + metadata to disk using dimension-stamped filenames.

Dimension-stamped filenames
---------------------------
The surrogate is saved as::

    surrogate_{type}_dim{d_total}.pt   (NN)
    surrogate_{type}_dim{d_total}.pkl  (PCE)

where ``d_total`` is the total input dimension (n_terms_E + 1 + n_terms_kv +
…).  Phase 3 searches for this file and validates that the dimension matches
its expected input dimension before loading.
"""

from __future__ import annotations

import glob as _glob
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config_manager import ConfigManager
from .field_manager import FieldManager
from .forward_solver import BiotSolver
from .surrogate_models import BaseSurrogate, build_surrogate, load_surrogate
from .training_schema import build_training_signal


class Phase2Surrogate:
    """Phase-2 LUT construction and surrogate training.

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
        self._fm = FieldManager(self._cfg)
        self._solver = BiotSolver(self._cfg)
        self._output_dir = Path(output_dir or self._cfg["phase2"]["output_dir"])
        self._surrogate: Optional[BaseSurrogate] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        X_train: Optional[np.ndarray] = None,
        Y_train: Optional[np.ndarray] = None,
    ) -> BaseSurrogate:
        """Run the full Phase-2 pipeline.

        Parameters
        ----------
        X_train, Y_train : Phase-1 data (optional; used to augment LUT).

        Returns
        -------
        Trained surrogate model.
        """
        # --- 1. LUT grid generation ---
        print("[Phase 2] Generating LUT grid ...")
        grid_X = self._generate_grid()
        print(f"[Phase 2]   Grid shape: {grid_X.shape}")

        # --- 2. Solver evaluations ---
        training_signal_type = self._cfg["phase2"]["training_signal"]
        if training_signal_type == "data":
            grid_Y = self._evaluate_solver(grid_X)
        else:
            # physics or hybrid: still evaluate solver for ground truth
            grid_Y = self._evaluate_solver(grid_X)

        # --- 3. Optionally combine with Phase-1 data ---
        if X_train is not None and Y_train is not None:
            print(f"[Phase 2] Augmenting with {len(X_train)} Phase-1 samples")
            # Project Phase-1 full-dim X to the reduced dimensions used by the LUT.
            # For Phase 2, the surrogate maps *reduced* params (same dim as total_input_dim
            # here, since no reducer has been trained yet).
            combined_X = np.concatenate([grid_X, X_train], axis=0)
            combined_Y = np.concatenate([grid_Y, Y_train], axis=0)
        else:
            combined_X, combined_Y = grid_X, grid_Y

        # --- 4. Surrogate fitting ---
        print(f"[Phase 2] Training {self._cfg['phase2']['surrogate_type']} surrogate ...")
        self._surrogate = self._build_and_fit_surrogate(combined_X, combined_Y)

        # --- 5. Save ---
        self._save(grid_X, grid_Y)
        print(f"[Phase 2] Surrogate saved to '{self._output_dir}'")
        return self._surrogate

    def load_surrogate(self) -> BaseSurrogate:
        """Load a previously trained Phase-2 surrogate.

        Searches for a dimension-stamped file
        ``surrogate_{type}_dim{d_total}.(pt|pkl)`` in the output directory.
        Falls back to the legacy ``surrogate/`` sub-directory for backward
        compatibility.

        Returns
        -------
        The loaded surrogate model.

        Raises
        ------
        FileNotFoundError
            If no surrogate file is found in the output directory.
        """
        d_total = self._fm.total_input_dim
        surr_path = _find_surrogate_file(self._output_dir, d_total)
        self._surrogate = load_surrogate(surr_path)
        return self._surrogate

    @property
    def surrogate(self) -> Optional[BaseSurrogate]:
        return self._surrogate

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_grid(self) -> np.ndarray:
        """Generate a grid of collocation points in the concatenated coefficient space.

        Uses a Latin Hypercube-like stratified sampling within a ±2σ range
        determined by the Matérn spectral variance of each field.
        """
        n_points = self._cfg["collocation_phase2"]["n_points"]
        total_dim = self._fm.total_input_dim
        rng = np.random.default_rng(seed=999)

        # Sample uniformly within a normalised range [-2, 2] for each dimension
        # then scale by the per-field spectral variance
        grid = rng.uniform(-2.0, 2.0, size=(n_points, total_dim))

        # Apply per-field spectral variance scaling
        scale = self._build_scale_vector()
        grid = grid * scale[np.newaxis, :]
        return grid

    def _build_scale_vector(self) -> np.ndarray:
        """Build a per-coefficient scaling vector from Matérn spectral std."""
        fm = self._fm
        scales = []
        for name in FieldManager.FIELD_NAMES:
            fc = fm.field_configs[name]
            if fc.n_terms == 0:
                scales.append(np.array([1.0]))
            else:
                var = fm.get_spectral_variance(
                    fc.n_terms, fc.nu_ref, fc.length_scale_ref
                )
                scales.append(np.sqrt(var))
        return np.concatenate(scales)

    def _evaluate_solver(self, X: np.ndarray) -> np.ndarray:
        """Run the Biot solver on all rows of *X*."""
        fields = self._fm.reconstruct_all_fields(X)
        return self._solver.run_batch(
            fields["E"], fields["k_h"], fields["k_v"]
        )

    def _build_and_fit_surrogate(
        self, X: np.ndarray, Y: np.ndarray
    ) -> BaseSurrogate:
        p2 = self._cfg["phase2"]
        surrogate = build_surrogate(
            surrogate_type=p2["surrogate_type"],
            input_dim=self._fm.total_input_dim,
            n_nodes_x=self._fm.n_nodes_x,
            output_repr=p2["output_repr"],
            n_output_modes=p2["n_output_modes"],
            nn_cfg=p2.get("nn", {}),
            pce_cfg=p2.get("pce", {}),
        )
        surrogate.fit(X, Y)
        return surrogate

    def _save(self, grid_X: np.ndarray, grid_Y: np.ndarray) -> None:
        d = self._output_dir
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "grid_points.npy", grid_X)
        np.save(d / "responses.npy", grid_Y)

        # Save surrogate with dimension-stamped filename
        surrogate_type = self._cfg["phase2"]["surrogate_type"]
        d_total = self._fm.total_input_dim
        ext = "pt" if surrogate_type == "nn" else "pkl"
        surr_filename = f"surrogate_{surrogate_type}_dim{d_total}.{ext}"
        self._surrogate.save(d / surr_filename)

        # Save config snapshot
        meta = {
            "input_dim": d_total,
            "n_nodes_x": self._fm.n_nodes_x,
            "surrogate_type": surrogate_type,
            "surrogate_filename": surr_filename,
            "output_repr": self._cfg["phase2"]["output_repr"],
            "n_output_modes": self._cfg["phase2"]["n_output_modes"],
            "config_hash": self._cm.config_hash(),
        }
        with open(d / "config.json", "w", encoding='utf-8') as f:
            json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Module-level helper (used by Phase 3 for dimension-validated loading)
# ---------------------------------------------------------------------------

def _find_surrogate_file(surr_dir: Path, d_expected: int) -> Path:
    """Find a dimension-stamped surrogate file in *surr_dir*.

    Searches for ``surrogate_*_dim{d_expected}.(pt|pkl)``.  Falls back to the
    legacy ``surrogate/`` sub-directory for backward compatibility.

    Parameters
    ----------
    surr_dir : Path
        Directory produced by :class:`Phase2Surrogate`.
    d_expected : int
        Expected input dimension of the surrogate.

    Returns
    -------
    Path
        Path to the surrogate file or directory.

    Raises
    ------
    ValueError
        If no matching surrogate is found, with a message listing available
        options to aid diagnosis.
    """
    surr_dir = Path(surr_dir)
    # Search for dimension-stamped files with known extensions only
    matches = []
    for ext in ("pt", "pkl"):
        pattern = str(surr_dir / f"surrogate_*_dim{d_expected}.{ext}")
        matches.extend(_glob.glob(pattern))
    if matches:
        return Path(matches[0])

    # Backward compatibility: legacy surrogate/ sub-directory
    legacy = surr_dir / "surrogate"
    if legacy.exists():
        return legacy

    # Provide a helpful error message with only relevant surrogate files
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
