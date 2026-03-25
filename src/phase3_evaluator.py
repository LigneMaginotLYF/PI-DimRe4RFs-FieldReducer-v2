"""
phase3_evaluator.py
===================
Phase 3: Dimension reducer evaluation.

Evaluation is ALWAYS physics-driven:
  - Ground truth: Y_full = P(E_full, k_h_full, k_v_full)  [Biot on full fields]
  - Prediction:   Y_red  = P(E_red, k_h_red, k_v_red)    [Biot on reduced fields]

The Phase-2 surrogate is NEVER used for evaluation, even if Phase 3 was
trained with the surrogate loss.  This ensures evaluation reflects true
physics preservation.

Output layout::

    <output_dir>/<model_name>_<YYYYMMDD_HHMMSS>/
        metrics.json
        plots/
            settlement_comparison.png
            E_field_comparison.png
            k_h_field_comparison.png
            k_v_field_comparison.png
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .config_manager import ConfigManager
from .field_manager import FieldManager
from .forward_solver import BiotSolver
from .phase3_reducer import Phase3Reducer
from .utils import compute_metrics
from .visualization_v2 import (
    plot_settlement_comparison_global_y,
    plot_all_material_fields,
)


class Phase3Evaluator:
    """Evaluate a Phase-3 reducer using direct Biot physics.

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
        self._base_dir = Path(output_dir or p3["output_dir"])

        # Full-space FieldManager (to reconstruct full fields)
        full_fields_cfg = p3.get("full_fields")
        self._fm_full = FieldManager(self._cfg, fields_override=full_fields_cfg)

        # Reduced-space FieldManager (to reconstruct reduced fields)
        reduced_fields_cfg = p3.get("reduced_fields")
        self._fm_reduced = FieldManager(self._cfg, fields_override=reduced_fields_cfg)

        self._solver = BiotSolver(self._cfg)

        p3_eval = p3.get("evaluation", {})
        self._test_fraction: float = float(p3_eval.get("test_fraction", 0.2))
        self._n_plot_samples: int = int(p3_eval.get("n_plot_samples", 5))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        X_test_full: np.ndarray,
        Y_test_full: np.ndarray,
        reducer: Phase3Reducer,
        model_name: str = "reducer",
    ) -> Dict[str, Any]:
        """Evaluate *reducer* using direct Biot physics.

        Steps:
          1. X_reduced = P3(X_test_full)
          2. Reconstruct reduced material fields
          3. Y_pred = P(E_red, k_h_red, k_v_red)  [Biot, NOT surrogate]
          4. Compare Y_test_full vs Y_pred

        Parameters
        ----------
        X_test_full : (n_test, full_dim)
            Held-out full-dimensional coefficient vectors.
        Y_test_full : (n_test, n_nodes_x)
            Ground-truth settlement profiles (from full-field Biot).
        reducer : Phase3Reducer
            Trained Phase-3 reducer.
        model_name : str
            Used to construct the timestamped output folder name.

        Returns
        -------
        dict with metrics, paths.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self._base_dir / f"{model_name}_{ts}"
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Phase 3 Eval] Reducing {len(X_test_full)} test samples ...")
        X_reduced = reducer.reduce(X_test_full)

        print("[Phase 3 Eval] Running Biot solver on reduced fields ...")
        fields_reduced = self._fm_reduced.reconstruct_all_fields(X_reduced)
        Y_pred = self._solver.run_batch(
            fields_reduced["E"], fields_reduced["k_h"], fields_reduced["k_v"]
        )
        print("[Phase 3 Eval] Biot evaluation complete.")

        # Settlement metrics
        settlement_metrics = compute_metrics(Y_test_full, Y_pred)
        print(
            f"[Phase 3 Eval] R²={settlement_metrics.get('R2', float('nan')):.4f} | "
            f"RMSE={settlement_metrics.get('RMSE', float('nan')):.4e} | "
            f"Rel-L2={settlement_metrics.get('relative_L2', float('nan')):.4e}"
        )

        plots: Dict[str, str] = {}

        # Settlement comparison plot
        n_nodes_x = Y_test_full.shape[1]
        plot_path = plots_dir / "settlement_comparison.png"
        plot_settlement_comparison_global_y(
            y_true=Y_test_full,
            y_pred=Y_pred,
            n_nodes_x=n_nodes_x,
            lx=self._fm_full.lx,
            save_path=plot_path,
            title="Phase 3 – settlement: P(full) vs P(reduced) [Biot]",
            n_samples=self._n_plot_samples,
            label_true="P(full fields) [ground truth]",
            label_pred="P(reduced fields) [Biot]",
        )
        plots["settlement_comparison"] = str(plot_path)

        # Material field comparisons
        fields_full = self._fm_full.reconstruct_all_fields(X_test_full)
        field_plots = plot_all_material_fields(
            fields_orig_dict=fields_full,
            fields_redu_dict=fields_reduced,
            n_nodes_x=self._fm_full.n_nodes_x,
            n_nodes_z=self._fm_full.n_nodes_z,
            lx=self._fm_full.lx,
            lz=self._fm_full.lz,
            output_dir=plots_dir,
            n_samples=self._n_plot_samples,
        )
        plots.update(field_plots)

        # Field-reconstruction metrics
        field_metrics: Dict[str, float] = {}
        for name in ("E", "k_h", "k_v"):
            if name in fields_full and name in fields_reduced:
                m = compute_metrics(fields_full[name], fields_reduced[name])
                for k, v in m.items():
                    field_metrics[f"{name}_{k}"] = v

        metrics = {**settlement_metrics, **field_metrics}

        # Persist metrics
        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metrics": metrics,
                    "n_test": int(len(X_test_full)),
                    "evaluation_mode": "physics_biot",
                    "training_signal": self._cfg["phase3"].get("training_signal", "surrogate"),
                    "config_hash": self._cm.config_hash(),
                },
                f,
                indent=2,
            )

        return {
            "metrics": metrics,
            "metrics_path": str(metrics_path),
            "plots": plots,
            "output_dir": str(run_dir),
        }
