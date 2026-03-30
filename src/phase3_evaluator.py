"""
phase3_evaluator.py
===================
Phase 3: Dimension reducer evaluation.

Evaluation uses **two** decode paths in parallel:

1. **Physics path** (primary, always run):
   full params → reducer → reduced params → Biot solver → settlement (GT comparison)

2. **Surrogate path** (when surrogate is available):
   full params → reducer → reduced params → Phase-2 surrogate → settlement

Showing both paths in the settlement comparison plot makes it easy to
attribute errors: if the surrogate path is accurate but the physics path
diverges, the issue is in physical reconstruction / field ordering.  If both
paths diverge, the reducer itself is the problem.

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
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .config_manager import ConfigManager
from .field_manager import FieldManager
from .forward_solver import BiotSolver
from .phase3_reducer import Phase3Reducer
from .surrogate_models import BaseSurrogate
from .utils import compute_metrics
from .visualization_v2 import (
    plot_settlement_comparison_global_y,
    plot_all_material_fields,
)

logger = logging.getLogger(__name__)


class Phase3Evaluator:
    """Evaluate a Phase-3 reducer using direct Biot physics (primary) and,
    optionally, the Phase-2 surrogate (secondary diagnostic path).

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
        surrogate: Optional[BaseSurrogate] = None,
    ) -> Dict[str, Any]:
        """Evaluate *reducer* using direct Biot physics and (optionally) the
        Phase-2 surrogate.

        Steps:
          1. X_reduced = P3(X_test_full)
          2. Reconstruct reduced material fields
          3. Y_biot  = P(E_red, k_h_red, k_v_red)  [Biot – primary metric]
          4. Y_surr  = P2(X_reduced)                [surrogate – diagnostic, if available]
          5. Compare Y_test_full vs Y_biot (and Y_surr when present)

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
        surrogate : BaseSurrogate, optional
            If provided, also evaluate the surrogate decode path and include
            metrics / third settlement curve in the comparison plot.

        Returns
        -------
        dict with metrics, paths.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self._base_dir / f"{model_name}_{ts}"
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        n_test = len(X_test_full)
        print(f"[Phase 3 Eval] Reducing {n_test} test samples ...")
        X_reduced = reducer.reduce(X_test_full)

        # --- Finite check on reducer output ---
        if not np.all(np.isfinite(X_reduced)):
            n_bad = int(np.sum(~np.isfinite(X_reduced)))
            pct_bad = 100.0 * n_bad / X_reduced.size
            if pct_bad > 10.0:
                raise ValueError(
                    f"[Phase 3 Eval] {pct_bad:.1f}% of reducer outputs are non-finite "
                    f"({n_bad}/{X_reduced.size} values). This indicates a serious problem "
                    "with the reducer model. Check training convergence and input scaling."
                )
            logger.warning(
                "[Phase 3 Eval] %d non-finite values (%.1f%%) in reducer output (X_reduced). "
                "Replacing with zeros for evaluation.",
                n_bad, pct_bad,
            )
            X_reduced = np.where(np.isfinite(X_reduced), X_reduced, 0.0)

        # Per-sample diagnostics
        for idx in range(min(self._n_plot_samples, n_test)):
            xr = X_reduced[idx]
            print(
                f"[Phase 3 Eval] Sample {idx}: X_reduced min={xr.min():.3e} "
                f"max={xr.max():.3e}"
            )

        print("[Phase 3 Eval] Running Biot solver on reduced fields ...")
        fields_reduced = self._fm_reduced.reconstruct_all_fields(X_reduced)

        # Finite check on reconstructed fields before Biot call
        for fname, farr in fields_reduced.items():
            if not np.all(np.isfinite(farr)):
                n_bad = int(np.sum(~np.isfinite(farr)))
                logger.warning(
                    "[Phase 3 Eval] %d non-finite values in reconstructed field '%s'. "
                    "Clamping to physical range.",
                    n_bad, fname,
                )
                # Clamp to the field's valid range from config
                fc = self._fm_reduced.field_configs[fname]
                if fname == "E":
                    lo_phys, hi_phys = fc.E_ref * 0.01, fc.E_ref * 100.0
                else:
                    lo_phys, hi_phys = float(fc.k_range[0]), float(fc.k_range[1])
                fields_reduced[fname] = np.clip(
                    np.where(np.isfinite(farr), farr, lo_phys), lo_phys, hi_phys
                )

        Y_biot = self._solver.run_batch(
            fields_reduced["E"], fields_reduced["k_h"], fields_reduced["k_v"]
        )
        print("[Phase 3 Eval] Biot evaluation complete.")

        # Per-sample Biot output diagnostics
        for idx in range(min(self._n_plot_samples, n_test)):
            print(
                f"[Phase 3 Eval] Sample {idx}: Y_biot  min={Y_biot[idx].min():.3e} "
                f"max={Y_biot[idx].max():.3e} | "
                f"Y_gt   min={Y_test_full[idx].min():.3e} "
                f"max={Y_test_full[idx].max():.3e}"
            )

        # --- Primary (Biot) metrics ---
        biot_metrics = compute_metrics(Y_test_full, Y_biot)
        print(
            f"[Phase 3 Eval] Physics path — "
            f"R²={biot_metrics.get('R2', float('nan')):.4f} | "
            f"RMSE={biot_metrics.get('RMSE', float('nan')):.4e} | "
            f"Rel-L2={biot_metrics.get('relative_L2', float('nan')):.4e}"
        )

        # --- Surrogate path (diagnostic) ---
        Y_surr: Optional[np.ndarray] = None
        surr_metrics: Dict[str, float] = {}
        if surrogate is not None:
            print("[Phase 3 Eval] Running surrogate decode path ...")
            try:
                Y_surr = surrogate.predict(X_reduced)
                surr_metrics = compute_metrics(Y_test_full, Y_surr)
                print(
                    f"[Phase 3 Eval] Surrogate path — "
                    f"R²={surr_metrics.get('R2', float('nan')):.4f} | "
                    f"RMSE={surr_metrics.get('RMSE', float('nan')):.4e} | "
                    f"Rel-L2={surr_metrics.get('relative_L2', float('nan')):.4e}"
                )
                # Per-sample surrogate diagnostics
                for idx in range(min(self._n_plot_samples, n_test)):
                    print(
                        f"[Phase 3 Eval] Sample {idx}: Y_surr  min={Y_surr[idx].min():.3e} "
                        f"max={Y_surr[idx].max():.3e}"
                    )
            except Exception as exc:
                logger.warning("[Phase 3 Eval] Surrogate path failed: %s", exc)
                Y_surr = None

        plots: Dict[str, str] = {}

        # --- Settlement comparison plot (3 curves when surrogate available) ---
        n_nodes_x = Y_test_full.shape[1]
        plot_path = plots_dir / "settlement_comparison.png"
        plot_settlement_comparison_global_y(
            y_true=Y_test_full,
            y_pred=Y_biot,
            n_nodes_x=n_nodes_x,
            lx=self._fm_full.lx,
            save_path=plot_path,
            title="Phase 3 – settlement: P(full) vs P(reduced) [Biot]",
            n_samples=self._n_plot_samples,
            label_true="P(full fields) [ground truth]",
            label_pred="P(reduced fields) [Biot]",
            y_pred_surrogate=Y_surr,
            label_pred_surrogate="P2(reduced params) [surrogate]",
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

        # Merge: physics metrics are canonical; surrogate metrics namespaced
        metrics: Dict[str, Any] = {**biot_metrics, **field_metrics}
        if surr_metrics:
            for k, v in surr_metrics.items():
                metrics[f"surrogate_{k}"] = v

        # Persist metrics
        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metrics": metrics,
                    "n_test": int(n_test),
                    "evaluation_mode": "physics_biot",
                    "surrogate_path_evaluated": surrogate is not None,
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
