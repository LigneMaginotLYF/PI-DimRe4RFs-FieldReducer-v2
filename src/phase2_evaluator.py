"""
phase2_evaluator.py
===================
Phase 2: Surrogate model evaluation.

Computes prediction metrics and generates diagnostic plots for a trained
Phase-2 surrogate, saving everything to a timestamped sub-folder.

Output layout::

    <output_dir>/<model_name>_<YYYYMMDD_HHMMSS>/
        metrics.json
        plots/
            settlement_comparison.png
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .config_manager import ConfigManager
from .field_manager import FieldManager
from .surrogate_models import BaseSurrogate
from .utils import compute_metrics
from .visualization_v2 import plot_settlement_comparison_global_y


class Phase2Evaluator:
    """Evaluate a Phase-2 surrogate and persist metrics + plots.

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
        self._base_dir = Path(output_dir or p2["output_dir"])

        # Reduced-space FieldManager (for domain length metadata)
        reduced_fields_cfg = p2.get("reduced_fields")
        self._fm = FieldManager(self._cfg, fields_override=reduced_fields_cfg)

        p2_eval = p2.get("evaluation", {})
        self._test_fraction: float = float(p2_eval.get("test_fraction", 0.2))
        self._n_plot_samples: int = int(p2_eval.get("n_plot_samples", 5))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        surrogate: BaseSurrogate,
        model_name: str = "surrogate",
    ) -> Dict[str, Any]:
        """Evaluate *surrogate* on *(X_test, Y_test)* and save all artefacts.

        Parameters
        ----------
        X_test : (n_test, reduced_dim)
        Y_test : (n_test, n_nodes_x)
        surrogate : BaseSurrogate
        model_name : str

        Returns
        -------
        dict with metrics, paths.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self._base_dir / f"{model_name}_{ts}"
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        Y_pred = surrogate.predict(X_test)
        metrics = compute_metrics(Y_test, Y_pred)
        n_nodes_x = Y_test.shape[1]

        print(
            f"[Phase 2 Eval] R²={metrics.get('R2', float('nan')):.4f} | "
            f"RMSE={metrics.get('RMSE', float('nan')):.4e} | "
            f"Rel-L2={metrics.get('relative_L2', float('nan')):.4e}"
        )

        plot_path = plots_dir / "settlement_comparison.png"
        plot_settlement_comparison_global_y(
            y_true=Y_test,
            y_pred=Y_pred,
            n_nodes_x=n_nodes_x,
            lx=self._fm.lx,
            save_path=plot_path,
            title="Phase 2 – surrogate vs Biot ground truth",
            n_samples=self._n_plot_samples,
            label_true="P(ξ') Biot [ground truth]",
            label_pred="P2(ξ') surrogate",
        )

        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metrics": metrics,
                    "n_test": int(len(X_test)),
                    "surrogate_type": self._cfg["phase2"]["surrogate_type"],
                    "output_repr": self._cfg["phase2"]["output_repr"],
                    "config_hash": self._cm.config_hash(),
                },
                f,
                indent=2,
            )

        plots: Dict[str, str] = {"settlement_comparison": str(plot_path)}
        return {
            "metrics": metrics,
            "metrics_path": str(metrics_path),
            "plots": plots,
            "output_dir": str(run_dir),
        }
