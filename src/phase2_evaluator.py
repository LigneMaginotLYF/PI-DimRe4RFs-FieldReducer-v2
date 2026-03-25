"""
phase2_evaluator.py
===================
Phase 2: Surrogate model evaluation.

Computes prediction metrics and generates diagnostic plots for a trained
Phase-2 surrogate, saving everything to a timestamped sub-folder of the
configured Phase-2 output directory.

Output layout::

    results/<run_id>/phase2_surrogate/<model_name>_<YYYYMMDD_HHMMSS>/
        metrics.json
        plots/
            settlement_comparison.png

Usage
-----
::

    from src.phase2_evaluator import Phase2Evaluator
    evaluator = Phase2Evaluator(config_manager)
    results = evaluator.run(X_test, Y_test, surrogate=trained_surrogate)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .config_manager import ConfigManager
from .surrogate_models import BaseSurrogate
from .utils import compute_metrics
from .visualization_v2 import plot_settlement_comparison_global_y


class Phase2Evaluator:
    """Evaluate a Phase-2 surrogate and persist metrics + plots.

    Parameters
    ----------
    config_manager : ConfigManager
    output_dir : str, optional
        Base directory for evaluation artefacts.  If *None*, falls back to
        ``phase2.output_dir`` from the config.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        output_dir: Optional[str] = None,
    ) -> None:
        self._cm = config_manager
        self._cfg = config_manager.cfg
        self._base_dir = Path(output_dir or self._cfg["phase2"]["output_dir"])

        p2_eval = self._cfg.get("phase2", {}).get("evaluation", {})
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
        X_test : (n_test, input_dim)
            Held-out coefficient vectors.
        Y_test : (n_test, n_nodes_x)
            Ground-truth settlement profiles.
        surrogate : BaseSurrogate
            Trained Phase-2 surrogate.
        model_name : str
            Used to construct the timestamped output folder name.

        Returns
        -------
        dict with keys:
          - ``metrics``       : dict of metric name → float
          - ``metrics_path``  : path to *metrics.json*
          - ``plots``         : dict of plot name → path string
          - ``output_dir``    : str path of the timestamped folder
        """
        # Timestamped output directory
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self._base_dir / f"{model_name}_{ts}"
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Predict
        Y_pred = surrogate.predict(X_test)

        # Metrics
        metrics = compute_metrics(Y_test, Y_pred)
        n_nodes_x = Y_test.shape[1]

        # --- Settlement comparison plot (global y-axis) ---
        plot_path = plots_dir / "settlement_comparison.png"
        plot_settlement_comparison_global_y(
            y_true=Y_test,
            y_pred=Y_pred,
            n_nodes_x=n_nodes_x,
            save_path=plot_path,
            title=f"Phase 2 – {model_name} – settlement comparison",
            n_samples=self._n_plot_samples,
        )

        # --- Persist metrics ---
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
