"""
phase3_evaluator.py
===================
Phase 3: Dimension reducer evaluation.

Computes prediction metrics (settlement) and generates diagnostic plots for
a trained Phase-3 reducer, including material-field heatmap comparisons.
Everything is stored in a timestamped sub-folder.

Output layout::

    results/<run_id>/phase3_reducer/<model_name>_<YYYYMMDD_HHMMSS>/
        metrics.json
        plots/
            settlement_comparison.png
            E_field_comparison.png
            k_h_field_comparison.png
            k_v_field_comparison.png

Usage
-----
::

    from src.phase3_evaluator import Phase3Evaluator
    evaluator = Phase3Evaluator(config_manager)
    results = evaluator.run(X_test, Y_test, reducer=trained_reducer,
                            surrogate=loaded_surrogate)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .config_manager import ConfigManager
from .field_manager import FieldManager
from .phase3_reducer import Phase3Reducer
from .surrogate_models import BaseSurrogate
from .utils import compute_metrics
from .visualization_v2 import (
    plot_settlement_comparison_global_y,
    plot_all_material_fields,
)


class Phase3Evaluator:
    """Evaluate a Phase-3 reducer and persist metrics + plots.

    Parameters
    ----------
    config_manager : ConfigManager
    output_dir : str, optional
        Base directory for evaluation artefacts.  Defaults to
        ``phase3.output_dir`` from the config.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        output_dir: Optional[str] = None,
    ) -> None:
        self._cm = config_manager
        self._cfg = config_manager.cfg
        self._base_dir = Path(output_dir or self._cfg["phase3"]["output_dir"])
        self._fm = FieldManager(self._cfg)

        p3_eval = self._cfg.get("phase3", {}).get("evaluation", {})
        self._test_fraction: float = float(p3_eval.get("test_fraction", 0.2))
        self._n_plot_samples: int = int(p3_eval.get("n_plot_samples", 5))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        reducer: Phase3Reducer,
        surrogate: Optional[BaseSurrogate] = None,
        model_name: str = "reducer",
    ) -> Dict[str, Any]:
        """Evaluate *reducer* on *(X_test, Y_test)* and save all artefacts.

        The settlement prediction is obtained by:
          1. Passing *X_test* through the reducer to get *X_reduced*.
          2. If *surrogate* is provided, predicting settlements from *X_reduced*
             via the surrogate.  Otherwise, the original settlement array *Y_test*
             is used as the reference (only field-reconstruction quality is
             measured).

        Material field comparisons reconstruct both the original (*X_test*)
        and reduced (*X_reduced*) physical fields using :class:`FieldManager`
        and generate heatmap grids.

        Parameters
        ----------
        X_test : (n_test, input_dim)
            Held-out full-dimensional coefficient vectors.
        Y_test : (n_test, n_nodes_x)
            Ground-truth settlement profiles.
        reducer : Phase3Reducer
            Trained Phase-3 reducer (must have ``reduce`` method available).
        surrogate : BaseSurrogate, optional
            Phase-2 surrogate for settlement prediction from reduced params.
            Required to compute settlement metrics; if *None* only field-
            reconstruction metrics are computed.
        model_name : str
            Used to construct the timestamped output folder name.

        Returns
        -------
        dict with keys:
          - ``metrics``       : dict of metric name → float (settlement if
                                surrogate provided, else field-reconstruction)
          - ``metrics_path``  : path to *metrics.json*
          - ``plots``         : dict of plot name → path string
          - ``output_dir``    : str path of the timestamped folder
        """
        # Timestamped output directory
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self._base_dir / f"{model_name}_{ts}"
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Reduce
        X_reduced = reducer.reduce(X_test)

        # Settlement prediction
        plots: Dict[str, str] = {}
        settlement_metrics: Dict[str, float] = {}
        n_nodes_x = Y_test.shape[1]

        if surrogate is not None:
            Y_pred = surrogate.predict(X_reduced)
            settlement_metrics = compute_metrics(Y_test, Y_pred)
            plot_path = plots_dir / "settlement_comparison.png"
            plot_settlement_comparison_global_y(
                y_true=Y_test,
                y_pred=Y_pred,
                n_nodes_x=n_nodes_x,
                save_path=plot_path,
                title=f"Phase 3 – {model_name} – settlement comparison",
                n_samples=self._n_plot_samples,
            )
            plots["settlement_comparison"] = str(plot_path)

        # Material field comparisons
        fields_orig = self._fm.reconstruct_all_fields(X_test)
        fields_redu = self._fm.reconstruct_all_fields(X_reduced)
        field_plots = plot_all_material_fields(
            fields_orig_dict=fields_orig,
            fields_redu_dict=fields_redu,
            n_nodes_x=self._fm.n_nodes_x,
            n_nodes_z=self._fm.n_nodes_z,
            output_dir=plots_dir,
            n_samples=self._n_plot_samples,
        )
        plots.update(field_plots)

        # Field-reconstruction L2 metrics (always computed)
        field_metrics: Dict[str, float] = {}
        for name in ("E", "k_h", "k_v"):
            if name in fields_orig and name in fields_redu:
                m = compute_metrics(fields_orig[name], fields_redu[name])
                for k, v in m.items():
                    field_metrics[f"{name}_{k}"] = v

        metrics = {**settlement_metrics, **field_metrics}

        # Persist metrics
        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metrics": metrics,
                    "n_test": int(len(X_test)),
                    "training_signal": self._cfg["phase3"]["training_signal"],
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
