"""
phase4_validation.py
====================
Phase 4: Validation and visualisation.

Loads trained Phase-2 and Phase-3 models, evaluates them on a test set, and
produces:
  - metrics.json  : R², RMSE, relative L², roughness, per-sample errors
  - plots/settlement_comparison.png
  - plots/E_field_original_vs_reduced.png
  - plots/k_h_field_original_vs_reduced.png
  - plots/k_v_field_original_vs_reduced.png
  - plots/sensitivity_heatmap.png
  - plots/error_distribution.png
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config_manager import ConfigManager
from .field_manager import FieldManager
from .forward_solver import BiotSolver
from .surrogate_models import BaseSurrogate, load_surrogate
from .phase3_reducer import Phase3Reducer
from .utils import compute_metrics, plot_settlement_comparison, plot_field_2d


class Phase4Validator:
    """Run validation and produce diagnostic plots.

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
        self._output_dir = Path(output_dir or self._cfg["phase4"]["output_dir"])
        self._plots_dir = self._output_dir / "plots"

        self._surrogate: Optional[BaseSurrogate] = None
        self._reducer: Optional[Phase3Reducer] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Run full Phase-4 validation.

        Parameters
        ----------
        X_test : (n_test, input_dim) — full-dimensional coefficients
        Y_test : (n_test, n_nodes_x) — ground-truth settlements

        Returns
        -------
        dict with metrics and paths to generated plots.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._plots_dir.mkdir(parents=True, exist_ok=True)

        # Load surrogate
        surr_dir = Path(self._cfg["phase3"]["surrogate_dir"]) / "surrogate"
        if surr_dir.exists():
            print(f"[Phase 4] Loading Phase-2 surrogate from '{surr_dir}' ...")
            self._surrogate = load_surrogate(surr_dir)
        else:
            print("[Phase 4] No Phase-2 surrogate found; will use direct solver.")

        # Load reducer
        reducer_dir = Path(self._cfg["phase3"]["output_dir"])
        if (reducer_dir / "reducer.pt").exists():
            print(f"[Phase 4] Loading Phase-3 reducer from '{reducer_dir}' ...")
            self._reducer = Phase3Reducer(self._cm)
            self._reducer.load()
        else:
            print("[Phase 4] No Phase-3 reducer found; evaluating surrogate directly.")

        # Predictions
        Y_pred = self._predict(X_test)

        # Metrics
        metrics = compute_metrics(Y_test, Y_pred)
        print(
            f"[Phase 4] R²={metrics['R2']:.4f}  RMSE={metrics['RMSE']:.4e}  "
            f"rel-L²={metrics['relative_L2']:.4f}"
        )

        # Save metrics (excluding per_sample_errors list for readability in summary)
        summary = {k: v for k, v in metrics.items() if k != "per_sample_errors"}
        summary["per_sample_mean_error"] = float(np.mean(metrics["per_sample_errors"]))
        summary["per_sample_std_error"] = float(np.std(metrics["per_sample_errors"]))
        metrics_path = self._output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Plots
        plot_paths = self._make_plots(X_test, Y_test, Y_pred)

        return {"metrics": summary, "plots": plot_paths, "metrics_path": str(metrics_path)}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Produce settlement predictions from full-dimensional coefficients."""
        use_physics = self._cfg["phase4"]["use_physics_for_plots"]

        if self._reducer is not None:
            X_reduced = self._reducer.reduce(X)
        else:
            X_reduced = X

        if use_physics or self._surrogate is None:
            # Direct solver
            fields = self._fm.reconstruct_all_fields(X_reduced)
            return self._solver.run_batch(
                fields["E"], fields["k_h"], fields["k_v"]
            )
        else:
            return self._surrogate.predict(X_reduced)

    def _make_plots(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        Y_pred: np.ndarray,
    ) -> Dict[str, str]:
        paths: Dict[str, str] = {}
        nx = self._fm.n_nodes_x
        nz = self._fm.n_nodes_z

        # Settlement comparison
        p = str(self._plots_dir / "settlement_comparison.png")
        plot_settlement_comparison(Y_test, Y_pred, nx, save_path=p)
        paths["settlement_comparison"] = p

        # Material field comparisons
        if self._reducer is not None:
            X_reduced = self._reducer.reduce(X_test[:5])
        else:
            X_reduced = X_test[:5]

        orig_fields = self._fm.reconstruct_all_fields(X_test[:5])
        reduced_fields = self._fm.reconstruct_all_fields(X_reduced)

        for name in ("E", "k_h", "k_v"):
            for i in range(min(2, len(X_test))):
                p_orig = str(self._plots_dir / f"{name}_sample{i}_original.png")
                p_red = str(self._plots_dir / f"{name}_sample{i}_reduced.png")
                plot_field_2d(
                    orig_fields[name][i],
                    nx, nz,
                    title=f"{name} sample {i} (original)",
                    save_path=p_orig,
                )
                plot_field_2d(
                    reduced_fields[name][i],
                    nx, nz,
                    title=f"{name} sample {i} (reduced)",
                    save_path=p_red,
                )
            paths[f"{name}_field_comparison"] = str(self._plots_dir / f"{name}_sample0_original.png")

        # Error distribution
        self._plot_error_distribution(Y_test, Y_pred)
        paths["error_distribution"] = str(self._plots_dir / "error_distribution.png")

        # Sensitivity heatmap
        self._plot_sensitivity_heatmap(X_test)
        paths["sensitivity_heatmap"] = str(self._plots_dir / "sensitivity_heatmap.png")

        return paths

    def _plot_error_distribution(
        self, Y_test: np.ndarray, Y_pred: np.ndarray
    ) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        errors = np.linalg.norm(Y_pred - Y_test, axis=1) / (
            np.linalg.norm(Y_test, axis=1) + 1e-12
        )
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(errors, bins=20, edgecolor="black")
        ax.set_xlabel("Relative L² error")
        ax.set_ylabel("Count")
        ax.set_title("Error distribution")
        plt.tight_layout()
        plt.savefig(self._plots_dir / "error_distribution.png", dpi=100)
        plt.close(fig)

    def _plot_sensitivity_heatmap(self, X_test: np.ndarray) -> None:
        """Variance of predicted settlement w.r.t. individual reduced parameters."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_params = self._fm.total_input_dim
        nx = self._fm.n_nodes_x
        heatmap = np.zeros((n_params, nx))

        if self._surrogate is None:
            return

        X_base = X_test.mean(axis=0, keepdims=True)
        for j in range(n_params):
            delta = np.zeros_like(X_base)
            delta[0, j] = X_test[:, j].std() if X_test[:, j].std() > 1e-12 else 1.0
            Y_plus = self._surrogate.predict(X_base + delta)
            Y_minus = self._surrogate.predict(X_base - delta)
            heatmap[j] = (Y_plus - Y_minus)[0]

        fig, ax = plt.subplots(figsize=(8, max(3, n_params // 2)))
        im = ax.imshow(heatmap, aspect="auto", cmap="RdBu_r")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("x-node")
        ax.set_ylabel("Parameter index")
        ax.set_title("Settlement sensitivity (finite diff)")
        plt.tight_layout()
        plt.savefig(self._plots_dir / "sensitivity_heatmap.png", dpi=100)
        plt.close(fig)
