"""
visualization_v2.py
===================
Unified visualization module for Phase 2 and Phase 3 evaluation.

New functions
-------------
- :func:`plot_settlement_comparison_global_y`
    Settlement profiles with a **global** y-axis range shared across all
    subplot columns for consistent visual comparison.

- :func:`plot_material_fields_comparison`
    Side-by-side 5-sample heatmap grid for a single material field:
      Top row: original fields
      Bottom row: reduced/reconstructed fields
    Each cell has its own colorbar; each column title shows per-sample stats.

- :func:`plot_all_material_fields`
    Convenience wrapper that calls :func:`plot_material_fields_comparison`
    for E, k_h, and k_v in sequence.

Backward compatibility
----------------------
All plot helpers originally defined in :mod:`src.utils` are re-exported here
so any code that imports from *visualization_v2* finds the complete API.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

# Re-export existing helpers for backward compatibility
from .utils import (
    plot_settlement_comparison,  # legacy name kept
    plot_field_2d,
    compute_metrics,
)

__all__ = [
    # New
    "plot_settlement_comparison_global_y",
    "plot_material_fields_comparison",
    "plot_all_material_fields",
    # Legacy re-exports
    "plot_settlement_comparison",
    "plot_field_2d",
    "compute_metrics",
]


# ---------------------------------------------------------------------------
# Settlement comparison with global y-axis
# ---------------------------------------------------------------------------

def plot_settlement_comparison_global_y(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_nodes_x: int,
    save_path: str | Path | None = None,
    title: str = "Settlement comparison",
    n_samples: int = 5,
    collocation_x: Optional[np.ndarray] = None,
) -> None:
    """Plot true vs predicted settlement profiles with a **shared** y-axis.

    The y-axis limits are derived from the global min/max across *all*
    displayed samples, ensuring every subplot uses the same scale so that
    differences in magnitude are visible.

    Parameters
    ----------
    y_true : (n_total, n_nodes_x)
        Ground-truth settlement profiles.
    y_pred : (n_total, n_nodes_x)
        Predicted/reconstructed settlement profiles.
    n_nodes_x : int
        Number of surface nodes (x direction).
    save_path : str or Path, optional
        If given, save the figure to this path (parent dirs created automatically).
    title : str
        Figure super-title.
    n_samples : int
        Number of samples to display (default: 5).
    collocation_x : (n_coll,) array, optional
        x-coordinates of collocation points; if supplied, vertical dashed
        lines are drawn at these locations.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = min(n_samples, len(y_true))
    x = np.linspace(0.0, 1.0, n_nodes_x)

    # Global y range across all shown samples
    shown_true = y_true[:n_show]
    shown_pred = y_pred[:n_show]
    y_min = min(shown_true.min(), shown_pred.min())
    y_max = max(shown_true.max(), shown_pred.max())
    y_margin = max(abs(y_max - y_min) * 0.05, 1e-10)

    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 3), sharey=True)
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(x, shown_true[i], "k-", label="True", linewidth=1.5)
        ax.plot(x, shown_pred[i], "r--", label="Pred", linewidth=1.5)
        if collocation_x is not None:
            for xc in collocation_x:
                ax.axvline(xc, color="gray", linewidth=0.5, linestyle=":", alpha=0.6)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_xlabel("x")
        ax.set_title(f"Sample {i}")
        if i == 0:
            ax.set_ylabel("Settlement")
            ax.legend(fontsize=8)

    fig.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Material field heatmap comparison
# ---------------------------------------------------------------------------

def plot_material_fields_comparison(
    fields_original: np.ndarray,
    fields_reduced: np.ndarray,
    n_nodes_x: int,
    n_nodes_z: int,
    field_name: str = "field",
    save_path: str | Path | None = None,
    n_samples: int = 5,
) -> None:
    """Side-by-side heatmap grid for one material field.

    Layout (2 rows × *n_samples* columns):
      - **Row 0**: original field heatmaps
      - **Row 1**: reduced/reconstructed field heatmaps
    Each cell has a shared colorbar per row.  Column titles report the
    per-sample absolute maximum difference between original and reduced.

    Parameters
    ----------
    fields_original : (n_total, n_nodes) array
        Original field realisations (physical units, shape n_nodes = n_nodes_x × n_nodes_z).
    fields_reduced : (n_total, n_nodes) array
        Reduced/reconstructed field realisations.
    n_nodes_x, n_nodes_z : int
        Grid dimensions.
    field_name : str
        Name used in the figure title and colorbar labels.
    save_path : str or Path, optional
        Save destination.
    n_samples : int
        Number of columns to show (default: 5).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    n_show = min(n_samples, len(fields_original))
    orig = fields_original[:n_show].reshape(n_show, n_nodes_x, n_nodes_z)
    redu = fields_reduced[:n_show].reshape(n_show, n_nodes_x, n_nodes_z)

    # Shared colour range across all shown samples (per-row shared colorbar)
    vmin = min(orig.min(), redu.min())
    vmax = max(orig.max(), redu.max())

    fig, axes = plt.subplots(
        2, n_show,
        figsize=(3 * n_show + 1, 5),
        gridspec_kw={"hspace": 0.4, "wspace": 0.3},
    )
    if n_show == 1:
        axes = axes[:, np.newaxis]

    for col in range(n_show):
        diff = float(np.max(np.abs(orig[col] - redu[col])))
        for row, data in enumerate((orig[col], redu[col])):
            ax = axes[row, col]
            im = ax.imshow(
                data.T,
                origin="lower",
                aspect="auto",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            row_label = "Original" if row == 0 else "Reduced"
            if col == 0:
                ax.set_ylabel(row_label, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"s{col}\n|Δ|={diff:.2g}", fontsize=8)

    # Shared colourbar on the right of the last column
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap="viridis",
                                norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label=field_name)

    fig.suptitle(f"{field_name}: original vs reduced", fontsize=11)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Convenience wrapper: all three fields
# ---------------------------------------------------------------------------

def plot_all_material_fields(
    fields_orig_dict: Dict[str, np.ndarray],
    fields_redu_dict: Dict[str, np.ndarray],
    n_nodes_x: int,
    n_nodes_z: int,
    output_dir: str | Path,
    n_samples: int = 5,
) -> Dict[str, str]:
    """Generate :func:`plot_material_fields_comparison` for E, k_h, k_v.

    Parameters
    ----------
    fields_orig_dict : dict mapping field name to (n_total, n_nodes) array
    fields_redu_dict : dict mapping field name to (n_total, n_nodes) array
    n_nodes_x, n_nodes_z : int
    output_dir : str or Path
        Directory where ``{name}_field_comparison.png`` files are saved.
    n_samples : int

    Returns
    -------
    dict mapping field name to saved file path string.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, str] = {}
    for name in ("E", "k_h", "k_v"):
        if name not in fields_orig_dict or name not in fields_redu_dict:
            continue
        save_path = output_dir / f"{name}_field_comparison.png"
        plot_material_fields_comparison(
            fields_orig_dict[name],
            fields_redu_dict[name],
            n_nodes_x=n_nodes_x,
            n_nodes_z=n_nodes_z,
            field_name=name,
            save_path=save_path,
            n_samples=n_samples,
        )
        saved[name] = str(save_path)
    return saved
