"""
visualization_v2.py
===================
Unified visualization module for Phase 2 and Phase 3 evaluation.

New functions
-------------
- :func:`plot_settlement_comparison_global_y`
    Settlement profiles with:
    * shared y-axis across all panels
    * y-axis STARTING FROM 0 (settlement is non-negative)
    * physical x-axis with units [m]
    * configurable axis labels and legend

- :func:`plot_material_fields_comparison`
    3-row × n_samples-column heatmap grid for one material field:
      Row 0: original fields (with physical (x,z) coordinates)
      Row 1: reduced/reconstructed fields
      Row 2: absolute difference |original − reduced|
    Shared colorbars per row, interpolated to smooth resolution.

- :func:`plot_all_material_fields`
    Convenience wrapper that calls :func:`plot_material_fields_comparison`
    for E, k_h, and k_v in a single figure each.

Backward compatibility
----------------------
All plot helpers originally defined in :mod:`src.utils` are re-exported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

# Re-export legacy helpers for backward compatibility
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
    lx: float = 1.0,
    save_path: str | Path | None = None,
    title: str = "Settlement comparison",
    n_samples: int = 5,
    collocation_x: Optional[np.ndarray] = None,
    label_true: str = "Ground truth",
    label_pred: str = "Prediction",
) -> None:
    """Plot true vs predicted settlement profiles with a **shared** y-axis.

    The y-axis always starts from 0 (settlement is non-negative by convention)
    and extends to the global maximum across all displayed samples plus 5 % margin.
    The x-axis uses physical coordinates in metres.

    Parameters
    ----------
    y_true : (n_total, n_nodes_x)
        Ground-truth settlement profiles [m].
    y_pred : (n_total, n_nodes_x)
        Predicted settlement profiles [m].
    n_nodes_x : int
        Number of surface nodes (x direction).
    lx : float
        Physical domain length in x [m].  Default 1.0.
    save_path : str or Path, optional
    title : str
    n_samples : int
    collocation_x : optional array of collocation x-positions [m]
    label_true, label_pred : legend labels
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = min(n_samples, len(y_true))
    x_phys = np.linspace(0.0, lx, n_nodes_x)

    shown_true = y_true[:n_show]
    shown_pred = y_pred[:n_show]

    # Global y range: start from 0, end at max + margin
    y_max = max(shown_true.max(), shown_pred.max(), 0.0)
    y_margin = max(abs(y_max) * 0.05, 1e-10)

    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 3.5), sharey=True)
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(x_phys, shown_true[i], "b-", label=label_true, linewidth=1.5)
        ax.plot(x_phys, shown_pred[i], "r--", label=label_pred, linewidth=1.5)
        if collocation_x is not None:
            for xc in collocation_x:
                ax.axvline(xc, color="gray", linewidth=0.5, linestyle=":", alpha=0.6)
        ax.set_ylim(0.0, y_max + y_margin)
        ax.set_xlabel("Position [m]", fontsize=9)
        ax.set_title(f"Sample {i}", fontsize=9)
        if i == 0:
            ax.set_ylabel("Settlement [m]", fontsize=9)
            ax.legend(fontsize=7)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Material field heatmap comparison (3-row × n_samples layout)
# ---------------------------------------------------------------------------

def plot_material_fields_comparison(
    fields_original: np.ndarray,
    fields_reduced: np.ndarray,
    n_nodes_x: int,
    n_nodes_z: int,
    lx: float = 1.0,
    lz: float = 0.5,
    field_name: str = "field",
    save_path: str | Path | None = None,
    n_samples: int = 5,
    interp_res: int = 200,
) -> None:
    """3-row × *n_samples*-column heatmap grid for one material field.

    Layout (3 rows × *n_samples* columns):
      - **Row 0**: Original field heatmaps
      - **Row 1**: Reduced/reconstructed field heatmaps
      - **Row 2**: Absolute difference |Original − Reduced|

    Axes use physical coordinates (metres, NOT node indices).
    Fields are interpolated to *interp_res* × *interp_res* for smooth rendering.
    Shared colorbars per row: rows 0–1 share one colorbar, row 2 has its own.

    Parameters
    ----------
    fields_original : (n_total, n_nodes) array
    fields_reduced : (n_total, n_nodes) array
    n_nodes_x, n_nodes_z : int
    lx, lz : float
        Physical domain dimensions [m].
    field_name : str
    save_path : optional
    n_samples : int
    interp_res : int
        Target interpolation resolution (pixels per side).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom

    n_show = min(n_samples, len(fields_original))
    orig = fields_original[:n_show].reshape(n_show, n_nodes_x, n_nodes_z)
    redu = fields_reduced[:n_show].reshape(n_show, n_nodes_x, n_nodes_z)
    diff = np.abs(orig - redu)

    # Interpolate for smooth rendering
    zoom_x = max(1, interp_res // n_nodes_x)
    zoom_z = max(1, interp_res // n_nodes_z)

    def _zoom(arr2d: np.ndarray) -> np.ndarray:
        return zoom(arr2d, (zoom_x, zoom_z), order=1)

    orig_zoom = [_zoom(orig[i]) for i in range(n_show)]
    redu_zoom = [_zoom(redu[i]) for i in range(n_show)]
    diff_zoom = [_zoom(diff[i]) for i in range(n_show)]

    # Colour limits
    vmin_field = min(np.min(orig), np.min(redu))
    vmax_field = max(np.max(orig), np.max(redu))
    vmax_diff = max(np.max(diff), 1e-30)

    # Physical extent: [x_min, x_max, z_min, z_max]
    extent = [0.0, lx, 0.0, lz]

    fig, axes = plt.subplots(
        3, n_show,
        figsize=(3.5 * n_show + 1.5, 7),
        gridspec_kw={"hspace": 0.45, "wspace": 0.25},
    )
    if n_show == 1:
        axes = axes[:, np.newaxis]

    row_labels = ["Original", "Reduced", "|Difference|"]

    for col in range(n_show):
        orig_stats = f"[{orig[col].min():.2g}, {orig[col].max():.2g}]"
        diff_max = float(np.max(diff[col]))

        for row, (data, label) in enumerate(
            zip((orig_zoom[col], redu_zoom[col], diff_zoom[col]), row_labels)
        ):
            ax = axes[row, col]
            if row < 2:
                im = ax.imshow(
                    data.T,
                    origin="lower",
                    aspect="auto",
                    cmap="viridis",
                    extent=extent,
                    vmin=vmin_field,
                    vmax=vmax_field,
                )
            else:
                im = ax.imshow(
                    data.T,
                    origin="lower",
                    aspect="auto",
                    cmap="hot_r",
                    extent=extent,
                    vmin=0.0,
                    vmax=vmax_diff,
                )

            if col == 0:
                ax.set_ylabel(f"{label}\nz [m]", fontsize=8)
            else:
                ax.set_yticks([])

            if row == 2:
                ax.set_xlabel("x [m]", fontsize=8)
            else:
                ax.set_xticks([])

            if row == 0:
                ax.set_title(f"Sample {col}\n{orig_stats}", fontsize=7)
            elif row == 2:
                ax.set_title(f"|Δ|_max={diff_max:.2g}", fontsize=7)

    # Colorbars
    # Rows 0-1: shared field colorbar
    fig.subplots_adjust(right=0.87)
    cbar_ax1 = fig.add_axes([0.89, 0.42, 0.015, 0.50])
    sm1 = plt.cm.ScalarMappable(
        cmap="viridis", norm=plt.Normalize(vmin=vmin_field, vmax=vmax_field)
    )
    sm1.set_array([])
    cb1 = fig.colorbar(sm1, cax=cbar_ax1)
    cb1.set_label(field_name, fontsize=8)

    # Row 2: difference colorbar
    cbar_ax2 = fig.add_axes([0.89, 0.06, 0.015, 0.28])
    sm2 = plt.cm.ScalarMappable(
        cmap="hot_r", norm=plt.Normalize(vmin=0.0, vmax=vmax_diff)
    )
    sm2.set_array([])
    cb2 = fig.colorbar(sm2, cax=cbar_ax2)
    cb2.set_label(f"|Δ {field_name}|", fontsize=8)

    fig.suptitle(f"{field_name}: original vs reduced (all {n_show} samples)", fontsize=11)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Convenience wrapper: all three fields
# ---------------------------------------------------------------------------

def plot_all_material_fields(
    fields_orig_dict: Dict[str, np.ndarray],
    fields_redu_dict: Dict[str, np.ndarray],
    n_nodes_x: int,
    n_nodes_z: int,
    lx: float = 1.0,
    lz: float = 0.5,
    output_dir: str | Path = ".",
    n_samples: int = 5,
    interp_res: int = 200,
) -> Dict[str, str]:
    """Generate :func:`plot_material_fields_comparison` for E, k_h, k_v.

    Parameters
    ----------
    fields_orig_dict : dict mapping field name to (n_total, n_nodes) array
    fields_redu_dict : dict mapping field name to (n_total, n_nodes) array
    n_nodes_x, n_nodes_z : int
    lx, lz : float
        Physical domain dimensions [m].
    output_dir : Path
    n_samples : int
    interp_res : int

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
            lx=lx,
            lz=lz,
            field_name=name,
            save_path=save_path,
            n_samples=n_samples,
            interp_res=interp_res,
        )
        saved[name] = str(save_path)
    return saved
