"""
config_manager.py
=================
YAML configuration loading, validation, and default injection.

All pipeline components receive a validated config dict produced here.

Config format support
---------------------
Three config formats are accepted (all translated internally to phase2/phase3):

**Canonical format** (recommended — single source of truth per model):
  Top-level sections: ``model_a`` (surrogate), ``model_b`` (reducer), ``solver``, ``grid``.
  Each model block contains field definitions, dataset generation counts,
  model hyperparameters, and evaluation/plotting settings.  There is no
  separate ``data_generation`` / ``models`` split, eliminating shadowing.
  Example::

      model_a:
        fields: { E: {...}, k_h: {...}, k_v: {...} }  # single canonical definition
        n_samples: 2000
        output_dir: "models/phase2_surrogate"
        type: "nn"
        nn: { ... }
        evaluation: { test_fraction: 0.1, n_plot_samples: 10 }
      model_b:
        fields: { E: {...}, k_h: {...}, k_v: {...} }  # full-space fields for reducer input
        n_samples: 5000
        output_dir: "models/phase3_reducer"
        nn: { ... }
        evaluation: { test_fraction: 0.1, n_plot_samples: 10, plot_mode: "three_curve" }

  Note: ``model_b`` never specifies a ``reduced_fields`` block — it is always
  derived from ``model_a.fields`` to guarantee consistency.

**Intermediate format** (still accepted for backward compatibility):
  Top-level sections:  ``data_generation``, ``models``, ``evaluation``, ``solver``, ``grid``.
  Dataset generation configs for both models appear first, then model configs.

**Old / legacy format** (still supported, no breaking change):
  Top-level sections: ``phase2``, ``phase3``, ``collocation_phase2``, ``collocation_phase3``.
  ``collocation_phase2`` / ``collocation_phase3`` are deprecated; move to
  ``phase2.collocation_n_points`` / ``phase3.collocation_n_points``.

Single-source-of-truth enforcement
------------------------------------
Regardless of which format is used, ``phase3.reduced_fields`` is ALWAYS
synchronised to equal ``phase2.reduced_fields`` after all translations and
merges are applied.  If they differ (e.g. because of stale duplicate blocks),
a ``UserWarning`` is emitted and the Phase-2 value is used for both, preventing
the reducer→surrogate dimension mismatch.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration (minimal, always valid)
# ---------------------------------------------------------------------------
_DEFAULTS: Dict[str, Any] = {
    "grid": {
        "n_nodes_x": 20,
        "n_nodes_z": 10,
        "lx": 1.0,
        "lz": 0.5,
    },
    # random_fields is used ONLY by Phase 1 (full-space dataset generation).
    # For Phase 2/3, use model_a.fields (reduced-space) and model_b.fields
    # (full-space) respectively.  n_terms here is intentionally the same as
    # model_b.fields.E.n_terms (=15, full-space dimension); it is different from
    # model_a.fields.E.n_terms (=8, reduced-space dimension).
    "random_fields": {
        "E": {
            "n_terms": 15,
            "basis": "dct",
            "seed": 42,
            "covariance": "matern",
            "nu_sampling": True,
            "nu_range": [0.5, 2.5],
            "nu_ref": 1.5,
            "length_scale_sampling": True,
            "length_scale_range": [0.1, 0.5],
            "length_scale_ref": 0.3,
            # Unified format (preferred)
            "mean": 10.0e6,
            "range": [5.0e6, 20.0e6],
            "fluctuation_std": 1.0,
            "force_identity_reduction": False,
            "mean_sampling": True,
            "mean_range": [5.0e6, 20.0e6],
        },
        "k_h": {
            "n_terms": 0,
            "basis": "dct",
            "seed": 43,
            "covariance": "matern",
            "nu_sampling": False,
            "nu_ref": 1.5,
            "length_scale_sampling": False,
            "length_scale_ref": 0.3,
            # Unified format (preferred)
            "mean": 1.0e-12,
            "range": [1.0e-13, 1.0e-10],
            "fluctuation_std": 0.5,
            "force_identity_reduction": False,
            "mean_sampling": True,
            "mean_range": [1.0e-13, 1.0e-10],
        },
        "k_v": {
            "n_terms": 0,
            "basis": "dct",
            "seed": 44,
            "covariance": "matern",
            "nu_sampling": True,
            "nu_range": [0.5, 2.5],
            "nu_ref": 1.5,
            "length_scale_sampling": True,
            "length_scale_range": [0.1, 0.5],
            "length_scale_ref": 0.3,
            # Unified format (preferred)
            "mean": 1.0e-12,
            "range": [1.0e-13, 1.0e-10],
            "fluctuation_std": 0.5,
            "force_identity_reduction": False,
            "mean_sampling": True,
            "mean_range": [1.0e-13, 1.0e-10],
        },
    },
    "solver": {
        "type": "2d",
        "mode": "steady",
        "nu_biot": 0.3,
        "fluid_viscosity": 1.0e-3,
        "fluid_compressibility": 4.5e-10,
        "load": 1.0e6,
        "transient": {"dt": 0.01, "n_steps": 100},
    },
    "phase1": {
        "n_samples": 2000,
        "val_fraction": 0.2,
        "output_dir": "data",
    },
    # ---------------------------------------------------------------------------
    # Phase 2 (surrogate / Model A): data generation + model + evaluation
    # ---------------------------------------------------------------------------
    "phase2": {
        # --- Dataset generation (reduced-space sampling) ---
        "n_training_samples": 2000,
        "collocation_n_points": 20,  # canonical path (was: top-level collocation_phase2)
        # reduced_fields defines the reduced-space coefficient dimension;
        # used by BOTH data generation AND the surrogate model.
        # This is the SINGLE SOURCE OF TRUTH — it is also propagated to
        # phase3.reduced_fields by _sync_reduced_fields().
        "reduced_fields": {
            "E": {
                "n_terms": 8, "basis": "dct", "seed": 142,
                "nu_sampling": False, "nu_ref": 1.5,
                "length_scale_sampling": False, "length_scale_ref": 0.3,
                "mean": 10.0e6, "range": [5.0e6, 20.0e6], "fluctuation_std": 1.0,
                "mean_sampling": True, "mean_range": [5.0e6, 20.0e6],
            },
            "k_h": {
                "n_terms": 0, "basis": "dct", "seed": 143,
                "nu_sampling": False, "nu_ref": 1.5,
                "length_scale_sampling": False, "length_scale_ref": 0.3,
                "mean": 1.0e-12, "range": [1.0e-13, 1.0e-10], "fluctuation_std": 0.5,
                "mean_sampling": True, "mean_range": [1.0e-13, 1.0e-10],
            },
            "k_v": {
                "n_terms": 0, "basis": "dct", "seed": 144,
                "nu_sampling": False, "nu_ref": 1.5,
                "length_scale_sampling": False, "length_scale_ref": 0.3,
                "mean": 1.0e-12, "range": [1.0e-13, 1.0e-10], "fluctuation_std": 0.5,
                "mean_sampling": True, "mean_range": [1.0e-13, 1.0e-10],
            },
        },
        # --- Model (surrogate architecture + training) ---
        "surrogate_type": "nn",
        "output_repr": "direct",
        "n_output_modes": 10,
        "training_signal": "hybrid",
        "hybrid_alpha": 0.1,
        "physics_check_interval": 10,
        "output_dir": "models/phase2_surrogate",
        "nn": {
            "hidden_dims": [128, 128, 64],
            "epochs": 200,
            "lr": 1.0e-3,
            "batch_size": 32,
            "patience": 50,
        },
        "pce": {"degree": 3},
        # --- Evaluation ---
        "evaluation": {
            "test_fraction": 0.1,
            "n_plot_samples": 10,
        },
    },
    # ---------------------------------------------------------------------------
    # Phase 3 (reducer / Model B): data generation + model + evaluation
    # ---------------------------------------------------------------------------
    "phase3": {
        # --- Dataset generation (full-space sampling) ---
        "n_training_samples": 5000,
        "collocation_n_points": 20,  # canonical path (was: top-level collocation_phase3)
        # full_fields defines the full-space coefficient dimension for reducer input
        "full_fields": {
            "E": {
                "n_terms": 15, "basis": "dct", "seed": 42,
                "nu_sampling": True, "nu_range": [0.5, 2.5], "nu_ref": 1.5,
                "length_scale_sampling": True, "length_scale_range": [0.1, 0.5],
                "length_scale_ref": 0.3,
                "mean": 10.0e6, "range": [5.0e6, 20.0e6], "fluctuation_std": 1.0,
                "mean_sampling": True, "mean_range": [5.0e6, 20.0e6],
            },
            "k_h": {
                "n_terms": 0, "basis": "dct", "seed": 43,
                "nu_sampling": False, "nu_ref": 1.5,
                "length_scale_sampling": False, "length_scale_ref": 0.3,
                "mean": 1.0e-12, "range": [1.0e-13, 1.0e-10], "fluctuation_std": 0.5,
                "mean_sampling": True, "mean_range": [1.0e-13, 1.0e-10],
            },
            "k_v": {
                "n_terms": 0, "basis": "dct", "seed": 44,
                "nu_sampling": True, "nu_range": [0.5, 2.5], "nu_ref": 1.5,
                "length_scale_sampling": True, "length_scale_range": [0.1, 0.5],
                "length_scale_ref": 0.3,
                "mean": 1.0e-12, "range": [1.0e-13, 1.0e-10], "fluctuation_std": 0.5,
                "mean_sampling": True, "mean_range": [1.0e-13, 1.0e-10],
            },
        },
        # reduced_fields is ALWAYS synchronised from phase2.reduced_fields
        # by _sync_reduced_fields().  Do NOT set this manually — it is listed
        # here only so that the defaults dict is self-consistent; it will be
        # overwritten at runtime by the phase2 value.
        "reduced_fields": {
            "E": {
                "n_terms": 8, "basis": "dct", "seed": 142,
                "nu_sampling": False, "nu_ref": 1.5,
                "length_scale_sampling": False, "length_scale_ref": 0.3,
                "mean": 10.0e6, "range": [5.0e6, 20.0e6], "fluctuation_std": 1.0,
                "mean_sampling": True, "mean_range": [5.0e6, 20.0e6],
            },
            "k_h": {
                "n_terms": 0, "basis": "dct", "seed": 143,
                "nu_sampling": False, "nu_ref": 1.5,
                "length_scale_sampling": False, "length_scale_ref": 0.3,
                "mean": 1.0e-12, "range": [1.0e-13, 1.0e-10], "fluctuation_std": 0.5,
                "mean_sampling": True, "mean_range": [1.0e-13, 1.0e-10],
            },
            "k_v": {
                "n_terms": 0, "basis": "dct", "seed": 144,
                "nu_sampling": False, "nu_ref": 1.5,
                "length_scale_sampling": False, "length_scale_ref": 0.3,
                "mean": 1.0e-12, "range": [1.0e-13, 1.0e-10], "fluctuation_std": 0.5,
                "mean_sampling": True, "mean_range": [1.0e-13, 1.0e-10],
            },
        },
        # --- Model (reducer architecture + training) ---
        "reducer_type": "nn",
        "training_signal": "surrogate",
        "output_dir": "models/phase3_reducer",
        "surrogate_dir": "models/phase2_surrogate",
        "load_phase2_model": None,
        "nn": {
            "hidden_dims": [256, 128, 64],
            "epochs": 300,
            "lr": 1.0e-3,
            "batch_size": 32,
            "patience": 50,
        },
        # --- Evaluation ---
        "evaluation": {
            "test_fraction": 0.1,
            "n_plot_samples": 10,
            # plot_mode: "two_curve"   → GT + reducer→Biot
            #             "three_curve" → GT + reducer→Biot + reducer→Surrogate
            "plot_mode": "three_curve",
        },
    },
    # NOTE: collocation_phase2 / collocation_phase3 are DEPRECATED top-level keys.
    # The canonical paths are now phase2.collocation_n_points / phase3.collocation_n_points.
    # These entries are kept here only for backward compatibility with old configs that
    # set them at the top level; new configs should use the phase-specific paths.
    "collocation_phase2": {"n_points": 20},
    "collocation_phase3": {"n_points": 5},
    "phase4": {
        "n_test_samples": 50,
        "output_dir": "results",
        "use_physics_for_plots": False,
        "random_seed": 0,
        "shuffle": True,
    },
}


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge *override* into a copy of *base*."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class ConfigManager:
    """Load, validate, and expose configuration.

    Parameters
    ----------
    path : str or Path, optional
        Path to YAML config file.  If *None*, default values are used.
    overrides : dict, optional
        Additional key-value overrides (applied after file loading).
    """

    def __init__(
        self,
        path: str | Path | None = None,
        overrides: Dict[str, Any] | None = None,
    ) -> None:
        cfg: Dict[str, Any] = copy.deepcopy(_DEFAULTS)

        if path is not None:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")
            with open(path, encoding='utf-8') as f:
                user_cfg = yaml.safe_load(f) or {}
            user_cfg = self._translate_new_format(user_cfg)
            user_cfg = self._handle_deprecated_keys(user_cfg)
            cfg = _deep_merge(cfg, user_cfg)

        if overrides:
            overrides = self._translate_new_format(overrides)
            overrides = self._handle_deprecated_keys(overrides)
            cfg = _deep_merge(cfg, overrides)

        self._cfg = self._coerce_numeric_types(cfg)
        self._sync_reduced_fields()
        self._validate()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def cfg(self) -> Dict[str, Any]:
        """Return a deep copy of the validated configuration dict."""
        return copy.deepcopy(self._cfg)

    def get(self, *keys: str, default: Any = None) -> Any:
        """Retrieve a nested value by key path, e.g. get('phase2', 'nn', 'lr')."""
        node = self._cfg
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    def config_hash(self) -> str:
        """SHA-256 hex digest of the serialised configuration (for model versioning)."""
        serialised = json.dumps(self._cfg, sort_keys=True, default=str)
        return hashlib.sha256(serialised.encode()).hexdigest()[:16]

    def save(self, path: str | Path) -> None:
        """Persist the merged configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding='utf-8') as f:
            yaml.safe_dump(self._cfg, f, default_flow_style=False)

    def warn_if_transient_mode(self) -> None:
        """Emit a prominent warning when ``solver.mode`` is 'transient'.

        The current two-model pipeline (surrogate + reducer) targets steady-state
        Biot consolidation.  Transient mode changes the forward solver internally
        but the training and evaluation pipelines are **not** wired for time-series
        data end-to-end.

        Call this method at the **start** of every training and validation entry
        point (before any pipeline execution begins) so the user is immediately
        informed of the limitation.  It is a no-op when ``solver.mode`` is
        'steady' (default).
        """
        mode = self._cfg.get("solver", {}).get("mode", "steady")
        if mode == "transient":
            warnings.warn(
                "\n"
                "╔══════════════════════════════════════════════════════════════╗\n"
                "║  TRANSIENT MODE WARNING                                      ║\n"
                "║  solver.mode = 'transient' is set, but the current pipeline  ║\n"
                "║  (surrogate + reducer) is designed for STEADY-STATE Biot     ║\n"
                "║  consolidation only.  Transient-aware training, loss terms,  ║\n"
                "║  and time-series evaluation are NOT yet implemented end-to-  ║\n"
                "║  end.  Your run will proceed with steady-style behaviour;    ║\n"
                "║  transient dt / n_steps settings are forwarded to the solver ║\n"
                "║  but have no effect on surrogate or reducer training.        ║\n"
                "╚══════════════════════════════════════════════════════════════╝",
                UserWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def n_terms_E(self) -> int:
        return self._cfg["random_fields"]["E"]["n_terms"]

    @property
    def n_terms_kh(self) -> int:
        return self._cfg["random_fields"]["k_h"]["n_terms"]

    @property
    def n_terms_kv(self) -> int:
        return self._cfg["random_fields"]["k_v"]["n_terms"]

    @property
    def total_input_dim(self) -> int:
        """Total concatenated coefficient dimension: n_terms_E + n_terms_kh + n_terms_kv.

        For homogeneous fields (n_terms == 0) a single scalar is used,
        so the effective per-field contribution is max(n_terms, 1).
        """
        return (
            max(self.n_terms_E, 1)
            + max(self.n_terms_kh, 1)
            + max(self.n_terms_kv, 1)
        )

    @property
    def n_nodes_x(self) -> int:
        return self._cfg["grid"]["n_nodes_x"]

    @property
    def n_nodes_z(self) -> int:
        return self._cfg["grid"]["n_nodes_z"]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _coerce_numeric_types(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all numeric config values are proper Python floats/ints.

        YAML may load numeric values as strings (especially on systems with
        non-default YAML parsers or when values are quoted in the file).
        This method converts all known numeric fields to the appropriate type.
        """
        _float_fields = {
            'E_ref', 'mean', 'fluctuation_std', 'logE_std',
            'nu_ref', 'length_scale_ref',
            'learning_rate', 'lr', 'hybrid_alpha',
            'val_fraction',
        }
        _int_fields = {
            'n_terms', 'seed', 'n_samples', 'n_output_modes',
            'epochs', 'batch_size', 'patience', 'degree',
            'n_points', 'n_test_samples',
        }
        _list_float_fields = {
            'range', 'nu_range', 'length_scale_range', 'k_range', 'mean_range',
        }

        def _coerce_dict(d: Dict) -> None:
            for key, val in d.items():
                if key in _float_fields and val is not None:
                    try:
                        d[key] = float(val)
                    except (ValueError, TypeError) as e:
                        logger.warning("Could not coerce %s=%r to float: %s", key, val, e)
                elif key in _int_fields and val is not None:
                    try:
                        d[key] = int(val)
                    except (ValueError, TypeError) as e:
                        logger.warning("Could not coerce %s=%r to int: %s", key, val, e)
                elif key in _list_float_fields and isinstance(val, (list, tuple)):
                    try:
                        d[key] = [float(x) for x in val]
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            "Could not coerce %s=%r to float list: %s", key, val, e
                        )
                elif isinstance(val, dict):
                    _coerce_dict(val)
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict):
                            _coerce_dict(item)

        _coerce_dict(config)

        # Explicitly handle grid fields
        grid = config.get('grid', {})
        for key in ('n_nodes_x', 'n_nodes_z'):
            if key in grid and grid[key] is not None:
                try:
                    grid[key] = int(grid[key])
                except (ValueError, TypeError):
                    pass
        for key in ('lx', 'lz'):
            if key in grid and grid[key] is not None:
                try:
                    grid[key] = float(grid[key])
                except (ValueError, TypeError):
                    pass

        # Explicitly handle solver fields
        solver = config.get('solver', {})
        for key in ('n_steps',):
            if key in solver and solver[key] is not None:
                try:
                    solver[key] = int(solver[key])
                except (ValueError, TypeError):
                    pass
        for key in ('nu_biot', 'fluid_viscosity', 'fluid_compressibility', 'load'):
            if key in solver and solver[key] is not None:
                try:
                    solver[key] = float(solver[key])
                except (ValueError, TypeError):
                    pass
        transient = solver.get('transient', {})
        if 'dt' in transient:
            try:
                transient['dt'] = float(transient['dt'])
            except (ValueError, TypeError):
                pass
        if 'n_steps' in transient:
            try:
                transient['n_steps'] = int(transient['n_steps'])
            except (ValueError, TypeError):
                pass

        return config

    def _validate(self) -> None:
        """Basic sanity checks on the merged configuration."""
        rf = self._cfg["random_fields"]
        for name in ("E", "k_h", "k_v"):
            if name not in rf:
                raise ValueError(f"random_fields.{name} section is missing from config")
            field_cfg = rf[name]
            if field_cfg.get("basis", "dct") != "dct":
                raise ValueError(
                    f"random_fields.{name}.basis must be 'dct'; "
                    f"got '{field_cfg.get('basis')}'"
                )
            if field_cfg["n_terms"] < 0:
                raise ValueError(
                    f"random_fields.{name}.n_terms must be >= 0"
                )

        grid = self._cfg["grid"]
        for key in ("n_nodes_x", "n_nodes_z"):
            if grid[key] < 1:
                raise ValueError(f"grid.{key} must be >= 1")

        phase2 = self._cfg["phase2"]
        valid_repr = {"direct", "dct", "poly", "bspline"}
        if phase2["output_repr"] not in valid_repr:
            raise ValueError(
                f"phase2.output_repr must be one of {valid_repr}; "
                f"got '{phase2['output_repr']}'"
            )

        valid_signals = {"data", "physics", "hybrid", "surrogate"}
        for phase_key in ("phase2", "phase3"):
            sig = self._cfg[phase_key]["training_signal"]
            if sig not in valid_signals:
                raise ValueError(
                    f"{phase_key}.training_signal must be one of {valid_signals}; "
                    f"got '{sig}'"
                )

        # After _sync_reduced_fields, phase2 and phase3 reduced_fields must match.
        p2_rf = self._cfg["phase2"].get("reduced_fields", {})
        p3_rf = self._cfg["phase3"].get("reduced_fields", {})
        for field_name in ("E", "k_h", "k_v"):
            p2_n = p2_rf.get(field_name, {}).get("n_terms")
            p3_n = p3_rf.get(field_name, {}).get("n_terms")
            if p2_n is not None and p3_n is not None and p2_n != p3_n:
                raise ValueError(
                    f"phase2.reduced_fields.{field_name}.n_terms ({p2_n}) != "
                    f"phase3.reduced_fields.{field_name}.n_terms ({p3_n}).  "
                    "This would cause a dimension mismatch between the reducer output "
                    "and surrogate input.  _sync_reduced_fields should have fixed this; "
                    "please report this as a bug."
                )

    def _sync_reduced_fields(self) -> None:
        """Enforce that phase3.reduced_fields always equals phase2.reduced_fields.

        This is the single-source-of-truth guarantee: the reducer's output
        dimension MUST match the surrogate's input dimension.  If any mismatch
        is detected (e.g. due to stale duplicate config blocks), a UserWarning
        is emitted and the Phase-2 definition is used for both.

        Call this method *after* ``_coerce_numeric_types`` and *before*
        ``_validate``.
        """
        p2_rf = self._cfg.get("phase2", {}).get("reduced_fields", {})
        p3_rf = self._cfg.get("phase3", {}).get("reduced_fields", {})

        mismatch_fields = []
        for field_name in ("E", "k_h", "k_v"):
            p2_n = p2_rf.get(field_name, {}).get("n_terms")
            p3_n = p3_rf.get(field_name, {}).get("n_terms")
            if p2_n is not None and p3_n is not None and p2_n != p3_n:
                mismatch_fields.append(
                    f"{field_name}: phase2 n_terms={p2_n}, phase3 n_terms={p3_n}"
                )

        if mismatch_fields:
            warnings.warn(
                "Dimension mismatch detected between phase2.reduced_fields and "
                "phase3.reduced_fields.  This would cause a broadcast error during "
                "surrogate decode in Phase-3 evaluation.  Synchronising to the "
                "Phase-2 definition (single source of truth).\n"
                "Mismatched fields: " + "; ".join(mismatch_fields) + "\n"
                "To silence this warning, remove the duplicate 'reduced_fields' block "
                "from phase3/model_b — it is always derived from phase2/model_a.fields.",
                UserWarning,
                stacklevel=3,
            )

        # Always overwrite phase3.reduced_fields with phase2.reduced_fields.
        if p2_rf:
            if "phase3" not in self._cfg:
                self._cfg["phase3"] = {}
            self._cfg["phase3"]["reduced_fields"] = copy.deepcopy(p2_rf)

    # ------------------------------------------------------------------
    # New-format translation + deprecated-key handling
    # ------------------------------------------------------------------

    @staticmethod
    def _translate_new_format(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Translate canonical / intermediate formats to the internal phase2/phase3 layout.

        Three source formats are handled:

        1. **Canonical** (``model_a`` / ``model_b`` top-level keys) — recommended.
           ``model_a.fields`` is the single field-dimension definition used by
           *both* Phase-2 data generation and the surrogate model.
           ``model_b.fields`` defines full-space fields for the reducer input;
           the reducer *output* dimension (``phase3.reduced_fields``) is always
           inherited from ``model_a.fields`` — it cannot be overridden here.

        2. **Intermediate** (``data_generation`` / ``models`` / ``evaluation``) —
           still accepted.  When both ``data_generation.surrogate.fields`` and
           ``models.surrogate.reduced_fields`` are present, the
           ``data_generation`` value takes precedence to avoid silent shadowing.

        3. **Legacy** (``phase2`` / ``phase3``) — passed through unchanged.
        """
        ma = cfg.get("model_a", {})
        mb = cfg.get("model_b", {})
        dg = cfg.get("data_generation", {})
        mods = cfg.get("models", {})
        eval_ = cfg.get("evaluation", {})

        has_canonical = bool(ma or mb)
        has_intermediate = bool(dg or mods or eval_)
        has_legacy = "phase2" in cfg or "phase3" in cfg

        if not (has_canonical or has_intermediate):
            return cfg  # purely legacy format – pass through unchanged

        if (has_canonical or has_intermediate) and has_legacy:
            warnings.warn(
                "Config mixes canonical/intermediate keys (model_a / model_b / "
                "data_generation / models) with legacy keys (phase2 / phase3).  "
                "Canonical/intermediate sections take precedence for overlapping settings.",
                UserWarning,
                stacklevel=5,
            )

        result = copy.deepcopy(cfg)

        # ===================================================================
        # 1. Canonical format: model_a / model_b
        # ===================================================================
        if has_canonical:
            # ---- Model A (surrogate / phase2) ----
            if ma:
                p2 = result.setdefault("phase2", {})
                # fields is the single canonical definition for both data gen and model
                if "fields" in ma:
                    p2["reduced_fields"] = copy.deepcopy(ma["fields"])
                if "n_samples" in ma:
                    p2["n_training_samples"] = ma["n_samples"]
                if "output_dir" in ma:
                    p2["output_dir"] = ma["output_dir"]
                if "collocation_n_points" in ma:
                    p2["collocation_n_points"] = ma["collocation_n_points"]
                if "type" in ma:
                    p2["surrogate_type"] = ma["type"]
                for k in ("output_repr", "training_signal", "hybrid_alpha",
                          "physics_check_interval", "n_output_modes", "nn", "pce"):
                    if k in ma:
                        p2[k] = copy.deepcopy(ma[k])
                if "evaluation" in ma:
                    p2["evaluation"] = copy.deepcopy(ma["evaluation"])

            # ---- Model B (reducer / phase3) ----
            if mb:
                p3 = result.setdefault("phase3", {})
                # model_b.fields = full-space fields (reducer INPUT)
                if "fields" in mb:
                    p3["full_fields"] = copy.deepcopy(mb["fields"])
                if "n_samples" in mb:
                    p3["n_training_samples"] = mb["n_samples"]
                if "output_dir" in mb:
                    p3["output_dir"] = mb["output_dir"]
                if "collocation_n_points" in mb:
                    p3["collocation_n_points"] = mb["collocation_n_points"]
                if "type" in mb:
                    p3["reducer_type"] = mb["type"]
                for k in ("training_signal", "surrogate_dir", "load_phase2_model", "nn"):
                    if k in mb:
                        p3[k] = copy.deepcopy(mb[k])
                if "evaluation" in mb:
                    p3["evaluation"] = copy.deepcopy(mb["evaluation"])
                # NOTE: reduced_fields is intentionally NOT read from model_b.
                # It is always inherited from model_a.fields / phase2.reduced_fields
                # by _sync_reduced_fields().  Any explicit model_b reduced_fields
                # would be silently ignored here and overridden later.
                if "reduced_fields" in mb:
                    warnings.warn(
                        "model_b.reduced_fields is ignored.  The reducer output "
                        "dimension is always derived from model_a.fields to guarantee "
                        "consistency with the surrogate input dimension.  "
                        "Remove model_b.reduced_fields from your config.",
                        UserWarning,
                        stacklevel=5,
                    )

            # Remove canonical top-level keys
            for k in ("model_a", "model_b"):
                result.pop(k, None)

        # ===================================================================
        # 2. Intermediate format: data_generation / models / evaluation
        # ===================================================================
        if has_intermediate:
            # ---- Surrogate (phase2) ----
            dg_surr = dg.get("surrogate", {})
            m_surr = mods.get("surrogate", {})
            eval_surr = eval_.get("surrogate", {})

            if dg_surr or m_surr or eval_surr:
                p2 = result.setdefault("phase2", {})

                # data_generation.surrogate settings (take precedence over models.surrogate
                # for reduced_fields to avoid silent shadowing)
                if "n_samples" in dg_surr:
                    p2["n_training_samples"] = dg_surr["n_samples"]
                if "output_dir" in dg_surr:
                    p2.setdefault("output_dir", dg_surr["output_dir"])
                if "collocation_n_points" in dg_surr:
                    p2["collocation_n_points"] = dg_surr["collocation_n_points"]
                # Stash data_generation.surrogate.fields before processing the model section,
                # so we can apply it AFTER models.surrogate.reduced_fields (last write wins).
                # This ensures data_generation always takes precedence over models.surrogate.
                dg_fields = copy.deepcopy(dg_surr["fields"]) if "fields" in dg_surr else None

                # models.surrogate settings (non-field keys only, to avoid shadowing)
                if "type" in m_surr:
                    p2["surrogate_type"] = m_surr["type"]
                for k in ("output_repr", "training_signal", "hybrid_alpha",
                          "physics_check_interval", "n_output_modes",
                          "output_dir", "nn", "pce"):
                    if k in m_surr:
                        p2[k] = copy.deepcopy(m_surr[k])
                # models.surrogate.reduced_fields is accepted but will be overridden
                # by data_generation.surrogate.fields if the latter is present.
                if "reduced_fields" in m_surr:
                    p2["reduced_fields"] = copy.deepcopy(m_surr["reduced_fields"])

                # data_generation.surrogate.fields wins over models.surrogate.reduced_fields
                if dg_fields is not None:
                    if "reduced_fields" in p2 and p2["reduced_fields"] != dg_fields:
                        warnings.warn(
                            "Both data_generation.surrogate.fields and "
                            "models.surrogate.reduced_fields are defined but differ.  "
                            "data_generation.surrogate.fields takes precedence.  "
                            "Remove the duplicate models.surrogate.reduced_fields block.",
                            UserWarning,
                            stacklevel=5,
                        )
                    p2["reduced_fields"] = dg_fields

                if eval_surr:
                    p2["evaluation"] = copy.deepcopy(eval_surr)

            # ---- Reducer (phase3) ----
            dg_red = dg.get("reducer", {})
            m_red = mods.get("reducer", {})
            eval_red = eval_.get("reducer", {})

            if dg_red or m_red or eval_red:
                p3 = result.setdefault("phase3", {})

                if "n_samples" in dg_red:
                    p3["n_training_samples"] = dg_red["n_samples"]
                if "output_dir" in dg_red:
                    p3.setdefault("output_dir", dg_red["output_dir"])
                if "fields" in dg_red:
                    p3["full_fields"] = copy.deepcopy(dg_red["fields"])
                if "collocation_n_points" in dg_red:
                    p3["collocation_n_points"] = dg_red["collocation_n_points"]

                if "type" in m_red:
                    p3["reducer_type"] = m_red["type"]
                for k in ("training_signal", "output_dir", "surrogate_dir",
                          "load_phase2_model", "nn"):
                    if k in m_red:
                        p3[k] = copy.deepcopy(m_red[k])

                # models.reducer.reduced_fields is accepted but will be overridden
                # by _sync_reduced_fields(); emit a warning to inform the user.
                if "reduced_fields" in m_red:
                    warnings.warn(
                        "models.reducer.reduced_fields is defined but will be "
                        "overridden to match phase2.reduced_fields (single source "
                        "of truth).  Remove this block to silence the warning.",
                        UserWarning,
                        stacklevel=5,
                    )
                    p3["reduced_fields"] = copy.deepcopy(m_red["reduced_fields"])

                if eval_red:
                    p3["evaluation"] = copy.deepcopy(eval_red)

            # Remove intermediate top-level keys to avoid confusion
            for k in ("data_generation", "models", "evaluation"):
                result.pop(k, None)

        return result

    @staticmethod
    def _handle_deprecated_keys(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Emit deprecation warnings and migrate obsolete top-level keys.

        ``collocation_phase2``/``collocation_phase3`` at the top level are
        deprecated; the canonical paths are now
        ``phase2.collocation_n_points`` / ``phase3.collocation_n_points``.
        """
        result = copy.deepcopy(cfg)

        if "collocation_phase2" in cfg:
            warnings.warn(
                "Top-level 'collocation_phase2' is deprecated.  "
                "Use 'phase2.collocation_n_points' instead.",
                DeprecationWarning,
                stacklevel=5,
            )
            n_pts = cfg["collocation_phase2"].get("n_points", 20)
            result.setdefault("phase2", {}).setdefault("collocation_n_points", n_pts)

        if "collocation_phase3" in cfg:
            warnings.warn(
                "Top-level 'collocation_phase3' is deprecated.  "
                "Use 'phase3.collocation_n_points' instead.",
                DeprecationWarning,
                stacklevel=5,
            )
            n_pts = cfg["collocation_phase3"].get("n_points", 5)
            result.setdefault("phase3", {}).setdefault("collocation_n_points", n_pts)

        return result
