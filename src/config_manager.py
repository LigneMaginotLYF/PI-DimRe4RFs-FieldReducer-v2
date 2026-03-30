"""
config_manager.py
=================
YAML configuration loading, validation, and default injection.

All pipeline components receive a validated config dict produced here.

Config format support
---------------------
Two config formats are accepted:

**New format** (recommended, as of this refactor):
  Top-level sections:  ``data_generation``, ``models``, ``evaluation``, ``solver``, ``grid``.
  Dataset generation configs for both models appear first, then model configs.
  Example::

      data_generation:
        surrogate:
          n_samples: 200
          fields: { E: {...}, k_h: {...}, k_v: {...} }
        reducer:
          n_samples: 500
          fields: { E: {...}, k_h: {...}, k_v: {...} }
      models:
        surrogate:
          type: "nn"
          reduced_fields: { E: {...}, ... }
          nn: { ... }
        reducer:
          type: "nn"
          surrogate_dir: "..."
          nn: { ... }
      evaluation:
        surrogate: { test_fraction: 0.2, n_plot_samples: 5 }
        reducer:   { test_fraction: 0.2, n_plot_samples: 5 }

**Old / legacy format** (still supported, no breaking change):
  Top-level sections: ``phase2``, ``phase3``, ``collocation_phase2``, ``collocation_phase3``.
  ``collocation_phase2`` / ``collocation_phase3`` are deprecated; move to
  ``phase2.collocation_n_points`` / ``phase3.collocation_n_points``.
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
    "random_fields": {
        "E": {
            "n_terms": 5,
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
            "mean_sampling": False,
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
            "mean_sampling": False,
            "mean_range": [1.0e-13, 1.0e-10],
        },
        "k_v": {
            "n_terms": 2,
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
            "mean_sampling": False,
            "mean_range": [1.0e-13, 1.0e-10],
        },
    },
    "solver": {
        "type": "1d",
        "mode": "steady",
        "nu_biot": 0.3,
        "fluid_viscosity": 1.0e-3,
        "fluid_compressibility": 4.5e-10,
        "load": 1.0e4,
        "transient": {"dt": 0.01, "n_steps": 100},
    },
    "phase1": {
        "n_samples": 200,
        "val_fraction": 0.2,
        "output_dir": "data",
    },
    # ---------------------------------------------------------------------------
    # Phase 2 (surrogate): data generation + model + evaluation
    # ---------------------------------------------------------------------------
    "phase2": {
        # --- Dataset generation (reduced-space sampling) ---
        "n_training_samples": 200,
        "collocation_n_points": 20,  # canonical path (was: top-level collocation_phase2)
        # reduced_fields defines the reduced-space coefficient dimension;
        # used by both data generation and the surrogate model.
        "reduced_fields": {
            "E": {
                "n_terms": 3, "basis": "dct", "seed": 142,
                "nu_sampling": False, "nu_ref": 1.5,
                "length_scale_sampling": False, "length_scale_ref": 0.3,
                "mean": 10.0e6, "range": [5.0e6, 20.0e6], "fluctuation_std": 1.0,
                "mean_sampling": True, "mean_range": [5.0e6, 20.0e6],
            },
            "k_h": {
                "n_terms": 2, "basis": "dct", "seed": 143,
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
            "test_fraction": 0.2,
            "n_plot_samples": 5,
        },
    },
    # ---------------------------------------------------------------------------
    # Phase 3 (reducer): data generation + model + evaluation
    # ---------------------------------------------------------------------------
    "phase3": {
        # --- Dataset generation (full-space sampling) ---
        "n_training_samples": 500,
        "collocation_n_points": 5,  # canonical path (was: top-level collocation_phase3)
        # full_fields defines the full-space coefficient dimension for reducer input
        "full_fields": {
            "E": {
                "n_terms": 5, "basis": "dct", "seed": 42,
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
                "n_terms": 2, "basis": "dct", "seed": 44,
                "nu_sampling": True, "nu_range": [0.5, 2.5], "nu_ref": 1.5,
                "length_scale_sampling": True, "length_scale_range": [0.1, 0.5],
                "length_scale_ref": 0.3,
                "mean": 1.0e-12, "range": [1.0e-13, 1.0e-10], "fluctuation_std": 0.5,
                "mean_sampling": True, "mean_range": [1.0e-13, 1.0e-10],
            },
        },
        # reduced_fields MUST match phase2.reduced_fields
        "reduced_fields": {
            "E": {
                "n_terms": 3, "basis": "dct", "seed": 142,
                "nu_sampling": False, "nu_ref": 1.5,
                "length_scale_sampling": False, "length_scale_ref": 0.3,
                "mean": 10.0e6, "range": [5.0e6, 20.0e6], "fluctuation_std": 1.0,
                "mean_sampling": True, "mean_range": [5.0e6, 20.0e6],
            },
            "k_h": {
                "n_terms": 2, "basis": "dct", "seed": 143,
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
            "test_fraction": 0.2,
            "n_plot_samples": 5,
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

    # ------------------------------------------------------------------
    # New-format translation + deprecated-key handling
    # ------------------------------------------------------------------

    @staticmethod
    def _translate_new_format(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Translate ``data_generation``/``models``/``evaluation`` format to
        the internal ``phase2``/``phase3`` representation.

        When the config uses the new structured layout, this method maps each
        section to the canonical internal key paths so that all downstream code
        continues to work without changes.  Both formats can coexist; new-format
        values take precedence over same-path old-format values.
        """
        dg = cfg.get("data_generation", {})
        mods = cfg.get("models", {})
        eval_ = cfg.get("evaluation", {})

        if not (dg or mods or eval_):
            return cfg  # purely old format – nothing to translate

        has_old_format = "phase2" in cfg or "phase3" in cfg
        if has_old_format:
            warnings.warn(
                "Config mixes new-format keys (data_generation / models / evaluation) "
                "with old-format keys (phase2 / phase3).  New-format sections take "
                "precedence for overlapping settings.",
                UserWarning,
                stacklevel=5,
            )

        result = copy.deepcopy(cfg)

        # ---- Surrogate (phase2) ----
        dg_surr = dg.get("surrogate", {})
        m_surr = mods.get("surrogate", {})
        eval_surr = eval_.get("surrogate", {})

        if dg_surr or m_surr or eval_surr:
            p2 = result.setdefault("phase2", {})

            # Data generation
            if "n_samples" in dg_surr:
                p2["n_training_samples"] = dg_surr["n_samples"]
            if "output_dir" in dg_surr:
                p2.setdefault("output_dir", dg_surr["output_dir"])
            if "fields" in dg_surr:
                p2["reduced_fields"] = copy.deepcopy(dg_surr["fields"])
            if "collocation_n_points" in dg_surr:
                p2["collocation_n_points"] = dg_surr["collocation_n_points"]

            # Model
            if "type" in m_surr:
                p2["surrogate_type"] = m_surr["type"]
            for k in ("output_repr", "training_signal", "hybrid_alpha",
                      "physics_check_interval", "n_output_modes",
                      "output_dir", "reduced_fields", "nn", "pce"):
                if k in m_surr:
                    p2[k] = copy.deepcopy(m_surr[k])

            # Evaluation
            if eval_surr:
                p2["evaluation"] = copy.deepcopy(eval_surr)

        # ---- Reducer (phase3) ----
        dg_red = dg.get("reducer", {})
        m_red = mods.get("reducer", {})
        eval_red = eval_.get("reducer", {})

        if dg_red or m_red or eval_red:
            p3 = result.setdefault("phase3", {})

            # Data generation
            if "n_samples" in dg_red:
                p3["n_training_samples"] = dg_red["n_samples"]
            if "output_dir" in dg_red:
                p3.setdefault("output_dir", dg_red["output_dir"])
            if "fields" in dg_red:
                p3["full_fields"] = copy.deepcopy(dg_red["fields"])
            if "collocation_n_points" in dg_red:
                p3["collocation_n_points"] = dg_red["collocation_n_points"]

            # Model
            if "type" in m_red:
                p3["reducer_type"] = m_red["type"]
            for k in ("training_signal", "output_dir", "surrogate_dir",
                      "load_phase2_model", "reduced_fields", "nn"):
                if k in m_red:
                    p3[k] = copy.deepcopy(m_red[k])

            # Inherit reduced_fields from surrogate if not specified in reducer
            if "reduced_fields" not in m_red and "surrogate" in mods:
                surr_rf = mods["surrogate"].get("reduced_fields")
                if surr_rf:
                    p3.setdefault("reduced_fields", copy.deepcopy(surr_rf))

            # Evaluation
            if eval_red:
                p3["evaluation"] = copy.deepcopy(eval_red)

        # Remove translated top-level keys to avoid confusion
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
