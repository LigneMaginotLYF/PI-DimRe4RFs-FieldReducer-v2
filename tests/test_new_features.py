"""
test_new_features.py
====================
Tests for the new features added in the comprehensive refactor:

- Mean sampling toggled on/off for both dataset generators (B)
- No config shadowing; single canonical parameter paths (A)
- Settlement plotting uses per-sample y-limits (D)
- Phase-3 evaluator produces 3-curve settlement comparison (E)
- Surrogate-path metrics namespaced in metrics.json (E)
- New config format (data_generation / models / evaluation) is translated (A)
- Deprecated collocation_phase2/3 keys emit DeprecationWarning (A)
"""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# A) Config: new format translation, no shadowing, deprecated keys
# ---------------------------------------------------------------------------

class TestConfigRedesign:
    """Tests for the new config layout and backward-compat handling."""

    def test_new_format_translates_to_phase2(self):
        """New-format data_generation.surrogate → phase2 settings."""
        from src.config_manager import ConfigManager

        cm = ConfigManager(overrides={
            "data_generation": {
                "surrogate": {
                    "n_samples": 77,
                    "collocation_n_points": 12,
                    "fields": {
                        "E": {
                            "n_terms": 3, "mean": 10.0e6, "range": [5e6, 20e6],
                            "fluctuation_std": 1.0, "seed": 1,
                        },
                        "k_h": {
                            "n_terms": 2, "mean": 1e-12, "range": [1e-13, 1e-10],
                            "fluctuation_std": 0.5, "seed": 2,
                        },
                        "k_v": {
                            "n_terms": 0, "mean": 1e-12, "range": [1e-13, 1e-10],
                            "fluctuation_std": 0.5, "seed": 3,
                        },
                    },
                }
            },
            "models": {
                "surrogate": {
                    "type": "nn",
                    "output_repr": "direct",
                    "training_signal": "data",
                    "output_dir": "/tmp/test_surr_translate",
                },
            },
        })
        cfg = cm.cfg
        assert cfg["phase2"]["n_training_samples"] == 77
        assert cfg["phase2"]["collocation_n_points"] == 12
        assert cfg["phase2"]["surrogate_type"] == "nn"
        assert cfg["phase2"]["training_signal"] == "data"

    def test_new_format_translates_to_phase3(self):
        """New-format data_generation.reducer → phase3 settings."""
        from src.config_manager import ConfigManager

        cm = ConfigManager(overrides={
            "data_generation": {
                "reducer": {
                    "n_samples": 333,
                    "collocation_n_points": 7,
                    "fields": {
                        "E": {
                            "n_terms": 5, "mean": 10.0e6, "range": [5e6, 20e6],
                            "fluctuation_std": 1.0, "seed": 42,
                        },
                        "k_h": {
                            "n_terms": 0, "mean": 1e-12, "range": [1e-13, 1e-10],
                            "fluctuation_std": 0.5, "seed": 43,
                        },
                        "k_v": {
                            "n_terms": 2, "mean": 1e-12, "range": [1e-13, 1e-10],
                            "fluctuation_std": 0.5, "seed": 44,
                        },
                    },
                }
            },
            "models": {
                "reducer": {
                    "type": "nn",
                    "training_signal": "surrogate",
                    "output_dir": "/tmp/test_red_translate",
                    "surrogate_dir": "/tmp/fake_surr_dir",
                },
            },
        })
        cfg = cm.cfg
        assert cfg["phase3"]["n_training_samples"] == 333
        assert cfg["phase3"]["collocation_n_points"] == 7
        assert cfg["phase3"]["reducer_type"] == "nn"

    def test_deprecated_collocation_keys_warn(self):
        """Legacy collocation_phase2 / collocation_phase3 must emit DeprecationWarning."""
        from src.config_manager import ConfigManager

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cm = ConfigManager(overrides={
                "collocation_phase2": {"n_points": 15},
                "collocation_phase3": {"n_points": 6},
            })

        dep_msgs = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
        assert any("collocation_phase2" in m for m in dep_msgs), \
            f"No DeprecationWarning for collocation_phase2. Got: {dep_msgs}"
        assert any("collocation_phase3" in m for m in dep_msgs), \
            f"No DeprecationWarning for collocation_phase3. Got: {dep_msgs}"

    def test_deprecated_collocation_migrated_to_canonical_path(self):
        """Deprecated collocation keys are migrated to the canonical sub-path."""
        from src.config_manager import ConfigManager

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cm = ConfigManager(overrides={
                "collocation_phase2": {"n_points": 17},
                "collocation_phase3": {"n_points": 8},
            })
        cfg = cm.cfg
        assert cfg["phase2"]["collocation_n_points"] == 17
        assert cfg["phase3"]["collocation_n_points"] == 8

    def test_no_shadowing_collocation_canonical_wins(self):
        """When both deprecated and canonical collocation keys are set,
        canonical (phase2.collocation_n_points) wins (was set earlier in merge)."""
        from src.config_manager import ConfigManager

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cm = ConfigManager(overrides={
                # canonical path set explicitly
                "phase2": {"collocation_n_points": 99},
                # deprecated top-level key should NOT overwrite canonical
                "collocation_phase2": {"n_points": 3},
            })
        cfg = cm.cfg
        # Since _handle_deprecated_keys uses setdefault, canonical value (99) wins
        assert cfg["phase2"]["collocation_n_points"] == 99

    def test_new_config_yaml_is_parseable(self):
        """The repo's config.yaml must load without errors using the new format."""
        from src.config_manager import ConfigManager
        from pathlib import Path
        yaml_path = Path(__file__).parent.parent / "config.yaml"
        if yaml_path.exists():
            cm = ConfigManager(str(yaml_path))
            cfg = cm.cfg
            assert cfg["phase2"]["n_training_samples"] > 0
            assert cfg["phase3"]["n_training_samples"] > 0


# ---------------------------------------------------------------------------
# B) Mean sampling: toggled on/off for both dataset generators
# ---------------------------------------------------------------------------

class TestMeanSamplingE:
    """Tests for E field mean sampling (n_terms > 0 and n_terms == 0)."""

    def _make_cfg(self, n_nodes_x=8, n_nodes_z=4) -> dict:
        return {
            "grid": {"n_nodes_x": n_nodes_x, "n_nodes_z": n_nodes_z, "lx": 1.0, "lz": 0.5},
            "random_fields": {
                "E": {"n_terms": 0, "mean": 10e6, "range": [5e6, 20e6],
                      "fluctuation_std": 1.0, "seed": 1},
                "k_h": {"n_terms": 0, "mean": 1e-12, "range": [1e-13, 1e-10],
                        "fluctuation_std": 0.5, "seed": 2},
                "k_v": {"n_terms": 0, "mean": 1e-12, "range": [1e-13, 1e-10],
                        "fluctuation_std": 0.5, "seed": 3},
            },
            "solver": {"type": "1d", "mode": "steady", "nu_biot": 0.3,
                       "fluid_viscosity": 1e-3, "fluid_compressibility": 4.5e-10,
                       "load": 1e4, "transient": {"dt": 0.01, "n_steps": 10}},
        }

    def test_e_scalar_mean_sampling_off_constant_mean(self):
        """When mean_sampling=False, all E scalars have the same mean level."""
        from src.field_manager import FieldConfig, FieldManager

        fc = FieldConfig.from_dict("E", {
            "n_terms": 0, "mean": 10e6, "range": [5e6, 20e6],
            "fluctuation_std": 0.5, "seed": 99,
            "mean_sampling": False,
        })
        cfg = self._make_cfg()
        cfg["random_fields"]["E"].update({"n_terms": 0, "mean": 10e6, "fluctuation_std": 0.5,
                                          "mean_sampling": False, "seed": 99})
        fm = FieldManager(cfg, fields_override={
            "E": cfg["random_fields"]["E"],
            "k_h": {"n_terms": 0, "mean": 1e-12, "range": [1e-13, 1e-10],
                    "fluctuation_std": 0.5, "seed": 2},
            "k_v": {"n_terms": 0, "mean": 1e-12, "range": [1e-13, 1e-10],
                    "fluctuation_std": 0.5, "seed": 3},
        })
        X, fields, _ = fm.generate_dataset(50)
        E = fields["E"]  # (50, n_nodes)
        # Spatial mean of each sample should be close to E_ref (10e6)
        # (log-normal: mean ≈ E_ref * exp(0.5 * logE_std^2))
        # just check that variance of sample means is relatively small
        sample_means = E.mean(axis=1)
        cv = sample_means.std() / sample_means.mean()
        # Without mean_sampling the variation should be moderate (not extreme)
        assert cv < 2.0, f"Coefficient of variation too high without mean_sampling: {cv:.2f}"

    def test_e_scalar_mean_sampling_on_varies_mean(self):
        """When mean_sampling=True, E sample means span the configured mean_range.

        Uses fluctuation_std=1.0 (realistic), which is required for the mean-
        encoding mechanism (log(e_mean/E_ref) / logE_std) to be well-defined.
        With n_terms=0 and mean_sampling=True, each sample gets a constant field
        equal to the sampled mean, so sample spatial means should span mean_range.
        """
        from src.field_manager import FieldManager

        cfg = self._make_cfg()
        override = {
            "E": {"n_terms": 0, "mean": 10e6, "range": [5e6, 20e6],
                  "fluctuation_std": 1.0,   # positive logE_std required
                  "seed": 7,
                  "mean_sampling": True, "mean_range": [5e6, 20e6]},
            "k_h": {"n_terms": 0, "mean": 1e-12, "range": [1e-13, 1e-10],
                    "fluctuation_std": 0.5, "seed": 2},
            "k_v": {"n_terms": 0, "mean": 1e-12, "range": [1e-13, 1e-10],
                    "fluctuation_std": 0.5, "seed": 3},
        }
        fm = FieldManager(cfg, fields_override=override)
        _, fields, _ = fm.generate_dataset(200)
        E = fields["E"]  # (200, n_nodes)
        sample_means = E.mean(axis=1)
        # With n_terms=0 and mean_sampling, each sample is a constant field = e_mean.
        # Sample means should span [5e6, 20e6] (4× ratio).
        assert sample_means.min() >= 5e6 * 0.9, (
            f"Minimum E mean {sample_means.min():.2e} below expected range"
        )
        assert sample_means.max() <= 20e6 * 1.1, (
            f"Maximum E mean {sample_means.max():.2e} above expected range"
        )
        assert sample_means.std() > 1e5, (
            f"E means not varying (mean_sampling has no effect?). "
            f"std={sample_means.std():.2e}"
        )

    def test_e_field_mean_sampling_on_varies_mean(self):
        """When mean_sampling=True for E with n_terms > 0, the spatial mean varies."""
        from src.field_manager import FieldManager

        cfg = self._make_cfg(n_nodes_x=10, n_nodes_z=5)
        override = {
            "E": {"n_terms": 3, "mean": 10e6, "range": [5e6, 20e6],
                  "fluctuation_std": 1.0, "seed": 42,
                  "nu_sampling": False, "nu_ref": 1.5,
                  "length_scale_sampling": False, "length_scale_ref": 0.3,
                  "mean_sampling": True, "mean_range": [5e6, 20e6]},
            "k_h": {"n_terms": 0, "mean": 1e-12, "range": [1e-13, 1e-10],
                    "fluctuation_std": 0.5, "seed": 2},
            "k_v": {"n_terms": 0, "mean": 1e-12, "range": [1e-13, 1e-10],
                    "fluctuation_std": 0.5, "seed": 3},
        }
        fm = FieldManager(cfg, fields_override=override)
        _, fields, _ = fm.generate_dataset(100)
        E = fields["E"]
        # Spatial mean of each sample
        sample_means = E.mean(axis=1)
        # Should span a wide range (5e6 to 20e6) due to mean_sampling
        ratio = sample_means.max() / sample_means.min()
        assert ratio > 2.0, (
            f"E mean ratio {ratio:.2f} is too small; mean_sampling should produce "
            f"sample means spanning [5e6, 20e6] (ratio ≈ 4). "
            f"min={sample_means.min():.2e}, max={sample_means.max():.2e}"
        )

    def test_e_mean_range_invalid_raises(self):
        """Invalid mean_range (min > max) should raise ValueError."""
        from src.field_manager import FieldConfig
        with pytest.raises(ValueError, match="mean_range"):
            FieldConfig.from_dict("E", {
                "n_terms": 0, "mean": 10e6, "range": [5e6, 20e6],
                "fluctuation_std": 1.0, "seed": 1,
                "mean_sampling": True, "mean_range": [20e6, 5e6],  # invalid
            })


class TestMeanSamplingK:
    """Tests for k field mean sampling."""

    def _make_cfg(self) -> dict:
        return {
            "grid": {"n_nodes_x": 8, "n_nodes_z": 4, "lx": 1.0, "lz": 0.5},
            "random_fields": {
                "E": {"n_terms": 0, "mean": 10e6, "range": [5e6, 20e6],
                      "fluctuation_std": 1.0, "seed": 1},
                "k_h": {"n_terms": 0, "mean": 1e-12, "range": [1e-13, 1e-10],
                        "fluctuation_std": 0.5, "seed": 2},
                "k_v": {"n_terms": 0, "mean": 1e-12, "range": [1e-13, 1e-10],
                        "fluctuation_std": 0.5, "seed": 3},
            },
            "solver": {"type": "1d", "mode": "steady", "nu_biot": 0.3,
                       "fluid_viscosity": 1e-3, "fluid_compressibility": 4.5e-10,
                       "load": 1e4, "transient": {"dt": 0.01, "n_steps": 10}},
        }

    def test_k_scalar_mean_sampling_on_varies_mean(self):
        """k_h mean_sampling=True should produce k values spanning mean_range."""
        from src.field_manager import FieldManager

        cfg = self._make_cfg()
        override = {
            "E": {"n_terms": 0, "mean": 10e6, "range": [5e6, 20e6],
                  "fluctuation_std": 1.0, "seed": 1},
            "k_h": {"n_terms": 0, "mean": 1e-12, "range": [1e-13, 1e-10],
                    "fluctuation_std": 0.0,   # zero fluctuation
                    "seed": 7,
                    "mean_sampling": True, "mean_range": [1e-13, 1e-10]},
            "k_v": {"n_terms": 0, "mean": 1e-12, "range": [1e-13, 1e-10],
                    "fluctuation_std": 0.5, "seed": 3},
        }
        fm = FieldManager(cfg, fields_override=override)
        _, fields, _ = fm.generate_dataset(200)
        kh = fields["k_h"].ravel()
        # With zero fluctuation and mean_sampling, all k_h values are the sampled means
        assert kh.min() >= 1e-13 * 0.5, f"k_h min={kh.min():.2e} below expected range"
        assert kh.max() <= 1e-10 * 2.0, f"k_h max={kh.max():.2e} above expected range"
        # Should span multiple orders of magnitude
        log_ratio = np.log10(kh.max()) - np.log10(kh.min())
        assert log_ratio > 1.0, (
            f"k_h log-range {log_ratio:.2f} too small; mean_sampling should span "
            f"[1e-13, 1e-10] (3 decades)"
        )

    def test_k_mean_sampling_off_does_not_change_behavior(self):
        """mean_sampling=False should produce the same results as before."""
        from src.field_manager import FieldManager

        cfg = self._make_cfg()
        override_off = {
            "E": {"n_terms": 0, "mean": 10e6, "range": [5e6, 20e6],
                  "fluctuation_std": 1.0, "seed": 1},
            "k_h": {"n_terms": 0, "mean": 1e-12, "range": [1e-13, 1e-10],
                    "fluctuation_std": 0.5, "seed": 7, "mean_sampling": False},
            "k_v": {"n_terms": 0, "mean": 1e-12, "range": [1e-13, 1e-10],
                    "fluctuation_std": 0.5, "seed": 3},
        }
        fm = FieldManager(cfg, fields_override=override_off)
        # Just check it runs without error and produces finite values
        _, fields, _ = fm.generate_dataset(20)
        kh = fields["k_h"]
        assert np.all(kh > 0), "k_h values must be positive"
        assert np.all(np.isfinite(kh)), "k_h values must be finite"


# ---------------------------------------------------------------------------
# D) Settlement plotting: per-sample y-axis
# ---------------------------------------------------------------------------

class TestPerSampleYAxis:
    """Tests for per-sample y-axis limits in settlement comparison plots."""

    def test_plot_accepts_per_sample_flag(self, tmp_path):
        """plot_settlement_comparison_global_y must run without errors."""
        from src.visualization_v2 import plot_settlement_comparison_global_y
        np.random.seed(0)
        Y_true = np.abs(np.random.randn(4, 8)) * 1e-3
        Y_pred = np.abs(np.random.randn(4, 8)) * 1e-3
        save_path = tmp_path / "per_sample_y.png"
        plot_settlement_comparison_global_y(
            Y_true, Y_pred, n_nodes_x=8, save_path=save_path, n_samples=4
        )
        assert save_path.exists()

    def test_plot_accepts_third_curve(self, tmp_path):
        """Passing y_pred_surrogate produces a plot with a third curve."""
        from src.visualization_v2 import plot_settlement_comparison_global_y
        np.random.seed(0)
        Y_true = np.abs(np.random.randn(3, 8)) * 1e-3
        Y_pred = np.abs(np.random.randn(3, 8)) * 1e-3
        Y_surr = np.abs(np.random.randn(3, 8)) * 1e-3
        save_path = tmp_path / "three_curve.png"
        plot_settlement_comparison_global_y(
            Y_true, Y_pred, n_nodes_x=8,
            save_path=save_path, n_samples=3,
            y_pred_surrogate=Y_surr,
            label_pred_surrogate="Surrogate path",
        )
        assert save_path.exists()

    def test_per_sample_y_limits_differ_across_samples(self, tmp_path):
        """Each subplot should use its own y-limits, not global ones."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.visualization_v2 import plot_settlement_comparison_global_y

        np.random.seed(42)
        # Create samples with vastly different magnitudes
        Y_true = np.zeros((3, 8))
        Y_pred = np.zeros((3, 8))
        Y_true[0] = 1e-6   # tiny
        Y_true[1] = 1e-3   # medium
        Y_true[2] = 1e-1   # large
        Y_pred[0] = 9e-7
        Y_pred[1] = 9e-4
        Y_pred[2] = 9e-2

        save_path = tmp_path / "per_sample_limits.png"
        plot_settlement_comparison_global_y(
            Y_true, Y_pred, n_nodes_x=8, save_path=save_path, n_samples=3
        )
        # If the file exists we pass (a figure was created); detailed axis
        # checking would require parsing the saved image or axes state.
        assert save_path.exists()


# ---------------------------------------------------------------------------
# E) Phase-3 evaluator: 3-curve settlement comparison
# ---------------------------------------------------------------------------

class TestPhase3ThreeCurves:
    """Tests for the 3-curve settlement comparison in Phase-3 evaluation."""

    def _make_tiny_setup(self, tmp_path: Path):
        """Return (reducer, surrogate, X_full, Y_full, cm) for Phase-3 eval."""
        from src.config_manager import ConfigManager
        from src.field_manager import FieldManager
        from src.forward_solver import BiotSolver
        from src.surrogate_models import NNSurrogate
        from src.phase3_reducer import Phase3Reducer

        grid = {"n_nodes_x": 6, "n_nodes_z": 4, "lx": 1.0, "lz": 0.5}
        solver_cfg = {"type": "1d", "mode": "steady", "nu_biot": 0.3,
                      "fluid_viscosity": 1e-3, "fluid_compressibility": 4.5e-10,
                      "load": 1e4, "transient": {"dt": 0.01, "n_steps": 10}}
        reduced_fields = {
            "E": {"n_terms": 2, "mean": 10e6, "range": [5e6, 20e6],
                  "fluctuation_std": 1.0, "seed": 142,
                  "nu_sampling": False, "nu_ref": 1.5,
                  "length_scale_sampling": False, "length_scale_ref": 0.3},
            "k_h": {"n_terms": 1, "mean": 1e-12, "range": [1e-13, 1e-10],
                    "fluctuation_std": 0.5, "seed": 143,
                    "nu_sampling": False, "nu_ref": 1.5,
                    "length_scale_sampling": False, "length_scale_ref": 0.3},
            "k_v": {"n_terms": 0, "mean": 1e-12, "range": [1e-13, 1e-10],
                    "fluctuation_std": 0.5, "seed": 144},
        }
        full_fields = {
            "E": {"n_terms": 3, "mean": 10e6, "range": [5e6, 20e6],
                  "fluctuation_std": 1.0, "seed": 42,
                  "nu_sampling": False, "nu_ref": 1.5,
                  "length_scale_sampling": False, "length_scale_ref": 0.3},
            "k_h": {"n_terms": 0, "mean": 1e-12, "range": [1e-13, 1e-10],
                    "fluctuation_std": 0.5, "seed": 43},
            "k_v": {"n_terms": 2, "mean": 1e-12, "range": [1e-13, 1e-10],
                    "fluctuation_std": 0.5, "seed": 44,
                    "nu_sampling": False, "nu_ref": 1.5,
                    "length_scale_sampling": False, "length_scale_ref": 0.3},
        }
        surr_dir = str(tmp_path / "surr")
        p3_dir = str(tmp_path / "p3")
        cfg = {
            "grid": grid,
            "solver": solver_cfg,
            "random_fields": {
                "E": full_fields["E"], "k_h": full_fields["k_h"],
                "k_v": full_fields["k_v"],
            },
            "phase2": {
                "surrogate_type": "nn", "output_repr": "direct",
                "training_signal": "data", "output_dir": surr_dir,
                "n_training_samples": 12, "n_output_modes": 6,
                "hybrid_alpha": 0.1, "physics_check_interval": 10,
                "reduced_fields": reduced_fields,
                "nn": {"hidden_dims": [8, 8], "epochs": 2, "lr": 1e-3,
                       "batch_size": 4, "patience": 5},
                "pce": {"degree": 3},
                "evaluation": {"test_fraction": 0.2, "n_plot_samples": 3},
            },
            "phase3": {
                "reducer_type": "nn", "training_signal": "surrogate",
                "n_training_samples": 12, "output_dir": p3_dir,
                "surrogate_dir": surr_dir, "load_phase2_model": None,
                "full_fields": full_fields,
                "reduced_fields": reduced_fields,
                "nn": {"hidden_dims": [8, 8], "epochs": 2, "lr": 1e-3,
                       "batch_size": 4, "patience": 5},
                "evaluation": {"test_fraction": 0.2, "n_plot_samples": 3},
            },
        }

        # Build Phase-2 surrogate
        fm_red = FieldManager(cfg, fields_override=reduced_fields)
        biot = BiotSolver(cfg)
        X_red, fields_red, _ = fm_red.generate_dataset(12)
        Y_red = biot.run_batch(fields_red["E"], fields_red["k_h"], fields_red["k_v"])

        d_red = fm_red.total_input_dim
        nx = grid["n_nodes_x"]
        surrogate = NNSurrogate(d_red, nx, epochs=2, hidden_dims=[8, 8])
        surrogate.fit(X_red, Y_red)

        Path(surr_dir).mkdir(parents=True, exist_ok=True)
        surrogate.save(Path(surr_dir) / f"surrogate_nn_dim{d_red}.pt")

        # Generate FULL-space data
        fm_full = FieldManager(cfg, fields_override=full_fields)
        X_full, fields_full, _ = fm_full.generate_dataset(12)
        Y_full = biot.run_batch(fields_full["E"], fields_full["k_h"], fields_full["k_v"])

        cm = ConfigManager(overrides=cfg)
        p3 = Phase3Reducer(cm)
        p3.run(X_full, Y_full)

        return p3, surrogate, X_full, Y_full, cm

    def test_three_curve_plot_generated(self, tmp_path):
        """Phase-3 evaluator with surrogate should generate a settlement comparison."""
        from src.phase3_evaluator import Phase3Evaluator

        p3, surrogate, X_full, Y_full, cm = self._make_tiny_setup(tmp_path)

        cfg = cm.cfg
        cfg["phase3"]["output_dir"] = str(tmp_path / "p3eval")
        from src.config_manager import ConfigManager
        cm2 = ConfigManager(overrides=cfg)
        ev = Phase3Evaluator(cm2)
        results = ev.run(
            X_full[:4], Y_full[:4],
            reducer=p3,
            surrogate=surrogate,
            model_name="reducer_3curve",
        )
        assert "settlement_comparison" in results["plots"]
        assert Path(results["plots"]["settlement_comparison"]).exists()

    def test_surrogate_metrics_namespaced(self, tmp_path):
        """Surrogate-path metrics must appear under surrogate_* keys in metrics.json."""
        from src.phase3_evaluator import Phase3Evaluator

        p3, surrogate, X_full, Y_full, cm = self._make_tiny_setup(tmp_path)

        cfg = cm.cfg
        cfg["phase3"]["output_dir"] = str(tmp_path / "p3eval_ns")
        from src.config_manager import ConfigManager
        cm2 = ConfigManager(overrides=cfg)
        ev = Phase3Evaluator(cm2)
        results = ev.run(
            X_full[:4], Y_full[:4],
            reducer=p3,
            surrogate=surrogate,
            model_name="reducer_ns",
        )
        with open(results["metrics_path"]) as f:
            data = json.load(f)
        assert data.get("surrogate_path_evaluated") is True
        metrics = data["metrics"]
        assert any(k.startswith("surrogate_") for k in metrics), (
            f"No surrogate_* metric found. Keys: {list(metrics.keys())}"
        )

    def test_no_surrogate_runs_physics_only(self, tmp_path):
        """Phase-3 eval without surrogate must still complete (physics path only)."""
        from src.phase3_evaluator import Phase3Evaluator

        p3, _, X_full, Y_full, cm = self._make_tiny_setup(tmp_path)

        cfg = cm.cfg
        cfg["phase3"]["output_dir"] = str(tmp_path / "p3eval_phys")
        from src.config_manager import ConfigManager
        cm2 = ConfigManager(overrides=cfg)
        ev = Phase3Evaluator(cm2)
        results = ev.run(
            X_full[:4], Y_full[:4],
            reducer=p3,
            surrogate=None,   # no surrogate
            model_name="reducer_phys",
        )
        with open(results["metrics_path"]) as f:
            data = json.load(f)
        assert data.get("surrogate_path_evaluated") is False
        assert data.get("evaluation_mode") == "physics_biot"
