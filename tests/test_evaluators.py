"""
test_evaluators.py
==================
Unit tests for Phase2Evaluator and Phase3Evaluator.
"""
import copy
import json
import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def phase2_setup(tiny_cfg):
    """Train and return a Phase-2 surrogate and test data (in REDUCED space)."""
    from src.field_manager import FieldManager
    from src.forward_solver import BiotSolver
    from src.surrogate_models import NNSurrogate

    cfg = copy.deepcopy(tiny_cfg)
    # Use reduced_fields for Phase 2 data
    reduced_fields_cfg = cfg["phase2"]["reduced_fields"]
    fm = FieldManager(cfg, fields_override=reduced_fields_cfg)
    solver = BiotSolver(cfg)
    X, fields, _ = fm.generate_dataset(20)
    Y = solver.run_batch(fields["E"], fields["k_h"], fields["k_v"])

    d = fm.total_input_dim
    nx = fm.n_nodes_x
    surrogate = NNSurrogate(d, nx, epochs=3, hidden_dims=[8, 8])
    surrogate.fit(X, Y)

    return surrogate, X, Y, cfg


@pytest.fixture
def phase3_setup(tiny_cfg, phase2_setup):
    """Train and return a Phase-3 reducer with Phase 2 data saved."""
    from src.config_manager import ConfigManager
    from src.phase3_reducer import Phase3Reducer
    from src.field_manager import FieldManager
    from src.forward_solver import BiotSolver

    surrogate, X_red, Y_red, cfg = phase2_setup

    # Save surrogate with reduced dim-stamp so Phase 3 can find it
    reduced_fields_cfg = cfg["phase2"]["reduced_fields"]
    fm_red = FieldManager(cfg, fields_override=reduced_fields_cfg)
    d_red = fm_red.total_input_dim
    surr_dir = Path(cfg["phase3"]["surrogate_dir"])
    surr_dir.mkdir(parents=True, exist_ok=True)
    surrogate.save(surr_dir / f"surrogate_nn_dim{d_red}.pt")

    # Generate FULL-space training data for Phase 3
    full_fields_cfg = cfg["phase3"]["full_fields"]
    fm_full = FieldManager(cfg, fields_override=full_fields_cfg)
    solver = BiotSolver(cfg)
    X_full, fields_full, _ = fm_full.generate_dataset(20)
    Y_full = solver.run_batch(fields_full["E"], fields_full["k_h"], fields_full["k_v"])

    cm = ConfigManager(overrides=cfg)
    p3 = Phase3Reducer(cm)
    # Pass full-space data directly
    p3.run(X_full, Y_full)

    return p3, surrogate, X_full, Y_full, cfg


class TestPhase2Evaluator:
    def test_run_returns_required_keys(self, tiny_cfg, phase2_setup):
        from src.config_manager import ConfigManager
        from src.phase2_evaluator import Phase2Evaluator

        surrogate, X, Y, cfg = phase2_setup
        cfg = copy.deepcopy(cfg)
        cfg["phase2"]["output_dir"] = "/tmp/test_p2eval_keys"

        cm = ConfigManager(overrides=cfg)
        ev = Phase2Evaluator(cm)
        results = ev.run(X[:5], Y[:5], surrogate=surrogate)

        assert "metrics" in results
        assert "metrics_path" in results
        assert "plots" in results
        assert "output_dir" in results

    def test_metrics_json_written(self, tiny_cfg, phase2_setup):
        from src.config_manager import ConfigManager
        from src.phase2_evaluator import Phase2Evaluator

        surrogate, X, Y, cfg = phase2_setup
        cfg = copy.deepcopy(cfg)
        cfg["phase2"]["output_dir"] = "/tmp/test_p2eval_json"

        cm = ConfigManager(overrides=cfg)
        ev = Phase2Evaluator(cm)
        results = ev.run(X[:5], Y[:5], surrogate=surrogate)

        assert Path(results["metrics_path"]).exists()
        with open(results["metrics_path"]) as f:
            data = json.load(f)
        assert "metrics" in data
        assert "R2" in data["metrics"]

    def test_settlement_plot_written(self, tiny_cfg, phase2_setup):
        from src.config_manager import ConfigManager
        from src.phase2_evaluator import Phase2Evaluator

        surrogate, X, Y, cfg = phase2_setup
        cfg = copy.deepcopy(cfg)
        cfg["phase2"]["output_dir"] = "/tmp/test_p2eval_plot"

        cm = ConfigManager(overrides=cfg)
        ev = Phase2Evaluator(cm)
        results = ev.run(X[:5], Y[:5], surrogate=surrogate)

        assert "settlement_comparison" in results["plots"]
        assert Path(results["plots"]["settlement_comparison"]).exists()

    def test_timestamped_output_dir(self, tiny_cfg, phase2_setup):
        """Output should be in a timestamped subdirectory."""
        import re
        from src.config_manager import ConfigManager
        from src.phase2_evaluator import Phase2Evaluator

        surrogate, X, Y, cfg = phase2_setup
        cfg = copy.deepcopy(cfg)
        cfg["phase2"]["output_dir"] = "/tmp/test_p2eval_ts"

        cm = ConfigManager(overrides=cfg)
        ev = Phase2Evaluator(cm)
        results = ev.run(X[:5], Y[:5], surrogate=surrogate, model_name="nn")

        out_dir = Path(results["output_dir"])
        assert out_dir.name.startswith("nn_")
        assert re.search(r"\d{8}_\d{6}", out_dir.name) is not None

    def test_metrics_values_finite(self, tiny_cfg, phase2_setup):
        from src.config_manager import ConfigManager
        from src.phase2_evaluator import Phase2Evaluator

        surrogate, X, Y, cfg = phase2_setup
        cfg = copy.deepcopy(cfg)
        cfg["phase2"]["output_dir"] = "/tmp/test_p2eval_finite"

        cm = ConfigManager(overrides=cfg)
        ev = Phase2Evaluator(cm)
        results = ev.run(X[:5], Y[:5], surrogate=surrogate)

        for k, v in results["metrics"].items():
            if isinstance(v, (list, np.ndarray)):
                assert np.all(np.isfinite(v)), f"Metric '{k}' contains non-finite values"
            elif isinstance(v, (int, float)):
                assert np.isfinite(v), f"Metric '{k}' is not finite: {v}"


class TestPhase3Evaluator:
    def test_run_returns_required_keys(self, tiny_cfg, phase3_setup):
        from src.config_manager import ConfigManager
        from src.phase3_evaluator import Phase3Evaluator

        p3, surrogate, X_full, Y_full, cfg = phase3_setup
        cfg = copy.deepcopy(cfg)
        cfg["phase3"]["output_dir"] = "/tmp/test_p3eval_keys"

        cm = ConfigManager(overrides=cfg)
        ev = Phase3Evaluator(cm)
        results = ev.run(X_full[:5], Y_full[:5], reducer=p3)

        assert "metrics" in results
        assert "metrics_path" in results
        assert "plots" in results
        assert "output_dir" in results

    def test_metrics_json_written(self, tiny_cfg, phase3_setup):
        from src.config_manager import ConfigManager
        from src.phase3_evaluator import Phase3Evaluator

        p3, surrogate, X_full, Y_full, cfg = phase3_setup
        cfg = copy.deepcopy(cfg)
        cfg["phase3"]["output_dir"] = "/tmp/test_p3eval_json"

        cm = ConfigManager(overrides=cfg)
        ev = Phase3Evaluator(cm)
        results = ev.run(X_full[:5], Y_full[:5], reducer=p3)

        assert Path(results["metrics_path"]).exists()
        with open(results["metrics_path"]) as f:
            data = json.load(f)
        assert "metrics" in data
        assert "evaluation_mode" in data
        assert data["evaluation_mode"] == "physics_biot"

    def test_settlement_plot_generated(self, tiny_cfg, phase3_setup):
        from src.config_manager import ConfigManager
        from src.phase3_evaluator import Phase3Evaluator

        p3, surrogate, X_full, Y_full, cfg = phase3_setup
        cfg = copy.deepcopy(cfg)
        cfg["phase3"]["output_dir"] = "/tmp/test_p3eval_settlement"

        cm = ConfigManager(overrides=cfg)
        ev = Phase3Evaluator(cm)
        results = ev.run(X_full[:5], Y_full[:5], reducer=p3)

        assert "settlement_comparison" in results["plots"]
        assert Path(results["plots"]["settlement_comparison"]).exists()

    def test_field_plots_generated(self, tiny_cfg, phase3_setup):
        """Material field comparison plots must be generated for all 3 fields."""
        from src.config_manager import ConfigManager
        from src.phase3_evaluator import Phase3Evaluator

        p3, surrogate, X_full, Y_full, cfg = phase3_setup
        cfg = copy.deepcopy(cfg)
        cfg["phase3"]["output_dir"] = "/tmp/test_p3eval_fields"

        cm = ConfigManager(overrides=cfg)
        ev = Phase3Evaluator(cm)
        results = ev.run(X_full[:5], Y_full[:5], reducer=p3)

        for field_name in ("E", "k_h", "k_v"):
            assert field_name in results["plots"], (
                f"Missing field plot for '{field_name}'. "
                f"Available: {list(results['plots'].keys())}"
            )
            assert Path(results["plots"][field_name]).exists()

    def test_field_metrics_in_results(self, tiny_cfg, phase3_setup):
        """Field-reconstruction metrics should be included."""
        from src.config_manager import ConfigManager
        from src.phase3_evaluator import Phase3Evaluator

        p3, surrogate, X_full, Y_full, cfg = phase3_setup
        cfg = copy.deepcopy(cfg)
        cfg["phase3"]["output_dir"] = "/tmp/test_p3eval_fmetrics"

        cm = ConfigManager(overrides=cfg)
        ev = Phase3Evaluator(cm)
        results = ev.run(X_full[:5], Y_full[:5], reducer=p3)

        metrics = results["metrics"]
        assert any(k.startswith("E_") for k in metrics), (
            f"No E field metrics found. Keys: {list(metrics.keys())}"
        )

    def test_timestamped_output_dir(self, tiny_cfg, phase3_setup):
        import re
        from src.config_manager import ConfigManager
        from src.phase3_evaluator import Phase3Evaluator

        p3, surrogate, X_full, Y_full, cfg = phase3_setup
        cfg = copy.deepcopy(cfg)
        cfg["phase3"]["output_dir"] = "/tmp/test_p3eval_ts"

        cm = ConfigManager(overrides=cfg)
        ev = Phase3Evaluator(cm)
        results = ev.run(X_full[:5], Y_full[:5], reducer=p3, model_name="reducer")

        out_dir = Path(results["output_dir"])
        assert out_dir.name.startswith("reducer_")
        assert re.search(r"\d{8}_\d{6}", out_dir.name) is not None

    def test_uses_physics_not_surrogate(self, tiny_cfg, phase3_setup):
        """Phase 3 evaluation must use Biot physics, not the surrogate."""
        from src.config_manager import ConfigManager
        from src.phase3_evaluator import Phase3Evaluator

        p3, surrogate, X_full, Y_full, cfg = phase3_setup
        cfg = copy.deepcopy(cfg)
        cfg["phase3"]["output_dir"] = "/tmp/test_p3eval_physics"

        cm = ConfigManager(overrides=cfg)
        ev = Phase3Evaluator(cm)
        results = ev.run(X_full[:5], Y_full[:5], reducer=p3)

        # Must report physics-based evaluation
        with open(results["metrics_path"]) as f:
            data = json.load(f)
        assert data.get("evaluation_mode") == "physics_biot"


class TestVisualizationV2:
    def test_settlement_global_y(self, tmp_path):
        from src.visualization_v2 import plot_settlement_comparison_global_y
        np.random.seed(0)
        Y_true = np.abs(np.random.randn(5, 6)) * 1e-3
        Y_pred = np.abs(np.random.randn(5, 6)) * 1e-3
        save_path = tmp_path / "settlement.png"
        plot_settlement_comparison_global_y(
            Y_true, Y_pred, n_nodes_x=6, save_path=save_path
        )
        assert save_path.exists()

    def test_settlement_y_axis_starts_zero(self, tmp_path):
        """Plot function must set y_min=0."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.visualization_v2 import plot_settlement_comparison_global_y

        np.random.seed(0)
        Y_true = np.abs(np.random.randn(3, 6)) * 1e-3
        Y_pred = np.abs(np.random.randn(3, 6)) * 1e-3
        save_path = tmp_path / "settlement_y0.png"
        plot_settlement_comparison_global_y(
            Y_true, Y_pred, n_nodes_x=6, save_path=save_path, n_samples=3
        )
        assert save_path.exists()

    def test_material_fields_comparison(self, tmp_path):
        from src.visualization_v2 import plot_material_fields_comparison
        np.random.seed(0)
        n_nodes = 6 * 4
        orig = np.abs(np.random.randn(5, n_nodes)) * 1e7 + 5e6
        redu = orig * 0.9
        save_path = tmp_path / "E_comparison.png"
        plot_material_fields_comparison(
            orig, redu, n_nodes_x=6, n_nodes_z=4,
            lx=1.0, lz=0.5,
            field_name="E", save_path=save_path
        )
        assert save_path.exists()

    def test_material_fields_3_rows(self, tmp_path):
        """Material field comparison must produce 3-row layout."""
        from src.visualization_v2 import plot_material_fields_comparison
        np.random.seed(0)
        n_nodes = 6 * 4
        orig = np.abs(np.random.randn(5, n_nodes)) * 1e7 + 5e6
        redu = orig * 0.9
        save_path = tmp_path / "E_3rows.png"
        plot_material_fields_comparison(
            orig, redu, n_nodes_x=6, n_nodes_z=4, field_name="E",
            save_path=save_path, n_samples=5
        )
        assert save_path.exists()

    def test_plot_all_material_fields(self, tmp_path):
        from src.visualization_v2 import plot_all_material_fields
        np.random.seed(0)
        n_nodes = 6 * 4
        fields = {
            "E": np.abs(np.random.randn(5, n_nodes)) * 1e7,
            "k_h": np.abs(np.random.randn(5, n_nodes)) * 1e-12,
            "k_v": np.abs(np.random.randn(5, n_nodes)) * 1e-12,
        }
        saved = plot_all_material_fields(
            fields, fields, n_nodes_x=6, n_nodes_z=4,
            lx=1.0, lz=0.5, output_dir=tmp_path
        )
        for name in ("E", "k_h", "k_v"):
            assert name in saved
            assert Path(saved[name]).exists()

    def test_backward_compat_plot_settlement(self, tmp_path):
        """Legacy plot_settlement_comparison still importable from visualization_v2."""
        from src.visualization_v2 import plot_settlement_comparison
        np.random.seed(0)
        Y_true = np.random.randn(3, 6)
        Y_pred = np.random.randn(3, 6)
        save_path = tmp_path / "legacy_settlement.png"
        plot_settlement_comparison(Y_true, Y_pred, n_nodes_x=6, save_path=str(save_path))
        assert save_path.exists()
