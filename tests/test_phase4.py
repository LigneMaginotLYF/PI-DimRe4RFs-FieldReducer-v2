"""
test_phase4.py
==============
Unit tests for Phase 4: validation and visualisation.
"""
import copy
import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def phase4_setup(tiny_cfg):
    """Set up trained Phase 2 surrogate and Phase 3 reducer for Phase 4 tests."""
    from src.field_manager import FieldManager
    from src.forward_solver import BiotSolver
    from src.surrogate_models import NNSurrogate
    from src.config_manager import ConfigManager
    from src.phase3_reducer import Phase3Reducer

    fm = FieldManager(tiny_cfg)
    solver = BiotSolver(tiny_cfg)
    X, fields, _ = fm.generate_dataset(20)
    Y = solver.run_batch(fields["E"], fields["k_h"], fields["k_v"])

    d = fm.total_input_dim
    nx = fm.n_nodes_x

    # Train and save Phase 2 surrogate with dimension-stamped name
    surrogate = NNSurrogate(d, nx, epochs=3, hidden_dims=[8, 8])
    surrogate.fit(X, Y)
    surr_dir = Path(tiny_cfg["phase3"]["surrogate_dir"])
    surr_dir.mkdir(parents=True, exist_ok=True)
    surrogate.save(surr_dir / f"surrogate_nn_dim{d}.pt")

    # Train Phase 3 reducer
    cm = ConfigManager(overrides=tiny_cfg)
    p3 = Phase3Reducer(cm)
    p3.run(X, Y)

    return X, Y, fm, solver


class TestPhase4Validator:
    def test_run_returns_metrics(self, tiny_cfg, phase4_setup):
        """Phase 4 run() returns a dict containing metrics."""
        from src.config_manager import ConfigManager
        from src.phase4_validation import Phase4Validator

        X, Y, _, _ = phase4_setup
        cm = ConfigManager(overrides=tiny_cfg)
        p4 = Phase4Validator(cm)
        results = p4.run(X[:5], Y[:5])

        assert "metrics" in results
        metrics = results["metrics"]
        assert "R2" in metrics
        assert "RMSE" in metrics
        assert "relative_L2" in metrics

    def test_metrics_json_saved(self, tiny_cfg, phase4_setup):
        """Phase 4 must save metrics.json to the output directory."""
        from src.config_manager import ConfigManager
        from src.phase4_validation import Phase4Validator

        X, Y, _, _ = phase4_setup
        cm = ConfigManager(overrides=tiny_cfg)
        p4 = Phase4Validator(cm)
        results = p4.run(X[:5], Y[:5])

        metrics_path = Path(tiny_cfg["phase4"]["output_dir"]) / "metrics.json"
        assert metrics_path.exists(), "metrics.json not found"
        assert "metrics_path" in results

    def test_plots_generated(self, tiny_cfg, phase4_setup):
        """Phase 4 must generate at least the settlement comparison plot."""
        from src.config_manager import ConfigManager
        from src.phase4_validation import Phase4Validator

        X, Y, _, _ = phase4_setup
        cm = ConfigManager(overrides=tiny_cfg)
        p4 = Phase4Validator(cm)
        results = p4.run(X[:5], Y[:5])

        plots_dir = Path(tiny_cfg["phase4"]["output_dir"]) / "plots"
        assert plots_dir.exists(), "plots directory not created"
        assert "settlement_comparison" in results.get("plots", {})
        settlement_plot = Path(results["plots"]["settlement_comparison"])
        assert settlement_plot.exists(), f"Settlement plot missing: {settlement_plot}"

    def test_metrics_values_finite(self, tiny_cfg, phase4_setup):
        """Metrics must be finite numbers."""
        from src.config_manager import ConfigManager
        from src.phase4_validation import Phase4Validator

        X, Y, _, _ = phase4_setup
        cm = ConfigManager(overrides=tiny_cfg)
        p4 = Phase4Validator(cm)
        results = p4.run(X[:5], Y[:5])

        for key, val in results["metrics"].items():
            assert np.isfinite(val), f"Metric '{key}' is not finite: {val}"

    def test_no_surrogate_fallback_to_solver(self, tiny_cfg):
        """Phase 4 should fall back to the solver when no surrogate is found."""
        from src.config_manager import ConfigManager
        from src.phase4_validation import Phase4Validator
        from src.field_manager import FieldManager
        from src.forward_solver import BiotSolver

        cfg = copy.deepcopy(tiny_cfg)
        cfg["phase3"]["surrogate_dir"] = "/tmp/test_p4_nosurr_dir"
        cfg["phase3"]["output_dir"] = "/tmp/test_p4_noreducer"
        cfg["phase4"]["output_dir"] = "/tmp/test_p4_nosurr_out"
        cfg["phase4"]["use_physics_for_plots"] = False

        # Ensure dirs are empty (no surrogate/reducer)
        Path(cfg["phase3"]["surrogate_dir"]).mkdir(parents=True, exist_ok=True)

        fm = FieldManager(cfg)
        solver = BiotSolver(cfg)
        X, fields, _ = fm.generate_dataset(6)
        Y = solver.run_batch(fields["E"], fields["k_h"], fields["k_v"])

        cm = ConfigManager(overrides=cfg)
        p4 = Phase4Validator(cm)
        # Should run without error, falling back to direct solver
        results = p4.run(X[:3], Y[:3])
        assert "metrics" in results

    def test_r2_bounded(self, tiny_cfg, phase4_setup):
        """R² should be ≤ 1 (can be negative if predictions are poor)."""
        from src.config_manager import ConfigManager
        from src.phase4_validation import Phase4Validator

        X, Y, _, _ = phase4_setup
        cm = ConfigManager(overrides=tiny_cfg)
        p4 = Phase4Validator(cm)
        results = p4.run(X[:5], Y[:5])

        assert results["metrics"]["R2"] <= 1.0

    def test_rmse_positive(self, tiny_cfg, phase4_setup):
        """RMSE must be non-negative."""
        from src.config_manager import ConfigManager
        from src.phase4_validation import Phase4Validator

        X, Y, _, _ = phase4_setup
        cm = ConfigManager(overrides=tiny_cfg)
        p4 = Phase4Validator(cm)
        results = p4.run(X[:5], Y[:5])

        assert results["metrics"]["RMSE"] >= 0.0

    def test_custom_output_dir(self, tiny_cfg, phase4_setup):
        """Phase 4 respects an overridden output directory."""
        from src.config_manager import ConfigManager
        from src.phase4_validation import Phase4Validator

        cfg = copy.deepcopy(tiny_cfg)
        cfg["phase4"]["output_dir"] = "/tmp/test_p4_custom_dir"

        X, Y, _, _ = phase4_setup
        cm = ConfigManager(overrides=cfg)
        p4 = Phase4Validator(cm, output_dir="/tmp/test_p4_custom_dir")
        results = p4.run(X[:5], Y[:5])

        assert Path("/tmp/test_p4_custom_dir/metrics.json").exists()
