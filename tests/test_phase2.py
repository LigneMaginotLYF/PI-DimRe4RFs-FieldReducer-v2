"""
test_phase2.py
==============
Unit tests for Phase 2: surrogate training in REDUCED parameter space.
"""
import numpy as np
import pytest
from pathlib import Path


def _make_reduced_data(cfg):
    """Helper: generate data from the REDUCED parameter space."""
    from src.field_manager import FieldManager
    from src.forward_solver import BiotSolver
    reduced_fields_cfg = cfg["phase2"]["reduced_fields"]
    fm = FieldManager(cfg, fields_override=reduced_fields_cfg)
    solver = BiotSolver(cfg)
    X, fields, _ = fm.generate_dataset(12)
    Y = solver.run_batch(fields["E"], fields["k_h"], fields["k_v"])
    return X, Y, fm


class TestPhase2Surrogate:
    def test_run_saves_artifacts(self, tiny_cfg):
        from src.config_manager import ConfigManager
        from src.phase2_surrogate import Phase2Surrogate
        import copy

        cfg = copy.deepcopy(tiny_cfg)
        cfg["phase2"]["output_dir"] = "/tmp/test_p2_artifacts"

        cm = ConfigManager(overrides=cfg)
        p2 = Phase2Surrogate(cm)
        p2.run()  # generates own reduced-space data

        out_dir = Path(cfg["phase2"]["output_dir"])
        assert (out_dir / "config.json").exists()
        assert (out_dir / "phase2_X_test.npy").exists()
        assert (out_dir / "phase2_Y_test.npy").exists()

    def test_dimension_stamped_filename(self, tiny_cfg):
        """Phase 2 surrogate must be saved with dimension-stamped filename."""
        from src.config_manager import ConfigManager
        from src.phase2_surrogate import Phase2Surrogate
        from src.field_manager import FieldManager
        import copy

        cfg = copy.deepcopy(tiny_cfg)
        cfg["phase2"]["output_dir"] = "/tmp/test_p2_dimstamp"
        cfg["phase2"]["surrogate_type"] = "nn"

        cm = ConfigManager(overrides=cfg)
        p2 = Phase2Surrogate(cm)
        p2.run()

        out_dir = Path(cfg["phase2"]["output_dir"])
        # Reduced dim: E:2, kh:scalar, kv:scalar → 4
        fm_red = FieldManager(cfg, fields_override=cfg["phase2"]["reduced_fields"])
        d_red = fm_red.total_input_dim
        expected_file = out_dir / f"surrogate_nn_dim{d_red}.pt"
        assert expected_file.exists(), (
            f"Dimension-stamped surrogate file not found: {expected_file}. "
            f"Files in dir: {list(out_dir.iterdir())}"
        )

    def test_surrogate_predict_shape(self, tiny_cfg):
        from src.config_manager import ConfigManager
        from src.phase2_surrogate import Phase2Surrogate
        from src.field_manager import FieldManager
        import copy

        cfg = copy.deepcopy(tiny_cfg)
        cfg["phase2"]["output_dir"] = "/tmp/test_p2_pred_shape"

        cm = ConfigManager(overrides=cfg)
        p2 = Phase2Surrogate(cm)
        surrogate = p2.run()

        # Predict from reduced-space samples
        fm_red = p2.field_manager
        X_red, _, _ = fm_red.generate_dataset(3)
        pred = surrogate.predict(X_red)
        assert pred.shape == (3, cfg["grid"]["n_nodes_x"])

    def test_load_surrogate(self, tiny_cfg):
        from src.config_manager import ConfigManager
        from src.phase2_surrogate import Phase2Surrogate
        from src.field_manager import FieldManager
        import copy

        cfg = copy.deepcopy(tiny_cfg)
        cfg["phase2"]["output_dir"] = "/tmp/test_p2_load"

        cm = ConfigManager(overrides=cfg)
        p2 = Phase2Surrogate(cm)
        p2.run()

        loaded = p2.load_surrogate()
        assert loaded is not None

        fm_red = p2.field_manager
        X_red, _, _ = fm_red.generate_dataset(2)
        pred = loaded.predict(X_red)
        assert pred.shape == (2, cfg["grid"]["n_nodes_x"])

    def test_pce_surrogate(self, tiny_cfg):
        from src.config_manager import ConfigManager
        from src.phase2_surrogate import Phase2Surrogate
        from src.field_manager import FieldManager
        import copy

        cfg = copy.deepcopy(tiny_cfg)
        cfg["phase2"] = dict(tiny_cfg["phase2"])
        cfg["phase2"]["surrogate_type"] = "pce"
        cfg["phase2"]["output_dir"] = "/tmp/test_p2_pce"

        cm = ConfigManager(overrides=cfg)
        p2 = Phase2Surrogate(cm)
        surrogate = p2.run()

        fm_red = p2.field_manager
        X_red, _, _ = fm_red.generate_dataset(3)
        pred = surrogate.predict(X_red)
        assert pred.shape == (3, cfg["grid"]["n_nodes_x"])

    def test_pce_dimension_stamped_filename(self, tiny_cfg):
        """PCE surrogate must also be saved with dimension-stamped filename."""
        from src.config_manager import ConfigManager
        from src.phase2_surrogate import Phase2Surrogate
        from src.field_manager import FieldManager
        import copy

        cfg = copy.deepcopy(tiny_cfg)
        cfg["phase2"]["surrogate_type"] = "pce"
        cfg["phase2"]["output_dir"] = "/tmp/test_p2_pce_dimstamp"

        cm = ConfigManager(overrides=cfg)
        p2 = Phase2Surrogate(cm)
        p2.run()

        out_dir = Path(cfg["phase2"]["output_dir"])
        fm_red = FieldManager(cfg, fields_override=cfg["phase2"]["reduced_fields"])
        d_red = fm_red.total_input_dim
        expected_file = out_dir / f"surrogate_pce_dim{d_red}.pkl"
        assert expected_file.exists(), (
            f"Dimension-stamped PCE surrogate file not found: {expected_file}. "
            f"Files in dir: {list(out_dir.iterdir())}"
        )

    def test_reduced_dim_is_smaller_than_full_dim(self, tiny_cfg):
        """Phase 2 must operate on REDUCED dim, not full dim."""
        from src.config_manager import ConfigManager
        from src.phase2_surrogate import Phase2Surrogate
        from src.field_manager import FieldManager
        import copy

        cfg = copy.deepcopy(tiny_cfg)
        cfg["phase2"]["output_dir"] = "/tmp/test_p2_dim_check"

        cm = ConfigManager(overrides=cfg)
        p2 = Phase2Surrogate(cm)

        # Full dim from random_fields
        fm_full = FieldManager(cfg)
        full_dim = fm_full.total_input_dim

        # Reduced dim from reduced_fields
        reduced_dim = p2.reduced_dim

        assert reduced_dim < full_dim, (
            f"Reduced dim ({reduced_dim}) must be < full dim ({full_dim})"
        )
