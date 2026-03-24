"""
test_phase2.py
==============
Unit tests for Phase 2: LUT generation and surrogate training.
"""
import numpy as np
import pytest
from pathlib import Path


class TestPhase2Surrogate:
    def test_run_saves_artifacts(self, tiny_cfg):
        from src.config_manager import ConfigManager
        from src.phase2_surrogate import Phase2Surrogate
        from src.field_manager import FieldManager
        from src.forward_solver import BiotSolver

        cm = ConfigManager(overrides=tiny_cfg)
        p2 = Phase2Surrogate(cm)

        # Build small training data
        fm = FieldManager(tiny_cfg)
        solver = BiotSolver(tiny_cfg)
        X, fields, _ = fm.generate_dataset(10)
        Y = solver.run_batch(fields["E"], fields["k_h"], fields["k_v"])

        p2.run(X, Y)

        out_dir = Path(tiny_cfg["phase2"]["output_dir"])
        assert (out_dir / "grid_points.npy").exists()
        assert (out_dir / "responses.npy").exists()
        assert (out_dir / "config.json").exists()

    def test_surrogate_predict_shape(self, tiny_cfg):
        from src.config_manager import ConfigManager
        from src.phase2_surrogate import Phase2Surrogate
        from src.field_manager import FieldManager
        from src.forward_solver import BiotSolver

        cm = ConfigManager(overrides=tiny_cfg)
        p2 = Phase2Surrogate(cm)

        fm = FieldManager(tiny_cfg)
        solver = BiotSolver(tiny_cfg)
        X, fields, _ = fm.generate_dataset(10)
        Y = solver.run_batch(fields["E"], fields["k_h"], fields["k_v"])

        surrogate = p2.run(X, Y)
        pred = surrogate.predict(X[:3])
        assert pred.shape == (3, tiny_cfg["grid"]["n_nodes_x"])

    def test_load_surrogate(self, tiny_cfg):
        from src.config_manager import ConfigManager
        from src.phase2_surrogate import Phase2Surrogate
        from src.field_manager import FieldManager
        from src.forward_solver import BiotSolver

        cm = ConfigManager(overrides=tiny_cfg)
        p2 = Phase2Surrogate(cm)

        fm = FieldManager(tiny_cfg)
        solver = BiotSolver(tiny_cfg)
        X, fields, _ = fm.generate_dataset(10)
        Y = solver.run_batch(fields["E"], fields["k_h"], fields["k_v"])

        p2.run(X, Y)
        loaded = p2.load_surrogate()
        assert loaded is not None

        pred = loaded.predict(X[:2])
        assert pred.shape == (2, 6)

    def test_pce_surrogate(self, tiny_cfg):
        from src.config_manager import ConfigManager
        from src.phase2_surrogate import Phase2Surrogate
        from src.field_manager import FieldManager
        from src.forward_solver import BiotSolver

        cfg = dict(tiny_cfg)
        cfg["phase2"] = dict(tiny_cfg["phase2"])
        cfg["phase2"]["surrogate_type"] = "pce"
        cfg["phase2"]["output_dir"] = "/tmp/test_p2_pce"

        cm = ConfigManager(overrides=cfg)
        p2 = Phase2Surrogate(cm)

        fm = FieldManager(cfg)
        solver = BiotSolver(cfg)
        X, fields, _ = fm.generate_dataset(10)
        Y = solver.run_batch(fields["E"], fields["k_h"], fields["k_v"])

        surrogate = p2.run(X, Y)
        pred = surrogate.predict(X[:3])
        assert pred.shape == (3, cfg["grid"]["n_nodes_x"])
