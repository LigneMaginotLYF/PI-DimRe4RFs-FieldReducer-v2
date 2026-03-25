"""
test_phase3.py
==============
Unit tests for Phase 3: dimension reducer (FULL → REDUCED).

Phase 3 maps full-dimensional parameters (from phase3.full_fields) to
reduced-dimensional parameters (from phase3.reduced_fields = phase2.reduced_fields).
"""
import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def saved_phase2_surrogate(tiny_cfg):
    """Train a Phase-2 surrogate in REDUCED space and save to surrogate_dir.

    Returns (surrogate, fm_reduced, surrogate_dir).
    """
    from src.field_manager import FieldManager
    from src.forward_solver import BiotSolver
    from src.surrogate_models import NNSurrogate

    # Phase 2 uses REDUCED fields
    reduced_fields_cfg = tiny_cfg["phase2"]["reduced_fields"]
    fm = FieldManager(tiny_cfg, fields_override=reduced_fields_cfg)
    solver = BiotSolver(tiny_cfg)
    X, fields, _ = fm.generate_dataset(15)
    Y = solver.run_batch(fields["E"], fields["k_h"], fields["k_v"])

    d = fm.total_input_dim  # 4 in tiny_cfg
    nx = fm.n_nodes_x
    surrogate = NNSurrogate(d, nx, epochs=3, hidden_dims=[8, 8])
    surrogate.fit(X, Y)

    # Save with dimension-stamped filename where Phase 3 expects it
    surr_dir = Path(tiny_cfg["phase3"]["surrogate_dir"])
    surr_dir.mkdir(parents=True, exist_ok=True)
    surrogate.save(surr_dir / f"surrogate_nn_dim{d}.pt")
    return surrogate, fm, surr_dir


@pytest.fixture
def full_space_data(tiny_cfg):
    """Generate data in FULL parameter space for Phase 3 training."""
    from src.field_manager import FieldManager
    from src.forward_solver import BiotSolver

    full_fields_cfg = tiny_cfg["phase3"]["full_fields"]
    fm = FieldManager(tiny_cfg, fields_override=full_fields_cfg)
    solver = BiotSolver(tiny_cfg)
    X_full, fields, _ = fm.generate_dataset(20)
    Y_full = solver.run_batch(fields["E"], fields["k_h"], fields["k_v"])
    return X_full, Y_full, fm


class TestPhase3Reducer:
    def test_run_saves_artifacts(self, tiny_cfg, saved_phase2_surrogate):
        from src.config_manager import ConfigManager
        from src.phase3_reducer import Phase3Reducer
        import copy

        cfg = copy.deepcopy(tiny_cfg)
        cfg["phase3"]["output_dir"] = "/tmp/test_p3_artifacts"
        cfg["phase3"]["surrogate_dir"] = str(saved_phase2_surrogate[2])

        # Generate FULL-space data
        from src.field_manager import FieldManager
        from src.forward_solver import BiotSolver
        fm_full = FieldManager(cfg, fields_override=cfg["phase3"]["full_fields"])
        solver = BiotSolver(cfg)
        X_full, fields, _ = fm_full.generate_dataset(12)
        Y_full = solver.run_batch(fields["E"], fields["k_h"], fields["k_v"])

        cm = ConfigManager(overrides=cfg)
        p3 = Phase3Reducer(cm)
        p3.run(X_full, Y_full)

        out_dir = Path(cfg["phase3"]["output_dir"])
        assert (out_dir / "reducer.pt").exists()
        assert (out_dir / "config.json").exists()
        assert (out_dir / "X_mean.npy").exists()

    def test_reduce_shape_maps_to_reduced_dim(self, tiny_cfg, saved_phase2_surrogate, full_space_data):
        """Reducer output must have shape (n_samples, reduced_dim), not full_dim."""
        from src.config_manager import ConfigManager
        from src.phase3_reducer import Phase3Reducer
        import copy

        cfg = copy.deepcopy(tiny_cfg)
        cfg["phase3"]["output_dir"] = "/tmp/test_p3_reduce_shape"
        cfg["phase3"]["surrogate_dir"] = str(saved_phase2_surrogate[2])

        X_full, Y_full, _ = full_space_data

        cm = ConfigManager(overrides=cfg)
        p3 = Phase3Reducer(cm)
        p3.run(X_full, Y_full)

        assert p3.full_dim == X_full.shape[1]    # 6 in tiny_cfg
        assert p3.reduced_dim < p3.full_dim       # 4 < 6

        X_red = p3.reduce(X_full[:5])
        # Shape must be (n_samples, reduced_dim)
        assert X_red.shape == (5, p3.reduced_dim)

    def test_reduce_single_sample(self, tiny_cfg, saved_phase2_surrogate, full_space_data):
        from src.config_manager import ConfigManager
        from src.phase3_reducer import Phase3Reducer
        import copy

        cfg = copy.deepcopy(tiny_cfg)
        cfg["phase3"]["output_dir"] = "/tmp/test_p3_single"
        cfg["phase3"]["surrogate_dir"] = str(saved_phase2_surrogate[2])

        X_full, Y_full, _ = full_space_data

        cm = ConfigManager(overrides=cfg)
        p3 = Phase3Reducer(cm)
        p3.run(X_full, Y_full)

        single = p3.reduce(X_full[0])
        assert single.shape == (p3.reduced_dim,)

    def test_load(self, tiny_cfg, saved_phase2_surrogate, full_space_data):
        from src.config_manager import ConfigManager
        from src.phase3_reducer import Phase3Reducer
        import copy

        cfg = copy.deepcopy(tiny_cfg)
        cfg["phase3"]["output_dir"] = "/tmp/test_p3_load"
        cfg["phase3"]["surrogate_dir"] = str(saved_phase2_surrogate[2])

        X_full, Y_full, _ = full_space_data

        cm = ConfigManager(overrides=cfg)

        p3_train = Phase3Reducer(cm)
        p3_train.run(X_full, Y_full)
        pred_train = p3_train.reduce(X_full[:3])

        p3_load = Phase3Reducer(cm)
        p3_load.load()
        pred_load = p3_load.reduce(X_full[:3])

        np.testing.assert_allclose(pred_train, pred_load, atol=1e-5)

    def test_dimension_mismatch_raises_value_error(self, tiny_cfg):
        """Phase 3 must raise ValueError when surrogate internal dim != expected dim."""
        import copy
        from src.config_manager import ConfigManager
        from src.phase3_reducer import Phase3Reducer
        from src.field_manager import FieldManager
        from src.forward_solver import BiotSolver
        from src.surrogate_models import NNSurrogate

        cfg = copy.deepcopy(tiny_cfg)
        cfg["phase3"]["output_dir"] = "/tmp/test_p3_dimcheck"
        cfg["phase3"]["surrogate_dir"] = "/tmp/test_p3_wrong_dim_surr"

        # The correct reduced dim is 4 (E:2, kh:scalar, kv:scalar)
        fm_red = FieldManager(cfg, fields_override=cfg["phase2"]["reduced_fields"])
        reduced_dim = fm_red.total_input_dim   # 4
        wrong_internal_dim = 2  # intentionally wrong

        nx = cfg["grid"]["n_nodes_x"]
        surrogate = NNSurrogate(wrong_internal_dim, nx, epochs=2, hidden_dims=[8, 8])
        surrogate.fit(
            np.random.randn(10, wrong_internal_dim),
            np.random.randn(10, nx),
        )

        surr_dir = Path(cfg["phase3"]["surrogate_dir"])
        surr_dir.mkdir(parents=True, exist_ok=True)
        # Filename says "dim{reduced_dim}" but internal model has wrong_internal_dim
        surrogate.save(surr_dir / f"surrogate_nn_dim{reduced_dim}.pt")

        # Generate FULL-space data
        fm_full = FieldManager(cfg, fields_override=cfg["phase3"]["full_fields"])
        solver = BiotSolver(cfg)
        X_full, fields, _ = fm_full.generate_dataset(8)
        Y_full = solver.run_batch(fields["E"], fields["k_h"], fields["k_v"])

        cm = ConfigManager(overrides=cfg)
        p3 = Phase3Reducer(cm)
        with pytest.raises(ValueError, match="dimension mismatch"):
            p3.run(X_full, Y_full)

    def test_missing_surrogate_raises_value_error(self, tiny_cfg):
        """Phase 3 must raise ValueError when no surrogate file exists."""
        import copy
        from src.config_manager import ConfigManager
        from src.phase3_reducer import Phase3Reducer
        from src.field_manager import FieldManager
        from src.forward_solver import BiotSolver

        cfg = copy.deepcopy(tiny_cfg)
        cfg["phase3"]["output_dir"] = "/tmp/test_p3_nosurr"
        cfg["phase3"]["surrogate_dir"] = "/tmp/test_p3_empty_surr_dir"

        empty_dir = Path(cfg["phase3"]["surrogate_dir"])
        empty_dir.mkdir(parents=True, exist_ok=True)

        fm_full = FieldManager(cfg, fields_override=cfg["phase3"]["full_fields"])
        solver = BiotSolver(cfg)
        X_full, fields, _ = fm_full.generate_dataset(8)
        Y_full = solver.run_batch(fields["E"], fields["k_h"], fields["k_v"])

        cm = ConfigManager(overrides=cfg)
        p3 = Phase3Reducer(cm)
        with pytest.raises(ValueError, match="No Phase-2 surrogate found"):
            p3.run(X_full, Y_full)

    def test_physics_training_signal(self, tiny_cfg):
        """Test that physics training signal runs without error."""
        import copy
        from src.config_manager import ConfigManager
        from src.phase3_reducer import Phase3Reducer
        from src.field_manager import FieldManager
        from src.forward_solver import BiotSolver

        cfg = copy.deepcopy(tiny_cfg)
        cfg["phase3"] = copy.deepcopy(tiny_cfg["phase3"])
        cfg["phase3"]["training_signal"] = "physics"
        cfg["phase3"]["output_dir"] = "/tmp/test_p3_phys"
        cfg["phase3"]["nn"]["epochs"] = 3

        fm_full = FieldManager(cfg, fields_override=cfg["phase3"]["full_fields"])
        solver = BiotSolver(cfg)
        X_full, fields, _ = fm_full.generate_dataset(10)
        Y_full = solver.run_batch(fields["E"], fields["k_h"], fields["k_v"])

        cm = ConfigManager(overrides=cfg)
        p3 = Phase3Reducer(cm)
        p3.run(X_full, Y_full)
        assert Path(cfg["phase3"]["output_dir"]).exists()

    def test_full_dim_and_reduced_dim_are_independent(self, tiny_cfg):
        """full_dim and reduced_dim must be different."""
        from src.config_manager import ConfigManager
        from src.phase3_reducer import Phase3Reducer

        cm = ConfigManager(overrides=tiny_cfg)
        p3 = Phase3Reducer(cm)
        assert p3.full_dim != p3.reduced_dim
