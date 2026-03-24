"""
test_phase3.py
==============
Unit tests for Phase 3: dimension reducer training.
"""
import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def trained_surrogate(tiny_cfg):
    """Return a fitted surrogate for use in Phase 3 tests.

    The surrogate is saved with a dimension-stamped filename so Phase 3
    can locate it via ``_find_surrogate_file``.
    """
    from src.field_manager import FieldManager
    from src.forward_solver import BiotSolver
    from src.surrogate_models import NNSurrogate

    fm = FieldManager(tiny_cfg)
    solver = BiotSolver(tiny_cfg)
    X, fields, _ = fm.generate_dataset(15)
    Y = solver.run_batch(fields["E"], fields["k_h"], fields["k_v"])

    d = fm.total_input_dim
    nx = fm.n_nodes_x
    surrogate = NNSurrogate(d, nx, epochs=3, hidden_dims=[8, 8])
    surrogate.fit(X, Y)

    # Save with dimension-stamped filename where Phase 3 expects it
    surr_dir = Path(tiny_cfg["phase3"]["surrogate_dir"])
    surr_dir.mkdir(parents=True, exist_ok=True)
    surrogate.save(surr_dir / f"surrogate_nn_dim{d}.pt")
    return surrogate, X, Y


class TestPhase3Reducer:
    def test_run_saves_artifacts(self, tiny_cfg, trained_surrogate):
        from src.config_manager import ConfigManager
        from src.phase3_reducer import Phase3Reducer

        cm = ConfigManager(overrides=tiny_cfg)
        _, X, Y = trained_surrogate

        p3 = Phase3Reducer(cm)
        p3.run(X, Y)

        out_dir = Path(tiny_cfg["phase3"]["output_dir"])
        assert (out_dir / "reducer.pt").exists()
        assert (out_dir / "config.json").exists()
        assert (out_dir / "X_mean.npy").exists()

    def test_reduce_shape(self, tiny_cfg, trained_surrogate):
        from src.config_manager import ConfigManager
        from src.phase3_reducer import Phase3Reducer

        cm = ConfigManager(overrides=tiny_cfg)
        _, X, Y = trained_surrogate

        p3 = Phase3Reducer(cm)
        p3.run(X, Y)

        X_red = p3.reduce(X[:5])
        assert X_red.shape == X[:5].shape  # same dimension as input

    def test_reduce_single_sample(self, tiny_cfg, trained_surrogate):
        from src.config_manager import ConfigManager
        from src.phase3_reducer import Phase3Reducer

        cm = ConfigManager(overrides=tiny_cfg)
        _, X, Y = trained_surrogate

        p3 = Phase3Reducer(cm)
        p3.run(X, Y)

        single = p3.reduce(X[0])
        assert single.shape == (6,)

    def test_load(self, tiny_cfg, trained_surrogate):
        from src.config_manager import ConfigManager
        from src.phase3_reducer import Phase3Reducer

        cm = ConfigManager(overrides=tiny_cfg)
        _, X, Y = trained_surrogate

        # Train and save
        p3_train = Phase3Reducer(cm)
        p3_train.run(X, Y)
        pred_train = p3_train.reduce(X[:3])

        # Load fresh
        p3_load = Phase3Reducer(cm)
        p3_load.load()
        pred_load = p3_load.reduce(X[:3])

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

        # Save a surrogate file whose filename matches the expected dimension (6)
        # but whose internal input_dim is WRONG (4) — this tests the internal
        # metadata validation inside _load_phase2_surrogate.
        correct_dim = 6   # expected by tiny_cfg (3+1+2)
        wrong_internal_dim = 4
        nx = cfg["grid"]["n_nodes_x"]
        surrogate = NNSurrogate(wrong_internal_dim, nx, epochs=2, hidden_dims=[8, 8])
        import numpy as _np
        surrogate.fit(
            _np.random.randn(10, wrong_internal_dim),
            _np.random.randn(10, nx),
        )

        surr_dir = Path(cfg["phase3"]["surrogate_dir"])
        surr_dir.mkdir(parents=True, exist_ok=True)
        # Filename says "dim6" but internal model has input_dim=4
        surrogate.save(surr_dir / f"surrogate_nn_dim{correct_dim}.pt")

        cm = ConfigManager(overrides=cfg)
        fm = FieldManager(cfg)
        solver = BiotSolver(cfg)
        X, fields, _ = fm.generate_dataset(8)
        Y = solver.run_batch(fields["E"], fields["k_h"], fields["k_v"])

        p3 = Phase3Reducer(cm)
        with pytest.raises(ValueError, match="dimension mismatch"):
            p3.run(X, Y)

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

        # Ensure surrogate dir exists but is empty
        empty_dir = Path(cfg["phase3"]["surrogate_dir"])
        empty_dir.mkdir(parents=True, exist_ok=True)

        cm = ConfigManager(overrides=cfg)
        fm = FieldManager(cfg)
        solver = BiotSolver(cfg)
        X, fields, _ = fm.generate_dataset(8)
        Y = solver.run_batch(fields["E"], fields["k_h"], fields["k_v"])

        p3 = Phase3Reducer(cm)
        with pytest.raises(ValueError, match="No Phase-2 surrogate found"):
            p3.run(X, Y)

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

        cm = ConfigManager(overrides=cfg)
        fm = FieldManager(cfg)
        solver = BiotSolver(cfg)
        X, fields, _ = fm.generate_dataset(10)
        Y = solver.run_batch(fields["E"], fields["k_h"], fields["k_v"])

        p3 = Phase3Reducer(cm)
        p3.run(X, Y)
        assert Path(cfg["phase3"]["output_dir"]).exists()
