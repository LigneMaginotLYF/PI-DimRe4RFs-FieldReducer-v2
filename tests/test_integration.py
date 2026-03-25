"""
test_integration.py
===================
Integration test: run the full 4-phase pipeline end-to-end.
"""
import numpy as np
import pytest
from pathlib import Path


def test_full_pipeline(tiny_cfg):
    """Full pipeline: Phase 1 → Phase 2 → Phase 3 → Phase 4."""
    import copy

    from src.config_manager import ConfigManager
    from src.phase1_dataset import Phase1DatasetGenerator
    from src.phase2_surrogate import Phase2Surrogate
    from src.phase3_reducer import Phase3Reducer
    from src.phase4_validation import Phase4Validator

    cfg = copy.deepcopy(tiny_cfg)
    cfg["phase1"]["output_dir"] = "/tmp/test_integration_p1"
    cfg["phase2"]["output_dir"] = "/tmp/test_integration_p2"
    cfg["phase3"]["output_dir"] = "/tmp/test_integration_p3"
    cfg["phase3"]["surrogate_dir"] = "/tmp/test_integration_p2"
    cfg["phase4"]["output_dir"] = "/tmp/test_integration_p4"

    cm = ConfigManager(overrides=cfg)

    # Phase 1
    p1 = Phase1DatasetGenerator(cm)
    p1.run()
    X_train, Y_train, X_val, Y_val = p1.load()

    assert X_train.shape[1] == 6
    assert Y_train.shape[1] == 6

    # Phase 2
    p2 = Phase2Surrogate(cm)
    surrogate = p2.run(X_train, Y_train)
    assert surrogate is not None

    pred = surrogate.predict(X_train[:3])
    assert pred.shape == (3, 6)

    # Phase 3
    p3 = Phase3Reducer(cm)
    p3.run(X_train, Y_train, X_val, Y_val)

    X_red = p3.reduce(X_val)
    assert X_red.shape == X_val.shape

    # Phase 4
    p4 = Phase4Validator(cm)
    results = p4.run(X_val, Y_val)

    assert "metrics" in results
    assert "R2" in results["metrics"]
    assert "RMSE" in results["metrics"]
    assert Path(cfg["phase4"]["output_dir"]).exists()


def test_pipeline_with_pce_surrogate(tiny_cfg):
    """Integration test with PCE surrogate."""
    import copy

    from src.config_manager import ConfigManager
    from src.phase1_dataset import Phase1DatasetGenerator
    from src.phase2_surrogate import Phase2Surrogate

    cfg = copy.deepcopy(tiny_cfg)
    cfg["phase1"]["output_dir"] = "/tmp/test_int_pce_p1"
    cfg["phase2"]["surrogate_type"] = "pce"
    cfg["phase2"]["output_dir"] = "/tmp/test_int_pce_p2"

    cm = ConfigManager(overrides=cfg)

    p1 = Phase1DatasetGenerator(cm)
    p1.run()
    X_train, Y_train, _, _ = p1.load()

    p2 = Phase2Surrogate(cm)
    surrogate = p2.run(X_train, Y_train)
    pred = surrogate.predict(X_train[:2])
    assert pred.shape == (2, 6)


def test_config_manager_defaults():
    from src.config_manager import ConfigManager
    cm = ConfigManager()
    assert cm.n_terms_E == 5
    assert cm.n_terms_kh == 0
    assert cm.n_terms_kv == 2
    # total_input_dim: max(5,1) + max(0,1) + max(2,1) = 5+1+2=8
    assert cm.total_input_dim == 8


def test_config_manager_hash_stability():
    from src.config_manager import ConfigManager
    cm1 = ConfigManager()
    cm2 = ConfigManager()
    assert cm1.config_hash() == cm2.config_hash()


def test_solver_with_string_config_values():
    """Verify BiotSolver handles string-typed config values without TypeError.

    When YAML loads quoted numbers (e.g. load: "1.0e4"), they arrive as strings.
    Both ConfigManager coercion and BiotSolver's defensive float() casts must
    prevent arithmetic failures such as 'ufunc divide not supported'.
    """
    import numpy as np
    from src.config_manager import ConfigManager
    from src.forward_solver import BiotSolver
    from src.field_manager import FieldManager

    # Simulate YAML loading quoted numeric strings
    string_cfg = {
        "grid": {
            "n_nodes_x": "6",
            "n_nodes_z": "4",
            "lx": "1.0",
            "lz": "0.5",
        },
        "random_fields": {
            "E": {
                "n_terms": "3",
                "basis": "dct",
                "seed": "42",
                "nu_ref": "1.5",
                "nu_sampling": False,
                "length_scale_ref": "0.3",
                "length_scale_sampling": False,
                "mean": "10000000.0",
                "range": ["5000000.0", "20000000.0"],
                "fluctuation_std": "1.0",
                "force_identity_reduction": False,
            },
            "k_h": {
                "n_terms": "0",
                "basis": "dct",
                "seed": "43",
                "nu_ref": "1.5",
                "nu_sampling": False,
                "length_scale_ref": "0.3",
                "length_scale_sampling": False,
                "mean": "1e-12",
                "range": ["1e-13", "1e-10"],
                "fluctuation_std": "0.5",
                "force_identity_reduction": False,
            },
            "k_v": {
                "n_terms": "2",
                "basis": "dct",
                "seed": "44",
                "nu_ref": "1.5",
                "nu_sampling": False,
                "length_scale_ref": "0.3",
                "length_scale_sampling": False,
                "mean": "1e-12",
                "range": ["1e-13", "1e-10"],
                "fluctuation_std": "0.5",
                "force_identity_reduction": False,
            },
        },
        "solver": {
            "type": "1d",
            "mode": "steady",
            "nu_biot": "0.3",
            "fluid_viscosity": "0.001",
            "fluid_compressibility": "4.5e-10",
            "load": "10000.0",
            "transient": {"dt": "0.01", "n_steps": "5"},
        },
        "phase1": {"n_samples": 10, "val_fraction": 0.2, "output_dir": "/tmp/test_str_cfg"},
        "phase2": {
            "surrogate_type": "nn",
            "output_repr": "direct",
            "n_output_modes": 5,
            "training_signal": "data",
            "hybrid_alpha": 0.5,
            "output_dir": "/tmp/test_str_cfg_p2",
            "nn": {"hidden_dims": [16], "epochs": 2, "lr": 1e-3, "batch_size": 4, "patience": 2},
            "pce": {"degree": 2},
        },
        "collocation_phase2": {"n_points": 4},
        "phase3": {
            "reducer_type": "nn",
            "training_signal": "surrogate",
            "hybrid_alpha": 0.5,
            "output_dir": "/tmp/test_str_cfg_p3",
            "surrogate_dir": "/tmp/test_str_cfg_p2",
            "nn": {"hidden_dims": [16], "epochs": 2, "lr": 1e-3, "batch_size": 4, "patience": 2},
        },
        "collocation_phase3": {"n_points": 2},
        "phase4": {"n_test_samples": 3, "output_dir": "/tmp/test_str_cfg_p4", "use_physics_for_plots": False},
    }

    cm = ConfigManager(overrides=string_cfg)

    # Verify grid values are proper types after coercion
    cfg = cm.cfg
    assert isinstance(cfg["grid"]["n_nodes_x"], int)
    assert isinstance(cfg["grid"]["n_nodes_z"], int)
    assert isinstance(cfg["grid"]["lx"], float)
    assert isinstance(cfg["grid"]["lz"], float)

    # Verify solver values are proper types after coercion
    solver_cfg = cfg["solver"]
    assert isinstance(solver_cfg["load"], float)
    assert isinstance(solver_cfg["nu_biot"], float)
    assert isinstance(solver_cfg["fluid_viscosity"], float)
    assert isinstance(solver_cfg["fluid_compressibility"], float)
    assert isinstance(solver_cfg["transient"]["dt"], float)
    assert isinstance(solver_cfg["transient"]["n_steps"], int)

    # Verify BiotSolver initialises without error and attributes are correct types
    solver = BiotSolver(cfg)
    assert isinstance(solver.load, float)
    assert isinstance(solver.nu_biot, float)
    assert isinstance(solver.dt, float)
    assert isinstance(solver.n_steps, int)
    assert isinstance(solver.lx, float)
    assert isinstance(solver.lz, float)
    assert isinstance(solver.n_nodes_x, int)
    assert isinstance(solver.n_nodes_z, int)

    # Verify the solver runs without arithmetic TypeError (the core bug)
    fm = FieldManager(cfg)
    rng = np.random.default_rng(0)
    xi_E = rng.standard_normal((1, fm.field_configs["E"].effective_dim))
    xi_kh = rng.standard_normal((1, fm.field_configs["k_h"].effective_dim))
    xi_kv = rng.standard_normal((1, fm.field_configs["k_v"].effective_dim))
    xi = np.concatenate([xi_E, xi_kh, xi_kv], axis=1)

    fields = fm.reconstruct_all_fields(xi)
    settlement = solver.run(fields["E"][0], fields["k_h"][0], fields["k_v"][0])
    assert settlement.dtype == np.float64
    assert np.all(np.isfinite(settlement))

