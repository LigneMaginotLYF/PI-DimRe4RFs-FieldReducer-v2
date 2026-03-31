"""
test_integration.py
===================
Integration test: run the full pipeline end-to-end.

Phase 2: surrogate in REDUCED parameter space
Phase 3: dimension reducer from FULL to REDUCED space
"""
import numpy as np
import pytest
from pathlib import Path


def test_full_pipeline(tiny_cfg):
    """Full pipeline: Phase 2 → Phase 3 (with independent data generation)."""
    import copy
    import os
    from src.config_manager import ConfigManager
    from src.phase2_surrogate import Phase2Surrogate
    from src.phase3_reducer import Phase3Reducer
    from src.phase2_evaluator import Phase2Evaluator
    from src.phase3_evaluator import Phase3Evaluator

    cfg = copy.deepcopy(tiny_cfg)
    cfg["phase2"]["output_dir"] = "/tmp/test_integration_p2"
    cfg["phase3"]["output_dir"] = "/tmp/test_integration_p3"
    cfg["phase3"]["surrogate_dir"] = "/tmp/test_integration_p2"

    cm = ConfigManager(overrides=cfg)

    # Phase 2: train surrogate in REDUCED space (dim=4: E:2, kh:scalar, kv:scalar)
    p2 = Phase2Surrogate(cm)
    surrogate = p2.run()
    assert surrogate is not None

    # Surrogate input dim = reduced dim
    assert surrogate.input_dim == p2.reduced_dim

    # Predict with reduced-space inputs
    from src.field_manager import FieldManager
    fm_red = p2.field_manager
    X_red, fields_red, _ = fm_red.generate_dataset(3)
    pred = surrogate.predict(X_red)
    assert pred.shape == (3, cfg["grid"]["n_nodes_x"])

    # Phase 3: train reducer FULL → REDUCED (full_dim=6, reduced_dim=4)
    p3 = Phase3Reducer(cm)
    p3.run()  # Generates its own full-space data
    assert p3.full_dim == 6   # E:3, kh:scalar, kv:2
    assert p3.reduced_dim == 4  # E:2, kh:scalar, kv:scalar

    # Reduce full-space samples
    fm_full = p3.field_manager_full
    X_full, _, _ = fm_full.generate_dataset(5)
    X_reduced = p3.reduce(X_full)
    assert X_reduced.shape == (5, p3.reduced_dim)

    # Phase 2 evaluation
    X_test_p2 = np.load(os.path.join(cfg["phase2"]["output_dir"], "phase2_X_test.npy"))
    Y_test_p2 = np.load(os.path.join(cfg["phase2"]["output_dir"], "phase2_Y_test.npy"))
    p2_eval = Phase2Evaluator(cm)
    p2_results = p2_eval.run(X_test_p2, Y_test_p2, surrogate=surrogate)
    assert "metrics" in p2_results
    assert "R2" in p2_results["metrics"]

    # Phase 3 evaluation (always Biot physics)
    X_test_p3 = np.load(os.path.join(cfg["phase3"]["output_dir"], "phase3_X_test_full.npy"))
    Y_test_p3 = np.load(os.path.join(cfg["phase3"]["output_dir"], "phase3_Y_test_full.npy"))
    p3_eval = Phase3Evaluator(cm)
    p3_results = p3_eval.run(X_test_p3, Y_test_p3, reducer=p3)
    assert "metrics" in p3_results
    assert "R2" in p3_results["metrics"]
    # Must report physics evaluation
    import json
    with open(p3_results["metrics_path"]) as f:
        data = json.load(f)
    assert data["evaluation_mode"] == "physics_biot"


def test_pipeline_with_pce_surrogate(tiny_cfg):
    """Phase 2 with PCE surrogate."""
    import copy
    from src.config_manager import ConfigManager
    from src.phase2_surrogate import Phase2Surrogate
    from src.field_manager import FieldManager

    cfg = copy.deepcopy(tiny_cfg)
    cfg["phase2"]["surrogate_type"] = "pce"
    cfg["phase2"]["output_dir"] = "/tmp/test_int_pce_p2"

    cm = ConfigManager(overrides=cfg)

    p2 = Phase2Surrogate(cm)
    surrogate = p2.run()

    # Predict with reduced-space inputs
    fm_red = p2.field_manager
    X_red, _, _ = fm_red.generate_dataset(2)
    pred = surrogate.predict(X_red)
    assert pred.shape == (2, cfg["grid"]["n_nodes_x"])


def test_config_manager_defaults():
    from src.config_manager import ConfigManager
    cm = ConfigManager()
    assert cm.n_terms_E == 15
    assert cm.n_terms_kh == 0
    assert cm.n_terms_kv == 0
    # total_input_dim: max(15,1) + max(0,1) + max(0,1) = 15+1+1=17
    assert cm.total_input_dim == 17


def test_config_manager_hash_stability():
    from src.config_manager import ConfigManager
    cm1 = ConfigManager()
    cm2 = ConfigManager()
    assert cm1.config_hash() == cm2.config_hash()


def test_solver_with_string_config_values():
    """Verify BiotSolver handles string-typed config values without TypeError."""
    import numpy as np
    from src.config_manager import ConfigManager
    from src.forward_solver import BiotSolver
    from src.field_manager import FieldManager

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
            "n_training_samples": 8,
            "output_dir": "/tmp/test_str_cfg_p2",
            "reduced_fields": {
                "E": {
                    "n_terms": "2", "seed": "42", "nu_ref": "1.5",
                    "length_scale_ref": "0.3", "nu_sampling": False,
                    "length_scale_sampling": False,
                    "mean": "10000000.0", "range": ["5000000.0", "20000000.0"],
                    "fluctuation_std": "1.0",
                },
                "k_h": {
                    "n_terms": "0", "seed": "43", "nu_ref": "1.5",
                    "length_scale_ref": "0.3", "nu_sampling": False,
                    "length_scale_sampling": False,
                    "mean": "1e-12", "range": ["1e-13", "1e-10"],
                    "fluctuation_std": "0.5",
                },
                "k_v": {
                    "n_terms": "0", "seed": "44", "nu_ref": "1.5",
                    "length_scale_ref": "0.3", "nu_sampling": False,
                    "length_scale_sampling": False,
                    "mean": "1e-12", "range": ["1e-13", "1e-10"],
                    "fluctuation_std": "0.5",
                },
            },
            "nn": {"hidden_dims": [16], "epochs": 2, "lr": 1e-3, "batch_size": 4, "patience": 2},
            "pce": {"degree": 2},
        },
        "collocation_phase2": {"n_points": 4},
        "phase3": {
            "reducer_type": "nn",
            "training_signal": "surrogate",
            "n_training_samples": 10,
            "output_dir": "/tmp/test_str_cfg_p3",
            "surrogate_dir": "/tmp/test_str_cfg_p2",
            "full_fields": {
                "E": {
                    "n_terms": "3", "seed": "42", "nu_ref": "1.5",
                    "length_scale_ref": "0.3", "nu_sampling": False,
                    "length_scale_sampling": False,
                    "mean": "10000000.0", "range": ["5000000.0", "20000000.0"],
                    "fluctuation_std": "1.0",
                },
                "k_h": {
                    "n_terms": "0", "seed": "43", "nu_ref": "1.5",
                    "length_scale_ref": "0.3", "nu_sampling": False,
                    "length_scale_sampling": False,
                    "mean": "1e-12", "range": ["1e-13", "1e-10"],
                    "fluctuation_std": "0.5",
                },
                "k_v": {
                    "n_terms": "2", "seed": "44", "nu_ref": "1.5",
                    "length_scale_ref": "0.3", "nu_sampling": False,
                    "length_scale_sampling": False,
                    "mean": "1e-12", "range": ["1e-13", "1e-10"],
                    "fluctuation_std": "0.5",
                },
            },
            "reduced_fields": {
                "E": {
                    "n_terms": "2", "seed": "42", "nu_ref": "1.5",
                    "length_scale_ref": "0.3", "nu_sampling": False,
                    "length_scale_sampling": False,
                    "mean": "10000000.0", "range": ["5000000.0", "20000000.0"],
                    "fluctuation_std": "1.0",
                },
                "k_h": {
                    "n_terms": "0", "seed": "43", "nu_ref": "1.5",
                    "length_scale_ref": "0.3", "nu_sampling": False,
                    "length_scale_sampling": False,
                    "mean": "1e-12", "range": ["1e-13", "1e-10"],
                    "fluctuation_std": "0.5",
                },
                "k_v": {
                    "n_terms": "0", "seed": "44", "nu_ref": "1.5",
                    "length_scale_ref": "0.3", "nu_sampling": False,
                    "length_scale_sampling": False,
                    "mean": "1e-12", "range": ["1e-13", "1e-10"],
                    "fluctuation_std": "0.5",
                },
            },
            "nn": {"hidden_dims": [16], "epochs": 2, "lr": 1e-3, "batch_size": 4, "patience": 2},
        },
        "collocation_phase3": {"n_points": 2},
        "phase4": {"n_test_samples": 3, "output_dir": "/tmp/test_str_cfg_p4", "use_physics_for_plots": False},
    }

    cm = ConfigManager(overrides=string_cfg)

    cfg = cm.cfg
    assert isinstance(cfg["grid"]["n_nodes_x"], int)
    assert isinstance(cfg["grid"]["n_nodes_z"], int)
    assert isinstance(cfg["grid"]["lx"], float)
    assert isinstance(cfg["grid"]["lz"], float)

    solver_cfg = cfg["solver"]
    assert isinstance(solver_cfg["load"], float)
    assert isinstance(solver_cfg["nu_biot"], float)
    assert isinstance(solver_cfg["fluid_viscosity"], float)
    assert isinstance(solver_cfg["fluid_compressibility"], float)
    assert isinstance(solver_cfg["transient"]["dt"], float)
    assert isinstance(solver_cfg["transient"]["n_steps"], int)

    from src.forward_solver import BiotSolver
    solver = BiotSolver(cfg)
    assert isinstance(solver.load, float)
    assert isinstance(solver.nu_biot, float)
    assert isinstance(solver.dt, float)
    assert isinstance(solver.n_steps, int)
    assert isinstance(solver.lx, float)
    assert isinstance(solver.lz, float)
    assert isinstance(solver.n_nodes_x, int)
    assert isinstance(solver.n_nodes_z, int)

    from src.field_manager import FieldManager
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
