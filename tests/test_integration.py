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
