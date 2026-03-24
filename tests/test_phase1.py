"""
test_phase1.py
==============
Unit tests for Phase 1 dataset generation.
"""
import os
import numpy as np
import pytest


class TestPhase1Dataset:
    def test_run_and_load(self, tiny_cfg):
        from src.config_manager import ConfigManager
        from src.phase1_dataset import Phase1DatasetGenerator

        cm = ConfigManager(overrides=tiny_cfg)
        p1 = Phase1DatasetGenerator(cm)
        paths = p1.run()

        assert os.path.exists(paths["X_train"])
        assert os.path.exists(paths["Y_train"])
        assert os.path.exists(paths["X_val"])
        assert os.path.exists(paths["Y_val"])
        assert os.path.exists(paths["metadata"])

        X_train, Y_train, X_val, Y_val = p1.load()
        assert X_train.ndim == 2
        assert Y_train.ndim == 2
        assert X_train.shape[1] == 6   # total_input_dim: 3+1+2
        assert Y_train.shape[1] == 6   # n_nodes_x

    def test_shapes_consistent(self, tiny_cfg):
        from src.config_manager import ConfigManager
        from src.phase1_dataset import Phase1DatasetGenerator

        cm = ConfigManager(overrides=tiny_cfg)
        p1 = Phase1DatasetGenerator(cm)
        p1.run()
        X_train, Y_train, X_val, Y_val = p1.load()

        assert X_train.shape[0] == Y_train.shape[0]
        assert X_val.shape[0] == Y_val.shape[0]
        assert X_train.shape[0] + X_val.shape[0] == tiny_cfg["phase1"]["n_samples"]

    def test_settlements_are_positive(self, tiny_cfg):
        from src.config_manager import ConfigManager
        from src.phase1_dataset import Phase1DatasetGenerator

        cm = ConfigManager(overrides=tiny_cfg)
        p1 = Phase1DatasetGenerator(cm)
        p1.run()
        _, Y_train, _, Y_val = p1.load()
        # Settlement under compressive load should be positive (downward)
        assert np.all(Y_train > 0)
        assert np.all(Y_val > 0)

    def test_metadata_fields(self, tiny_cfg):
        import json
        from pathlib import Path
        from src.config_manager import ConfigManager
        from src.phase1_dataset import Phase1DatasetGenerator

        cm = ConfigManager(overrides=tiny_cfg)
        p1 = Phase1DatasetGenerator(cm)
        p1.run()
        meta_path = Path(tiny_cfg["phase1"]["output_dir"]) / "dataset_metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)

        assert "n_train" in meta
        assert "n_val" in meta
        assert "input_dim" in meta
        assert "config_hash" in meta
        assert meta["input_dim"] == 6
