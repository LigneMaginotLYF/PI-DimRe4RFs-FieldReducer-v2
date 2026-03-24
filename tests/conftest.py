"""
conftest.py
===========
Shared pytest fixtures for all test modules.
"""
from __future__ import annotations

import pytest
import numpy as np
import sys
import os

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def tiny_cfg() -> dict:
    """Minimal configuration for fast tests."""
    return {
        "grid": {"n_nodes_x": 6, "n_nodes_z": 4, "lx": 1.0, "lz": 0.5},
        "random_fields": {
            "E": {
                "n_terms": 3,
                "basis": "dct",
                "seed": 42,
                "covariance": "matern",
                "nu_sampling": False,
                "nu_ref": 1.5,
                "length_scale_sampling": False,
                "length_scale_ref": 0.3,
                "logE_std": 1.0,
                "E_ref": 10.0e6,
                "force_identity_reduction": False,
            },
            "k_h": {
                "n_terms": 0,
                "basis": "dct",
                "seed": 43,
                "covariance": "matern",
                "nu_sampling": False,
                "nu_ref": 1.5,
                "length_scale_sampling": False,
                "length_scale_ref": 0.3,
                "k_range": [1.0e-13, 1.0e-10],
                "force_identity_reduction": False,
            },
            "k_v": {
                "n_terms": 2,
                "basis": "dct",
                "seed": 44,
                "covariance": "matern",
                "nu_sampling": False,
                "nu_ref": 1.5,
                "length_scale_sampling": False,
                "length_scale_ref": 0.3,
                "k_range": [1.0e-13, 1.0e-10],
                "force_identity_reduction": False,
            },
        },
        "solver": {
            "type": "1d",
            "mode": "steady",
            "nu_biot": 0.3,
            "fluid_viscosity": 1.0e-3,
            "fluid_compressibility": 4.5e-10,
            "load": 1.0e4,
            "transient": {"dt": 0.01, "n_steps": 5},
        },
        "phase1": {"n_samples": 20, "val_fraction": 0.2, "output_dir": "/tmp/test_p1"},
        "phase2": {
            "surrogate_type": "nn",
            "output_repr": "direct",
            "n_output_modes": 5,
            "training_signal": "data",
            "hybrid_alpha": 0.5,
            "output_dir": "/tmp/test_p2",
            "nn": {
                "hidden_dims": [16, 16],
                "epochs": 5,
                "lr": 1e-3,
                "batch_size": 8,
                "patience": 3,
            },
            "pce": {"degree": 2},
        },
        "collocation_phase2": {"n_points": 8},
        "phase3": {
            "reducer_type": "nn",
            "training_signal": "surrogate",
            "hybrid_alpha": 0.5,
            "output_dir": "/tmp/test_p3",
            "surrogate_dir": "/tmp/test_p2",
            "nn": {
                "hidden_dims": [16, 16],
                "epochs": 5,
                "lr": 1e-3,
                "batch_size": 8,
                "patience": 3,
            },
        },
        "collocation_phase3": {"n_points": 3},
        "phase4": {
            "n_test_samples": 5,
            "output_dir": "/tmp/test_p4",
            "use_physics_for_plots": False,
        },
    }


@pytest.fixture
def field_manager(tiny_cfg):
    from src.field_manager import FieldManager
    return FieldManager(tiny_cfg)


@pytest.fixture
def biot_solver(tiny_cfg):
    from src.forward_solver import BiotSolver
    return BiotSolver(tiny_cfg)
