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
    # Define common field configs for reduced/full spaces
    _reduced_E = {
        "n_terms": 2, "basis": "dct", "seed": 142, "covariance": "matern",
        "nu_sampling": False, "nu_ref": 1.5,
        "length_scale_sampling": False, "length_scale_ref": 0.3,
        "mean": 10.0e6, "range": [5.0e6, 20.0e6], "fluctuation_std": 1.0,
        "force_identity_reduction": False,
    }
    _reduced_kh = {
        "n_terms": 0, "basis": "dct", "seed": 143, "covariance": "matern",
        "nu_sampling": False, "nu_ref": 1.5,
        "length_scale_sampling": False, "length_scale_ref": 0.3,
        "mean": 1.0e-12, "range": [1.0e-13, 1.0e-10], "fluctuation_std": 0.5,
        "force_identity_reduction": False,
    }
    _reduced_kv = {
        "n_terms": 0, "basis": "dct", "seed": 144, "covariance": "matern",
        "nu_sampling": False, "nu_ref": 1.5,
        "length_scale_sampling": False, "length_scale_ref": 0.3,
        "mean": 1.0e-12, "range": [1.0e-13, 1.0e-10], "fluctuation_std": 0.5,
        "force_identity_reduction": False,
    }
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
                "mean": 10.0e6,
                "range": [5.0e6, 20.0e6],
                "fluctuation_std": 1.0,
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
                "mean": 1.0e-12,
                "range": [1.0e-13, 1.0e-10],
                "fluctuation_std": 0.5,
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
                "mean": 1.0e-12,
                "range": [1.0e-13, 1.0e-10],
                "fluctuation_std": 0.5,
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
            "hybrid_alpha": 0.1,
            "physics_check_interval": 2,
            "n_training_samples": 12,
            "output_dir": "/tmp/test_p2",
            # Reduced space: E:2, k_h:scalar, k_v:scalar → dim=4
            "reduced_fields": {
                "E": _reduced_E,
                "k_h": _reduced_kh,
                "k_v": _reduced_kv,
            },
            "nn": {
                "hidden_dims": [16, 16],
                "epochs": 5,
                "lr": 1e-3,
                "batch_size": 8,
                "patience": 3,
            },
            "pce": {"degree": 2},
            "evaluation": {"test_fraction": 0.2, "n_plot_samples": 3},
        },
        "collocation_phase2": {"n_points": 8},
        "phase3": {
            "reducer_type": "nn",
            "training_signal": "surrogate",
            "n_training_samples": 16,
            "output_dir": "/tmp/test_p3",
            "surrogate_dir": "/tmp/test_p2",
            # Full space: E:3, k_h:scalar, k_v:2 → dim=6
            "full_fields": {
                "E": {
                    "n_terms": 3, "basis": "dct", "seed": 42,
                    "nu_sampling": False, "nu_ref": 1.5,
                    "length_scale_sampling": False, "length_scale_ref": 0.3,
                    "mean": 10.0e6, "range": [5.0e6, 20.0e6], "fluctuation_std": 1.0,
                    "force_identity_reduction": False,
                },
                "k_h": {
                    "n_terms": 0, "basis": "dct", "seed": 43,
                    "nu_sampling": False, "nu_ref": 1.5,
                    "length_scale_sampling": False, "length_scale_ref": 0.3,
                    "mean": 1.0e-12, "range": [1.0e-13, 1.0e-10], "fluctuation_std": 0.5,
                    "force_identity_reduction": False,
                },
                "k_v": {
                    "n_terms": 2, "basis": "dct", "seed": 44,
                    "nu_sampling": False, "nu_ref": 1.5,
                    "length_scale_sampling": False, "length_scale_ref": 0.3,
                    "mean": 1.0e-12, "range": [1.0e-13, 1.0e-10], "fluctuation_std": 0.5,
                    "force_identity_reduction": False,
                },
            },
            # Reduced space (must match phase2.reduced_fields)
            "reduced_fields": {
                "E": _reduced_E,
                "k_h": _reduced_kh,
                "k_v": _reduced_kv,
            },
            "nn": {
                "hidden_dims": [16, 16],
                "epochs": 5,
                "lr": 1e-3,
                "batch_size": 8,
                "patience": 3,
            },
            "evaluation": {"test_fraction": 0.2, "n_plot_samples": 3},
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
