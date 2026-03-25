"""
PI-DimRe4RFs-FieldReducer-v2
Physics-Informed Dimension Reduction for Material Random Fields.

Package providing a multi-phase pipeline for constructing dimension reduction
mappings for material random fields (E, k_h, k_v) using a shared 2D DCT-II
basis and physics-driven training signals.
"""

from .config_manager import ConfigManager
from .field_manager import FieldManager, FieldConfig
from .forward_solver import BiotSolver
from .utils import compute_dct_basis, matern_spectral_variance
from .phase2_evaluator import Phase2Evaluator
from .phase3_evaluator import Phase3Evaluator

__all__ = [
    "ConfigManager",
    "FieldManager",
    "FieldConfig",
    "BiotSolver",
    "compute_dct_basis",
    "matern_spectral_variance",
    "Phase2Evaluator",
    "Phase3Evaluator",
]
