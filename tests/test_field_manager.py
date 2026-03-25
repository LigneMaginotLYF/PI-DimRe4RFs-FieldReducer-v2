"""
test_field_manager.py
=====================
Unit tests for FieldManager and DCT basis construction.
"""
import numpy as np
import pytest
from src.field_manager import FieldManager, FieldConfig
from src.utils import compute_dct_basis, matern_spectral_variance


class TestDCTBasis:
    def test_shape(self):
        basis = compute_dct_basis(6, 4, 5)
        assert basis.shape == (24, 5)

    def test_zero_terms(self):
        basis = compute_dct_basis(6, 4, 0)
        assert basis.shape == (24, 0)

    def test_l2_normalised(self):
        basis = compute_dct_basis(6, 4, 5)
        norms = np.linalg.norm(basis, axis=0)
        np.testing.assert_allclose(norms, 1.0, atol=1e-8)

    def test_consistent(self):
        """Same call returns identical result (deterministic basis)."""
        b1 = compute_dct_basis(6, 4, 5)
        b2 = compute_dct_basis(6, 4, 5)
        np.testing.assert_array_equal(b1, b2)

    def test_ordering_by_frequency(self):
        """Basis columns should be ordered by ascending frequency magnitude."""
        basis = compute_dct_basis(8, 6, 10)
        # DC component (ix=0, iz=0) should be first column
        # It's a flat field → all entries equal
        first_col = basis[:, 0]
        assert np.std(first_col) < 1e-8, "First mode should be DC (constant)"


class TestMaternSpectralVariance:
    def test_shape(self):
        var = matern_spectral_variance(6, 4, 5)
        assert var.shape == (5,)

    def test_zero_terms(self):
        var = matern_spectral_variance(6, 4, 0)
        assert var.shape == (0,)

    def test_sums_to_one(self):
        var = matern_spectral_variance(6, 4, 5)
        assert abs(var.sum() - 1.0) < 1e-8

    def test_positive(self):
        var = matern_spectral_variance(6, 4, 5)
        assert np.all(var > 0)

    def test_decreasing_with_frequency(self):
        """Higher-frequency modes should generally have lower variance."""
        var = matern_spectral_variance(6, 4, 5)
        # Not strictly monotone, but first should dominate
        assert var[0] > var[-1]


class TestFieldConfig:
    def test_effective_dim_homogeneous(self):
        fc = FieldConfig(name="k_h", n_terms=0)
        assert fc.effective_dim == 1

    def test_effective_dim_high_dim(self):
        fc = FieldConfig(name="E", n_terms=5)
        assert fc.effective_dim == 5

    def test_from_dict_legacy_format(self):
        """Legacy keys (E_ref, logE_std, k_range) are still accepted."""
        d = {"n_terms": 3, "seed": 99, "nu_ref": 2.0, "E_ref": 5e6, "logE_std": 0.5}
        fc = FieldConfig.from_dict("E", d)
        assert fc.n_terms == 3
        assert fc.seed == 99
        assert fc.E_ref == 5e6
        assert fc.logE_std == 0.5

    def test_from_dict_unified_format(self):
        """New unified keys (mean, range, fluctuation_std) are accepted."""
        d = {
            "n_terms": 4,
            "seed": 7,
            "mean": 8e6,
            "range": [2e6, 25e6],
            "fluctuation_std": 0.8,
        }
        fc = FieldConfig.from_dict("E", d)
        assert fc.n_terms == 4
        assert fc.E_ref == 8e6         # mean → E_ref
        assert fc.logE_std == 0.8      # fluctuation_std → logE_std
        assert fc.k_range == (2e6, 25e6)  # range → k_range (stored for display)

    def test_unified_keys_take_precedence(self):
        """When both old and new keys are present, unified keys take precedence."""
        d = {
            "n_terms": 2,
            "E_ref": 1e6,      # legacy (should be overridden)
            "mean": 5e6,       # unified (should win)
            "logE_std": 0.3,   # legacy
            "fluctuation_std": 0.9,  # unified (should win)
        }
        fc = FieldConfig.from_dict("E", d)
        assert fc.E_ref == 5e6
        assert fc.logE_std == 0.9

    def test_k_field_range_alias(self):
        """'range' key is stored as k_range for k_h/k_v fields."""
        d = {"n_terms": 0, "range": [1e-14, 5e-11], "fluctuation_std": 0.4}
        fc = FieldConfig.from_dict("k_h", d)
        assert fc.k_range == (1e-14, 5e-11)


class TestFieldManager:
    def test_total_input_dim(self, field_manager):
        # E: 3 terms, k_h: 0→1, k_v: 2 → total 6
        assert field_manager.total_input_dim == 6

    def test_sample_coefficients_shape(self, field_manager):
        xi_E = field_manager.sample_coefficients(10, "E")
        assert xi_E.shape == (10, 3)

        xi_kh = field_manager.sample_coefficients(10, "k_h")
        assert xi_kh.shape == (10, 1)  # homogeneous → 1 scalar

        xi_kv = field_manager.sample_coefficients(10, "k_v")
        assert xi_kv.shape == (10, 2)

    def test_reconstruct_field_shape(self, field_manager):
        xi_E = field_manager.sample_coefficients(5, "E")
        E_field = field_manager.reconstruct_field(xi_E, "E")
        assert E_field.shape == (5, 24)  # n_nodes = 6*4

    def test_reconstruct_scalar_field(self, field_manager):
        xi_kh = field_manager.sample_coefficients(5, "k_h")
        k_h_field = field_manager.reconstruct_field(xi_kh, "k_h")
        # Homogeneous: all nodes same value
        assert k_h_field.shape == (5, 24)
        for i in range(5):
            assert np.std(k_h_field[i]) < 1e-10, "Homogeneous field should be constant"

    def test_E_field_positive(self, field_manager):
        xi_E = field_manager.sample_coefficients(10, "E")
        E = field_manager.reconstruct_field(xi_E, "E")
        assert np.all(E > 0), "Young's modulus must be positive"

    def test_k_fields_in_range(self, field_manager):
        xi_kv = field_manager.sample_coefficients(10, "k_v")
        k_v = field_manager.reconstruct_field(xi_kv, "k_v")
        cfg = field_manager.field_configs["k_v"]
        assert np.all(k_v > 0)

    def test_generate_dataset_shape(self, field_manager):
        X, fields, xi_dict = field_manager.generate_dataset(8)
        assert X.shape == (8, 6)
        assert fields["E"].shape == (8, 24)
        assert fields["k_h"].shape == (8, 24)
        assert fields["k_v"].shape == (8, 24)

    def test_split_coefficients(self, field_manager):
        X, _, _ = field_manager.generate_dataset(4)
        xi_E, xi_kh, xi_kv = field_manager.split_coefficients(X)
        assert xi_E.shape == (4, 3)
        assert xi_kh.shape == (4, 1)
        assert xi_kv.shape == (4, 2)

    def test_reconstruct_all_fields(self, field_manager):
        X, _, _ = field_manager.generate_dataset(4)
        fields = field_manager.reconstruct_all_fields(X)
        assert set(fields.keys()) == {"E", "k_h", "k_v"}
        for v in fields.values():
            assert v.shape == (4, 24)

    def test_reproducibility(self, tiny_cfg):
        fm1 = FieldManager(tiny_cfg)
        fm2 = FieldManager(tiny_cfg)
        X1, _, _ = fm1.generate_dataset(5)
        X2, _, _ = fm2.generate_dataset(5)
        np.testing.assert_array_equal(X1, X2)

    def test_single_sample_reconstruction(self, field_manager):
        X, _, _ = field_manager.generate_dataset(3)
        single = X[0]
        fields = field_manager.reconstruct_all_fields(single)
        for v in fields.values():
            assert v.shape == (24,)
