"""
test_surrogate_models.py
========================
Unit tests for NN and PCE surrogate models and all output representations.
"""
import numpy as np
import pytest


@pytest.fixture
def small_dataset():
    np.random.seed(0)
    n, d, nx = 30, 6, 6
    X = np.random.randn(n, d)
    Y = np.random.randn(n, nx)
    return X, Y, d, nx


class TestNNSurrogate:
    def test_fit_predict_direct(self, small_dataset):
        from src.surrogate_models import NNSurrogate
        X, Y, d, nx = small_dataset
        model = NNSurrogate(d, nx, output_repr="direct", epochs=5, hidden_dims=[8, 8], patience=2)
        model.fit(X, Y)
        pred = model.predict(X[:3])
        assert pred.shape == (3, nx)

    def test_fit_predict_dct(self, small_dataset):
        from src.surrogate_models import NNSurrogate
        X, Y, d, nx = small_dataset
        model = NNSurrogate(d, nx, output_repr="dct", n_output_modes=4, epochs=5, hidden_dims=[8, 8])
        model.fit(X, Y)
        pred = model.predict(X[:3])
        assert pred.shape == (3, nx)

    def test_fit_predict_poly(self, small_dataset):
        from src.surrogate_models import NNSurrogate
        X, Y, d, nx = small_dataset
        model = NNSurrogate(d, nx, output_repr="poly", n_output_modes=4, epochs=5, hidden_dims=[8, 8])
        model.fit(X, Y)
        pred = model.predict(X[:3])
        assert pred.shape == (3, nx)

    def test_fit_predict_bspline(self, small_dataset):
        from src.surrogate_models import NNSurrogate
        X, Y, d, nx = small_dataset
        model = NNSurrogate(d, nx, output_repr="bspline", n_output_modes=5, epochs=5, hidden_dims=[8, 8])
        model.fit(X, Y)
        pred = model.predict(X[:3])
        assert pred.shape == (3, nx)

    def test_save_load(self, small_dataset, tmp_path):
        from src.surrogate_models import NNSurrogate
        X, Y, d, nx = small_dataset
        model = NNSurrogate(d, nx, output_repr="direct", epochs=5, hidden_dims=[8, 8])
        model.fit(X, Y)
        pred_before = model.predict(X[:5])

        model.save(tmp_path / "nn_surr")
        loaded = NNSurrogate.load(tmp_path / "nn_surr")
        pred_after = loaded.predict(X[:5])
        np.testing.assert_allclose(pred_before, pred_after, atol=1e-5)


class TestPCESurrogate:
    def test_fit_predict_direct(self, small_dataset):
        from src.surrogate_models import PCESurrogate
        X, Y, d, nx = small_dataset
        model = PCESurrogate(d, nx, output_repr="direct", degree=2)
        model.fit(X, Y)
        pred = model.predict(X[:3])
        assert pred.shape == (3, nx)

    def test_save_load(self, small_dataset, tmp_path):
        from src.surrogate_models import PCESurrogate
        X, Y, d, nx = small_dataset
        model = PCESurrogate(d, nx, output_repr="direct", degree=2)
        model.fit(X, Y)
        pred_before = model.predict(X[:5])

        model.save(tmp_path / "pce_surr")
        loaded = PCESurrogate.load(tmp_path / "pce_surr")
        pred_after = loaded.predict(X[:5])
        np.testing.assert_allclose(pred_before, pred_after, atol=1e-5)


class TestBuildSurrogate:
    def test_build_nn(self):
        from src.surrogate_models import build_surrogate
        m = build_surrogate("nn", 6, 10, nn_cfg={"hidden_dims": [8], "epochs": 2})
        assert m.input_dim == 6

    def test_build_pce(self):
        from src.surrogate_models import build_surrogate
        m = build_surrogate("pce", 6, 10, pce_cfg={"degree": 2})
        assert m.input_dim == 6

    def test_unknown_type(self):
        from src.surrogate_models import build_surrogate
        with pytest.raises(ValueError):
            build_surrogate("unknown", 6, 10)
