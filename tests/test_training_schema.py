"""
test_training_schema.py
=======================
Unit tests for TrainingSignal implementations.
"""
import numpy as np
import pytest


class TestDirectPhysicsSignal:
    def test_evaluate_from_fields(self, field_manager, biot_solver):
        from src.training_schema import DirectPhysicsSignal
        sig = DirectPhysicsSignal(biot_solver, field_manager)
        n = 3
        X, fields, _ = field_manager.generate_dataset(n)
        settlements = sig.evaluate(
            E_field=fields["E"],
            k_h_field=fields["k_h"],
            k_v_field=fields["k_v"],
        )
        assert settlements.shape == (n, 6)

    def test_evaluate_from_xi(self, field_manager, biot_solver):
        from src.training_schema import DirectPhysicsSignal
        sig = DirectPhysicsSignal(biot_solver, field_manager)
        X, _, _ = field_manager.generate_dataset(3)
        settlements = sig.evaluate(xi_concat=X)
        assert settlements.shape == (3, 6)

    def test_single_sample(self, field_manager, biot_solver):
        from src.training_schema import DirectPhysicsSignal
        sig = DirectPhysicsSignal(biot_solver, field_manager)
        X, _, _ = field_manager.generate_dataset(1)
        result = sig.evaluate(xi_concat=X[0])
        assert result.shape == (6,)


class TestSurrogateSignal:
    def test_evaluate(self, field_manager):
        from src.training_schema import SurrogateSignal
        from src.surrogate_models import NNSurrogate

        d = field_manager.total_input_dim
        nx = field_manager.n_nodes_x
        np.random.seed(0)
        X = np.random.randn(20, d)
        Y = np.random.randn(20, nx)

        surrogate = NNSurrogate(d, nx, epochs=3, hidden_dims=[8, 8])
        surrogate.fit(X, Y)

        sig = SurrogateSignal(surrogate)
        pred = sig.evaluate(xi_concat=X[:5])
        assert pred.shape == (5, nx)


class TestHybridSignal:
    def test_evaluate(self, field_manager, biot_solver):
        from src.training_schema import DirectPhysicsSignal, SurrogateSignal, HybridPhysicsSignal
        from src.surrogate_models import NNSurrogate

        d = field_manager.total_input_dim
        nx = field_manager.n_nodes_x
        np.random.seed(0)
        X = np.random.randn(20, d)
        Y = np.random.randn(20, nx)

        surrogate = NNSurrogate(d, nx, epochs=3, hidden_dims=[8, 8])
        surrogate.fit(X, Y)

        phys = DirectPhysicsSignal(biot_solver, field_manager)
        surr = SurrogateSignal(surrogate)
        hybrid = HybridPhysicsSignal(phys, surr, alpha=0.5)

        X_test, _, _ = field_manager.generate_dataset(3)
        pred = hybrid.evaluate(xi_concat=X_test)
        assert pred.shape == (3, nx)


class TestBuildTrainingSignal:
    def test_surrogate_signal(self, field_manager):
        from src.training_schema import build_training_signal
        from src.surrogate_models import NNSurrogate

        d = field_manager.total_input_dim
        nx = field_manager.n_nodes_x
        surrogate = NNSurrogate(d, nx, epochs=2, hidden_dims=[8])

        sig = build_training_signal("surrogate", surrogate=surrogate)
        from src.training_schema import SurrogateSignal
        assert isinstance(sig, SurrogateSignal)

    def test_physics_signal(self, field_manager, biot_solver):
        from src.training_schema import build_training_signal, DirectPhysicsSignal
        sig = build_training_signal("physics", solver=biot_solver, field_manager=field_manager)
        assert isinstance(sig, DirectPhysicsSignal)

    def test_unknown_type(self):
        from src.training_schema import build_training_signal
        with pytest.raises(ValueError):
            build_training_signal("unknown")
