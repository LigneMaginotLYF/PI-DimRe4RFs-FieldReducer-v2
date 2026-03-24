"""
training_schema.py
==================
Physics-driven training signal abstractions.

All training phases use a :class:`TrainingSignal` to evaluate the response
(settlement profile) given either physical fields or reduced parameters.

Concrete implementations
------------------------
- :class:`DirectPhysicsSignal`  — calls the Biot solver directly on (E, k_h, k_v)
- :class:`SurrogateSignal`      — calls the Phase-2 surrogate on (xi_E, xi_kh, xi_kv)
- :class:`HybridPhysicsSignal`  — weighted combination of both

Usage
-----
In Phase 3 the reducer outputs (xi_E', xi_kh', xi_kv') which are passed through
the training signal to obtain a predicted settlement.  The loss is then the MSE
between this prediction and the ground-truth settlement from Phase 1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class TrainingSignal(ABC):
    """Base class for training signals used in surrogate / reducer training."""

    @abstractmethod
    def evaluate(self, **kwargs) -> np.ndarray:
        """Evaluate the training signal and return predicted settlements.

        Returns
        -------
        settlements : (n_samples, n_nodes_x) or (n_nodes_x,) array
        """

    def __call__(self, **kwargs) -> np.ndarray:
        return self.evaluate(**kwargs)


# ---------------------------------------------------------------------------
# Direct physics signal
# ---------------------------------------------------------------------------

class DirectPhysicsSignal(TrainingSignal):
    """Evaluate settlements by running the Biot solver directly.

    Parameters
    ----------
    solver : BiotSolver
        Initialised solver instance.
    field_manager : FieldManager
        Used to reconstruct physical fields from coefficients.
    """

    def __init__(self, solver, field_manager) -> None:
        self._solver = solver
        self._fm = field_manager

    def evaluate(
        self,
        xi_concat: Optional[np.ndarray] = None,
        E_field: Optional[np.ndarray] = None,
        k_h_field: Optional[np.ndarray] = None,
        k_v_field: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Run solver on given fields or reconstruct them from *xi_concat*.

        Either supply pre-computed fields or xi_concat (not both).
        """
        if E_field is None:
            if xi_concat is None:
                raise ValueError("Either xi_concat or physical fields must be provided")
            fields = self._fm.reconstruct_all_fields(xi_concat)
            E_field = fields["E"]
            k_h_field = fields["k_h"]
            k_v_field = fields["k_v"]

        single = E_field.ndim == 1
        if single:
            E_field = E_field[np.newaxis, :]
            k_h_field = k_h_field[np.newaxis, :]
            k_v_field = k_v_field[np.newaxis, :]

        settlements = self._solver.run_batch(E_field, k_h_field, k_v_field)
        return settlements[0] if single else settlements


# ---------------------------------------------------------------------------
# Surrogate signal
# ---------------------------------------------------------------------------

class SurrogateSignal(TrainingSignal):
    """Evaluate settlements via the Phase-2 surrogate model.

    Parameters
    ----------
    surrogate : object with ``predict(X)`` method
        Trained Phase-2 surrogate (NN or PCE).
    """

    def __init__(self, surrogate) -> None:
        self._surrogate = surrogate

    def evaluate(
        self,
        xi_concat: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Predict settlements from reduced parameters via surrogate.

        Parameters
        ----------
        xi_concat : (n_samples, reduced_dim) or (reduced_dim,) array
        """
        if xi_concat is None:
            raise ValueError("xi_concat must be provided for SurrogateSignal")
        single = xi_concat.ndim == 1
        if single:
            xi_concat = xi_concat[np.newaxis, :]
        pred = self._surrogate.predict(xi_concat)
        return pred[0] if single else pred


# ---------------------------------------------------------------------------
# Hybrid signal
# ---------------------------------------------------------------------------

class HybridPhysicsSignal(TrainingSignal):
    """Weighted combination of direct physics and surrogate signals.

    Loss = alpha * physics_loss + (1 - alpha) * surrogate_loss

    Parameters
    ----------
    physics_signal : DirectPhysicsSignal
    surrogate_signal : SurrogateSignal
    alpha : float
        Weight on the physics loss component (0 → pure surrogate, 1 → pure physics).
    """

    def __init__(
        self,
        physics_signal: DirectPhysicsSignal,
        surrogate_signal: SurrogateSignal,
        alpha: float = 0.5,
    ) -> None:
        self._physics = physics_signal
        self._surrogate = surrogate_signal
        self.alpha = alpha

    def evaluate(
        self,
        xi_concat: Optional[np.ndarray] = None,
        E_field: Optional[np.ndarray] = None,
        k_h_field: Optional[np.ndarray] = None,
        k_v_field: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Return alpha-weighted average of physics and surrogate responses."""
        phys = self._physics.evaluate(
            xi_concat=xi_concat,
            E_field=E_field,
            k_h_field=k_h_field,
            k_v_field=k_v_field,
        )
        surr = self._surrogate.evaluate(xi_concat=xi_concat)
        return self.alpha * phys + (1.0 - self.alpha) * surr


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_training_signal(
    signal_type: str,
    solver=None,
    field_manager=None,
    surrogate=None,
    alpha: float = 0.5,
) -> TrainingSignal:
    """Construct a training signal from a string type identifier.

    Parameters
    ----------
    signal_type : "data" | "physics" | "surrogate" | "hybrid"
    solver : BiotSolver (required for physics / hybrid)
    field_manager : FieldManager (required for physics / hybrid)
    surrogate : surrogate model (required for surrogate / hybrid)
    alpha : hybrid weighting
    """
    if signal_type in ("data", "surrogate"):
        if surrogate is None:
            raise ValueError(f"signal_type='{signal_type}' requires a surrogate model")
        return SurrogateSignal(surrogate)
    elif signal_type == "physics":
        if solver is None or field_manager is None:
            raise ValueError("signal_type='physics' requires solver and field_manager")
        return DirectPhysicsSignal(solver, field_manager)
    elif signal_type == "hybrid":
        if solver is None or field_manager is None or surrogate is None:
            raise ValueError(
                "signal_type='hybrid' requires solver, field_manager, and surrogate"
            )
        phys = DirectPhysicsSignal(solver, field_manager)
        surr = SurrogateSignal(surrogate)
        return HybridPhysicsSignal(phys, surr, alpha=alpha)
    else:
        raise ValueError(f"Unknown training signal type: '{signal_type}'")
