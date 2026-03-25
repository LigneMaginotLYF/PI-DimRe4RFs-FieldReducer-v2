"""
field_manager.py
================
Unified management of the three material random fields:
  E  (Young's modulus)
  k_h (horizontal permeability)
  k_v (vertical permeability)

All three fields share the **same** fixed 2D DCT-II basis (computed once per
spatial resolution) and differ only in:
  - number of DCT terms (n_terms), which determines field dimensionality;
    n_terms == 0 means a homogeneous scalar field sampled from a log-uniform
    or log-normal distribution.
  - random seed (for reproducibility and independence)
  - Matérn covariance parameters (nu, length_scale) used to shape the spectral
    variance of the DCT coefficients.

## Why DCT and not KL?

KL expansion depends on the covariance kernel, so changing the correlation
length or smoothness changes the basis vectors themselves — making runs with
different kernels incomparable.  DCT-II is a *fixed*, physics-agnostic basis
that is consistent across all runs and all fields.  The Matérn spectral
variance is used only to *weight* the random coefficients, not to define the
basis, so any change in covariance only changes the sampling distribution,
never the basis vectors.  This guarantees that the dimension-reduction mapping
(Phase 3) is valid for any correlation structure encountered during deployment.

## Homogeneous fields (n_terms == 0)

A homogeneous (constant) random field is parameterised by a single scalar.
The FieldManager returns a (n_nodes,) array whose entries are all equal to that
scalar.  Importantly, the reducer is still expected to learn a non-identity
mapping — the scalar may need to take on a *different* value in the reduced
space so that the aggregate response (settlement) is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .utils import compute_dct_basis, matern_spectral_variance


# ---------------------------------------------------------------------------
# Per-field configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class FieldConfig:
    """Configuration for a single material random field."""

    name: str                         # "E", "k_h", or "k_v"
    n_terms: int = 0                  # 0 = homogeneous scalar
    seed: int = 0
    nu_ref: float = 1.5               # reference nu for spectral variance
    nu_sampling: bool = False
    nu_range: Tuple[float, float] = (0.5, 2.5)
    length_scale_ref: float = 0.3
    length_scale_sampling: bool = False
    length_scale_range: Tuple[float, float] = (0.1, 0.5)
    force_identity_reduction: bool = False

    # E-specific
    logE_std: float = 1.0
    E_ref: float = 10.0e6

    # k_h / k_v specific
    k_range: Tuple[float, float] = (1.0e-13, 1.0e-10)

    @property
    def effective_dim(self) -> int:
        """Number of scalar parameters representing this field.

        Even homogeneous fields (n_terms == 0) contribute exactly 1 parameter.
        """
        return max(self.n_terms, 1)

    @classmethod
    def from_dict(cls, name: str, d: Dict[str, Any]) -> "FieldConfig":
        """Construct a FieldConfig from a dict.

        Supports both the unified format (``mean``, ``range``,
        ``fluctuation_std``) and the legacy format (``E_ref``, ``logE_std``,
        ``k_range``).  Unified keys take precedence over legacy keys when
        both are present.
        """
        # --- Unified → legacy key resolution ---
        # E_ref / mean
        if "mean" in d:
            E_ref = float(d["mean"])
        else:
            E_ref = float(d.get("E_ref", 10.0e6))

        # logE_std / fluctuation_std
        if "fluctuation_std" in d:
            logE_std = float(d["fluctuation_std"])
        else:
            logE_std = float(d.get("logE_std", 1.0))

        # k_range / range
        if "range" in d:
            k_range = tuple(float(x) for x in d["range"])
        else:
            k_range = tuple(float(x) for x in d.get("k_range", [1.0e-13, 1.0e-10]))

        return cls(
            name=name,
            n_terms=int(d.get("n_terms", 0)),
            seed=int(d.get("seed", 0)),
            nu_ref=float(d.get("nu_ref", 1.5)),
            nu_sampling=d.get("nu_sampling", False),
            nu_range=tuple(float(x) for x in d.get("nu_range", [0.5, 2.5])),
            length_scale_ref=float(d.get("length_scale_ref", 0.3)),
            length_scale_sampling=d.get("length_scale_sampling", False),
            length_scale_range=tuple(float(x) for x in d.get("length_scale_range", [0.1, 0.5])),
            force_identity_reduction=d.get("force_identity_reduction", False),
            logE_std=logE_std,
            E_ref=E_ref,
            k_range=k_range,
        )


# ---------------------------------------------------------------------------
# FieldManager
# ---------------------------------------------------------------------------

class FieldManager:
    """Generate and reconstruct material random fields from DCT coefficients.

    Parameters
    ----------
    cfg : dict
        Full configuration dict (from ConfigManager.cfg).
    """

    FIELD_NAMES = ("E", "k_h", "k_v")

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg
        grid = cfg["grid"]
        self.n_nodes_x: int = grid["n_nodes_x"]
        self.n_nodes_z: int = grid["n_nodes_z"]
        self.n_nodes: int = self.n_nodes_x * self.n_nodes_z

        # Build per-field configs
        rf_cfg = cfg["random_fields"]
        self.field_configs: Dict[str, FieldConfig] = {
            name: FieldConfig.from_dict(name, rf_cfg[name])
            for name in self.FIELD_NAMES
        }

        # Per-field independent RNGs (seeded once, advanced per call)
        self._rngs: Dict[str, np.random.Generator] = {
            name: np.random.default_rng(self.field_configs[name].seed)
            for name in self.FIELD_NAMES
        }

        # Cached DCT bases — computed lazily, indexed by n_terms
        self._basis_cache: Dict[int, np.ndarray] = {}
        self._variance_cache: Dict[Tuple, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Basis access
    # ------------------------------------------------------------------

    def get_basis(self, n_terms: int) -> np.ndarray:
        """Return (n_nodes, n_terms) DCT basis matrix (L2-normalised columns)."""
        if n_terms not in self._basis_cache:
            self._basis_cache[n_terms] = compute_dct_basis(
                self.n_nodes_x, self.n_nodes_z, n_terms
            )
        return self._basis_cache[n_terms]

    def get_spectral_variance(
        self, n_terms: int, nu: float, length_scale: float
    ) -> np.ndarray:
        """Return Matérn spectral variance weights for *n_terms* DCT modes."""
        key = (n_terms, round(nu, 6), round(length_scale, 6))
        if key not in self._variance_cache:
            self._variance_cache[key] = matern_spectral_variance(
                self.n_nodes_x, self.n_nodes_z, n_terms, nu, length_scale
            )
        return self._variance_cache[key]

    # ------------------------------------------------------------------
    # Coefficient sampling
    # ------------------------------------------------------------------

    def sample_coefficients(
        self,
        n_samples: int,
        field_name: str,
    ) -> np.ndarray:
        """Sample DCT coefficients for *field_name* over *n_samples* realisations.

        Returns
        -------
        xi : (n_samples, effective_dim) float64 array
            For homogeneous fields (n_terms == 0) this is a (n_samples, 1)
            array of log-scaled scalar parameters.
            For high-dimensional fields (n_terms > 0) this is a
            (n_samples, n_terms) array of weighted standard-normal draws.
        """
        fc = self.field_configs[field_name]
        rng = self._rngs[field_name]

        if fc.n_terms == 0:
            # --- Homogeneous field: sample a single scalar ---
            return self._sample_scalar(n_samples, field_name, rng)

        # --- High-dimensional field: sample DCT coefficients ---
        # Optionally sample Matérn parameters per-realisation
        nu_vals = self._sample_nu(n_samples, fc, rng)
        ls_vals = self._sample_length_scale(n_samples, fc, rng)

        xi = np.zeros((n_samples, fc.n_terms), dtype=np.float64)
        for i in range(n_samples):
            var = self.get_spectral_variance(fc.n_terms, nu_vals[i], ls_vals[i])
            std = np.sqrt(var)
            xi[i] = rng.standard_normal(fc.n_terms) * std

        return xi

    def _sample_scalar(
        self, n_samples: int, field_name: str, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample a single scalar per realisation (log-uniform for k_h/k_v,
        log-normal for E).
        """
        fc = self.field_configs[field_name]
        if field_name == "E":
            # log-normal: E = E_ref * exp(logE_std * N(0,1))
            z = rng.standard_normal(n_samples)
            return (z * fc.logE_std).reshape(n_samples, 1)
        else:
            # log-uniform in k_range
            lo, hi = np.log10(fc.k_range[0]), np.log10(fc.k_range[1])
            log_k = rng.uniform(lo, hi, size=n_samples)
            return log_k.reshape(n_samples, 1)

    def _sample_nu(
        self, n_samples: int, fc: FieldConfig, rng: np.random.Generator
    ) -> np.ndarray:
        if fc.nu_sampling:
            return rng.uniform(fc.nu_range[0], fc.nu_range[1], size=n_samples)
        return np.full(n_samples, fc.nu_ref)

    def _sample_length_scale(
        self, n_samples: int, fc: FieldConfig, rng: np.random.Generator
    ) -> np.ndarray:
        if fc.length_scale_sampling:
            lo, hi = fc.length_scale_range
            return rng.uniform(lo, hi, size=n_samples)
        return np.full(n_samples, fc.length_scale_ref)

    # ------------------------------------------------------------------
    # Coefficient → physical field reconstruction
    # ------------------------------------------------------------------

    def reconstruct_field(
        self, xi: np.ndarray, field_name: str
    ) -> np.ndarray:
        """Map DCT coefficients *xi* to a physical field on the spatial grid.

        Parameters
        ----------
        xi : (effective_dim,) or (n_samples, effective_dim) array

        Returns
        -------
        field : (n_nodes,) or (n_samples, n_nodes) array of physical values
        """
        fc = self.field_configs[field_name]
        xi = np.asarray(xi, dtype=np.float64)
        single = xi.ndim == 1
        if single:
            xi = xi[np.newaxis, :]

        if fc.n_terms == 0:
            field = self._reconstruct_scalar(xi, field_name)
        else:
            basis = self.get_basis(fc.n_terms)  # (n_nodes, n_terms)
            field = xi @ basis.T  # (n_samples, n_nodes)
            field = self._apply_physical_transform(field, field_name)

        field = np.asarray(field, dtype=np.float64)
        if np.any(np.isnan(field)):
            raise ValueError(
                f"Field '{field_name}' contains NaN values after reconstruction."
            )
        if np.any(np.isinf(field)):
            raise ValueError(
                f"Field '{field_name}' contains Inf values after reconstruction."
            )

        return field[0] if single else field

    def _apply_physical_transform(
        self, raw: np.ndarray, field_name: str
    ) -> np.ndarray:
        """Convert the raw linear combination to physical units.

        E  : E = E_ref * exp(raw)     [log-normal]
        k_h, k_v : k = exp( log(k_mid) + raw * scale )  [log-uniform-like]
        """
        fc = self.field_configs[field_name]
        raw = np.asarray(raw, dtype=np.float64)
        if field_name == "E":
            E_ref = float(fc.E_ref)
            logE_std = float(fc.logE_std)
            return E_ref * np.exp(raw * logE_std)
        else:
            lo = float(np.log10(fc.k_range[0]))
            hi = float(np.log10(fc.k_range[1]))
            k_mid = (lo + hi) / 2.0
            k_scale = (hi - lo) / 2.0
            return 10.0 ** (k_mid + raw * k_scale)

    def _reconstruct_scalar(
        self, xi: np.ndarray, field_name: str
    ) -> np.ndarray:
        """Reconstruct a homogeneous (constant) field from a single scalar xi.

        Returns (n_samples, n_nodes) where every spatial entry is equal to
        the physical value derived from the scalar parameter.
        """
        fc = self.field_configs[field_name]
        n_samples = xi.shape[0]
        scalar = np.asarray(xi[:, 0], dtype=np.float64)

        if field_name == "E":
            E_ref = float(fc.E_ref)
            logE_std = float(fc.logE_std)
            phys = E_ref * np.exp(scalar * logE_std)
        else:
            lo = float(np.log10(fc.k_range[0]))
            hi = float(np.log10(fc.k_range[1]))
            k_mid = (lo + hi) / 2.0
            k_scale = (hi - lo) / 2.0
            phys = 10.0 ** (k_mid + scalar * k_scale)

        return np.tile(phys[:, np.newaxis], (1, self.n_nodes))

    # ------------------------------------------------------------------
    # Full dataset generation
    # ------------------------------------------------------------------

    def generate_dataset(
        self, n_samples: int
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Generate concatenated coefficient array + individual fields.

        Returns
        -------
        X : (n_samples, total_input_dim)  — concatenated coefficients
        fields : dict with keys "E", "k_h", "k_v" each (n_samples, n_nodes)
        xi_dict : dict with keys "E", "k_h", "k_v" each (n_samples, eff_dim)
        """
        xi_dict: Dict[str, np.ndarray] = {}
        for name in self.FIELD_NAMES:
            xi_dict[name] = self.sample_coefficients(n_samples, name)

        X = np.concatenate(
            [xi_dict[name] for name in self.FIELD_NAMES], axis=1
        )

        fields: Dict[str, np.ndarray] = {}
        for name in self.FIELD_NAMES:
            fields[name] = self.reconstruct_field(xi_dict[name], name)

        return X, fields, xi_dict

    # ------------------------------------------------------------------
    # Slice helpers (split concatenated X back into per-field coefficients)
    # ------------------------------------------------------------------

    @property
    def slice_E(self) -> slice:
        d_E = self.field_configs["E"].effective_dim
        return slice(0, d_E)

    @property
    def slice_kh(self) -> slice:
        d_E = self.field_configs["E"].effective_dim
        d_kh = self.field_configs["k_h"].effective_dim
        return slice(d_E, d_E + d_kh)

    @property
    def slice_kv(self) -> slice:
        d_E = self.field_configs["E"].effective_dim
        d_kh = self.field_configs["k_h"].effective_dim
        d_kv = self.field_configs["k_v"].effective_dim
        return slice(d_E + d_kh, d_E + d_kh + d_kv)

    @property
    def total_input_dim(self) -> int:
        return sum(fc.effective_dim for fc in self.field_configs.values())

    def split_coefficients(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split concatenated coefficient array into (xi_E, xi_kh, xi_kv)."""
        return X[:, self.slice_E], X[:, self.slice_kh], X[:, self.slice_kv]

    def reconstruct_all_fields(
        self, X: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Reconstruct E, k_h, k_v fields from concatenated coefficients X.

        Parameters
        ----------
        X : (n_samples, total_input_dim) or (total_input_dim,)

        Returns
        -------
        dict with keys "E", "k_h", "k_v", each (n_samples, n_nodes) or (n_nodes,)
        """
        single = X.ndim == 1
        if single:
            X = X[np.newaxis, :]
        xi_E, xi_kh, xi_kv = self.split_coefficients(X)
        result = {
            "E": self.reconstruct_field(xi_E, "E"),
            "k_h": self.reconstruct_field(xi_kh, "k_h"),
            "k_v": self.reconstruct_field(xi_kv, "k_v"),
        }
        if single:
            result = {k: v[0] for k, v in result.items()}
        return result
