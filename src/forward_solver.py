"""
forward_solver.py
=================
Biot consolidation solver interface supporting 1-D and 2-D geometries in both
steady-state and transient modes.

The solver accepts physical material fields (E, k_h, k_v) discretised on the
shared spatial grid and returns the surface settlement profile at n_nodes_x
surface nodes.

## Physics Background

Biot's consolidation equations couple:
  - Mechanical equilibrium (linear elasticity with pore pressure)
  - Fluid flow (Darcy's law + fluid mass conservation)

In the simplified forms implemented here:

### 1-D Steady-State (Terzaghi-like consolidation column)
  σ'(z) + dp/dz = 0
  k / γ_w * d²p/dz² = 0  (steady: no time derivative)
  Applied surface load q → settlement s = q * H / E  (oedometric)

For a heterogeneous column the settlement is computed using the principle of
virtual work under drained conditions:

  s(x) = ∫₀ᴴ ε(x,z) dz  where  ε(x,z) = σ'(z) / E(x,z)

### 2-D (plane strain, simplified)
  The 2-D solver uses a simplified FEM-like assembly; for a computationally
  tractable placeholder it integrates the vertical strain over the depth for
  each x-column.

All computations are performed with double precision float64.

Note
----
This is a pedagogical / research implementation.  For production use replace
the solver body with a fully validated finite-element solver.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class BiotSolver:
    """Biot consolidation solver.

    Parameters
    ----------
    cfg : dict
        Full configuration dict.  Only the 'solver' and 'grid' sub-dicts are
        used.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg
        grid = cfg["grid"]
        solver_cfg = cfg["solver"]

        self.n_nodes_x: int = grid["n_nodes_x"]
        self.n_nodes_z: int = grid["n_nodes_z"]
        self.lx: float = grid["lx"]
        self.lz: float = grid["lz"]

        self.solver_type: str = solver_cfg.get("type", "1d")
        self.mode: str = solver_cfg.get("mode", "steady")
        self.nu_biot: float = solver_cfg.get("nu_biot", 0.3)
        self.fluid_viscosity: float = solver_cfg.get("fluid_viscosity", 1.0e-3)
        self.fluid_compressibility: float = solver_cfg.get("fluid_compressibility", 4.5e-10)
        self.load: float = solver_cfg.get("load", 1.0e4)
        self.dt: float = solver_cfg.get("transient", {}).get("dt", 0.01)
        self.n_steps: int = solver_cfg.get("transient", {}).get("n_steps", 100)

        self.dz: float = self.lz / max(self.n_nodes_z - 1, 1)
        self.dx: float = self.lx / max(self.n_nodes_x - 1, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_inputs(
        self,
        E_field: np.ndarray,
        k_h_field: np.ndarray,
        k_v_field: np.ndarray,
    ) -> tuple:
        """Validate and coerce solver inputs to float64.

        Parameters
        ----------
        E_field, k_h_field, k_v_field : array-like, shape (n_nodes,)

        Returns
        -------
        Tuple of validated (E_field, k_h_field, k_v_field) as float64 arrays.

        Raises
        ------
        ValueError
            If any field contains NaN/Inf or non-positive values, or has the
            wrong shape.
        """
        expected_size = self.n_nodes_x * self.n_nodes_z
        validated = []
        for name, field in (
            ("E", E_field),
            ("k_h", k_h_field),
            ("k_v", k_v_field),
        ):
            field = np.asarray(field, dtype=np.float64)
            if field.size != expected_size:
                raise ValueError(
                    f"{name}_field has {field.size} elements, "
                    f"expected {expected_size} (n_nodes_x={self.n_nodes_x} "
                    f"× n_nodes_z={self.n_nodes_z})"
                )
            field = field.ravel()
            if np.any(np.isnan(field)):
                raise ValueError(f"{name}_field contains NaN values")
            if np.any(np.isinf(field)):
                raise ValueError(f"{name}_field contains Inf values")
            if np.any(field <= 0):
                raise ValueError(
                    f"{name}_field must be strictly positive; "
                    f"got min={np.min(field):.3g}"
                )
            validated.append(field)
        return tuple(validated)

    def run(
        self,
        E_field: np.ndarray,
        k_h_field: np.ndarray,
        k_v_field: np.ndarray,
    ) -> np.ndarray:
        """Run the Biot solver for a single set of material fields.

        Parameters
        ----------
        E_field : (n_nodes,) array  — Young's modulus at each grid node [Pa]
        k_h_field : (n_nodes,) array  — horizontal permeability [m²]
        k_v_field : (n_nodes,) array  — vertical permeability [m²]

        Returns
        -------
        settlement : (n_nodes_x,) array  — surface settlement at each x node [m]
        """
        E_field, k_h_field, k_v_field = self.validate_inputs(
            E_field, k_h_field, k_v_field
        )
        E = E_field.reshape(self.n_nodes_x, self.n_nodes_z)
        k_h = k_h_field.reshape(self.n_nodes_x, self.n_nodes_z)
        k_v = k_v_field.reshape(self.n_nodes_x, self.n_nodes_z)

        if self.solver_type == "1d":
            return self._solve_1d(E, k_h, k_v)
        else:
            return self._solve_2d(E, k_h, k_v)

    def run_batch(
        self,
        E_fields: np.ndarray,
        k_h_fields: np.ndarray,
        k_v_fields: np.ndarray,
    ) -> np.ndarray:
        """Vectorised wrapper around :meth:`run`.

        Parameters
        ----------
        E_fields : (n_samples, n_nodes) array
        k_h_fields : (n_samples, n_nodes) array
        k_v_fields : (n_samples, n_nodes) array

        Returns
        -------
        settlements : (n_samples, n_nodes_x) array
        """
        n_samples = E_fields.shape[0]
        settlements = np.zeros((n_samples, self.n_nodes_x), dtype=np.float64)
        for i in range(n_samples):
            settlements[i] = self.run(E_fields[i], k_h_fields[i], k_v_fields[i])
        return settlements

    # ------------------------------------------------------------------
    # 1-D solver
    # ------------------------------------------------------------------

    def _solve_1d(
        self,
        E: np.ndarray,
        k_h: np.ndarray,
        k_v: np.ndarray,
    ) -> np.ndarray:
        """1-D oedometric consolidation column for each x-node.

        For each x-column the effective-stress approach gives:

            ε(x,z) = q / E(x,z)            (oedometric strain)
            s(x)   = ∫₀ᴴ ε(x,z) dz

        Permeability k_v controls the time-to-consolidation; in steady state
        we compute the *drained* settlement (final state).  Transient mode
        uses the 1-D Terzaghi equation.
        """
        if self.mode == "steady":
            return self._steady_1d(E, k_v)
        else:
            return self._transient_1d(E, k_v)

    def _steady_1d(self, E: np.ndarray, k_v: np.ndarray) -> np.ndarray:
        """Drained steady-state settlement (oedometric)."""
        # E shape: (n_nodes_x, n_nodes_z)
        # Oedometric strain at each node
        q = self.load
        strain = q / E  # (n_nodes_x, n_nodes_z)
        # Trapezoidal integration over z
        z_coords = np.linspace(0.0, self.lz, self.n_nodes_z)
        settlement = np.trapezoid(strain, z_coords, axis=1)  # (n_nodes_x,)
        return settlement

    def _transient_1d(self, E: np.ndarray, k_v: np.ndarray) -> np.ndarray:
        """1-D Terzaghi consolidation: return settlement at t = n_steps * dt.

        Uses explicit finite differences for the diffusion equation:
            ∂u/∂t = c_v ∂²u/∂z²
        where c_v = k_v * E / γ_w  (coefficient of consolidation).
        """
        gamma_w = 9810.0  # water unit weight [N/m³]
        dz = self.dz
        dt = self.dt
        n_steps = self.n_steps
        nz = self.n_nodes_z
        nx = self.n_nodes_x
        q = self.load

        # Column-average c_v for each x-column
        c_v = k_v * E / gamma_w  # (nx, nz)

        settlements = np.zeros(nx, dtype=np.float64)
        for ix in range(nx):
            c_v_col = c_v[ix]  # (nz,)
            E_col = E[ix]      # (nz,)
            k_v_col = k_v[ix]  # (nz,)

            # Initial excess pore pressure = applied load
            u = np.full(nz, q, dtype=np.float64)
            u[0] = 0.0     # drained at surface
            u[-1] = 0.0    # drained at base

            c_v_mean = np.mean(c_v_col)
            r = c_v_mean * dt / (dz ** 2)
            if r > 0.5:
                # Reduce dt internally if stability violated
                r = 0.49

            for _ in range(n_steps):
                u_new = u.copy()
                u_new[1:-1] += r * (u[:-2] - 2 * u[1:-1] + u[2:])
                u_new[0] = 0.0
                u_new[-1] = 0.0
                u = u_new

            # Settlement at this time step
            effective_stress = q - u
            strain = effective_stress / E_col
            z_coords = np.linspace(0.0, self.lz, nz)
            settlements[ix] = np.trapezoid(strain, z_coords)

        return settlements

    # ------------------------------------------------------------------
    # 2-D solver
    # ------------------------------------------------------------------

    def _solve_2d(
        self,
        E: np.ndarray,
        k_h: np.ndarray,
        k_v: np.ndarray,
    ) -> np.ndarray:
        """Simplified 2-D plane-strain Biot solver.

        Implements a column-by-column 1-D integration approach as a tractable
        approximation.  In a full FEM implementation this would assemble the
        global stiffness and flow matrices.

        The settlement at surface node *x* is:

            s(x) = ∫₀ᴴ σ'_zz(x,z) / M(x,z)  dz

        where M is the constrained modulus  M = E(1-ν) / ((1+ν)(1-2ν)).
        """
        nu = self.nu_biot
        M = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))

        if self.mode == "steady":
            return self._steady_2d(M)
        else:
            return self._transient_1d(E, k_v)

    def _steady_2d(self, M: np.ndarray) -> np.ndarray:
        """Drained 2-D settlement (column integration with constrained modulus)."""
        q = self.load
        strain = q / M
        z_coords = np.linspace(0.0, self.lz, self.n_nodes_z)
        return np.trapezoid(strain, z_coords, axis=1)
