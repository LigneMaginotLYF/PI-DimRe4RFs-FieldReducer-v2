# config_schema.md — Configuration Reference

All configuration parameters for PI-DimRe4RFs-FieldReducer-v2.

---

## `grid` — Spatial Discretization

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_nodes_x` | int | 20 | Number of nodes along the x-axis (horizontal) |
| `n_nodes_z` | int | 10 | Number of nodes along the z-axis (vertical / depth) |
| `lx` | float | 1.0 | Domain length in x [m] |
| `lz` | float | 0.5 | Domain depth in z [m] |

---

## `random_fields` — Material Random Field Definitions

All three fields (`E`, `k_h`, `k_v`) share the same 2D DCT-II basis and differ only in the parameters below.

### Common keys (all fields)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_terms` | int | — | Number of DCT modes. `0` = homogeneous scalar field. |
| `basis` | str | `"dct"` | Always `"dct"`; kept for documentation clarity. |
| `seed` | int | — | Random seed for reproducibility of this field's sampling. |
| `covariance` | str | `"matern"` | Covariance kernel type (`"matern"` only in v2). |
| `nu_sampling` | bool | `false` | Sample smoothness parameter ν per realisation. |
| `nu_range` | [float, float] | `[0.5, 2.5]` | Uniform range for ν when `nu_sampling=true`. |
| `nu_ref` | float | `1.5` | Reference ν for spectral variance (always used for basis). |
| `length_scale_sampling` | bool | `false` | Sample length-scale ℓ per realisation. |
| `length_scale_range` | [float, float] | `[0.1, 0.5]` | Uniform range for ℓ when sampling. |
| `length_scale_ref` | float | `0.3` | Reference ℓ for spectral variance. |
| `force_identity_reduction` | bool | `false` | If `true`, Phase-3 mapper is encouraged to be identity. |

### `E`-specific keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `logE_std` | float | `1.0` | Std of log(E); controls E variability. |
| `E_ref` | float | `10.0e6` | Reference Young's modulus [Pa]. |

Physical mapping: `E(x) = E_ref * exp(ξ · Φ · logE_std)`

### `k_h` and `k_v`-specific keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `k_range` | [float, float] | `[1e-13, 1e-10]` | [min, max] permeability [m²]. |

Physical mapping: `k(x) = 10^(k_mid + ξ · Φ · k_scale)` where `k_mid = (log₁₀ k_max + log₁₀ k_min)/2`.

---

## `solver` — Biot Consolidation Solver

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | str | `"1d"` | Solver dimensionality: `"1d"` or `"2d"`. |
| `mode` | str | `"steady"` | Time regime: `"steady"` or `"transient"`. |
| `nu_biot` | float | `0.3` | Poisson's ratio. |
| `fluid_viscosity` | float | `1e-3` | Fluid dynamic viscosity [Pa·s]. |
| `fluid_compressibility` | float | `4.5e-10` | Fluid compressibility [1/Pa]. |
| `load` | float | `1e4` | Applied surface load [Pa]. |
| `transient.dt` | float | `0.01` | Time step [s] (transient mode). |
| `transient.n_steps` | int | `100` | Number of time steps (transient mode). |

---

## `phase1` — Dataset Generation

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_samples` | int | `200` | Total number of random field realisations. |
| `val_fraction` | float | `0.2` | Fraction of samples reserved for validation. |
| `output_dir` | str | `"data"` | Directory for saving Phase-1 artefacts. |

**Outputs:** `X_train.npy`, `Y_train.npy`, `X_val.npy`, `Y_val.npy`, `dataset_metadata.json`

---

## `phase2` — LUT and Surrogate Training

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `surrogate_type` | str | `"nn"` | `"nn"` (neural network) or `"pce"` (polynomial chaos). |
| `output_repr` | str | `"direct"` | Output representation: `"direct"`, `"dct"`, `"poly"`, `"bspline"`. |
| `n_output_modes` | int | `10` | Modes for `dct`/`poly`/`bspline` representations. |
| `training_signal` | str | `"data"` | `"data"`, `"physics"`, or `"hybrid"`. |
| `hybrid_alpha` | float | `0.5` | Physics weight in hybrid mode. |
| `output_dir` | str | `"models/phase2_surrogate"` | Directory for surrogate artefacts. |

### `phase2.nn` — Neural Network Config

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `hidden_dims` | list[int] | `[128, 128, 64]` | Hidden layer sizes. |
| `epochs` | int | `500` | Maximum training epochs. |
| `lr` | float | `1e-3` | Learning rate. |
| `batch_size` | int | `32` | Mini-batch size. |
| `patience` | int | `50` | Early stopping patience (validation loss). |

### `phase2.pce` — PCE Config

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `degree` | int | `3` | Maximum total polynomial degree. |

---

## `collocation_phase2` — Phase-2 Collocation

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_points` | int | `20` | Number of LUT grid points (dense for diagnostics). |

---

## `phase3` — Dimension Reducer Training

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `reducer_type` | str | `"nn"` | Currently only `"nn"`. |
| `training_signal` | str | `"surrogate"` | `"surrogate"`, `"physics"`, or `"hybrid"`. |
| `hybrid_alpha` | float | `0.5` | Physics weight in hybrid mode. |
| `output_dir` | str | `"models/phase3_reducer"` | Directory for reducer artefacts. |
| `surrogate_dir` | str | `"models/phase2_surrogate"` | Path to load Phase-2 surrogate. |

### `phase3.nn` — Reducer NN Config

Same keys as `phase2.nn`: `hidden_dims`, `epochs`, `lr`, `batch_size`, `patience`.

---

## `collocation_phase3` — Phase-3 Collocation

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_points` | int | `5` | Number of collocation points (sparse for training efficiency). |

---

## `phase4` — Validation

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_test_samples` | int | `50` | Samples from Phase-1 validation set used for evaluation. |
| `output_dir` | str | `"results"` | Directory for metrics and plots. |
| `use_physics_for_plots` | bool | `false` | If `true`, bypass surrogate and use direct solver for ground-truth plots. |

---

## Output Representations

| Value | Description |
|-------|-------------|
| `"direct"` | Surrogate outputs settlement at all `n_nodes_x` nodes directly. |
| `"dct"` | Surrogate outputs first `n_output_modes` DCT-II coefficients; inverse DCT reconstructs the profile. |
| `"poly"` | Surrogate outputs polynomial coefficients (degree = `n_output_modes - 1`); evaluated at `n_nodes_x` nodes. |
| `"bspline"` | Surrogate outputs `n_output_modes` B-spline control points; evaluated at `n_nodes_x` nodes. |

---

## Training Signals

| Value | Used in | Description |
|-------|---------|-------------|
| `"data"` | Phase 2 | Pure data-driven: fit surrogate to (X, Y) from Phase 1. |
| `"physics"` | Phase 2, 3 | Direct Biot solver evaluations (accurate, slow). |
| `"surrogate"` | Phase 3 | Use Phase-2 surrogate as a differentiable oracle. |
| `"hybrid"` | Phase 2, 3 | Weighted combination: `α * physics + (1-α) * surrogate/data`. |
