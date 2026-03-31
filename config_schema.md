# config_schema.md — Configuration Reference

All configuration parameters for PI-DimRe4RFs-FieldReducer-v2.

## Config formats

Three YAML layouts are supported (all translated internally to `phase2`/`phase3`):

**Canonical format** (recommended — single source of truth per model):
```yaml
grid: ...
solver: ...
model_a:              # Model A — surrogate (reduced params → settlement)
  fields: ...         # SINGLE canonical field definition: controls both
                      # dataset generation AND the surrogate input dimension
  n_samples: 2000
  output_dir: "models/phase2_surrogate"
  type: "nn"
  nn: { ... }
  evaluation: { test_fraction: 0.1, n_plot_samples: 10 }
model_b:              # Model B — reducer (full params → reduced params)
  fields: ...         # Full-space fields for reducer INPUT (dataset gen only)
                      # Reducer OUTPUT dimension = model_a.fields dimensions (auto)
  n_samples: 5000
  output_dir: "models/phase3_reducer"
  nn: { ... }
  evaluation: { plot_mode: "three_curve", ... }
```

**Key design rule**: `model_a.fields` controls **both** dataset generation and the
surrogate's input dimension.  The reducer's output dimension is **always derived
automatically** from `model_a.fields` — you never need to specify it separately.
This eliminates the shadowing / mismatch problem.

**Intermediate format** (still accepted):
```yaml
data_generation:
  surrogate: ...    # dataset-generation params for Model A (surrogate)
  reducer: ...      # dataset-generation params for Model B (reducer)
models:
  surrogate: ...    # architecture/training params for Model A
  reducer: ...      # architecture/training params for Model B
evaluation:
  surrogate: ...
  reducer: ...
```

When both `data_generation.surrogate.fields` and `models.surrogate.reduced_fields`
are present, `data_generation.surrogate.fields` takes precedence.

**Legacy format** (still supported, no breaking change):
```yaml
phase2: ...      # merged surrogate data-gen + model config
phase3: ...      # merged reducer data-gen + model config
collocation_phase2: ...   # DEPRECATED → phase2.collocation_n_points
collocation_phase3: ...   # DEPRECATED → phase3.collocation_n_points
```

### Single-source-of-truth enforcement

Regardless of format, `phase3.reduced_fields` is **always synchronised** to equal
`phase2.reduced_fields` at runtime (in `_sync_reduced_fields()`).  If a mismatch
is detected (e.g. stale duplicate config blocks), a `UserWarning` is emitted and
the Phase-2 value wins.  This prevents the reducer→surrogate broadcast error.

---

## `grid` — Spatial Discretization

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_nodes_x` | int | 20 | Number of nodes along the x-axis (horizontal) |
| `n_nodes_z` | int | 10 | Number of nodes along the z-axis (vertical / depth) |
| `lx` | float | 1.0 | Domain length in x [m] |
| `lz` | float | 0.5 | Domain depth in z [m] |

---

## `random_fields` — Material Random Field Definitions (Phase-1 only)

Base field definitions used by Phase 1 only.  For Phase 2 and Phase 3, use
`model_a.fields` / `model_b.fields` (or `data_generation.surrogate/reducer.fields`).

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
| `mean_sampling` | bool | `false` | Enable per-sample mean sampling (see below). |
| `mean_range` | [float, float] | field-specific | Physical range `[min, max]` for mean sampling. For E: [Pa]; for k: [m²]. |

### Mean sampling

When `mean_sampling: true`, each training sample draws its mean value
independently from `mean_range` (uniform in physical space for E; log-uniform
for k fields). This produces diverse training data where both the mean level
and spatial pattern vary, which is critical for training generalising surrogates
and reducers.

| Field | Requirement | Notes |
|-------|-------------|-------|
| `E` | `fluctuation_std > 0` | logE_std = `fluctuation_std`; mean encoding requires logE_std > 0 |
| `k_h`, `k_v` | none | Mean is encoded in log10-space; works for any `fluctuation_std` |

### `E`-specific keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `logE_std` | float | `1.0` | Std of log(E); controls E variability. Alias: `fluctuation_std`. |
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
| `type` | str | `"2d"` | Solver dimensionality: `"1d"` or `"2d"`. |
| `mode` | str | `"steady"` | Time regime: `"steady"` or `"transient"`. Setting `"transient"` emits a UserWarning (transient pipeline not yet implemented). |
| `nu_biot` | float | `0.3` | Poisson's ratio. |
| `fluid_viscosity` | float | `1e-3` | Fluid dynamic viscosity [Pa·s]. |
| `fluid_compressibility` | float | `4.5e-10` | Fluid compressibility [1/Pa]. |
| `load` | float | `1e6` | Applied surface load [Pa]. |
| `transient.dt` | float | `0.01` | Time step [s] (transient mode only). |
| `transient.n_steps` | int | `100` | Number of time steps (transient mode only). |

---

## `model_a` — Surrogate (Model A, reduced params → settlement)

### `model_a.fields` — Canonical field definitions (SINGLE SOURCE OF TRUTH)

Same key schema as `random_fields`.  These settings control:
1. The **reduced-space coefficient dimension** (n_terms per field, each contributing
   max(n_terms, 1) parameters).
2. **Dataset generation** sampling for Phase 2 training data.
3. The **surrogate model's input dimension** (= sum of effective dims).
4. The **reducer model's output dimension** (inherited automatically).

Do **not** duplicate these in `models.surrogate.reduced_fields` or `model_b`
— they are the single canonical definition.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_samples` | int | `2000` | Number of reduced-space training samples. |
| `output_dir` | str | `"models/phase2_surrogate"` | Directory for training data + model artefacts. |
| `collocation_n_points` | int | `20` | Collocation points for physics regularisation. |
| `type` | str | `"nn"` | `"nn"` or `"pce"`. |
| `output_repr` | str | `"direct"` | `"direct"`, `"dct"`, `"poly"`, `"bspline"`. |
| `n_output_modes` | int | `10` | Modes for `dct`/`poly`/`bspline` output representations. |
| `training_signal` | str | `"hybrid"` | `"data"`, `"physics"`, or `"hybrid"`. |
| `hybrid_alpha` | float | `0.1` | Physics weight in hybrid mode. |
| `physics_check_interval` | int | `10` | Run Biot check every N epochs (hybrid/physics). |
| `nn` | dict | see defaults | NN hyperparams: `hidden_dims`, `epochs`, `lr`, `batch_size`, `patience`. |
| `pce` | dict | `{degree: 3}` | PCE hyperparams. |

#### `model_a.evaluation`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `test_fraction` | float | `0.1` | Fraction of data used for evaluation. |
| `n_plot_samples` | int | `10` | Number of samples shown in comparison plots. |

---

## `model_b` — Reducer (Model B, full params → reduced params)

### `model_b.fields` — Full-space field definitions

Controls the reducer **input** space (full-dimensional fields).  Same key schema
as `random_fields`.  The reducer **output** dimension is automatically derived from
`model_a.fields` — do not set a `reduced_fields` key here.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_samples` | int | `5000` | Number of full-space training samples. |
| `output_dir` | str | `"models/phase3_reducer"` | Directory for training data + model artefacts. |
| `collocation_n_points` | int | `20` | Collocation points for physics regularisation. |
| `surrogate_dir` | str | `"models/phase2_surrogate"` | Path to load the Phase-2 surrogate. |
| `load_phase2_model` | str or null | `null` | Path to a specific `.pt` file; overrides auto-detection. |
| `type` | str | `"nn"` | Currently only `"nn"`. |
| `training_signal` | str | `"surrogate"` | `"surrogate"` or `"physics"`. |
| `nn` | dict | see defaults | NN hyperparams: `hidden_dims`, `epochs`, `lr`, `batch_size`, `patience`. |

#### `model_b.evaluation`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `test_fraction` | float | `0.1` | Fraction of data used for evaluation. |
| `n_plot_samples` | int | `10` | Number of samples shown in comparison plots. |
| `plot_mode` | str | `"three_curve"` | Settlement comparison figure: `"two_curve"` (GT + Biot) or `"three_curve"` (GT + Biot + Surrogate). |

**`plot_mode` details**:
- `"two_curve"`: Plots ground truth (GT) and reducer→Biot path.  Always available.
- `"three_curve"`: Adds the reducer→Surrogate path as a third curve.  Requires a trained Phase-2 surrogate to be passed to `Phase3Evaluator.run()`.

---

## `phase4` — End-to-End Validation (legacy)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_test_samples` | int | `50` | Samples from Phase-1 validation set used for evaluation. |
| `output_dir` | str | `"results"` | Directory for metrics and plots. |
| `use_physics_for_plots` | bool | `true` | If `true`, bypass surrogate and use direct solver for ground-truth plots. |
| `random_seed` | int | `0` | Seed for reproducible random sampling of test set. Change to get a different set of test samples. |
| `shuffle` | bool | `true` | If `true`, randomise sample order using `random_seed`. If `false`, take first `n_test_samples` in order. |

---

## Deprecated keys

| Deprecated key | Canonical replacement | Status |
|---|---|---|
| `collocation_phase2.n_points` | `model_a.collocation_n_points` | DeprecationWarning; still accepted |
| `collocation_phase3.n_points` | `model_b.collocation_n_points` | DeprecationWarning; still accepted |
| `data_generation.surrogate.*` | `model_a.*` | Intermediate format; still accepted |
| `data_generation.reducer.*` | `model_b.*` | Intermediate format; still accepted |
| `models.surrogate.*` | `model_a.*` | Intermediate format; still accepted |
| `models.reducer.*` | `model_b.*` | Intermediate format; still accepted |
| `phase2.*` | `model_a.*` | Legacy format; still accepted |
| `phase3.*` | `model_b.*` | Legacy format; still accepted |

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
| `"data"` | Model A | Pure data-driven: fit surrogate to (X, Y). |
| `"physics"` | Model A, B | Direct Biot solver evaluations (accurate, slow). |
| `"surrogate"` | Model B | Use Phase-2 surrogate as a differentiable oracle. |
| `"hybrid"` | Model A, B | Weighted combination: `α * physics + (1-α) * surrogate/data`. |
