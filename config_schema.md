# config_schema.md — Configuration Reference

All configuration parameters for PI-DimRe4RFs-FieldReducer-v2.

## Config formats

Two YAML layouts are supported:

**New format** (recommended):
```yaml
grid: ...
solver: ...
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

**Legacy format** (still supported, no breaking change):
```yaml
phase2: ...      # merged surrogate data-gen + model config
phase3: ...      # merged reducer data-gen + model config
collocation_phase2: ...   # DEPRECATED → use data_generation.surrogate.collocation_n_points
collocation_phase3: ...   # DEPRECATED → use data_generation.reducer.collocation_n_points
```

The new format is translated to the internal `phase2`/`phase3` representation
automatically; all downstream code is unaffected.

---

## `grid` — Spatial Discretization

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_nodes_x` | int | 20 | Number of nodes along the x-axis (horizontal) |
| `n_nodes_z` | int | 10 | Number of nodes along the z-axis (vertical / depth) |
| `lx` | float | 1.0 | Domain length in x [m] |
| `lz` | float | 0.5 | Domain depth in z [m] |

---

## `random_fields` — Material Random Field Definitions (base template)

Base field definitions used across the pipeline (Phase 1, and as the structural
template for `data_generation.surrogate.fields` and `data_generation.reducer.fields`).
Each phase-specific field block may add `mean_sampling`/`mean_range` on top of these
shared keys.

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
| `type` | str | `"1d"` | Solver dimensionality: `"1d"` or `"2d"`. |
| `mode` | str | `"steady"` | Time regime: `"steady"` or `"transient"`. |
| `nu_biot` | float | `0.3` | Poisson's ratio. |
| `fluid_viscosity` | float | `1e-3` | Fluid dynamic viscosity [Pa·s]. |
| `fluid_compressibility` | float | `4.5e-10` | Fluid compressibility [1/Pa]. |
| `load` | float | `1e4` | Applied surface load [Pa]. |
| `transient.dt` | float | `0.01` | Time step [s] (transient mode only). |
| `transient.n_steps` | int | `100` | Number of time steps (transient mode only). |

---

## `data_generation` — Dataset Generation for Both Models

### `data_generation.surrogate` — Surrogate Training Data (reduced-space)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_samples` | int | `200` | Number of reduced-space training samples. |
| `output_dir` | str | `"data/phase2"` | Directory for surrogate training data. |
| `collocation_n_points` | int | `20` | Collocation points for physics regularisation. |
| `fields` | dict | See defaults | Per-field configs for reduced-space sampling. Each field supports all keys from `random_fields` plus `mean_sampling`/`mean_range`. |

### `data_generation.reducer` — Reducer Training Data (full-space)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_samples` | int | `500` | Number of full-space training samples. |
| `output_dir` | str | `"data/phase3"` | Directory for reducer training data. |
| `collocation_n_points` | int | `5` | Collocation points for physics regularisation. |
| `fields` | dict | See defaults | Per-field configs for full-space sampling. |

---

## `models` — Model Architecture and Training

### `models.surrogate` — Surrogate (Model A)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | str | `"nn"` | `"nn"` or `"pce"`. |
| `output_repr` | str | `"direct"` | Output representation: `"direct"`, `"dct"`, `"poly"`, `"bspline"`. |
| `n_output_modes` | int | `10` | Modes for `dct`/`poly`/`bspline` representations. |
| `training_signal` | str | `"hybrid"` | `"data"`, `"physics"`, or `"hybrid"`. |
| `hybrid_alpha` | float | `0.1` | Physics weight in hybrid mode. |
| `physics_check_interval` | int | `10` | Run Biot check every N epochs (hybrid/physics). |
| `output_dir` | str | `"models/phase2_surrogate"` | Directory for surrogate artefacts. |
| `reduced_fields` | dict | See defaults | Per-field dimensionality config (n_terms defines input dim). |

#### `models.surrogate.nn` / `models.surrogate.pce`

Same as legacy `phase2.nn` / `phase2.pce`.

### `models.reducer` — Reducer (Model B)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | str | `"nn"` | Currently only `"nn"`. |
| `training_signal` | str | `"surrogate"` | `"surrogate"` or `"physics"`. |
| `output_dir` | str | `"models/phase3_reducer"` | Directory for reducer artefacts. |
| `surrogate_dir` | str | `"models/phase2_surrogate"` | Path to load Phase-2 surrogate. |
| `load_phase2_model` | str or null | `null` | Path to a specific `.pt` file; overrides auto-detection. |
| `reduced_fields` | dict or null | inherited from `models.surrogate.reduced_fields` | If omitted, inherited from the surrogate model config. |

---

## `evaluation` — Evaluation and Plotting

### `evaluation.surrogate`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `test_fraction` | float | `0.2` | Fraction of data used for evaluation. |
| `n_plot_samples` | int | `5` | Number of samples shown in comparison plots. |

### `evaluation.reducer`

Same keys as `evaluation.surrogate`.

**Settlement comparison plots** use per-sample y-axis: each subplot has its own
`[0, y_max_sample]` range, making it easy to compare profiles at different scales.

**Phase-3 settlement plot** shows three curves per sample:
1. Ground truth (full-field Biot)
2. Reducer → reduced fields → Biot (**primary evaluation**)
3. Reducer → reduced params → surrogate (**diagnostic only**)

Metrics for curves 2 and 3 are both written to `metrics.json`, with curve-3
metrics namespaced under `surrogate_*` keys.

---

## `phase4` — End-to-End Validation (legacy)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_test_samples` | int | `50` | Samples from Phase-1 validation set used for evaluation. |
| `output_dir` | str | `"results"` | Directory for metrics and plots. |
| `use_physics_for_plots` | bool | `false` | If `true`, bypass surrogate and use direct solver for ground-truth plots. |

---

## Deprecated keys

| Deprecated key | Canonical replacement | Status |
|---|---|---|
| `collocation_phase2.n_points` | `data_generation.surrogate.collocation_n_points` | DeprecationWarning emitted; still accepted |
| `collocation_phase3.n_points` | `data_generation.reducer.collocation_n_points` | DeprecationWarning emitted; still accepted |
| `phase2.*` | `data_generation.surrogate.*` / `models.surrogate.*` | Still accepted (old format) |
| `phase3.*` | `data_generation.reducer.*` / `models.reducer.*` | Still accepted (old format) |

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
| `"data"` | Phase 2 | Pure data-driven: fit surrogate to (X, Y). |
| `"physics"` | Phase 2, 3 | Direct Biot solver evaluations (accurate, slow). |
| `"surrogate"` | Phase 3 | Use Phase-2 surrogate as a differentiable oracle. |
| `"hybrid"` | Phase 2, 3 | Weighted combination: `α * physics + (1-α) * surrogate/data`. |
