# PI-DimRe4RFs-FieldReducer-v2

**Physics-Informed Dimension Reduction for Material Random Fields — Version 2**

A comprehensive, architecturally clean framework for constructing dimension reduction mappings for material random fields using a multi-field approach.  Handles Young's modulus (**E**), horizontal permeability (**k_h**), and vertical permeability (**k_v**) as parameterised fields with flexible dimensionality — from homogeneous scalars to high-dimensional random fields.

---

## Architecture Overview

```
X = [ξ_E | ξ_kh | ξ_kv]       ← full-dim coefficients (Phase 1)
         ↓  Phase-3 Reducer
X' = [ξ'_E | ξ'_kh | ξ'_kv]  ← reduced coefficients
         ↓  Phase-2 Surrogate or Biot Solver
Y = settlement profile (n_nodes_x,)
```

### Key Design Decisions

1. **Fixed 2D DCT-II basis** for all fields — unlike KL expansion, the DCT basis is independent of the covariance kernel, ensuring consistent physical meaning across different runs and different correlation structures.

2. **Independent field configurations** — E, k_h, k_v each have their own random seed, Matérn covariance parameters, and number of DCT terms (`n_terms`).

3. **Homogeneous fields as a special case** — setting `n_terms = 0` gives a single scalar parameter.  The reducer still learns a non-identity mapping because the aggregate settlement response depends on all three fields jointly.

4. **Physics-driven training abstraction** — all phases support `"data"`, `"physics"`, `"surrogate"`, and `"hybrid"` training signals via a clean `TrainingSignal` base class.

5. **Phase independence** — Phase 2 and Phase 3 can be run separately; models are saved and reloaded with config-hash validation.

---

## Directory Structure

```
PI-DimRe4RFs-FieldReducer-v2/
├── src/
│   ├── __init__.py
│   ├── config_manager.py         # YAML loading, validation, defaults
│   ├── field_manager.py          # Unified E, k_h, k_v DCT field generation
│   ├── forward_solver.py         # Biot solver (1D/2D, steady/transient)
│   ├── phase1_dataset.py         # Phase 1: dataset generation
│   ├── phase2_surrogate.py       # Phase 2: LUT + surrogate training
│   ├── phase3_reducer.py         # Phase 3: dimension reducer training
│   ├── phase4_validation.py      # Phase 4: metrics, plots, sensitivity
│   ├── surrogate_models.py       # NN and PCE surrogates
│   ├── training_schema.py        # TrainingSignal abstractions
│   └── utils.py                  # DCT basis, metrics, plots
├── scripts/
│   ├── train_full.py             # Run all 4 phases
│   ├── train_phase2_only.py      # Phases 1+2 (independent surrogate)
│   ├── train_phase3_only.py      # Phase 3 only (loads pre-trained Phase-2)
│   └── validate.py               # Phase 4 validation only
├── tests/
│   ├── conftest.py
│   ├── test_field_manager.py
│   ├── test_phase1.py
│   ├── test_phase2.py
│   ├── test_phase3.py
│   ├── test_surrogate_models.py
│   ├── test_training_schema.py
│   └── test_integration.py
├── config.yaml                   # Default configuration
├── config_schema.md              # Full config reference
├── requirements.txt
└── .gitignore
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Run the full 4-phase pipeline
python scripts/train_full.py --config config.yaml

# Or run phases independently
python scripts/train_phase2_only.py   # Phases 1+2
python scripts/train_phase3_only.py   # Phase 3 (requires Phase 2)
python scripts/validate.py            # Phase 4 (requires Phases 2+3)
```

---

## Configuration

See [`config_schema.md`](config_schema.md) for a full parameter reference.

Key config sections:

```yaml
random_fields:
  E:     { n_terms: 5, seed: 42, ... }   # high-dimensional random field
  k_h:   { n_terms: 0, seed: 43, ... }   # homogeneous scalar
  k_v:   { n_terms: 2, seed: 44, ... }   # 2-dimensional random field

phase2:
  surrogate_type: "nn"    # "nn" or "pce"
  output_repr: "direct"   # "direct" | "dct" | "poly" | "bspline"
  training_signal: "data" # "data" | "physics" | "hybrid"

phase3:
  training_signal: "surrogate"  # "surrogate" | "physics" | "hybrid"
```

---

## Testing

```bash
python -m pytest tests/ -v
```

---

## Mathematical Background

### DCT Basis vs KL Expansion

The KL basis depends on the covariance kernel — changing ν or ℓ changes the basis vectors themselves, making runs with different kernels incomparable.  The 2D DCT-II basis is **fixed** regardless of the covariance structure.  The Matérn spectral variance shapes only the **sampling distribution** of coefficients, not the basis.

This guarantees the dimension-reduction mapping (Phase 3) is valid for any correlation structure encountered during deployment.

### Homogeneous Fields and Non-Identity Reduction

A homogeneous field (`n_terms = 0`) is parameterised by a single scalar.  The reducer is still expected to learn a **non-identity** mapping: even if k_h is a scalar constant in the original space, the reduced-space parameter ξ'_kh may take a different value that better satisfies the target settlement response when combined with the other reduced fields.

### Matérn Spectral Variance

For a 2D Matérn(ν, ℓ) kernel, the spectral density at frequency ω is:

    S(ω) ∝ (2ν/ℓ² + ‖ω‖²)^{-(ν+1)}

The first `n_terms` DCT modes are ordered by ascending frequency magnitude; their spectral weights shape the standard deviation of the sampled coefficients.

### Physics-Driven Training Signals

| Signal | Phase 2 | Phase 3 | Differentiable |
|--------|---------|---------|----------------|
| `"data"` | ✓ | — | ✓ |
| `"surrogate"` | — | ✓ | ✓ (through NN surrogate) |
| `"physics"` | ✓ | ✓ | ✗ (ES gradient estimator) |
| `"hybrid"` | ✓ | ✓ | Partially |

For non-differentiable signals (direct Biot solver), Phase 3 uses an **Evolution Strategies (ES)** gradient estimator with random parameter perturbations.
