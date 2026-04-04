# Session Notes — 2026-04-03

## Issue: Kernel crash after Logistic Regression cell (`phase_3_wide_and_deep.ipynb`)

### Symptom
- LR cell printed results successfully, then kernel crashed
- Jupyter error appeared above the next cell
- Crash was reproducible

---

## Root Cause Analysis

### Attempt 1: OpenMP via `solver="saga"` (partially correct)
- `saga` solver uses OpenMP-backed C extensions internally
- On macOS, Jupyter uses a fork-based process model that is unsafe with OpenMP threads
- Same root cause as the existing `n_jobs=1` fix on XGBoost in the same notebook
- **Fix applied:** changed `solver="saga"` → `solver="lbfgs"` (scipy-based, fork-safe)

### Attempt 2: OOM (ruled out)
- Memory usage checked: **46% (9.3 GB / 34.4 GB)** — plenty of headroom
- OOM eliminated as cause

### Attempt 3: Thread env vars not set before library imports (root cause)
- `lbfgs` still calls into numpy/scipy which use OpenMP/BLAS/MKL threads internally
- Env vars limiting threads must be set **before** numpy/sklearn/torch are imported — these libraries read thread counts at import time
- Setting them after import has no effect

---

## Fixes Applied

### 1. `phase_3_wide_and_deep.ipynb` — Cell 27 (LR cell)
```python
# Before
lr_model = LogisticRegression(
    max_iter=1000, class_weight="balanced", solver="saga", random_state=SEED
)

# After
lr_model = LogisticRegression(
    max_iter=1000, class_weight="balanced",
    solver="lbfgs",  # WHY: saga uses OpenMP internally -> crashes macOS Jupyter via fork; lbfgs uses scipy (fork-safe)
    random_state=SEED
)
```

### 2. `phase_3_wide_and_deep.ipynb` — Cell 2 (imports cell, top of notebook)
Added before all imports:
```python
# WHY: must be set BEFORE numpy/sklearn/torch import — these libs read thread counts at import time
# Prevents OpenMP fork-safety crash in macOS Jupyter (same root cause as XGBoost n_jobs=1)
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # macOS Accelerate framework
os.environ["NUMEXPR_NUM_THREADS"] = "1"
```

**Requires kernel restart + run all** to take effect.

---

## Other Discussion

### BERT embeddings ablation
- Concern: BERT embeddings may not add meaningful signal
- Key risks identified:
  - **Dimensionality dominance** — 768-dim embeddings numerically dwarf tabular features
  - **Noise injection** — generic (non-fine-tuned) encoder adds variance without domain signal
  - **Leakage risk** — text may implicitly encode the label
- Ablation not yet run; suggested interpreting results by AUC vs F1 delta

---

## Pattern to Remember
Any sklearn/numpy/torch operation on macOS Jupyter can crash via OpenMP fork unsafety. Always set thread env vars at the very top of the notebook, before any scientific library imports.
