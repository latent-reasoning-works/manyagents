# Diffusion Gauge Demo Notebook

**Date:** 2026-03-25
**Status:** Draft
**Author:** cesar.valdez (for Pablo Gutierrez)

## Goal

A standalone Jupyter notebook that demonstrates the manyLatents diffusion gauge on actual LLM reasoning data. Pablo wants to see what the gauge looks like on real activations from a QWEN 3 model doing chain-of-thought math reasoning.

## Approach

Direct functional calls (Approach 2) — no Hydra, no async adapters. Import building blocks from `manyagents.inference` and `manylatents.callbacks.diffusion_operator`, wire them together with inline matplotlib. Linear, readable narrative.

## Notebook: `notebooks/diffusion_gauge_demo.ipynb`

### Cell 1 — Setup

Imports and GPU check.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from manyagents.inference import (
    load_model,
    generate_with_hidden_states,
    segment,
    pool_hidden_states_per_step,
)
from manylatents.callbacks.diffusion_operator import DiffusionGauge, TrajectoryVisualizer

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen3-4B"
```

### Cell 2 — Load Model

```python
model, tokenizer = load_model(MODEL_ID, device=device)
```

Uses `inference.load_model()` which handles HuggingFace Hub download, dtype, and device placement.

### Cell 3 — Math Problem & Generation

Hardcoded MATH500-style problem (algebra/number theory with 5-10 natural reasoning steps).

```python
prompt = "Find all integers n such that n^2 + 3n + 1 is a perfect square. Show your reasoning step by step."

result = generate_with_hidden_states(
    model, tokenizer,
    prompt=prompt,
    max_new_tokens=2048,
    temperature=0.6,
    layers=None,  # all layers
)
# result["token_hidden_states"]: (n_tokens, n_layers, d_model) float16
# result["text"]: raw CoT output
```

Backup problems in comments in case the primary one produces too few/many steps.

### Cell 4 — Segment & Pool

```python
steps = segment(result["text"], method="tags")  # <think>...</think> aware
pooled = pool_hidden_states_per_step(
    result["token_hidden_states"], steps
)  # (n_steps, n_layers, d_model)

n_layers = pooled.shape[1]
print(f"{len(steps)} reasoning steps, {n_layers} layers, {pooled.shape[2]}d")
```

If `<think>` tags are absent, falls back to sentence-level segmentation within the output.

### Cell 5 — Diffusion Gauge per Layer

Three layers: early (2), mid (n_layers // 2), late (-1).

```python
gauge = DiffusionGauge(knn=None, alpha=1.0, symmetric=False)
# knn=None: global bandwidth (median pairwise distance)
# n_steps is small (5-10), k-NN adaptive bandwidth inappropriate

layer_picks = {"early": 2, "mid": n_layers // 2, "late": n_layers - 1}
operators = {}
for name, idx in layer_picks.items():
    activations = pooled[:, idx, :]   # (n_steps, d_model)
    operators[name] = gauge(activations)  # (n_steps, n_steps)
```

Each operator is a row-stochastic transition matrix on the reasoning step manifold.

### Cell 6 — Visualization

3x4 subplot grid, `figsize=(20, 12)`.

**Rows:** early / mid / late layers.

**Columns:**

1. **Heatmap** — `imshow` of the diffusion operator with diverging colormap, step labels, transition probability annotations.
2. **2D MDS embedding** — apply MDS (via `sklearn.manifold.MDS`) to the diffusion operator treated as a similarity matrix: `distance[i,j] = ||row_i - row_j||`. Scatter with step indices labeled, arrows showing sequential step order. (Note: `TrajectoryVisualizer` is designed for trajectories of operators over training time, not steps within a single operator — so we compute MDS directly here.)
3. **Eigenspectrum** — bar chart of top-k eigenvalues, spectral gap highlighted.
4. **Stationary distribution** — bar chart of pi over steps, showing which steps are representation-space attractors.

Shared title with the math problem and model name.

### Cell 7 — Summary Table

```
Layer   | Spectral Gap | Spectral Radius | Stationary Entropy | Spread
--------|-------------|-----------------|-------------------|-------
Early   | ...         | ...             | ...               | ...
Mid     | ...         | ...             | ...               | ...
Late    | ...         | ...             | ...               | ...
```

Spectral metrics computed inline:
- **Spectral gap:** `1 - lambda_2` (second eigenvalue). Larger = more separated clusters.
- **Spectral radius:** largest non-trivial eigenvalue. Mixing speed.
- **Stationary entropy:** `-sum(pi * log(pi))`. High = uniform spread, low = some steps dominate.
- **Spread:** `TrajectoryVisualizer.compute_spread()`. Average pairwise operator distance.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Model | Qwen/Qwen3-4B | Fits single A100, fast for demo. Pablo can swap to larger variant. |
| Data | Hardcoded math problem | Demo purpose — no loader needed. 1-3 problems in comments. |
| Gauge bandwidth | `knn=None` (global) | n_steps is 5-10, too few for k-NN adaptive (default k=15). |
| Normalization | Row-stochastic (`symmetric=False`) | Transition probability interpretation is more intuitive for the demo. |
| Segmentation | `method="tags"` | QWEN 3 uses `<think>` tags for CoT. Falls back to sentence-level. |
| Layers captured | All | Capture everything, pick 3 for display. Avoids re-running if Pablo wants different layers. |
| Layer picks | early=2, mid=n//2, late=-1 | Standard early/mid/late comparison. Avoids layer 0 (embedding layer, no transformation). |
| Visualization | 3x4 grid | Compact, all info visible at once, easy to compare across depth. |

## Dependencies

Already available in the environment:
- `manyagents` (this package) — `inference.py` functions
- `manylatents` — `DiffusionGauge`, `TrajectoryVisualizer`
- `torch`, `transformers` — model loading
- `matplotlib`, `numpy`, `scipy` — viz and computation
- `scikit-learn` — MDS embedding of operator rows

## File Placement

```
agents/
  notebooks/
    diffusion_gauge_demo.ipynb    # the notebook
```

## Out of Scope

- MATH500 dataset loader (not needed for hardcoded demo)
- Adapter/Hydra integration (direct calls are clearer)
- WandB logging (local notebook only)
- Zhou Reasoning-Flow comparison (separate effort)
- SLURM submission script (Pablo runs interactively on GPU node)
