# Diffusion Gauge Demo Notebook — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a standalone Jupyter notebook showing the manyLatents diffusion gauge applied to QWEN 3 hidden states during math CoT reasoning, with cross-layer comparison.

**Architecture:** Direct functional calls — import `load_model`, `build_prompt`, `generate_with_hidden_states`, `segment`, `pool_hidden_states_per_step` from `manyagents.inference`, `DiffusionGauge` from `manylatents.callbacks.diffusion_operator`. Wire in a linear 7-cell notebook with inline matplotlib visualization. No Hydra, no async.

**Tech Stack:** PyTorch, transformers, manyagents.inference, manylatents DiffusionGauge, matplotlib, scikit-learn (MDS), numpy, scipy

**Branch:** `feat/diffusion-gauge-demo`

---

## File Structure

```
agents/
  notebooks/
    diffusion_gauge_demo.ipynb    # NEW — the demo notebook (7 cells)
```

Single file. No supporting modules needed — everything is imported from existing packages.

---

### Task 1: Create branch and notebook skeleton

**Files:**
- Create: `notebooks/diffusion_gauge_demo.ipynb`

- [ ] **Step 1: Create feature branch**

```bash
git checkout -b feat/diffusion-gauge-demo
```

- [ ] **Step 2: Create notebooks directory**

```bash
mkdir -p notebooks
```

- [ ] **Step 3: Create notebook with Cell 1 — Setup**

Create `notebooks/diffusion_gauge_demo.ipynb` with a single markdown cell (title + description) and Cell 1 (imports + GPU check):

**Markdown cell:**
```markdown
# Diffusion Gauge on QWEN 3 Reasoning Traces

Demonstrates the manyLatents **diffusion gauge** on hidden-state activations
extracted from a QWEN 3 model during chain-of-thought math reasoning.

**Pipeline:** Load model → Generate CoT with hidden states → Segment into
reasoning steps → Pool hidden states per step → Apply diffusion gauge per
layer → Visualize cross-layer comparison.

**Requirements:** GPU node (A100 recommended). Run with `uv run jupyter lab`.
```

**Code cell 1 — Setup:**
```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.manifold import MDS
from scipy import linalg

from manyagents.inference import (
    load_model,
    build_prompt,
    generate_with_hidden_states,
    segment,
    pool_hidden_states_per_step,
)
from manylatents.callbacks.diffusion_operator import DiffusionGauge

assert torch.cuda.is_available(), "This notebook requires a GPU"
device = "cuda"
MODEL_ID = "Qwen/Qwen3-4B"
print(f"Using {MODEL_ID} on {torch.cuda.get_device_name()}")
```

- [ ] **Step 4: Verify notebook is valid JSON**

```bash
uv run python -c "import json; json.load(open('notebooks/diffusion_gauge_demo.ipynb'))"
```

Expected: no output (valid JSON).

- [ ] **Step 5: Commit**

```bash
git add notebooks/diffusion_gauge_demo.ipynb
git commit -m "feat: scaffold diffusion gauge demo notebook with imports"
```

---

### Task 2: Model loading cell

**Files:**
- Modify: `notebooks/diffusion_gauge_demo.ipynb`

- [ ] **Step 1: Add Cell 2 — Load Model**

```python
model, tokenizer, _ = load_model(MODEL_ID, device_map="auto")
print(f"Loaded {MODEL_ID}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
print(f"  Layers: {model.config.num_hidden_layers}")
print(f"  Hidden dim: {model.config.hidden_size}")
```

Note: `load_model` returns a 3-tuple `(model, tokenizer, hf_module)`. We discard the third element.

- [ ] **Step 2: Verify notebook is valid JSON**

```bash
uv run python -c "import json; json.load(open('notebooks/diffusion_gauge_demo.ipynb'))"
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/diffusion_gauge_demo.ipynb
git commit -m "feat: add model loading cell to demo notebook"
```

---

### Task 3: Math problem and CoT generation cell

**Files:**
- Modify: `notebooks/diffusion_gauge_demo.ipynb`

- [ ] **Step 1: Add Cell 3 — Math Problem & Generate**

```python
MATH_PROBLEM = (
    "Find all integers n such that n^2 + 3n + 1 is a perfect square. "
    "Show your reasoning step by step."
)

# Backup problems if this one produces too few steps:
# "Prove that for any prime p > 3, p^2 - 1 is divisible by 24."
# "Find the last three digits of 7^2025."

prompt = build_prompt(tokenizer, MATH_PROBLEM)

print("Generating CoT (this may take a minute)...")
result = generate_with_hidden_states(
    model,
    tokenizer,
    prompt,
    max_new_tokens=2048,
    temperature=0.6,
    layers=None,  # capture ALL layers
)

print(f"Generated {result['n_new_tokens']} tokens in {result['generation_time_ms']}ms")
print(f"Hidden states shape: {result['token_hidden_states'].shape}")
print(f"  (n_tokens, n_layers, d_model)")
print()
print("--- Response ---")
print(result["text"][:2000])
```

- [ ] **Step 2: Verify notebook is valid JSON**

```bash
uv run python -c "import json; json.load(open('notebooks/diffusion_gauge_demo.ipynb'))"
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/diffusion_gauge_demo.ipynb
git commit -m "feat: add CoT generation cell with hidden state extraction"
```

---

### Task 4: Segmentation and pooling cell

**Files:**
- Modify: `notebooks/diffusion_gauge_demo.ipynb`

- [ ] **Step 1: Add Cell 4 — Segment & Pool**

```python
# Segment using <think> tag awareness (QWEN 3 emits <think>...</think>)
steps = segment(result["text"], tokenizer, segmentation="tags")

print(f"Found {len(steps)} reasoning steps:")
for i, step in enumerate(steps):
    kind = step["kind"]
    preview = step["text"][:80].replace("\n", " ")
    print(f"  [{i}] ({kind}) {preview}...")

# Pool token-level hidden states into per-step representations
pooled = pool_hidden_states_per_step(result["token_hidden_states"], steps)
n_steps, n_layers, d_model = pooled.shape
print(f"\nPooled shape: ({n_steps}, {n_layers}, {d_model})")
print(f"  = (n_steps, n_layers, d_model)")

# If too few steps (<3), suggest switching segmentation
if n_steps < 3:
    print("\n⚠ Very few steps detected. Try segmentation='delimiter' or a different problem.")
```

- [ ] **Step 2: Verify notebook is valid JSON**

```bash
uv run python -c "import json; json.load(open('notebooks/diffusion_gauge_demo.ipynb'))"
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/diffusion_gauge_demo.ipynb
git commit -m "feat: add segmentation and hidden state pooling cell"
```

---

### Task 5: Diffusion gauge application cell

**Files:**
- Modify: `notebooks/diffusion_gauge_demo.ipynb`

- [ ] **Step 1: Add Cell 5 — Diffusion Gauge per Layer**

```python
# Pick early / mid / late layers
layer_picks = {
    "early": 2,
    "mid": n_layers // 2,
    "late": n_layers - 1,
}
print(f"Layer indices: {layer_picks}")

# Global bandwidth (knn=None) because n_steps is small (< 15)
gauge = DiffusionGauge(knn=None, alpha=1.0, symmetric=False)

operators = {}
for name, idx in layer_picks.items():
    activations = pooled[:, idx, :].astype(np.float64)  # (n_steps, d_model)
    operators[name] = gauge(activations)  # (n_steps, n_steps)
    print(f"  {name} (layer {idx}): operator shape {operators[name].shape}, "
          f"row sums ≈ {operators[name].sum(axis=1).mean():.4f}")
```

- [ ] **Step 2: Verify notebook is valid JSON**

```bash
uv run python -c "import json; json.load(open('notebooks/diffusion_gauge_demo.ipynb'))"
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/diffusion_gauge_demo.ipynb
git commit -m "feat: add diffusion gauge application across layers"
```

---

### Task 6: Visualization cell

**Files:**
- Modify: `notebooks/diffusion_gauge_demo.ipynb`

- [ ] **Step 1: Add helper functions markdown cell**

```markdown
### Visualization

3x4 grid comparing early/mid/late layers. Each row shows:
1. **Heatmap** — diffusion operator transition probabilities
2. **MDS embedding** — reasoning steps in 2D, arrows show step order
3. **Eigenspectrum** — top eigenvalues with spectral gap highlighted
4. **Stationary distribution** — which steps are representation-space attractors
```

- [ ] **Step 2: Add Cell 6 — Visualization**

```python
def compute_spectral_metrics(operator):
    """Compute spectral gap, radius, stationary distribution, and entropy."""
    eigenvalues = np.sort(np.real(linalg.eigvals(operator)))[::-1]

    # Stationary distribution: left eigenvector of eigenvalue 1
    # For row-stochastic matrix, solve pi @ P = pi, sum(pi) = 1
    n = operator.shape[0]
    A = (operator.T - np.eye(n))
    A[-1, :] = 1.0  # replace last equation with sum constraint
    b = np.zeros(n)
    b[-1] = 1.0
    try:
        pi = linalg.solve(A, b)
        pi = np.abs(pi)
        pi /= pi.sum()
    except linalg.LinAlgError:
        pi = np.ones(n) / n  # uniform fallback

    # Entropy of stationary distribution
    pi_safe = np.clip(pi, 1e-12, None)
    entropy = -np.sum(pi_safe * np.log(pi_safe))

    return {
        "eigenvalues": eigenvalues,
        "spectral_gap": 1.0 - eigenvalues[1] if len(eigenvalues) > 1 else 1.0,
        "spectral_radius": eigenvalues[1] if len(eigenvalues) > 1 else 0.0,
        "stationary_dist": pi,
        "stationary_entropy": entropy,
    }


# Compute metrics for all layers
metrics = {name: compute_spectral_metrics(op) for name, op in operators.items()}

# --- Plot ---
fig, axes = plt.subplots(3, 4, figsize=(20, 12))
layer_names = ["early", "mid", "late"]
step_labels = [f"S{i}" for i in range(n_steps)]

for row, name in enumerate(layer_names):
    op = operators[name]
    m = metrics[name]
    layer_idx = layer_picks[name]

    # Column 1: Heatmap
    ax = axes[row, 0]
    im = ax.imshow(op, cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(n_steps))
    ax.set_yticks(range(n_steps))
    ax.set_xticklabels(step_labels, fontsize=7)
    ax.set_yticklabels(step_labels, fontsize=7)
    ax.set_title(f"Diffusion Operator — Layer {layer_idx} ({name})", fontsize=9)
    ax.set_xlabel("to step")
    ax.set_ylabel("from step")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Column 2: MDS embedding
    ax = axes[row, 1]
    row_dists = np.zeros((n_steps, n_steps))
    for i in range(n_steps):
        for j in range(n_steps):
            row_dists[i, j] = np.linalg.norm(op[i] - op[j])
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress="auto")
    coords = mds.fit_transform(row_dists)
    ax.scatter(coords[:, 0], coords[:, 1], c=range(n_steps), cmap="viridis", s=80, zorder=3)
    for i in range(n_steps):
        ax.annotate(step_labels[i], coords[i], fontsize=8, ha="center", va="bottom")
    for i in range(n_steps - 1):
        ax.annotate("", xy=coords[i + 1], xytext=coords[i],
                     arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5))
    ax.set_title(f"MDS of Operator Rows — Layer {layer_idx}", fontsize=9)
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")

    # Column 3: Eigenspectrum
    ax = axes[row, 2]
    eigs = m["eigenvalues"][:min(10, n_steps)]
    colors = ["#2196F3"] * len(eigs)
    if len(eigs) > 1:
        colors[0] = "#4CAF50"  # lambda_1 (≈1)
        colors[1] = "#FF5722"  # lambda_2 (spectral gap)
    ax.bar(range(len(eigs)), np.real(eigs), color=colors)
    if len(eigs) > 1:
        ax.axhline(y=np.real(eigs[1]), color="#FF5722", linestyle="--", alpha=0.4)
        ax.text(len(eigs) - 1, np.real(eigs[1]), f"gap={m['spectral_gap']:.3f}",
                fontsize=8, ha="right", va="bottom", color="#FF5722")
    ax.set_title(f"Eigenspectrum — Layer {layer_idx}", fontsize=9)
    ax.set_xlabel("Index")
    ax.set_ylabel("Eigenvalue")

    # Column 4: Stationary distribution
    ax = axes[row, 3]
    ax.bar(range(n_steps), m["stationary_dist"], color="#9C27B0", alpha=0.8)
    ax.set_xticks(range(n_steps))
    ax.set_xticklabels(step_labels, fontsize=7)
    ax.set_title(f"Stationary Dist π — Layer {layer_idx} (H={m['stationary_entropy']:.2f})",
                 fontsize=9)
    ax.set_xlabel("Step")
    ax.set_ylabel("π")

fig.suptitle(f"Diffusion Gauge: {MODEL_ID} on Math CoT\n\"{MATH_PROBLEM[:80]}...\"",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("diffusion_gauge_demo.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: diffusion_gauge_demo.png")
```

- [ ] **Step 3: Verify notebook is valid JSON**

```bash
uv run python -c "import json; json.load(open('notebooks/diffusion_gauge_demo.ipynb'))"
```

- [ ] **Step 4: Commit**

```bash
git add notebooks/diffusion_gauge_demo.ipynb
git commit -m "feat: add 3x4 cross-layer diffusion gauge visualization"
```

---

### Task 7: Summary table cell

**Files:**
- Modify: `notebooks/diffusion_gauge_demo.ipynb`

- [ ] **Step 1: Add Cell 7 — Summary Table**

```python
# Spread: average pairwise Frobenius distance between operator rows
def compute_spread(operator):
    n = operator.shape[0]
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += np.linalg.norm(operator[i] - operator[j], "fro") if operator.ndim > 1 else abs(operator[i] - operator[j])
            count += 1
    return total / max(count, 1)

print(f"{'Layer':<8} {'Idx':>4} {'Spectral Gap':>13} {'λ₂':>10} {'H(π)':>8} {'Spread':>8}")
print("-" * 58)
for name in layer_names:
    m = metrics[name]
    idx = layer_picks[name]
    spread = compute_spread(operators[name])
    print(f"{name:<8} {idx:>4} {m['spectral_gap']:>13.4f} {m['spectral_radius']:>10.4f} "
          f"{m['stationary_entropy']:>8.3f} {spread:>8.4f}")
```

- [ ] **Step 2: Verify notebook is valid JSON**

```bash
uv run python -c "import json; json.load(open('notebooks/diffusion_gauge_demo.ipynb'))"
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/diffusion_gauge_demo.ipynb
git commit -m "feat: add spectral metrics summary table"
```

---

### Task 8: Final cleanup and push

**Files:**
- Modify: `notebooks/diffusion_gauge_demo.ipynb`

- [ ] **Step 1: Verify complete notebook structure**

```bash
uv run python -c "
import json
nb = json.load(open('notebooks/diffusion_gauge_demo.ipynb'))
cells = nb['cells']
print(f'{len(cells)} cells total')
for i, c in enumerate(cells):
    t = c['cell_type']
    src = ''.join(c['source'])[:60].replace('\n', ' ')
    print(f'  [{i}] {t}: {src}...')
"
```

Expected: 8 cells (1 markdown + 7 code).

- [ ] **Step 2: Run ruff on any .py files touched**

```bash
uv run ruff check manyagents/
```

Expected: no new errors (we only added a notebook, no .py changes).

- [ ] **Step 3: Final commit if any uncommitted changes**

```bash
git status
# If clean, skip. Otherwise:
git add -A && git commit -m "chore: final cleanup for diffusion gauge demo"
```

- [ ] **Step 4: Push branch**

```bash
git push -u origin feat/diffusion-gauge-demo
```
