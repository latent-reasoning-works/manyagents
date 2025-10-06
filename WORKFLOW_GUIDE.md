# ManyAgents + ManyLatents Workflow Guide

This guide shows how to create workflows using the new `manylatents` API integration.

## Architecture Overview

```
┌─────────────────┐
│   ManyAgents    │
│   Orchestrator  │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ ManyLatents     │
│ Adapter         │
└────────┬────────┘
         │ (direct Python call, no subprocess)
         v
┌─────────────────┐
│ manylatents     │
│ .api.run()      │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ run_algorithm() │
│ or              │
│ run_pipeline()  │
└─────────────────┘
```

## Key Benefits

✅ **No subprocess overhead** - Direct Python API calls
✅ **In-memory data passing** - Embeddings passed as numpy arrays
✅ **Automatic mode detection** - Single algorithm or pipeline based on config
✅ **Consistent interface** - Same API for all workflow patterns

---

## Usage Patterns

### Pattern 1: Single Algorithm via Adapter

```python
from manyagents.adapters import ManyLatentsAdapter

adapter = ManyLatentsAdapter()
result = await adapter.run(
    objective="PCA on swissroll",
    input_files={}
)

embeddings = result.output_files['embeddings']  # numpy array
scores = result.output_files['scores']          # dict
```

**When to use:** Simple, high-level objective format ("ALGORITHM on DATASET")

---

### Pattern 2: Pipeline via Direct API Call

```python
from manylatents.api import run

result = await asyncio.get_event_loop().run_in_executor(
    None,
    lambda: run(
        data='swissroll',
        debug=True,
        pipeline=[
            {
                'name': 'pca_reduction',
                'overrides': {
                    'algorithms': {
                        'latent': {
                            '_target_': 'manylatents.algorithms.latent.pca.PCAModule',
                            'n_components': 50
                        }
                    }
                }
            },
            {
                'name': 'phate_embedding',
                'overrides': {
                    'algorithms': {
                        'latent': {
                            '_target_': 'manylatents.algorithms.latent.phate.PHATEModule',
                            'n_components': 2
                        }
                    }
                }
            }
        ]
    )
)

final_embeddings = result['embeddings']  # shape: (n_samples, 2)
```

**When to use:** Multi-step sequential workflows with automatic chaining

---

### Pattern 3: Config-Driven Workflow

```python
from manylatents.api import run

# Uses manylatents/configs/experiment/pca_phate_pipeline.yaml
result = run(experiment='pca_phate_pipeline')
```

**When to use:** Predefined, reusable pipeline configurations

---

## Complete Workflow Example

```python
import asyncio
from manyagents.models import WorkflowState
from manyagents.adapters import ManyLatentsAdapter

async def scientific_workflow():
    """Multi-step scientific analysis workflow."""

    state = WorkflowState(goal="Analyze genomic data with sequential DR")
    adapter = ManyLatentsAdapter()

    # Step 1: Initial dimensionality reduction
    print("Step 1: PCA reduction...")
    result1 = await adapter.run("PCA on hgdp_split", {})
    state.add_result(result1)

    if result1.status == 'failure':
        print(f"Failed: {result1.error_message}")
        return state

    print(f"PCA complete: {result1.summary}")

    # Step 2: Further analysis (future: pass result1 embeddings)
    # For now, use pipeline configs for multi-step (see Pattern 2)

    return state

asyncio.run(scientific_workflow())
```

---

## Available Algorithms

Configure algorithms with their full target paths:

```python
# PCA
'_target_': 'manylatents.algorithms.latent.pca.PCAModule'

# PHATE
'_target_': 'manylatents.algorithms.latent.phate.PHATEModule'

# UMAP
'_target_': 'manylatents.algorithms.latent.umap.UMAPModule'

# t-SNE
'_target_': 'manylatents.algorithms.latent.tsne.TSNEModule'

# MDS
'_target_': 'manylatents.algorithms.latent.mds.MDSModule'
```

---

## Available Datasets

```python
# Synthetic
data='swissroll'

# Genomic
data='hgdp_split'
data='aou_split'
data='ukbb_split'

# See manylatents/configs/data/ for complete list
```

---

## Testing the Integration

### 1. Update manyAgents dependencies

```bash
cd /network/scratch/c/cesar.valdez/manyAgents
uv sync  # Pulls manylatents from add_agentic_API branch
```

### 2. Run the example workflow

```bash
python run_discrete_workflow.py
```

### 3. Run comprehensive examples

```bash
python example_workflows.py
```

---

## Return Value Structure

All workflows return a dictionary with:

```python
{
    'embeddings': np.ndarray,      # Shape: (n_samples, n_components)
    'label': np.ndarray,            # Sample labels (if available)
    'metadata': {
        'source': 'single_algorithm' | 'pipeline',
        'algorithm_type': str,
        'num_steps': int,           # For pipelines
        ...
    },
    'scores': {
        'metric_name': float,
        ...
    }
}
```

---

## Next Steps

1. **Enable `input_data` parameter** - Pass embeddings between calls
2. **Add PrecomputedDataModule** - In-memory dataset for chained steps
3. **Enhance adapter** - Support custom hyperparameters in objective string
4. **Add caching** - Avoid recomputing identical steps

---

## Troubleshooting

**Issue:** `ModuleNotFoundError: No module named 'manylatents'`
**Solution:** Run `uv sync` in manyAgents directory

**Issue:** `ConfigKeyError: Key 'latent' is not in struct`
**Solution:** This is fixed in manylatents v0.1.0+ on `add_agentic_API` branch

**Issue:** Pipeline doesn't chain data
**Solution:** Use pipeline config (Pattern 2) - `input_data` support coming soon
