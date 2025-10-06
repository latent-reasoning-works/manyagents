# Usage Guide

## Entry Points

### Console Command
```bash
# Direct command (recommended)
uv run manyagents

# Module execution (alternative)  
uv run python -m manyagents.main
```

### Programmatic Usage
```python
from manyagents.adapters.manylatents_adapter import ManyLatentsAdapter

# Create adapter
adapter = ManyLatentsAdapter()

# Execute workflow using direct API
result = await adapter.run(
    task_config={
        "data": "swissroll",
        "algorithm": "pca",
        "n_components": 2
    },
    input_files={}
)

print(f"Success: {result['success']}")
print(f"Embeddings shape: {result['output_files']['embeddings'].shape}")
```

## Configuration

### Hydra Overrides
ManyAgents passes overrides directly to ManyLatents. Available configuration groups:

**Data Sources**:
- `data=swissroll` - Swiss roll synthetic dataset
- `data=GaussianBlobs` - Gaussian blob clusters
- `data=test_data` - Simple test dataset

**Algorithms**:
- `algorithm=latent/pca` - Principal Component Analysis
- `algorithm=latent/umap` - UMAP dimensionality reduction
- `algorithm=latent/tsne` - t-SNE embedding
- `algorithm=latent/phate` - PHATE algorithm

**Metrics**:
- `metrics=default` - Basic evaluation metrics
- `metrics=participation_ratio` - Participation ratio calculation
- `metrics=manifold_suite` - Full manifold analysis

### Resource Configuration
```python
from manyagents.core.models import AnalysisSpec

spec = AnalysisSpec(
    name="my_analysis",
    inputs={"data": "swissroll", "algorithm": "latent/pca"},
    params={"seed": 42, "debug": False},
    resources={"time_min": 10, "mem_gb": 4, "cpus": 2},
    est_cost_s=300.0
)
```

## Sequential Workflows

Execute multiple analyses in sequence:

```python
executor = ManyLatentsExecutor()

workflow_list = [
    "pca_analysis", 
    "umap_analysis", 
    "tsne_analysis"
]

base_overrides = [
    "data=swissroll",
    "metrics=default", 
    "seed=42"
]

results = executor.execute_sequential_workflows(
    workflow_list=workflow_list,
    base_overrides=base_overrides
)

for result in results:
    print(f"Workflow {result['workflow']}: {result['success']}")
```

## SLURM Integration

For cluster execution, use the Hydra SLURM launcher:

```bash
# Submit to SLURM cluster
uv run manyagents -m hydra/launcher=submitit_slurm

# Override SLURM parameters
uv run manyagents -m hydra/launcher=submitit_slurm \
    hydra/launcher.partition=gpu \
    hydra/launcher.time=60 \
    hydra/launcher.mem=8GB
```

## Output Management

All outputs are saved to the `outputs/` directory:

- **Hydra logs**: `outputs/{name}_{timestamp}/`
- **Experiment results**: Individual result JSON files
- **SLURM logs**: `slurm_*.out` and `slurm_*.err` files
- **Wandb logs**: `wandb/` directory for experiment tracking

## Environment Variables

- `HYDRA_FULL_ERROR=1` - Enable full Hydra error traces
- `WANDB_MODE=offline` - Disable Wandb syncing
- `UV_LINK_MODE=copy` - Use copy mode for UV installations

## Common Patterns

### Quick Test Run
```bash
uv run manyagents data=swissroll algorithm=latent/pca metrics=default trainer.max_epochs=1
```

### Dry Run Testing  
```python
executor = ManyLatentsExecutor(dry_run=True)
result = executor.execute_workflow("test", ["data=swissroll"])
# Shows command without executing
```

### Error Handling
```python
result = executor.execute_workflow("", overrides)
if not result['success']:
    print(f"Error: {result['stderr']}")
    print(f"Command: {' '.join(result['cmd'])}")
```