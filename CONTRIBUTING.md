# Contributing to manyAgents

Thank you for contributing to manyAgents! This document provides guidelines for adding new components and ensuring they integrate properly with the orchestration framework.

## Table of Contents

- [Integration Testing Philosophy](#integration-testing-philosophy)
- [Adding New Metrics](#adding-new-metrics)
- [Adding New Algorithms](#adding-new-algorithms)
- [Adding New Datasets](#adding-new-datasets)
- [Adding New Adapters](#adding-new-adapters)
- [Testing Your Changes](#testing-your-changes)
- [CI/CD Pipeline](#cicd-pipeline)

---

## Integration Testing Philosophy

manyAgents orchestrates external tools (manylatents, BioDiscoveryAgent, etc.). When you add or modify components, you must ensure they work **end-to-end** through the orchestration layer, not just in isolation.

**Key Principle**: If it works in manylatents CLI but breaks when called through manyAgents, the integration is broken.

---

## Adding New Metrics

### 1. Implement in manylatents

Add your metric following the [manylatents metrics architecture](https://github.com/cmvcordova/manyLatents/blob/main/docs/metrics_architecture.md):

```python
# manylatents/metrics/your_metric.py
def YourMetric(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[object] = None,
    **kwargs
) -> Union[float, tuple[float, np.ndarray], dict[str, Any]]:
    """Your metric implementation."""
    # Compute metric
    scalar_value = compute_something(embeddings)
    per_sample_values = compute_per_sample(embeddings)
    return (scalar_value, per_sample_values)
```

Create config in appropriate group:
```yaml
# manylatents/configs/metrics/embedding/your_metric.yaml
your_metric:
  _target_: manylatents.metrics.your_metric.YourMetric
  _partial_: True
  param1: value1
```

### 2. Test in manylatents CLI

```bash
cd /path/to/manyLatents
source .venv/bin/activate
python -m manylatents.main experiment=single_algorithm metrics/embedding=your_metric
```

Verify:
- Metric computes without errors
- Wandb logs scalar and table (if per-sample values returned)
- Output is correct

### 3. Test through manyAgents Integration

**REQUIRED**: Run the integration test to ensure your metric works through orchestration:

```bash
cd /path/to/manyAgents
source .venv/bin/activate
python -m manyagents.main experiment=manylatents_pipeline_with_metrics
```

This validates:
- ✅ Config overrides work through adapter
- ✅ Metric computes in multi-step pipelines
- ✅ Results are properly returned to orchestrator
- ✅ Wandb logging works end-to-end

**If this test fails, your metric is not ready for merge.**

### 4. Update manylatents dependency

After your metric is merged into manylatents `main`:

```bash
cd /path/to/manyAgents
uv lock --upgrade-package manylatents
uv sync
```

### 5. Create example config (optional)

If your metric is expensive or specialized, create a dedicated example:

```yaml
# manyagents/configs/experiment/example_with_your_metric.yaml
name: example_with_your_metric

workflow:
  steps:
    - name: compute_embeddings
      agent: manylatents
      config:
        experiment: single_algorithm
        algorithms:
          latent:
            n_components: 2
        metrics/embedding: your_metric  # Use your new metric
        project: manyagents_examples
```

---

## Adding New Algorithms

### 1. Implement in manylatents

Follow manylatents algorithm structure:

```python
# manylatents/algorithms/latent/your_algorithm.py
from manylatents.algorithms.latent_module_base import LatentModule

class YourAlgorithmModule(LatentModule):
    def fit(self, X: torch.Tensor) -> None:
        # Training logic
        pass

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        # Inference logic
        return embeddings
```

Create config:
```yaml
# manylatents/configs/algorithms/latent/your_algorithm.yaml
_target_: manylatents.algorithms.latent.your_algorithm.YourAlgorithmModule
n_components: 2
param1: value1
```

### 2. Test in manylatents CLI

```bash
python -m manylatents.main \
  algorithms/latent=your_algorithm \
  data=swissroll \
  metrics=test_metric
```

### 3. Test through manyAgents Integration

```bash
# Create a test workflow
python -m manyagents.main experiment=manylatents_single_algorithm \
  workflow.steps.0.config.algorithms.latent._target_=manylatents.algorithms.latent.your_algorithm.YourAlgorithmModule
```

Or create a dedicated config and test:
```bash
python -m manyagents.main experiment=example_with_your_algorithm
```

---

## Adding New Datasets

### 1. Implement in manylatents

```python
# manylatents/data/your_dataset.py
from lightning import LightningDataModule

class YourDataModule(LightningDataModule):
    def setup(self, stage=None):
        # Load/generate data
        self.train_dataset = ...
        self.test_dataset = ...
```

Config:
```yaml
# manylatents/configs/data/your_dataset.yaml
_target_: manylatents.data.your_dataset.YourDataModule
param1: value1
```

### 2. Test in manylatents CLI

```bash
python -m manylatents.main \
  data=your_dataset \
  algorithms/latent=pca \
  metrics=test_metric
```

### 3. Test through manyAgents Integration

```bash
python -m manyagents.main experiment=manylatents_single_algorithm \
  workflow.steps.0.config.data=your_dataset
```

---

## Adding New Adapters

When integrating a new external tool (e.g., BioDiscoveryAgent, CellForge):

### 1. Create Adapter Class

```python
# manyagents/adapters/your_tool_adapter.py
from manyagents.adapters.base import AgentAdapter

class YourToolAdapter(AgentAdapter):
    def run(self, config: dict) -> dict:
        """
        Execute your tool and return standardized results.

        Returns:
            dict with keys: summary, output_files, success, metadata
        """
        # Call your tool
        result = your_tool.execute(config)

        return {
            'summary': 'Tool executed successfully',
            'output_files': {'result': result},
            'success': True,
            'metadata': {}
        }
```

### 2. Register Adapter

```python
# manyagents/adapters/__init__.py
from manyagents.adapters.your_tool_adapter import YourToolAdapter

ADAPTERS = {
    'manylatents': ManyLatentsAdapter,
    'your_tool': YourToolAdapter,  # Add here
}
```

### 3. Create Example Workflow

```yaml
# manyagents/configs/experiment/your_tool_example.yaml
name: your_tool_example

workflow:
  steps:
    - name: run_your_tool
      agent: your_tool
      config:
        param1: value1
        param2: value2
```

### 4. Test Integration

```bash
python -m manyagents.main experiment=your_tool_example
```

---

## Testing Your Changes

### Local Testing Checklist

Before submitting a PR, run these tests:

#### 1. Component Works in Isolation
```bash
# For manylatents components
cd /path/to/manyLatents
python -m manylatents.main experiment=single_algorithm <your-overrides>
```

#### 2. Component Works Through Orchestration
```bash
# For manylatents integration
cd /path/to/manyAgents
python -m manyagents.main experiment=manylatents_pipeline_with_metrics
```

#### 3. Check Outputs
- [ ] No errors in console
- [ ] Output files created in `outputs/`
- [ ] Wandb logs populated (if not in debug mode)
- [ ] Adapter returns success=True

### What to Test Based on Component Type

| Component Type | Test Command | What to Verify |
|----------------|--------------|----------------|
| **Metric** | `manylatents_pipeline_with_metrics` | Metric computes, wandb table logged |
| **Algorithm** | `manylatents_single_algorithm` with override | Algorithm runs, embeddings generated |
| **Dataset** | `manylatents_single_algorithm` with override | Data loads, correct shape |
| **Adapter** | Custom workflow config | Tool executes, results returned |

---

## CI/CD Pipeline

### GitHub Actions

The repository has automated CI that runs on every push to `main`:

**Workflow**: `.github/workflows/integration-tests.yml`

**What it tests**:
- Runs `manylatents_pipeline_with_metrics`
- 2-step pipeline: PCA (50D) → UMAP (2D)
- Uses `test_metric` (fast, validates all 3 metric groups)
- Validates adapter, config overrides, data passing

**When it runs**:
- Every push to `main`
- Every pull request targeting `main`

**How to check**:
- Go to GitHub Actions tab
- Look for "Integration Tests" workflow
- Green checkmark = passed, red X = failed

### What CI Validates

✅ manyAgents → manylatents adapter works
✅ Multi-step pipeline execution
✅ Metrics computation (dataset/embedding/module)
✅ Config overrides propagate correctly
✅ In-memory data passing between steps
✅ Wandb integration (in debug mode)

### If CI Fails

1. **Check the workflow logs** on GitHub Actions
2. **Reproduce locally**:
   ```bash
   python -m manyagents.main experiment=manylatents_pipeline_with_metrics
   ```
3. **Debug the error**
4. **Fix and push again** (CI will re-run automatically)

---

## Code Style

- Follow PEP 8
- Use type hints for function signatures
- Add docstrings for public functions
- Keep functions focused and small

---

## Questions?

- Check existing adapters/configs for examples
- See `docs/` for architecture documentation
- Ask in issues or discussions

Thank you for contributing to manyAgents!
