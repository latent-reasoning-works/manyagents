# Experiment Configurations

This directory contains **pattern-based example configs** that demonstrate manyAgents workflow structures.

These configs are **teaching tools** - they show you how to structure your own experiments. Each file is heavily commented to be self-documenting.

## Available Patterns

### `single_algorithm.yaml`
The simplest workflow: run one algorithm on one dataset.

```bash
uv run manyagents experiment=single_algorithm
```

**Learn from this:**
- Basic workflow structure
- How to specify adapter and config
- Parameter passing to manylatents

## Creating Your Own Experiments

For project-specific experiments, create a new directory:

```
your_project/
  configs/
    experiment/
      hgdp_pca.yaml        # Your actual experiments
      ukbb_umap.yaml
      ...
```

Then use Hydra's search path to include them:

```python
# In your project's main.py
@hydra.main(config_path="configs", ...)
```

## Pattern vs Project Configs

| Type | Purpose | Location | Naming |
|------|---------|----------|--------|
| **Pattern** | Teaching examples | `manyagents/configs/experiment/` | `single_algorithm.yaml` |
| **Project** | Actual experiments | `your_project/configs/experiment/` | `hgdp_pca.yaml` |

Keep patterns generic and heavily commented. Make project configs specific and concise.
