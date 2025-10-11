# Experiment Configurations

This directory contains **pattern-based example configs** that demonstrate manyAgents workflow structures.

These configs are **teaching tools** - they show you how to structure your own experiments. Each file is heavily commented to be self-documenting.

## Available Patterns

### `manylatents_single_algorithm.yaml` ⭐ DEFINITIVE EXAMPLE 1
The simplest workflow: manyAgents orchestrating a single manyLatents DR experiment.

```bash
python -m manyagents.main experiment=manylatents_single_algorithm
```

**Learn from this:**
- Basic workflow structure with one step
- How to specify manylatents adapter
- Direct algorithm configuration (no experiment reference)
- Wandb step tagging (run named: "step0_pca_reduction")

### `manylatents_multi_step_pipeline.yaml` ⭐ DEFINITIVE EXAMPLE 2
Multi-step workflow: chaining PCA → UMAP with in-memory data passing.

```bash
python -m manyagents.main experiment=manylatents_multi_step_pipeline
```

**Learn from this:**
- Multi-step workflows with automatic data chaining
- Each step creates a separate wandb run
- Geometric metrics computation
- Embedding visualization (plots, no CSV)
- Preparing geometric features for downstream agents

### `single_algorithm.yaml` (Legacy)
Original simple example - kept for backward compatibility.

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
