# manyAgents

**Multi-Tool Orchestration Framework for Scientific Workflows**

## Overview

manyAgents is an **orchestration layer** that coordinates complex scientific workflows across specialized tools. It serves as the execution engine for automated discovery systems like [Geomancy](https://github.com/latent-reasoning-works/geomancy), providing a unified interface for heterogeneous computational tools.

### The "Many" Suite

Part of the **[Latent Reasoning Works](https://github.com/latent-reasoning-works)** ecosystem:

- **[manyLatents](https://github.com/latent-reasoning-works/manylatents)**: Discover **many** latent representations (PCA, UMAP, PHATE, autoencoders)
- **manyAgents** (this project): Orchestrate **many** tools and workflows
- **[Geomancy](https://github.com/latent-reasoning-works/geomancy)**: Train **many** RL policies via geometric reward signals

## Architecture

```
┌─────────────────────────────────────────┐
│  Geomancy (RL Agent / Expert Workflows) │
└──────────────────┬──────────────────────┘
                   ↓
┌──────────────────────────────────────────┐
│  manyAgents (Orchestration Layer)        │  ← YOU ARE HERE
│  • Coordinate multi-step workflows       │
│  • Standardized adapter abstraction      │
│  • In-memory data passing                │
│  • Metrics computation & aggregation     │
└──────────────────┬───────────────────────┘
                   ↓
        ┌──────────┴────────┬──────────┬──────────┐
        ↓                   ↓          ↓          ↓
  ┌──────────┐      ┌──────────┐  ┌──────┐  ┌──────────┐
  │ManyLatents│      │CellForge │  │STELLA│  │BioBridge │
  │(DR algos)│      │ (tools)  │  │(tools)│  │ (agents) │
  └──────────┘      └──────────┘  └──────┘  └──────────┘
```

### What manyAgents Does

✅ **Coordinate multi-tool workflows** - Chain outputs between steps  
✅ **Provide adapter abstraction** - Unified interface for diverse tools  
✅ **Compute comparative rewards** - Evaluate workflow quality via metrics  
✅ **Handle execution context** - Logging, metrics tracking, data passing

### What manyAgents Does NOT Do

❌ **Implement algorithms** - Delegates to specialized tools (manyLatents, etc.)  
❌ **Train RL policies** - That's Geomancy's responsibility  
❌ **Define expert workflows** - That's the user's domain knowledge

## Key Components

### Adapter System

All tools are accessed through standardized adapters:

```python
from manyagents.adapters import AgentAdapter

class YourToolAdapter(AgentAdapter):
    """Adapter for external tool integration."""
    
    def run(self, input_data: Dict) -> Dict:
        # Execute tool
        # Return standardized result format
        pass
```

**Available Adapters:**
- **ManyLatentsAdapter**: Dimensionality reduction algorithms (PCA, UMAP, PHATE, etc.)
- **PlaceholderAdapter**: Template for new tool integrations

### Workflow Execution

Sequential execution with in-memory data passing:

```yaml
# configs/experiment/example_workflow.yaml
workflow:
  steps:
    - agent: manylatents
      config:
        algorithms:
          latent: pca
          n_components: 50
    - agent: manylatents
      config:
        algorithms:
          latent: umap
          n_neighbors: 15
```

### Metrics System

Three-level metrics hierarchy:

1. **Dataset Metrics**: Properties of input data
2. **Embedding Metrics**: Quality of learned representations (Preservation Ratio, Anisotropy, etc.)
3. **Module Metrics**: Training dynamics (loss, convergence)

## Configuration System

Powered by [Hydra](https://hydra.cc/):

```
manyagents/configs/
├── experiment/              # Complete workflow definitions
│   ├── manylatents_pipeline_with_metrics.yaml
│   └── multi_workflow_comparison.yaml
├── agent/                   # Tool adapter configs
│   ├── manylatents.yaml
│   └── placeholder.yaml
└── workflow/               # Workflow structure
    └── sequential.yaml
```

## Quick Start

```bash
# Install dependencies
uv sync
source .venv/bin/activate

# Run a two-step pipeline (PCA → UMAP)
python -m manyagents.main experiment=manylatents_pipeline_with_metrics

# Override parameters
python -m manyagents.main \
  experiment=manylatents_pipeline_with_metrics \
  +workflow.steps.0.config.algorithms.latent.n_components=10
```

## Integration with Geomancy

manyAgents serves as Geomancy's **action executor**:

### Expert Mode
```python
# User defines reference workflow
expert_workflow = load_config("expert_workflow.yaml")

# Execute via manyAgents → Get G_target
G_target = manyagents.execute_workflow(expert_workflow)
```

### Training Mode
```python
# RL agent selects action
action = policy.select_action(observation)

# Execute via manyAgents
result = manyagents.execute_step(action, current_data)

# Compute reward from geometric metrics
reward = compute_geometric_reward(result.metrics, G_target)
```

See [Geomancy documentation](https://github.com/latent-reasoning-works/geomancy) for full RL training details.

## Development Phases

### ✅ Phase 1: The Principled Orchestrator (COMPLETE)

- ✅ Direct Python API integration with manyLatents
- ✅ In-memory data passing between steps
- ✅ Three-level metrics system
- ✅ Adapter abstraction for tool integration
- ✅ CI/CD with integration tests

### 🚧 Phase 2: Multi-Workflow Comparison (IN PROGRESS)

- 🚧 Execute anchor vs comparison workflows
- 🚧 Compute differential rewards (G_comparison - G_anchor)
- 🚧 Enable method benchmarking

### 📋 Phase 3: Full Geomancy Integration

- 📋 Provide action space for RL agent
- 📋 Execute RL-selected actions dynamically
- 📋 Return geometric metrics for reward computation
- 📋 Support episode-based execution patterns

## Documentation

- **[Design Decisions](design_decisions.md)** - Architectural choices and rationale
- **[Usage Guide](usage.md)** - Examples and workflows
- **[TODO](TODO.md)** - Future work and pre-release checklist
- **[Contributing](CONTRIBUTING.md)** - Development guidelines

## Technical Stack

- **Package Manager**: [UV](https://astral.sh/uv) - Fast Python package installer
- **Configuration**: [Hydra](https://hydra.cc) - Flexible config management
- **Validation**: [Pydantic](https://pydantic.dev) - Type-safe data models
- **HPC**: [SLURM](https://slurm.schedmd.com/) via hydra-submitit-launcher
- **Python**: 3.10+ (tested on 3.10-3.12)

## Next Steps

1. **Read the [Usage Guide](usage.md)** - Learn how to define workflows
2. **Check [Design Decisions](design_decisions.md)** - Understand architectural choices
3. **See [Contributing](../CONTRIBUTING.md)** - Join the development
4. **Explore Examples** - Review `manyagents/configs/experiment/` for workflow templates

## License

MIT License - see [LICENSE](../LICENSE) for details.
