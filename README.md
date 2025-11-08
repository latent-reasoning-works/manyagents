# manyAgents

**Multi-Tool Orchestration Framework for Scientific Workflows**

Version 0.1.0 - Phase 1 Complete

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What is manyAgents?

manyAgents is an **orchestration layer** that coordinates complex scientific workflows across specialized tools. It acts as the execution engine for automated discovery systems, providing:

- **Direct Python API integration** (no subprocess overhead)
- **Multi-step pipelines** with in-memory data passing
- **Standardized adapters** for heterogeneous tools
- **Metrics computation** for workflow evaluation
- **SLURM job submission** for HPC environments

## The "Many" Suite

**Part of the [Latent Reasoning Works](https://github.com/latent-reasoning-works) ecosystem:**

- **[manyLatents](https://github.com/latent-reasoning-works/manylatents)**: Discover **many** latent representations via dimensionality reduction algorithms (PCA, UMAP, PHATE, autoencoders, etc.)
- **manyAgents** (this repo): Orchestrate **many** tools and execute **many** workflows
- **[Geomancy](https://github.com/latent-reasoning-works/geomancy)**: Train **many** RL policies to discover workflows via geometric reward signals

## Architecture: The Orchestration Layer

```
Geomancy (RL Agent / Expert Workflows)
         ↓
   manyAgents (Orchestration Layer) ← YOU ARE HERE
         ↓
   ┌────────┴────────┬────────────┬──────────┐
   ↓                 ↓            ↓          ↓
ManyLatents      CellForge    STELLA    BioDiscovery
(DR algorithms)  (tools)      (tools)   (agents)
```

**manyAgents' Responsibilities:**

✅ **Coordinate multi-tool workflows** - Chain outputs from step to step  
✅ **Provide adapter abstraction** - Unified interface for diverse tools  
✅ **Compute comparative rewards** - Evaluate workflow quality via metrics  
✅ **Handle execution context** - Logging, metrics tracking, data passing  

**What manyAgents Does NOT Do:**

❌ Implement algorithms (delegates to tools like manyLatents)  
❌ Train RL policies (that's Geomancy's job)  
❌ Define expert workflows (that's the user's job)

## Quick Start

```bash
# Install dependencies
uv sync
source .venv/bin/activate

# Run a simple two-step pipeline (PCA → UMAP)
python -m manyagents.main experiment=manylatents_pipeline_with_metrics

# Run with custom algorithm parameters
python -m manyagents.main \
  experiment=manylatents_pipeline_with_metrics \
  +workflow.steps.0.config.algorithms.latent.n_components=10
```

## Available Adapters

### ManyLatentsAdapter
Orchestrates dimensionality reduction algorithms:
- **Algorithms**: PCA, UMAP, PHATE, t-SNE, MDS, Diffusion Maps, Autoencoders
- **Metrics**: Preservation Ratio, Anisotropy, Trust/Continuity, Reconstruction loss
- **Config**: Full Hydra integration for all manyLatents parameters

See [`manyagents/adapters/manylatents_adapter.py`](manyagents/adapters/manylatents_adapter.py)

### PlaceholderAdapter
Template for integrating new tools:
- Defines adapter contract
- Shows how to handle tool-specific configs
- Example of in-memory data passing

See [`manyagents/adapters/placeholder_adapter.py`](manyagents/adapters/placeholder_adapter.py)

## Integration with Geomancy

manyAgents serves as **Geomancy's action executor**:

1. **Expert Mode**: Execute reference workflow → `G_target` (geometric signature)
2. **Training Mode**: 
   - Geomancy RL agent selects action (e.g., "run PCA with n_components=10")
   - manyAgents executes via appropriate adapter
   - Returns geometric metrics for reward computation
3. **Evaluation Mode**: Replay learned policy through manyAgents orchestrator

See [Geomancy documentation](https://github.com/latent-reasoning-works/geomancy) for RL training details.

## Development Phases

### ✅ Phase 1: The Principled Orchestrator (COMPLETE)

Reliable foundation for executing static, human-written workflow plans:

- ✅ `AgentAdapter` abstraction for standardized tool integration
- ✅ Direct Python API calls (no subprocess overhead)
- ✅ In-memory data passing between workflow steps
- ✅ Three-level metrics system (dataset/embedding/module)
- ✅ CI/CD pipelines with integration tests

### 🚧 Phase 2: Multi-Workflow Comparison (IN PROGRESS)

Compare different workflows in parallel:

- 🚧 Execute anchor vs comparison workflows
- 🚧 Compute differential rewards (G_comparison - G_anchor)
- 🚧 Enable method benchmarking

### 📋 Phase 3: Integration with Geomancy RL Loop

Full integration as Geomancy's execution backend:

- 📋 Provide action space for RL agent
- 📋 Execute RL-selected actions
- 📋 Return geometric metrics for reward computation
- 📋 Support episode-based execution patterns

## Documentation

Comprehensive documentation in `docs/` directory:

- **[index.md](docs/index.md)** - Architecture overview and concepts
- **[design_decisions.md](docs/design_decisions.md)** - Architectural choices and rationale
- **[usage.md](docs/usage.md)** - Usage examples and workflows
- **[TODO.md](docs/TODO.md)** - Future work and pre-release checklist

## Configuration

manyAgents uses [Hydra](https://hydra.cc/) for configuration management:

```
manyagents/configs/
├── experiment/              # Complete workflow definitions
│   ├── manylatents_pipeline_with_metrics.yaml
│   └── multi_workflow_comparison.yaml
├── agent/                   # Tool adapter configs
│   ├── manylatents.yaml
│   └── placeholder.yaml
└── workflow/               # Workflow structure configs
    └── sequential.yaml
```

See individual config files for detailed parameter documentation.

## Testing

```bash
# Run all tests
pytest

# Run integration tests only
pytest tests/test_orchestration_correspondence.py

# Run with coverage
pytest --cov=manyagents tests/
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development workflow
- Code style guidelines
- Testing requirements
- PR process

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use manyAgents in your research, please cite:

```bibtex
@software{manyagents2025,
  title={manyAgents: Multi-Tool Orchestration for Scientific Workflows},
  author={Latent Reasoning Works},
  year={2025},
  url={https://github.com/latent-reasoning-works/manyagents}
}
```

## Contact

- **Issues**: [GitHub Issues](https://github.com/latent-reasoning-works/manyagents/issues)
- **Discussions**: [GitHub Discussions](https://github.com/latent-reasoning-works/manyagents/discussions)
