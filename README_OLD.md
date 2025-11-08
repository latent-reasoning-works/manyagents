# manyAgents

**Multi-Agent Orchestration Framework for Automated Scientific Discovery**

Version 0.1.0 - Phase 1 Complete

## Overview

manyAgents is a sophisticated orchestration framework that automates complex scientific workflows by coordinating specialized tools. It provides:

- **Direct Python API integration** with tools like manyLatents
- **Multi-step pipelines** with in-memory data passing
- **Hydra configuration system** for flexible workflow definition
- **Three-level metrics system** (dataset/embedding/module)
- **SLURM job submission** for HPC environments

## Quick Start

```bash
# Install dependencies
source .venv/bin/activate
uv sync

# Run a simple two-step pipeline (PCA → UMAP)
python -m manyagents.main experiment=manylatents_pipeline_with_metrics

# Run with custom algorithm configs
python -m manyagents.main \
  experiment=manylatents_pipeline_with_metrics \
  +workflow.steps.0.config.algorithms.latent.n_components=10
```

## Phase 1: The Principled Orchestrator ✅

Phase 1 establishes a reliable foundation for executing static, human-written workflow plans:

- ✅ AgentAdapter abstraction for standardized tool integration
- ✅ Direct Python API calls (no subprocess overhead)
- ✅ In-memory data passing between workflow steps
- ✅ Three-level metrics system fully functional
- ✅ CI/CD pipelines with integration tests

## What's Next

**Phase 2: The Agentic Planner** - Introduce autonomous workflow generation with LLM-driven planning

**Phase 3: The Learning Agent** - Enable learning from experience via Reinforcement Learning

## Documentation

See `docs/` directory for:
- `index.md` - Architecture overview
- `design_decisions.md` - Architectural choices and rationale
- `usage.md` - Usage examples and workflows
- `TODO.md` - Pre-release checklist and future work