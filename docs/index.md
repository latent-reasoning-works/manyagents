# ManyAgents

**ManyAgents** is a minimal prototype that orchestrates machine learning experiments via the [ManyLatents](https://github.com/cmvcordova/manyLatents.git) framework using Hydra configuration management and SLURM job submission.

## Overview

ManyAgents follows a simple **propose → execute → collect** workflow for ML experiment orchestration:

1. **Propose**: Define analysis specifications with parameters and resource requirements
2. **Execute**: Run experiments via ManyLatents subprocess calls with Hydra overrides  
3. **Collect**: Aggregate results for decision-making and further analysis

## Architecture

- **Package Manager**: UV (Astral)
- **Configuration**: Hydra with YAML configs
- **Execution**: Subprocess calls to `manylatents.main`
- **Job Submission**: SLURM via hydra-submitit-launcher
- **Python**: 3.10 (pinned for dependency compatibility)

## Key Components

### Core Models
- `AnalysisSpec`: Specification for individual analysis runs
- `AnalysisResult`: Results from executed analyses  
- `Metric`: Individual metric results with metadata

### Execution
- `ManyLatentsExecutor`: Handles subprocess calls to ManyLatents
- Sequential workflow support for discrete analysis lists
- Configurable timeouts and dry-run capabilities

### Configuration
Hydra-based configuration system with support for:
- Analysis parameters and overrides
- Resource allocation (CPU, memory, time)
- Output directory management
- SLURM job submission settings

## Quick Start

```bash
# Install dependencies
uv sync

# Run basic test
uv run manyagents

# Run with custom config
uv run manyagents --config-name=custom
```

## Project Status

✅ **Core Integration**: ManyLatents import and execution working  
✅ **Configuration**: Hydra setup with YAML configs  
✅ **Execution Skeleton**: Subprocess calls with parameter overrides  
🚧 **Sequential Workflows**: Planned for discrete analysis lists  

## Dependencies

Built on top of:
- [ManyLatents](https://github.com/cmvcordova/manyLatents.git) - Core ML framework
- [Hydra](https://hydra.cc) - Configuration management
- [Pydantic](https://pydantic.dev) - Data validation and models
- [UV](https://astral.sh/uv) - Package management