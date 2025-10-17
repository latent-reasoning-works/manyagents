# manyLatents Exploration - Complete Analysis for Phase 3 Design

## Overview

This analysis provides a comprehensive exploration of the manyLatents package architecture, training patterns, and configuration system. The goal is to inform the design of Phase 3 (RL/Learning Agent) in manyAgents.

**Key Finding**: manyLatents provides a production-ready template for Phase 3 architecture, with minimal additions needed to support RL-based workflow generation.

---

## Deliverable Documents

All documents are in markdown format with code examples and can be read sequentially or by topic.

### 1. MANYLATENTS_ARCHITECTURE.md (477 lines, 17KB)
**What it covers**: Deep dive into manyLatents' technical architecture

**Key sections**:
1. Algorithm base class structure - LatentModule abstract base
2. LightningModule integration - deferred initialization pattern
3. Training loop infrastructure - execute_step, dual dispatch, pipelines
4. Hydra config → algorithm instantiation mapping
5. Programmatic API design - how input_data chaining works
6. Metrics & evaluation - three-level metrics system
7. Design patterns for Phase 3 - 5 key patterns extracted
8. Existing but unused RL infrastructure - what's ready to use
9. Proposed RL integration - minimal changes needed

**Use this for**: Understanding how algorithms are trained, configured, and evaluated

### 2. PHASE3_IMPLICATIONS.md (485 lines, 17KB)
**What it covers**: How to use manyLatents patterns for Phase 3 RL design

**Key sections**:
1. RL environment as evolved orchestrator - from LLM planner to policy network
2. State representation - embeddings + metrics as latent state
3. Action space architecture - algorithm selection + hyperparameters
4. Reward function - normalize manyLatents metrics to scalar rewards
5. Experience replay infrastructure - trajectory buffer for learning
6. Policy architecture - PPO/A3C agent design with code
7. Integration with manyAgents - minimal changes to Phase 1
8. Comparison table - manyLatents vs Phase 3 architecture
9. Implementation roadmap - detailed 8-week plan

**Use this for**: Planning Phase 3 implementation with concrete code patterns

### 3. MANYLATENTS_KEY_FILES.md (219 lines, 9.4KB)
**What it covers**: Reference guide to critical file paths and their roles

**Key sections**:
- Algorithm implementations (8 latent + Lightning-based)
- Orchestration & execution entry points
- Configuration system (schema + presets)
- Data utilities and callbacks
- Design patterns reference (5 patterns explained)
- Critical entry points for manyAgents integration
- Total LOC analysis

**Use this for**: Quickly locating specific functionality when reading manyLatents source

---

## Quick Start Guide

### Understanding manyLatents (30 minutes)

1. **Read**: PHASE3_IMPLICATIONS.md Section 1-3 (state, actions, rewards)
2. **Reference**: MANYLATENTS_KEY_FILES.md for file locations
3. **Read**: MANYLATENTS_ARCHITECTURE.md Section 3-4 (execution + configuration)

### Planning Phase 3 Implementation (1 hour)

1. **Read**: PHASE3_IMPLICATIONS.md Section 6-9 (policy, integration, roadmap)
2. **Reference**: Code snippets in PHASE3_IMPLICATIONS.md Section 3-6
3. **Check**: MANYLATENTS_KEY_FILES.md for integration points

### Deep Dive into Specific Topic

**Topic: Config-Driven Algorithm Instantiation**
- Read: MANYLATENTS_ARCHITECTURE.md Section 4
- Reference: MANYLATENTS_KEY_FILES.md "Key Design Patterns" section 1
- Files: experiment.py, api.py

**Topic: Training Loop Patterns**
- Read: MANYLATENTS_ARCHITECTURE.md Section 3
- Reference: Code in Section 3 of MANYLATENTS_ARCHITECTURE.md
- Files: experiment.py (execute_step, run_algorithm, run_pipeline)

**Topic: Metrics & Rewards**
- Read: MANYLATENTS_ARCHITECTURE.md Section 6
- Read: PHASE3_IMPLICATIONS.md Section 4
- Files: experiment.py (evaluate function), utils/metrics.py

---

## Key Architectural Insights

### 1. Config-Driven Instantiation
All algorithm hyperparameters come from Hydra configs, enabling:
- Zero-code parameter tuning
- Dynamic config generation (perfect for RL agents outputting configs)
- Programmatic overrides via `manylatents.api.run(**config_overrides)`

**Phase 3 Use**: RL policy outputs algorithm config → api.run() executes it

### 2. State Threading
Pipeline steps chain automatically: Step 1 output → Step 2 input
- In-memory (no serialization)
- Supports arbitrary dimensionality changes
- Perfect for feedback loops

**Phase 3 Use**: Agent loop: observe state → select action → execute → observe result

### 3. Singledispatch Evaluation
Single evaluate() function handles different algorithm types:
- LatentModule (dict output) - computes embedding metrics
- LightningModule - runs trainer.test() + computes metrics

**Phase 3 Use**: Uniform reward computation for diverse algorithm types

### 4. Deferred Initialization
Network input dimensions inferred from first batch in setup():
- No need to hardcode input_dim in config
- Flexible for different dataset sizes
- Pattern enables meta-learning

**Phase 3 Use**: Adapts to different problem dimensionalities automatically

### 5. Programmatic API Layer
api.run() provides clean Python interface over Hydra complexity:
- Accepts input_data for chaining
- Smart routes to run_algorithm() or run_pipeline()
- Handles numpy array serialization workarounds

**Phase 3 Use**: Single entry point for agent decisions

---

## Critical Paths for Integration

### manyAgents → manyLatents Integration Points

**Main Entry Point**:
```python
# In ManyLatentsAdapter.run():
from manylatents.api import run

result = run(
    input_data=state.get('embeddings'),  # From previous step
    algorithms={'latent': agent_action},   # From RL policy
    seed=cfg.seed,
    debug=cfg.debug
)
# Returns: {'embeddings': ndarray, 'scores': dict, 'metadata': dict}
```

**File Locations**:
- `.venv/lib/python3.10/site-packages/manylatents/api.py` - Main entry
- `.venv/lib/python3.10/site-packages/manylatents/experiment.py` - Core logic
- `.venv/lib/python3.10/site-packages/manylatents/algorithms/latent_module_base.py` - Base class

---

## Algorithm Support (Production Ready)

### Latent Algorithms (Fit/Transform Pattern)
- **PCA** - Principal component analysis
- **PHATE** - Potential of heat-diffusion affinity-based transition embedding
- **UMAP** - Uniform manifold approximation and projection
- **t-SNE** - t-distributed stochastic neighbor embedding
- **MDS** - Multidimensional scaling
- **DiffusionMap** - Diffusion map embedding
- **AA** - Adversarial autoencoder
- **NoOp** - Identity (no transformation)

### Neural Network Algorithms (Lightning-based)
- **Reconstruction** - Generic autoencoder with modular networks
- **Autoencoder** - Standard encoder-decoder architecture
- **AANet** - Adversarial autoencoder variant

### Loss Functions (for neural networks)
- **MSE** - Mean squared error reconstruction
- **PR** - Persistent rank loss
- **Anisotropy** - Anisotropy loss
- **TSA** - Trustworthiness-continuity loss
- **AllGeom** - Combined geometric losses

---

## Metrics Available

### Embedding-Level Metrics
- Trustworthiness - How well k-nearest neighbors are preserved
- Continuity - How well k-nearest neighbors in original space map to embedding
- Local dimensionality - Intrinsic dimensionality at each point
- Reconstruction error - Autoencoder reconstruction loss

### Dataset-Level Metrics
- Variance explained - % of variance retained
- Effective dimensionality - Intrinsic dataset dimensionality

### Module-Level Metrics
- Training loss - Convergence during training
- Test loss - Generalization performance

**Phase 3 Use**: These metrics are normalized and weighted to compute RL rewards

---

## Implementation Checklist for Phase 3

### Phase 3.1: RL Infrastructure
- [ ] Create `EnvironmentState` class (embeddings + metrics + metadata)
- [ ] Create `Transition` dataclass (s, a, r, s', done, info)
- [ ] Create `TrajectoryBuffer` with experience replay
- [ ] Create `RewardFunction` to normalize metrics → scalar rewards
- [ ] Write unit tests for each component

### Phase 3.2: Policy Network
- [ ] Create `ActionSpace` class mapping actions to Hydra configs
- [ ] Create `RLPolicy` (PyTorch module with actor/critic heads)
- [ ] Create `PPOAgent` with policy gradient training
- [ ] Add checkpointing for policy parameters
- [ ] Write integration tests with mock algorithms

### Phase 3.3: Agent Integration
- [ ] Create `AdaptiveOrchestrator` calling api.run() from policy
- [ ] Integrate with existing `execute_workflow_chain()`
- [ ] Add trajectory logging to WandB
- [ ] Implement episode-based training loop
- [ ] Write end-to-end tests

### Phase 3.4: Evaluation & Benchmarking
- [ ] Compare RL policy vs LLM planner on benchmark problems
- [ ] Profile: execution time, memory usage, convergence speed
- [ ] Ablation studies: reward weights, policy architecture
- [ ] Write performance report

---

## Key Dependencies

- **PyTorch Lightning** - Neural network training (already in manyLatents)
- **Hydra** - Configuration management (already in manyLatents)
- **WandB** - Experiment tracking (already in manyLatents)
- **numpy, scipy, sklearn** - Scientific computing (already in manyLatents)

**For Phase 3**:
- **torch** - RL policy network (already available)
- **optional**: gym/gymnasium for standardized RL interfaces

---

## File Statistics

| File | Lines | Size | Purpose |
|------|-------|------|---------|
| MANYLATENTS_ARCHITECTURE.md | 477 | 17KB | Technical deep dive |
| PHASE3_IMPLICATIONS.md | 485 | 17KB | Phase 3 design guide |
| MANYLATENTS_KEY_FILES.md | 219 | 9.4KB | File reference guide |
| **Total** | **1181** | **43.4KB** | Complete analysis |

---

## How to Use These Documents

### For Architecture Review
Read in order:
1. MANYLATENTS_ARCHITECTURE.md (understand current state)
2. PHASE3_IMPLICATIONS.md Section 1-2 (understand RL adaptation)
3. PHASE3_IMPLICATIONS.md Section 9 (see roadmap)

### For Implementation
Read in order:
1. MANYLATENTS_KEY_FILES.md (locate critical files)
2. PHASE3_IMPLICATIONS.md Section 3-6 (understand design patterns)
3. PHASE3_IMPLICATIONS.md Section 9 (follow roadmap)

### For Reference
Use as needed:
- MANYLATENTS_KEY_FILES.md - When locating specific functionality
- MANYLATENTS_ARCHITECTURE.md Section 4-6 - When understanding config/metrics
- PHASE3_IMPLICATIONS.md Code sections - When implementing components

---

## Next Steps

1. **This Week**: Read all three documents to build mental model
2. **Next Week**: Create Phase 3.1 components (EnvironmentState, RewardFunction, TrajectoryBuffer)
3. **Following Week**: Implement RLPolicy and test with single algorithm
4. **Following Week**: Build AdaptiveOrchestrator and end-to-end test
5. **Following Week**: Benchmark against LLM planner

---

## Questions or Clarifications?

Refer to:
- MANYLATENTS_ARCHITECTURE.md for "how does manyLatents work?"
- PHASE3_IMPLICATIONS.md for "how do we build Phase 3?"
- MANYLATENTS_KEY_FILES.md for "where is this code?"

All documents include inline code examples and can be searched for specific terms.

