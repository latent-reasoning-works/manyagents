"""
ManyAgents: LLM agent testing framework.

Tests whether LLMs can reason about data geometry by dispatching
prompts to multiple agents (Claude, GPT-4, local LLMs) and evaluating
their responses against ground truth methods.

For pipeline orchestration, use Geomancy instead.

## Core Design Principles

1. **Schema-on-Read**: Flexible dict-based configs with runtime validation
   instead of rigid dataclasses. See docs/design_decisions.md Decision 003.

2. **Minimal Contracts**: Adapters validate only what they need, enabling
   future agents without schema changes.

3. **EmbeddingOutputs Pattern**: Universal data interchange format inspired
   by manylatents, allowing arbitrary field injection.

## Configuration Utilities

For config merging and building, see `manyagents.config_utils`:
- `load_manylatents_experiment()`: Load experiment configs with overrides
- `deep_merge()`: Recursive dict merging
- `build_hydra_overrides()`: Convert dicts to Hydra override strings
- `build_manylatents_config_with_hydra_zen()`: Experimental hydra-zen builder

### Hydra-Zen: When and Why?

**Use hydra-zen for**: Loading known schemas (manylatents experiments)
**Don't use for**: General config building (defeats flexibility)

**What hydra-zen buys us**:
- Type hints from function signatures
- Validation at build time
- Structured merging with interpolation

**What we give up**:
- Flexibility to inject arbitrary fields
- Simplicity (more complex API)
- Agent autonomy (LLMs need to learn hydra-zen)

**Current approach**: Use plain Hydra compose + OmegaConf.merge for experiment
loading. Keep hydra-zen as experimental prototype for potential future use.
"""
__version__ = "0.1.0"