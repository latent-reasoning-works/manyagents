"""
ManyAgents: LLM agent testing framework.

Tests whether LLMs can reason about data geometry by dispatching
prompts to multiple agents (Claude, GPT-4, local LLMs) and evaluating
their responses against ground truth methods.

Downstream consumers can compose adapters into their own workflows.

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
"""
__version__ = "0.2.0"
