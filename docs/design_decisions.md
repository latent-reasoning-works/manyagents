# Design Decisions

This document tracks architectural and design decisions made during manyAgents development. It serves as a reference for understanding why certain approaches were chosen and helps maintain consistency as the project evolves.

---

## Phase 1: Principled Orchestrator

### Decision 001: Explicit Algorithm Registry over Auto-Discovery
**Date**: 2025-10-06
**Status**: Adopted
**Context**: When integrating manylatents algorithms into manyAgents, we needed to decide between auto-discovering available algorithms via filesystem scanning versus maintaining an explicit registry.

**Decision**: Use explicit algorithm registration in the adapter layer.

**Rationale**:
- **Stable API Contract**: manyAgents is an orchestration layer that needs a well-defined, versioned interface
- **Selective Exposure**: Not all algorithms in manylatents may be suitable for agent use (some experimental)
- **Better Error Messages**: Early validation of algorithm names with helpful suggestions
- **IDE Support**: Explicit definitions enable better autocomplete and type checking
- **Documentation**: Clear, maintainable list of supported operations
- **Debugging**: Easier to trace issues when the connector layer is explicit

**Alternatives Considered**:
- Auto-discovery via `inspect` or filesystem scanning
  - Pros: Zero maintenance, automatic updates
  - Cons: Unclear API surface, potential for breaking changes, harder to version

**Implementation Notes**:
- Add a `list_available_algorithms()` helper in manylatents API for discoverability
- Document supported algorithms in adapter docstrings
- Version the adapter API separately from manylatents version

---

### Decision 002: Direct Python API over CLI/Subprocess
**Date**: 2025-10-06
**Status**: Adopted
**Context**: Initial prototype used subprocess calls to invoke manylatents CLI. This added overhead and made data passing inefficient.

**Decision**: Refactor to use direct Python API calls via `manylatents.api.run()`.

**Rationale**:
- **Performance**: Eliminates subprocess overhead and serialization costs
- **In-Memory Data**: Enable direct numpy array passing between workflow steps
- **Error Handling**: Direct access to Python exceptions and stack traces
- **Type Safety**: Full IDE support and type checking
- **Simplicity**: Fewer moving parts, easier to debug

**Changes Made**:
- Created `manylatents.api.run()` as programmatic entry point
- Refactored `ManyLatentsAdapter` to use API directly
- Deprecated and removed `ManyLatentsExecutor` class
- Updated all documentation

**Migration Path**:
- Since we're pre-release, full removal of deprecated code is acceptable
- No backward compatibility needed

---

### Decision 003: Schema-on-Read with Flexible Dicts over Rigid Dataclasses
**Date**: 2025-10-10
**Status**: Adopted
**Context**: ManyAgents orchestrates diverse agents (manylatents, BioDiscoveryAgent, CellForge) where each agent has completely different parameter requirements. We needed to decide how to type and validate configuration data passed between the orchestrator and adapters.

**Decision**: Use flexible `dict[str, Any]` with runtime validation instead of rigid dataclasses or per-agent schemas.

**Rationale**:

**The Problem with Dataclasses:**
- **Schema Explosion**: Would require one dataclass per agent type, plus one per algorithm within each agent (PCAConfig, PHATEConfig, UMAPConfig, etc.)
- **Maintenance Burden**: Hundreds of dataclasses requiring constant updates as agents evolve
- **Breaking Changes**: Adding new fields to agents would break all existing workflows
- **Unpredictable Needs**: We cannot predict the configuration needs of all future agents (especially LLM-driven agents that write their own configs)

**The Schema-on-Read Approach:**
Instead of enforcing schemas upfront (schema-on-write), we validate at runtime (schema-on-read):
1. Workflow configs can pass ANY parameters to agents (no schema restrictions)
2. Each adapter validates ONLY what it needs (minimal contract)
3. Agents can inject custom outputs without schema changes
4. Future agents can be added without modifying manyagents core

**Example Comparison:**

Rigid dataclass (brittle):
```python
@dataclass
class ManyLatentsConfig:
    algorithm: str
    data: str
    n_components: int
    # Breaking change when new field added:
    callbacks: dict  # All workflows must update!
```

Schema-on-read (flexible):
```python
# v1.0 workflow - still works
{"algorithm": "pca", "data": "swissroll", "n_components": 2}

# v2.0 workflow - adds callbacks naturally
{"algorithm": "pca", "callbacks": {"save": {"format": "csv"}}}
# Both work! Validation happens only when adapter needs specific fields.
```

**Inspiration from manylatents:**
This mirrors manylatents' successful `EmbeddingOutputs` pattern:
- `EmbeddingOutputs = dict[str, Any]` (not a dataclass!)
- Requires only "embeddings" key, everything else optional
- Algorithms inject custom fields freely (participation_ratio, curvature, etc.)
- Runtime validation via `validate_embedding_outputs()` function

**TypedDict for IDE Support:**
We use `TypedDict` with `total=False` to get IDE autocomplete while maintaining flexibility:
```python
class AdapterResult(TypedDict, total=False):
    success: bool  # IDE knows about this
    summary: str
    # But agents can add custom fields at runtime
```

**Enabling Autonomous Agents:**
When LLM agents need to write configs programmatically:
1. They build plain Python dicts (no schema constraints)
2. Use templates/guides as context
3. Validation happens at execution (fail-fast with clear errors)
4. Agent learns from validation errors (RL feedback loop)

This enables agents to explore configuration space without being locked into predetermined schemas.

**Validation Strategy:**
Rather than enforcing schemas upfront, we:
1. Accept flexible dicts
2. Validate at execution time (adapter-specific validators)
3. Provide clear, actionable error messages
4. Enable agents to learn from failures

Overhead is minimal (dict validation is ~microseconds) while preventing schema mismatches between agents.

**Alternatives Considered**:
- **Pydantic Models**: Powerful validation but still requires schema per agent
  - Pros: Runtime validation, JSON schema generation
  - Cons: Schema explosion, breaking changes on updates
- **Hydra Structured Configs**: Better type safety but too rigid
  - Pros: Compile-time checking, better IDE support
  - Cons: Requires defining all possible configs upfront
- **Protocol Types**: Runtime duck-typing
  - Pros: More Pythonic, flexible
  - Cons: Harder to document, unclear failure modes

**Implementation Notes**:
- Core types defined in `manyagents/types.py`
- Validation functions mirror manylatents pattern
- Each adapter implements its own specific validators
- Shipped Hydra config examples live in `manyagents/configs/`

---

## Decision Log Index

| ID | Title | Status | Phase |
|----|-------|--------|-------|
| 001 | Explicit Algorithm Registry | Adopted | Phase 1 |
| 002 | Direct Python API over CLI | Adopted | Phase 1 |
| 003 | Schema-on-Read with Flexible Dicts | Adopted | Phase 1 |
