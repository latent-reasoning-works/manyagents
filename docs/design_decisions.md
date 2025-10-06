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

## Future Decisions

This section will be populated as we progress through Phase 2 (Agentic Planner) and Phase 3 (Learning Agent).

### Template for New Decisions

```markdown
### Decision XXX: [Title]
**Date**: YYYY-MM-DD
**Status**: [Proposed | Adopted | Deprecated | Superseded]
**Context**: [What situation led to this decision?]

**Decision**: [What did we decide?]

**Rationale**:
- [Reason 1]
- [Reason 2]

**Alternatives Considered**:
- [Alternative 1]: Pros/Cons
- [Alternative 2]: Pros/Cons

**Implementation Notes**:
- [Note 1]
- [Note 2]
```

---

## Decision Log Index

| ID | Title | Status | Phase |
|----|-------|--------|-------|
| 001 | Explicit Algorithm Registry | Adopted | Phase 1 |
| 002 | Direct Python API over CLI | Adopted | Phase 1 |
