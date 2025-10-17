# Session Continuation Prompt

## Current State

We're implementing the **3-way golden test architecture** for the Geomancer project ecosystem. This session completed **Test 1** and established the foundation for multi-workflow comparison functionality.

### Completed This Session ✅

1. **Test 1: Orchestration Correspondence**
   - Created `tests/test_orchestration_correspondence.py`
   - Validates: `manylatents.api.run() ≡ manyagents.orchestration()`
   - Live comparison (no stored references needed)
   - Auto-discovery of test configs via pytest parametrization

2. **Test Infrastructure**
   - `tests/conftest.py` with pytest fixtures
   - `tests/configs/test_pca_50d.yaml` as first test case
   - Pytest auto-parametrization from discovered configs

3. **Architecture Decision**
   - Removed regression tests (redundant with live correspondence)
   - Decided on `manyagents.workflows` module naming (aligns with manylatents terminology)

4. **Branches Pushed**
   - `geomancer_init_test` in manyagents repo
   - `geomancer_init_test` in Geomancer repo
   - Development plan committed to Geomancer

## Next Steps (Priority Order)

### Immediate: Multi-Workflow Functionality in manyagents

**Goal:** Enable manyagents to run multiple workflows and perform operations on their outputs.

**Tasks:**
1. Create `manyagents/workflows.py` module with:
   ```python
   def execute_multiple(workflow_configs: List[Dict]) -> List[WorkflowResult]
   def subtract_results(result_A, result_B) -> Dict
   def concatenate_results(results: List) -> Dict
   def compare_results(result_A, result_B) -> Dict
   ```

2. Create experiment config: `manyagents/configs/experiment/multi_workflow_comparison.yaml`
   ```yaml
   name: pca_vs_umap_comparison
   workflows:
     baseline:
       steps: [...]
     comparison:
       steps: [...]
   operations:
     - type: subtract
       operands: [baseline, comparison]
       output_file: outputs/g_vector_diff.npy
   ```

3. Update `manyagents/main.py` to support multi-workflow execution mode

4. Add tests for multi-workflow functionality

**Design Question to Resolve:**
- How should multi-workflow results be structured?
- Should operations be configurable via YAML or always available as API functions?

### After Multi-Workflow: Geomancer Tests 2 & 3

**Test 2: Geomancer Dual Invocation**
- Location: `geomancer/tests/test_invocation_paths.py`
- Validates: `geomancer(via_manylatents) ≡ geomancer(via_manyagents)`

**Test 3: Seurat Domain Workflow**
- Location: `geomancer/tests/test_expert_workflows.py`
- Validates: Seurat single-cell workflow produces expected outputs
- Requires: Synthetic PBMC dataset generator

## Key Architectural Decisions

### Naming Conventions
- **`manyagents.workflows`**: Multi-workflow orchestration and result operations (decided)
- Keeps terminology aligned: manylatents has "workflow" (singular), manyagents has "workflows" (plural)
- Reinforces the "many" concept in manyagents

### Testing Strategy
- **Correspondence tests**: Live comparison between orchestration levels (primary)
- **No regression tests**: Redundant when ground truth can be regenerated on-the-fly
- **Config-driven**: Tests auto-discovered from `tests/configs/test_*.yaml`

### Repository Responsibilities
- **manylatents**: Pure geometric computation (single workflows)
- **manyagents**: Multi-agent + multi-workflow orchestration (plural workflows, operations on results)
- **Geomancer**: Domain-specific RL (uses both manylatents and manyagents)

## Files Modified This Session

### manyagents
```
tests/
├── configs/
│   └── test_pca_50d.yaml          [NEW]
├── conftest.py                     [NEW]
└── test_orchestration_correspondence.py  [NEW]
```

### Geomancer
```
GEOMANCER_DEVELOPMENT_PLAN.md       [NEW]
uv.lock                             [UPDATED]
```

## Commands to Resume

```bash
# Activate environment
cd /network/scratch/c/cesar.valdez/manyAgents
source .venv/bin/activate

# Verify branch
git branch --show-current  # Should be: geomancer_init_test

# Run current tests (should pass when manylatents API is ready)
pytest tests/test_orchestration_correspondence.py -v

# Check dependencies
python -c "import manyagents; import manylatents; print('✅ Ready')"
```

## Context for Next Session

Start with: **"Let's continue implementing the multi-workflow functionality in manyagents. We need to create the `manyagents.workflows` module that can execute multiple workflows and perform operations (subtract, concatenate) on their outputs."**

Key context:
- We're on `geomancer_init_test` branch in both repos
- Test 1 (correspondence) is implemented but not yet run (needs manylatents API verification)
- Multi-workflow comparison is THE core manyagents feature for Phase 3
- This enables baseline vs variant comparison and RL reward computation

## Open Questions for Next Session

1. Should `manyagents.workflows.execute_multiple()` be sequential or parallel?
2. How should we handle errors when one workflow fails in multi-workflow execution?
3. Should G_vector operations (subtract, concatenate) support arbitrary numpy operations or just predefined ones?
4. Do we need a `WorkflowResult` dataclass or keep using dictionaries?

---

**Session Date:** 2025-10-17
**Branch:** `geomancer_init_test` (both manyagents and Geomancer)
**Status:** Test infrastructure complete, ready for multi-workflow implementation
