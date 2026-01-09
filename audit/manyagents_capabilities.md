# manyAgents Capability Audit

**Date:** 2026-01-08
**Purpose:** Assess manyAgents infrastructure for pipeline invariance experiment
**Status:** COMPLETE

---

## Executive Summary

manyAgents provides a **solid foundation** for running the pipeline invariance experiment, but requires **moderate infrastructure additions** to fully support the experiment. The main gaps are:

1. **No Claude adapter** - Need to add for direct Claude API access
2. **Sequential execution only** - Parallel agent dispatch not yet implemented
3. **No JSON schema validation** - Manual post-processing required
4. **No cross-agent aggregation** - Need to build metrics computation

**Estimated effort to unblock golden test:** 2-3 hours
**Estimated effort for full experiment:** 4-6 hours

---

## 1. Parallel Agent Dispatch

### Current Status: ❌ NOT SUPPORTED (Sequential Only)

**What exists:**
- Sequential workflow execution via `execute_workflow_chain()`
- Individual adapters use async/await but steps run one-after-another
- SLURM integration available via `hydra/launcher=submitit_slurm` for cluster-level parallelism

**What's needed for experiment:**
- Concurrent dispatch of same prompt to multiple AI systems
- `asyncio.gather()` wrapper for parallel adapter execution
- Results collection from all parallel runs

**Complexity:** MODERATE (~50-100 lines of code)

---

## 2. Structured Output Collection

### Current Status: ⚠️ PARTIAL SUPPORT

**What exists:**
- Standard `AdapterResult` TypedDict format
- OpenAI adapter supports `response_format: "json_object"` with JSON parsing
- Pydantic v2 is a dependency but minimally used

**What's missing:**
- No JSON Schema generation or validation
- No standardized extraction of "recommended methods" from LLM responses

**Complexity:** TRIVIAL to MODERATE (~100-150 lines of code)

---

## 3. Validation Hooks

### Current Status: ⚠️ PARTIAL SUPPORT

**What exists:**
- Input validation: `validate_task_config()`, `validate_adapter_result()`
- Generic scoring/filtering pipelines in `utils/scoring.py` and `utils/validation.py`

**What's missing:**
- Post-completion schema validation hooks
- Ground-truth matching logic

**Complexity:** MODERATE (~100 lines of code)

---

## 4. Results Aggregation

### Current Status: ⚠️ PARTIAL SUPPORT

**What exists:**
- Workflow state accumulates step results
- Multi-workflow comparison (anchor vs comparison) in Phase 2 configs

**What's missing:**
- Cross-agent summary statistics (Jaccard similarity, match rates)
- Markdown table generation for paper

**Complexity:** TRIVIAL to MODERATE (~100 lines of code)

---

## 5. Available Adapters

### Current Registry:

| Adapter | Status | API Type | Suitable for Experiment |
|---------|--------|----------|------------------------|
| `manylatents` | ✅ Mature | Python API | No (DR algorithms, not LLM) |
| `openai` | ✅ Mature | AsyncOpenAI | ✅ Yes (GPT-4) |
| `cellforge` | ✅ Integrated | CLI subprocess | Possible (multi-agent) |
| `biomni` | ✅ New | Python API | ⚠️ Uses Claude internally |
| `kosmos` | ✅ New | CLI subprocess | ⚠️ Research agent, not direct LLM |

### Missing Adapters:

| Needed | Status | Effort |
|--------|--------|--------|
| **Claude (direct)** | ❌ Missing | TRIVIAL - mirror OpenAIAdapter |
| **Local models** | ❌ Missing | MODERATE - need inference setup |

---

## 6. Gap Summary

| Capability | Status | Complexity | Lines of Code |
|------------|--------|------------|---------------|
| Claude Adapter | ❌ Missing | TRIVIAL | ~150 |
| Parallel Dispatch | ❌ Missing | MODERATE | ~100 |
| Method Extraction | ⚠️ Missing | TRIVIAL | ~80 |
| Ground Truth Matching | ⚠️ Missing | TRIVIAL | ~60 |
| Jaccard Similarity | ⚠️ Missing | TRIVIAL | ~40 |
| Summary Generation | ⚠️ Missing | TRIVIAL | ~50 |
| **TOTAL** | | | **~480 lines** |

---

## 7. Recommendations

### Immediate (Unblock Golden Test)

1. **Create `ClaudeAdapter`** - Mirror OpenAIAdapter with Anthropic SDK
2. **Create experiment runner script** - Orchestrate parallel dispatch manually
3. **Create method extractor** - Regex/keyword-based extraction

### Short-term (Full Experiment)

4. **Add parallel dispatch wrapper** - `async def run_parallel_agents()`
5. **Add results aggregation** - Compute metrics and generate summary

---

*Audit completed: 2026-01-08*
