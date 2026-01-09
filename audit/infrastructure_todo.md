# Infrastructure TODO

**Date:** 2026-01-08
**Status:** COMPLETE

---

## Completed Components

### 1. Claude Adapter [DONE]
- [x] Created `manyagents/adapters/claude_adapter.py`
- [x] Mirrors OpenAI adapter structure with Anthropic SDK
- [x] Registered in `ADAPTER_REGISTRY` in main.py
- [x] Added to `__init__.py` exports
- [x] Added `anthropic>=0.20.0` to dependencies

### 2. Experiment Infrastructure [DONE]
- [x] Created `manyagents/experiment/` module
- [x] `runner.py`: Parallel dispatch and experiment orchestration
- [x] `extractor.py`: Method extraction from LLM responses
- [x] `metrics.py`: Jaccard similarity, ground truth matching, summary tables

### 3. Test Configurations [DONE]
- [x] Created `tests/golden_trajectory_test.json`
- [x] Created `tests/experiment_config.json` (4 scenarios)

### 4. Run Script [DONE]
- [x] Created `scripts/run_invariance_experiment.py`

---

## Verification

All imports verified working:
```
Available adapters: ['manylatents', 'openai', 'cellforge', 'biomni', 'kosmos', 'claude']
```

---

## To Run Tests

```bash
# Activate environment
source .venv/bin/activate

# Install dependencies (if needed)
uv sync

# Run golden test
python scripts/run_invariance_experiment.py --golden --systems claude,openai

# Run full experiment
python scripts/run_invariance_experiment.py --full
```

---

## Files Created

```
manyagents/
├── adapters/
│   └── claude_adapter.py          # NEW: Claude API adapter
├── experiment/
│   ├── __init__.py                # NEW: Module exports
│   ├── runner.py                  # NEW: Parallel dispatch & orchestration
│   ├── extractor.py               # NEW: Method extraction
│   └── metrics.py                 # NEW: Jaccard, ground truth metrics
audit/
├── manyagents_capabilities.md     # NEW: Capability audit
└── infrastructure_todo.md         # NEW: This file
tests/
├── golden_trajectory_test.json    # NEW: Golden test spec
└── experiment_config.json         # NEW: Full experiment config
scripts/
└── run_invariance_experiment.py   # NEW: Run script
results/                           # NEW: Output directory (will contain results)
```

---

*Completed: 2026-01-08*
