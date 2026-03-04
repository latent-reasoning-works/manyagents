# Trace Extraction Consolidation Design

**Date:** 2026-03-04
**Status:** Approved
**Scope:** manyagents core — inference.py, adapters, Hydra configs

## Problem

Trace extraction logic is scattered across three files with incompatible implementations:

| Concern | extract_traces.py | LocalLLMAdapter | ClaudeAdapter |
|---|---|---|---|
| ReasoningTrace construction | Multi-step (correct) | Single-step (wrong) | None |
| Hidden state schema | `{pooled_steps, token_level}` | `{layer_path: raw}` | N/A |
| Step splitting + pooling | Yes | No | No |
| Hydra integration | None (hand-rolled CLI) | Yes (adapter) | Yes (adapter) |

Geomancy needs a clean, callable interface to produce `(ReasoningTrace, hidden_states)` from any backend, consumable by `ReasoningTraceDataModule` in manylatents.

## Design

### 1. Functional API additions to `inference.py`

Three new functions. No new classes.

```python
# --- Module-level model cache ---

_model_cache: dict[str, tuple] = {}

def get_model(model: str, dtype=None) -> tuple[nn.Module, Any, Any]:
    """Cached model loading. Resolves name/HF ID, loads once per process."""
    model_path = resolve_model_path(model)
    if model_path not in _model_cache:
        _model_cache[model_path] = load_model(model_path, dtype=dtype)
    return _model_cache[model_path]

def clear_model_cache():
    """Free GPU memory. Call between runs or at process exit."""
    _model_cache.clear()
    torch.cuda.empty_cache()


# --- Trace building ---

def build_reasoning_trace(
    text: str,
    gen_metadata: dict,
    model_name: str,
    model_path: str,
    task: TaskInfo,
    step_defs: list[dict],
    generation_config: dict,
) -> ReasoningTrace:
    """Single source of truth for local-model trace construction.

    Replaces: extract_traces.py lines 117-153, LocalLLMAdapter._build_and_save_trace()
    """


def extract_trace(
    model, tokenizer,
    prompt: str,
    task: TaskInfo,
    model_name: str,
    model_path: str,
    *,
    system_prompt: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    layers: list[int] | None = None,
    step_delimiter: str = "\n",
) -> tuple[ReasoningTrace, dict[str, np.ndarray]]:
    """Full pipeline: prompt -> generate -> split -> pool -> (trace, hidden_states).

    Composes existing functions:
        build_prompt() -> generate_with_hidden_states() ->
        split_into_steps() -> pool_hidden_states_per_step() ->
        build_reasoning_trace()

    Returns:
        trace: ReasoningTrace with multi-step decomposition
        hidden_states: {
            "pooled_steps": ndarray (n_steps, n_layers, d_model) float16,
            "token_level": ndarray (n_tokens, n_layers, d_model) float16,
        }

    This output format matches what ReasoningTraceDataModule expects.
    """
```

### 2. Adapter cleanup

**Rename:** `LocalLLMAdapter` -> `HFAdapter`

File rename: `local_llm_adapter.py` -> `hf_adapter.py`

Changes:
- Delete `_build_and_save_trace()` — replaced by `inference.build_reasoning_trace()`
- When `build_trace=True`: delegate to `inference.extract_trace()`
- Hidden state output always uses `{pooled_steps, token_level}` schema
- Model loading uses `inference.get_model()` (shared cache) instead of own global lock

Registry backward compat:
```python
ADAPTER_REGISTRY = {
    "hf": HFAdapter,
    "local_llm": HFAdapter,  # alias
    ...
}
```

**ClaudeAdapter trace wiring:**
- When `build_trace=True` in task_config: call existing `trace_from_anthropic()` helper
- Returns `ReasoningTrace` in `AdapterResult` metadata
- `trace.has_tensors = False` (no hidden states from API)
- Dual-channel: HF traces have hidden states, Claude traces have thinking blocks. Same schema.

**Config rename:** `configs/agent/local_llm.yaml` -> `configs/agent/hf.yaml`
- Keep `local_llm.yaml` as import alias for backward compat

### 3. Hydra experiment config for trace extraction

Trace extraction becomes a standard experiment, not a standalone script.

```yaml
# configs/experiment/trace_extraction.yaml
defaults:
  - /agent: hf

agent:
  config:
    model: Qwen/Qwen3-4B
    build_trace: true
    capture_hidden_states: true
    layers: [-1]
    step_delimiter: "\n"
    temperature: 0.7
    max_new_tokens: 512

dataset: gsm8k
n_samples: 10
output_dir: outputs/traces/${agent.config.model}_${dataset}
```

Run via existing entry point:
```bash
python -m manyagents experiment=trace_extraction agent.config.model=Qwen/Qwen3-4B n_samples=500
```

`experiment.py::run_experiment()` gets a trace-extraction code path:
when `build_trace=True`, uses `extract_trace()` + `TraceStore` instead of prompt-dispatch-evaluate.

`scripts/extract_traces.py` is deleted.

### 4. End-to-end data flow

```
PATH 1: Batch (Hydra/SLURM sweep)
    python -m manyagents experiment=trace_extraction ...
      -> HFAdapter.run(build_trace=True)
      -> inference.extract_trace()
      -> TraceStore on disk

    Later: ReasoningTraceDataModule(trace_store_path="...")
      -> (N_total, D) tensor -> manylatents geometric analysis

PATH 2: Tight loop (geomancy programmatic)
    model, tok, _ = inference.get_model("Qwen/Qwen3-4B")
    for prompt in prompts:
        trace, hs = inference.extract_trace(model, tok, prompt, task=...)
        dm = ReasoningTraceDataModule(
            hidden_states=[hs["pooled_steps"]],
            trace_ids=[trace.trace_id],
        )
        # -> manylatents -> G-vector -> reward
    inference.clear_model_cache()

PATH 3: API traces (Claude/OpenAI)
    ClaudeAdapter.run(build_trace=True)
      -> trace_from_anthropic(response, task)
      -> ReasoningTrace (has_tensors=False)
      -> Text-based analysis (step structure, no geometry)
```

All paths produce `ReasoningTrace`. HF paths additionally produce
`{"pooled_steps", "token_level"}` hidden states consumable by
`ReasoningTraceDataModule`.

## Files changed

| File | Action |
|---|---|
| `manyagents/inference.py` | Add `get_model()`, `clear_model_cache()`, `build_reasoning_trace()`, `extract_trace()` |
| `manyagents/adapters/local_llm_adapter.py` | Rename to `hf_adapter.py`, class to `HFAdapter`, delegate to inference functions |
| `manyagents/adapters/__init__.py` | Update registry: `hf` + `local_llm` alias |
| `manyagents/adapters/claude_adapter.py` | Wire `trace_from_anthropic()` when `build_trace=True` |
| `manyagents/configs/agent/local_llm.yaml` | Rename to `hf.yaml`, keep alias |
| `manyagents/configs/experiment/trace_extraction.yaml` | New Hydra experiment config |
| `manyagents/experiment.py` | Add trace-extraction code path |
| `scripts/extract_traces.py` | Delete |
| `tests/` | Update imports, add tests for new inference functions |

## What this does NOT change

- `ReasoningTrace`, `TraceStore`, `ReasoningTraceDataModule` schemas — unchanged
- `manylatents` — no changes needed
- `geomancy` — no changes needed (calls `inference.extract_trace()` directly)
- Other adapters (mock, openai, manylatents, cellforge, kosmos, biomni) — unchanged
