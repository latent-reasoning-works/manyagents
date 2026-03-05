# manyAgents

Multi-agent orchestration for scientific workflows. Hydra + pydantic + uv.

## What belongs here

Adapters, orchestration, LLM metrics, reasoning trace capture. Anything that coordinates external tools or models.

**Do NOT put here:**
- Geometric metrics or DR algorithms (manyLatents)
- Reward computation, G-vectors, RL training (Geomancy)
- Cluster configs, SLURM launchers (Shop)

## Entry Points

```bash
# CLI — primary interface
manyagents experiment=geometric_reasoning

# With specific agents
manyagents experiment=geometric_reasoning active_agents=[claude,openai]

# Extract reasoning traces
manyagents experiment=trace_extraction agent=claude

# Multirun sweep
manyagents --multirun agent=claude,openai,hf experiment=invariance_full

# SLURM submission
manyagents experiment=geometric_reasoning cluster=mila_remote resources=api
```

```python
from manyagents.adapters import ManyLatentsAdapter

adapter = ManyLatentsAdapter()
adapter.setup_metrics(["participation_ratio", "trustworthiness"])
result = await adapter.execute_cached(algorithm="UMAP", params={"n_neighbors": 15}, data=X)
# result.scores: {"participation_ratio": 12.4, "trustworthiness": 0.92}
```

## Core Abstractions

**Adapter protocol** — all adapters implement:

```python
class AgentAdapter(ABC):
    async def run(self, task_config: dict[str, Any], input_files: dict[str, Path]) -> AdapterResult
```

`AdapterResult` is a TypedDict: `{success: bool, summary: str, output_files?: dict, metadata?: dict, embeddings?: dict}`.

Helper methods: `success_response()`, `error_response()`, `save_text_output()`, `save_json_output()`.

**Type system** — schema-on-read, not rigid dataclasses:

```python
TaskConfig = dict[str, Any]           # Flexible agent config
EmbeddingOutputs = dict[str, Any]     # Geometric data interchange (from manyLatents)
```

Validation at boundaries via `validate_task_config()`, `validate_adapter_result()`.

**Reasoning traces** — structured CoT capture:

```python
from manyagents.schemas import ReasoningTrace

trace.steps        # list[ReasoningStep] — each CoT step (thinking, output, tool_call)
trace.model_info   # ModelInfo — model, temperature, tokens
trace.task_info    # TaskInfo — prompt, dataset, experiment
```

Two storage layers: metadata (JSONL per trace) + tensors (npz per model).

## Config System

Hydra config groups under `manyagents/configs/`:

```
agent/          claude, openai, hf, local_llm, local_llm_70b, mock, biomni
experiment/     geometric_reasoning, invariance_golden, invariance_full,
                llm_reasoning_sweep, baseline_sweep, trace_extraction, ...
cluster/        local, mila_remote, mila_slurm, mila_sweep
logger/         minimal, wandb
prompts/        discrete, geometric_reasoning/...
main.yaml       Root config (merges all groups)
```

## Adapters

13 adapters behind `AgentAdapter`:

| Adapter | Type | Import guard |
|---------|------|-------------|
| `ClaudeAdapter` | API | always |
| `OpenAIAdapter` | API | always |
| `HFAdapter` | Local | always (alias: `local_llm`) |
| `ManyLatentsAdapter` | Python | optional (`manylatents`) |
| `CellForgeAdapter` | CLI | always |
| `KosmosAdapter` | CLI | always |
| `BiomniAdapter` | CLI | optional (`biomni>=0.0.2`) |
| `MockAdapter` | Testing | always |
| `PlaceholderAdapter` | Stub | always |

Get an adapter by name: `from manyagents.adapters import get_adapter`.

## Adding a New Adapter

4 files:

1. **Adapter**: `manyagents/adapters/<name>_adapter.py`
   - Subclass `AgentAdapter`
   - Implement `async run(task_config, input_files) -> AdapterResult`
   - Use `success_response()` / `error_response()` helpers
   - Lazy-import optional deps inside methods

2. **Export**: `manyagents/adapters/__init__.py`
   - Add to imports and `get_adapter()` registry

3. **Config**: `manyagents/configs/agent/<name>.yaml`
   - Adapter-specific defaults

4. **Test**: `tests/test_<name>_adapter.py` or add to `manyagents/adapters/test_adapters.py`

## Key Files

| File | What it does |
|------|-------------|
| `main.py` | Hydra CLI entry point |
| `experiment.py` | `run_experiment()` — prompt dispatch, metric extraction, aggregation |
| `inference.py` | Model loading, prompt building, generation (plain functions, no classes) |
| `config_utils.py` | `load_manylatents_experiment()`, `deep_merge()`, `build_hydra_overrides()` |
| `types.py` | `TaskConfig`, `AdapterResult`, `EmbeddingOutputs`, validation functions |
| `adapters/base.py` | `AgentAdapter` ABC, result helpers |
| `metrics/extractor.py` | Parse LLM responses for method recommendations |
| `metrics/llm.py` | LLM-based metric evaluation |
| `schemas/reasoning.py` | `ReasoningTrace`, `ReasoningStep`, `ModelInfo`, `StepKind` |
| `workflows/sequence.py` | DR workflow execution with G-vector tracking |

## Ecosystem Boundary Rules

- **Never import from geomancy.** manyAgents sits below geomancy in the dependency graph.
- **manyLatents is optional.** Guard with `try/except ImportError`. The adapter handles this.
- **GlobalHydra clearing** is handled inside `manylatents.api.run()` — do NOT clear it in adapters.
- **`compute_metric()` returns `float`** since March 2026. Use `compute_metric_detailed()` for per-sample arrays.
- Run `geomancy/scripts/check_boundaries.sh` to verify all rules.

## Gotchas

- **`uv run`, not `python`** — always prefix with `uv run` or activate the venv.
- **Async adapters** — `run()` is async. Use `asyncio.run()` or `await`.
- **Schema-on-read** — configs are dicts, not dataclasses. Validate at boundaries only.
- **`inference.py` is functional** — plain functions, module-level model cache, no classes.
- **`EmbeddingOutputs` is a deprecated alias** — it's just `dict[str, Any]` now.

## Tests

```bash
uv run pytest tests/ -v                    # all tests
uv run pytest tests/test_smoke.py -v       # quick smoke test
uv run pytest manyagents/adapters/test_adapters.py -v  # adapter tests
```

## Pre-push Checklist

```bash
uv run pytest tests/ -x -q
uv run ruff check manyagents/
```
