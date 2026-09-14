# manyAgents

Multi-agent orchestration for scientific workflows. Hydra + pydantic + uv. Version **0.2.0** (unreleased; 0.1.1 is the public tag).

manyAgents asks many language models which analysis fits a dataset's geometry, scores the text against configured expectations, and for local models records hidden-state trajectories that manyLatents measures. This file is for agents editing the repo. Newcomer material is `README.md`; the 3×3 design and scoring vocabulary are `manyagents/configs/experiment/README.md`; the adapter table, result contract, tool loop, and DR workflows are `docs/python-api.md`. Point at those; do not duplicate them here.

## What belongs here

Adapters, orchestration, LLM metrics, reasoning trace capture, the tool-calling loop. Anything that coordinates external tools or models.

**Do NOT put here:**
- Geometric metrics or DR algorithms (manyLatents)
- Run orchestration, recipe/dataset catalogs, artifact stores (manyRuns)
- Cluster configs, SLURM launchers (Shop)

## The contract: failure never becomes a number

Every bug in this repo's release history is the same bug: something that could not be measured turned into a plausible number. Fixed instances: a dropped mock response scored as an empty success; exit 0 after every API call failed; Claude multi-block responses truncated to the first block; Biomni scoring its own log as the answer; zero-token trace segments pooled into finite vectors; random scores when no scorer was configured; GVector padding zeros read as measurements; and a model that *rejected* every method scoring as a correct answer. When touching scoring, aggregation, adapters, or trace capture, hold this line:

- An unavailable measurement is `None` in Python, `null` in `results.json`, `n/a` in summaries (`metrics.llm.format_metric`). Never 0, never an empty set counted as agreement.
- A run that produced nothing exits nonzero: bare `manyagents`, empty or unknown `active_agents`, unknown adapter, compute adapter used for text evaluation, zero successful evaluations, zero persisted traces. Partial failures are counted (`prompts_failed`, `traces_failed`), never padded.
- A missing criterion makes the whole rate unavailable; the denominator is never quietly narrowed. Jaccard needs two successful prompts.
- Adapters return `error_response(...)` on failure; a missing dependency is a failed `AdapterResult` naming the extra, not an import error. Text adapters put nonempty text at `output_files["raw_response"]`; the evaluator does not fall back to `summary`.
- Compute/artifact adapters set `PRODUCES_TEXT_RESPONSE = False`; `run_experiment` refuses them before dispatch.
- `GVector` numeric fields are schema padding; provenance lives in `measurements`. Read through `metric_value(name)`, which raises for failed/unrequested/unknown. Legacy records without `measurements` load as `unknown`, never certified.
- Pooling rejects empty, reversed, or out-of-range token intervals with `ValueError`; hidden-state traces need at least two steps to persist. Generated traces are unjudged (`success=None`, `judge="none"`): there is no shipped answer judge.
- `SystemExit` propagates through `manyagents.api.run()`; `except Exception` does not catch it.

Tests protect this by executing the *documented* commands, not workarounds: a green suite once coexisted with seven experiments broken by their own README commands. `tests/test_cli_commands.py` runs every shipped `experiment=<name>` bare, both mock quickstarts, and the `example:evaluation-sweep` marker block from **both** `README.md` and this file; `tests/test_trace_geometry_bridge.py` executes README's `example:trace-geometry` block. Edit those blocks knowing they run: exactly one fenced block per marker pair, bash continuations allowed, and the literal `<!-- example:NAME -->` string may appear nowhere else in the document, prose included, or extraction fails. Regression tests for the list above: `tests/test_fail_closed.py`, `test_experiment_exit.py`, `test_result_contract.py`, `test_gvector_contract.py`.

## Scoring semantics (changed in 0.2.0)

`metrics/extractor.py` drops a method mention under a local English rejection cue ("do not use", "avoid", "instead of", "rather than", "neither … nor", "not appropriate", …) scoped to the sentence/contrast boundary, eight words after a prefix cue, or the next affirmative cue; a separate unrejected mention still counts. `check_ground_truth_match` passes only with at least one extracted expected method **and no extracted `failure_indicators` match** — the failure match is a veto, not a diagnostic. `ground_truth_matches` and `match_ratio` remain coverage diagnostics. Consequences:

- Mock 3×3 match rate is **3/9 (33.3%)**, down from 6/9; `test_wandb` mock is 0.5; the shipped sweep is 0.25 per agent. Tests pin all three.
- Rates are **not comparable across 0.1.1 → 0.2.0**. Re-extract from saved `raw_response` text; `extracted_methods` alone cannot recover rejections.
- **Jaccard is an invariance signal, not an objective.** It averages *all* successful prompt pairs, same-geometry pairs included; two empty sets score 1.0; one consistent method set per geometry, disjoint across geometries, floors at 0.25 on the 3×3 design; inconsistency *within* a geometry lowers it. Do not "improve" it, and do not add a geometry-aware variant without deliberately changing the published definition.
- It is a vocabulary heuristic, not an answer judge: hedges, quoted advice, and distant negation are not adjudicated. Never describe it as a judge.

Design table and vocabulary: `manyagents/configs/experiment/README.md`. Migration: `CHANGELOG.md`.

## Install and Hardware

Python **3.11–3.12**. `uv sync` = core (API adapters, mock, local HF generation; accelerate pulls torch). `--extra traces` = manylatents (`>=0.1.7,<0.2`), datasets (GSM8K), scipy: required for any hidden-state capture, `ManyLatentsAdapter`, and DR workflows. `--extra vllm` = vLLM generation only; add `traces` for replay. `--extra full` = traces + wandb + Biomni, **not** vLLM. pytest/ruff come from the default dev group; `--extra dev` adds pre-commit.

Sync every needed extra in one command, then activate `.venv` or prefix with `uv run --no-sync` so they stay put. Laptop: API clients, Ollama, mock, small HF models. GPU: large HF, vLLM. vLLM defaults to `dtype: bfloat16` with no fallback; on V100/RTX 8000 pass `agent.config.dtype=float16`.

## Entry Points

```bash
manyagents                                                        # lists experiments, exits nonzero
manyagents experiment=test_wandb                                  # two mock prompts; no keys, GPU, downloads
manyagents experiment=geometric_reasoning 'active_agents=[mock]'
manyagents experiment=geometric_reasoning 'active_agents=[claude,openai]'
manyagents experiment=trace_extraction agent=claude agent.config.capture_hidden_states=false
manyagents experiment=trace_extraction --cfg job                  # inspect only; rejected with --multirun
manyagents experiment=geometric_reasoning 'active_agents=[claude,openai]' cluster=mila_remote resources=api  # needs Shop + site access
```

<!-- example:evaluation-sweep -->
```bash
manyagents --multirun experiment=invariance_full 'active_agents=[claude],[openai],[local_llm]' 'output_dir=${hydra:runtime.output_dir}'
```
<!-- /example:evaluation-sweep -->

`invariance_full` defines agents `claude`, `openai`, `local_llm`, `biomni` — not `hf` or `mock`. `local_llm` loads HF with `Qwen/Qwen3-0.6B`; override `agents.local_llm.agent.config.model`. The `output_dir` override keeps every sweep job's results. Evaluation (3×3 scenarios) and `trace_extraction` (GSM8K) are separate workflows; nothing measures hidden states while making biological recommendations.

From Python, `manyagents.api.run(overrides)` composes and runs one job (never a multirun). `ManyLatentsAdapter.execute_cached(...)`, `agent_loop.run_agent_loop(...)`, and `workflows.sequence.execute_sequence(...)` are documented in `docs/python-api.md`.

## Core Abstractions

**Adapter protocol** — defined in `adapters/base.py` (`types.py` re-exports `AdapterResult`):

```python
class AgentAdapter(ABC):
    PRODUCES_TEXT_RESPONSE: bool = True
    async def run(self, task_config: dict[str, Any], input_files: dict[str, Path]) -> AdapterResult
```

`AdapterResult` is a TypedDict `{success: bool, summary: str, output_files: dict, metadata?: dict, embeddings?: dict}`; the first three are required even on failure and `validate_adapter_result()` rejects otherwise. Helpers: `success_response()`, `error_response()`, `save_response()` (returns `{"raw_response": Path}`), `save_text_output()`, `save_json_output()`. Response files are overwritten by later calls; `results.json` is the durable record.

**Type system** — schema-on-read: `TaskConfig = dict[str, Any]`, validated at boundaries by `validate_task_config()` / `validate_adapter_result()` in `types.py`. `EmbeddingOutputs` is a deprecated alias for `dict[str, Any]`.

**Reasoning traces** (`schemas/reasoning.py`):

```python
from manyagents.schemas import ReasoningTrace, TraceStore

trace.steps    # list[ReasoningStep]: index, text, kind (StepKind), token_count, has_hidden_states, layers_captured
trace.model    # ModelInfo: name, backend (ModelBackend), path, revision, generation_config
trace.task     # TaskInfo: dataset, task_id, prompt, expected_answer, domain, logic_type, metadata
trace.success  # Optional[bool]; None until an external judge sets it
```

Adapters put trace JSON at `output_files["trace"]` and NPZ states at `output_files["hidden_states"]`; load with `ReasoningTrace.from_json(path.read_text())`. `TraceStore` writes `traces.jsonl` plus `tensors/{trace_id}.npz`. `GVector` and `TransformationTrajectory` (`schemas/gvector.py`, `trajectory.py`) carry per-step geometry for DR workflows under the contract above.

## Config System

Hydra groups under `manyagents/configs/` (details: `docs/config_groups.md`):

```
agent/       claude, openai, ollama, hf, local_llm, local_llm_70b, vllm, mock, biomni
experiment/  test_wandb, geometric_reasoning, invariance_golden, invariance_full,
             invariance_compare_models, trace_extraction, reasoning_baseline,
             baseline_sweep, llm_reasoning_sweep           (its README.md documents the 3×3)
cluster/     local, mila_remote, mila_slurm, mila_sweep
resources/   api, cpu, gpu, local_llm                        (optional; pairs with cluster=)
logger/      minimal, wandb
prompts/     discrete, periodic, spatial, trajectory, geometric_reasoning/*, reasoning_baseline/*
main.yaml    root: cluster=local, optional resources/experiment, output_dir=outputs/${name}_${now:...}
```

Experiment configs are `# @package _global_`. Evaluation names agents under `agents.<name>.agent.{adapter,config}` and selects with `active_agents`; single-agent trace extraction uses `agent=<group>` and `agent.config.*`. Shipped defaults: Claude and Biomni `claude-opus-5` (adapter `DEFAULT_MODEL`/`DEFAULT_LLM` agree), OpenAI `gpt-4o`, HF/vLLM `Qwen/Qwen3-0.6B`, Ollama tag `qwen3`. Pin models explicitly when results must be comparable.

## Adapters

11 classes, 12 keys in `ADAPTER_REGISTRY` (`adapters/__init__.py`): `mock claude openai ollama hf local_llm(=hf) vllm cellforge kosmos placeholder`, plus `manylatents` and `biomni` when their imports succeed. Full table: `docs/python-api.md`. What not to get wrong:

- **Hidden states** come only from `hf` (captured during generation) and `vllm` (generate, then one HF forward pass over the exact emitted token ids); both need `traces`. `ollama` is an OpenAI-compatible client against a local server: quantized GGUF, no hidden states, and bolting a replay onto it reintroduces the model-loading cost plus quantization/tokenizer confounds. `claude`/`openai` give text-only traces.
- **HF trace path:** `HFAdapter → inference.extract_trace → generate_with_hidden_states → generate(output_hidden_states=True)`, last position per decode step. Shipped `layers: [-1]` is *post*-final-norm. `inference.forward_hidden_states(capture_prenorm=True)` and `extract_trace(state_dtype="float32")` exist, but neither `HFAdapter` nor `VLLMAdapter` forwards `capture_prenorm` or `state_dtype`: the Hydra path stores float16 post-norm only. `inference.generate_with_hooks` (manylatents `ActivationExtractor`) has **no callers**; keep it documented as unwired, not as the HF trace path.
- **Compute adapters** (`manylatents`, `cellforge`, `kosmos`, `placeholder`) set `PRODUCES_TEXT_RESPONSE = False`. `ManyLatentsAdapter.execute_cached(algorithm, params, data)` returns `scores` with `None` for each failed metric.
- **Trust boundary:** Biomni runs `biomni.agent.A1.go` in-process via `asyncio.to_thread` (no isolation; a timed-out await does not stop the thread). CellForge and Kosmos spawn subprocesses. `agent_loop` tool bodies run with caller permissions. Trusted task configs only.
- `MetricRegistry()` (`adapters/metric_registry.py`) discovers manylatents metric YAML in memory at use time; `manyagents-generate-registry` prints JSON and only `--output` writes a cache. Wheels build without manylatents.

## Adding a New Adapter

1. `manyagents/adapters/<name>_adapter.py`: subclass `AgentAdapter`; `async run(task_config, input_files) -> AdapterResult`; return `success_response(summary=..., output_files=self.save_response(text))` or `error_response(...)`; lazy-import optional deps and name the extra in the error; set `PRODUCES_TEXT_RESPONSE = False` if it does not answer in text.
2. `manyagents/adapters/__init__.py`: import and `ADAPTER_REGISTRY` entry, inside `try/except ImportError` when the dependency is optional.
3. `manyagents/configs/agent/<name>.yaml`: `# @package _global_` with `agent: {name, adapter, config}`.
4. Test success **and** failure in `tests/test_<name>_adapter.py` or `manyagents/adapters/test_adapters.py`. A failure is `success: False`; never a success with empty text.

## Key Files

| File | What it does |
|------|-------------|
| `main.py`, `api.py` | Hydra CLI entry; `api.run(overrides)` is the in-process equivalent (one job, `SystemExit` propagates) |
| `experiment.py` | `run_experiment()`: selection/agent validation and exits, prompt dispatch, `_add_ground_truth_matching`, `_run_trace_extraction`, `results.json`/`summary.md` |
| `inference.py` | Plain functions, module-level model cache: `load_model`, `generate*`, `forward_hidden_states` (+ `_batched`, `forward_recurrent_states`, `capture_prenorm`), `segment_*`, `pool_hidden_states_per_step`, `extract_trace(_batch)` |
| `metrics/extractor.py` | Vocabulary/phrase extraction with rejection filtering; `check_ground_truth_match` (veto) |
| `metrics/llm.py` | `compute_system_metrics`, pairwise Jaccard, `format_metric` (`n/a`), summary tables |
| `agent_loop.py`, `tools.py` | Provider-agnostic tool loop over adapters exposing `chat()`; `Tool` = JSON schema + trusted callable |
| `adapters/base.py` | `AgentAdapter`, `AdapterResult`, response helpers |
| `types.py` | `TaskConfig`, boundary validators, embedding-output helpers |
| `schemas/reasoning.py`, `gvector.py`, `trajectory.py` | `ReasoningTrace`/`TraceStore`; `GVector` measurements contract; `TransformationTrajectory` |
| `workflows/sequence.py` | `execute_sequence`: chained manylatents DR with a `GVector` per step |
| `utils/scoring.py`, `utils/validation.py` | Scorer/validator lookup by `module:function`; missing ones raise; `random_scores` only by explicit selection and rows carry `score_simulated=True` |
| `config_utils.py` | `load_manylatents_experiment()`, `deep_merge()`, `build_hydra_overrides()` |

## Ecosystem Boundary Rules

- **Never import from manyRuns.** manyRuns (run harness, not yet public) imports `manyagents.adapters` and `manyagents.agent_loop`; the arrow never points back. Its predecessor Geomancy is deprecated; do not reintroduce it either. No boundary script ships here: `grep -rn "manyruns\|geomancy" manyagents/` must stay empty.
- **manyLatents is optional.** Lazy-import inside methods, guard registry entries with `try/except ImportError`, mark tests `requires_manylatents`. Compute belongs there; manyagents coordinates it.
- **GlobalHydra clearing** happens inside `manylatents.api.run()`; do not clear it in adapters.
- **`compute_metric()` returns `float`** (manylatents, since March 2026). Use `compute_metric_detailed()` for per-sample arrays.

## Gotchas

- **`uv run --no-sync`, never bare `python`**, or activate `.venv`. Plain `uv run` re-syncs and drops extras.
- **`run()` is async**: `asyncio.run()` or `await`. `api.run()` uses `asyncio.run` itself; inside a running loop call `experiment.run_experiment(cfg)`.
- **`--cfg job`** inspects only and is rejected alongside `--multirun`.
- **Hidden-state `state_dtype` defaults to `"float16"`**, which overflows massive-activation channels (Sun et al. 2024). Faithful geometry needs direct `inference.extract_trace(state_dtype="float32")`; the adapter path cannot request it yet.
- **Shipped defaults are executed by tests:** `test_shipped_default_invocation_completes` runs every experiment bare (`baseline_sweep` = 36 jobs, `llm_reasoning_sweep` = 4). Changing a config's defaults changes what that test runs.
- **Mock is deterministic and pinned:** `tests/test_mock_fixture_scores.py` and `test_cli_commands.py` assert its rates; changing the fixture text changes numbers quoted in README and CHANGELOG.
- **Version lives in two places:** `pyproject.toml` and `manyagents/__init__.py`; releases also update `uv.lock` and `CHANGELOG.md`.

## Tests

```bash
uv run --no-sync pytest -q                                        # tests/ + manyagents/ (adapter tests)
uv run --no-sync pytest tests/test_smoke.py -q
uv run --no-sync pytest manyagents/adapters/test_adapters.py -q
```

Baseline on 2026-09-14 with `traces` installed and no vLLM: **499 passed, 5 skipped** (all five need vLLM or CUDA). Core-only installs also skip `requires_manylatents` cases. Reproduce a behavior change with a failing test that executes the real command or result, not one that only composes config.

## Pre-push Checklist

```bash
uv run --no-sync pytest -q
uv run --no-sync ruff check manyagents/ tests/ scripts/            # ruff select: E4, E7, E9, F
```

CI (`.github/workflows/ci.yml`) runs exactly these on locked core and `traces` installs. `uv run --no-sync pre-commit install` enables the same ruff hook locally.
