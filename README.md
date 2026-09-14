<div align="center">

<pre>
    ∿ · ─ · ∿ · ─
  · ─ · ∿ · ─ · ∿ ·  ──▶  Σ(·)
    ─ · ∿ · ─ · ∿

        m a n y a g e n t s

    coordinate, dispatch, aggregate
</pre>

[![CI](https://github.com/latent-reasoning-works/manyagents/actions/workflows/ci.yml/badge.svg)](https://github.com/latent-reasoning-works/manyagents/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-8B5CF6.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11–3.12-8B5CF6.svg)](https://www.python.org)
[![uv](https://img.shields.io/badge/pkg-uv-8B5CF6.svg)](https://docs.astral.sh/uv/)

</div>

[manyagents](https://github.com/latent-reasoning-works/manyagents) provides multi-agent evaluation and reasoning trace extraction for scientific workflows. Dispatch prompts through a shared adapter interface, extract method recommendations, and compare them against expected data geometry.

**Upgrading:** 0.1.1 is not a drop-in upgrade from 0.1.0. Read the [release notes and migration guide](CHANGELOG.md) for adapter results, exit semantics, model pinning, and legacy GVector data.

## Install

Requires Python **3.11–3.12**. From a source checkout:

```bash
uv sync                       # Core: API adapters, mock, local HF generation
uv sync --extra traces        # manylatents hooks/geometry, segmentation, datasets (GSM8K)
uv sync --extra vllm          # Plain vLLM generation on a supported GPU system
uv sync --extra traces --extra vllm  # vLLM generation + HF hidden-state replay
uv sync --extra full          # traces + W&B + Biomni; does not include vLLM
```

Core HF generation does not require manylatents. A core install is still large: `accelerate` pulls in **torch**. Plain vLLM generation needs only the `vllm` extra beyond core and does not load an HF model or use manylatents. Add `--extra wandb` for optional experiment logging. Development tools are synced by default; `--extra dev` also adds pre-commit.

Activate the environment with `source .venv/bin/activate` before the commands below. Alternatively, prefix commands with `uv run --no-sync` to retain the extras you installed.

**What runs where:** laptop = Claude/OpenAI API clients, Ollama server, mock, and small HF models; GPU = large HF models and vLLM. API adapters require `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`; Ollama requires a running local server and a downloaded model.

## Quickstart

```bash
# No API keys, GPU, or model download
manyagents experiment=test_wandb

# Run all nine geometric-reasoning prompts with mock responses
manyagents experiment=geometric_reasoning 'active_agents=[mock]'

# Evaluate API agents (requires both keys)
manyagents experiment=geometric_reasoning 'active_agents=[claude,openai]'

# Three separate jobs; retain each job's results
manyagents --multirun experiment=invariance_full 'active_agents=[claude],[openai],[local_llm]' 'output_dir=${hydra:runtime.output_dir}'
```

The mock deliberately answers every prompt identically: expect Jaccard 1.0 and clustering-for-all 100%. That is the invariance failure this evaluation is designed to detect.

The sweep uses names defined by `invariance_full`: `claude`, `openai`, and `local_llm` (it also defines `biomni`). The local job uses HF with `Qwen/Qwen3-0.6B` by default; select another accessible model with `agents.local_llm.agent.config.model=<Hub-ID-or-path>`. The output override saves each job under Hydra's numbered multirun directory. `--cfg job` only displays configuration; it cannot be combined with `--multirun` or verify execution.

Bare `manyagents` exits with a command hint and the available experiment names. Successful evaluations save `results.json` and `summary.md` under the configured `output_dir`. Zero successful evaluations exit nonzero; partial failures remain recorded alongside successes.

## The 3×3 experiment

`geometric_reasoning` tests three domains—immunology (discrete clusters), cancer (branching trajectories), and development (continuous manifolds)—under three conditions: biology alone, biology with geometry hints, and abstract geometry alone. Its nine prompts ask whether recommendations adapt to the data structure. **Ground-truth match rate up is good:** more answers mention expected methods. **Jaccard similarity up is bad in this evaluation:** the same method sets recur across prompts with different geometries. **Clustering-for-all up is bad:** clustering keeps being recommended regardless of structure. These are method-extraction scores, not a proof of scientific correctness; undefined measurements are `null` in JSON and `n/a` in summaries.

## Adapters and results

**11 adapter classes, 12 registry keys**, including the `placeholder` stub. `local_llm` is a registry alias for `HFAdapter`. Registration does not mean an external dependency or executable is installed.

| Registry key | Class | Execution / requirements |
|-------------|-------|--------------------------|
| `claude` | `ClaudeAdapter` | Anthropic API; text traces, no hidden states |
| `openai` | `OpenAIAdapter` | OpenAI-compatible Chat Completions API; generation |
| `ollama` | `OllamaAdapter` | Local HTTP server; generation, no hidden states |
| `hf`, `local_llm` | `HFAdapter` | Local HF generation in core; hidden-state extraction with `traces` |
| `vllm` | `VLLMAdapter` | `vllm` extra; HF trace replay also needs `traces` |
| `manylatents` | `ManyLatentsAdapter` | In-process DR algorithms and geometric metrics; `traces` |
| `biomni` | `BiomniAdapter` | In-process biomedical agent; `full` and Anthropic API access |
| `cellforge` | `CellForgeAdapter` | External local CLI installation; cell analysis |
| `kosmos` | `KosmosAdapter` | External local CLI installation |
| `mock` | `MockAdapter` | Deterministic configured responses; no external service |
| `placeholder` | `PlaceholderAdapter` | Stub for development |

```python
import asyncio
from manyagents.adapters import ADAPTER_REGISTRY

adapter = ADAPTER_REGISTRY["claude"]()
result = asyncio.run(adapter.run({
    "prompt": "Which methods suit a branching cell trajectory?",
    "build_trace": True,
}, {}))
if not result["success"]:
    raise RuntimeError(result["summary"])
response = result["output_files"]["raw_response"].read_text()
```

`run(task_config, input_files)` is async and returns `{success, summary, output_files, metadata}` (`metadata` is optional in the type contract). Text adapters expose a response path under `output_files["raw_response"]`; the evaluator also accepts inline strings. Compute/artifact adapters are rejected by the text-evaluation runner.

Claude and Biomni default to `claude-opus-5`. OpenAI retains `gpt-4o`, a supported ID verified against the [official API model documentation](https://developers.openai.com/api/docs/models/gpt-4o) on 2026-09-06; this is not a claim that it is OpenAI's newest model.

## Reasoning traces

Traces record model-emitted steps and metadata. Claude can store API text/thinking blocks with `build_trace=True`, but API adapters expose no hidden states. HF and vLLM can capture local hidden states with the extras above.

Continuing the successful Claude example:

```python
from manyagents.schemas import ReasoningTrace

trace = ReasoningTrace.from_json(result["output_files"]["trace"].read_text())
trace.steps  # list[ReasoningStep]
trace.model  # ModelInfo
trace.task   # TaskInfo
```

For local capture, `output_files["hidden_states"]` is a compressed NPZ path. The trace-extraction experiment gathers individual adapter outputs into a `TraceStore`:

```text
<output_dir>/traces/
├── traces.jsonl               # One ReasoningTrace per line
└── tensors/
    └── <trace_id>.npz         # Optional hidden-state arrays
```

```bash
# Requires --extra traces; downloads the model and GSM8K
manyagents experiment=trace_extraction agent=hf agent.config.model=Qwen/Qwen3-0.6B

# GPU trace extraction: requires --extra traces --extra vllm
manyagents experiment=trace_extraction agent=vllm
```

**vLLM hardware assumption:** `configs/agent/vllm.yaml` defaults to `dtype: bfloat16`, with no automatic fallback. Use bf16-capable hardware (Ampere or newer). On V100 or RTX 8000, pass **`agent.config.dtype=float16`**; this leaves Ampere+ defaults and numerics unchanged. See [Running Experiments](docs/running_experiments.md) for the full command.

vLLM trace replay was verified finite on an L40S: float16 `token_level` arrays have shape `(n_tokens, 1, d_model)` and `pooled_steps` arrays have shape `(n_steps, 1, d_model)` with the default single captured layer. The stored replay dtype is distinct from the vLLM engine's generation dtype.

## From traces to geometry

After the capture command above, replace `<output_dir>` with that run's output directory. `TraceStore.load_tensors(trace_id)` reads `tensors/{trace_id}.npz`. The stored `pooled_steps` array is **(steps, captured layers, hidden dimension)**; select a layer by its position in `layers_captured` and cast to float32 to obtain the **(steps, hidden dimension)** input expected by manyLatents.

Use **manyLatents directly** for trajectory velocity and curvature, as below. `ManyLatentsAdapter.run(input_data=...)` supports 2-D arrays for DR without a dummy `data` field, but its cached `setup_metrics` registry discovers YAML-backed metrics. manylatents 0.1.7 supplies no trajectory metric YAMLs, and `execute_cached` cannot consume a 3-D trace tensor. It is not the trace-to-trajectory-metrics interface.

```python
import numpy as np
from manyagents.schemas import TraceStore
from manylatents.metrics import compute_metric
from manylatents.metrics.trajectory_geometry import compute_cosine_velocity, compute_menger_curvature
from manylatents.api import run as ml_run

store = TraceStore("<output_dir>/traces", mode="r")
per_trace, ids = [], []
for trace in store:
    tensors = store.load_tensors(trace.trace_id)           # tensors/{trace_id}.npz
    if tensors is None or not trace.steps:                # text-only traces
        continue
    layers = trace.steps[0].layers_captured               # e.g. [28] -> axis-1 index
    steps = tensors["pooled_steps"][:, layers.index(layers[-1]), :].astype(np.float32)
    if len(steps) < 3:                                    # velocity needs 2, curvature needs 3
        continue
    print(trace.trace_id, compute_cosine_velocity(steps), compute_menger_curvature(steps))
    per_trace.append(steps)
    ids += [trace.trace_id] * len(steps)

if not per_trace:
    raise SystemExit("No traces with at least three steps and tensors; capture longer responses.")
X, ids = np.concatenate(per_trace), np.array(ids)
class _Grouped:
    step_trace_ids = ids                                  # preserves trace boundaries
print(compute_metric("trajectory_velocity", X, dataset=_Grouped()))
print(compute_metric("trajectory_curvature", X, dataset=_Grouped()))
r = ml_run(input_data=X, algorithm="pca", metrics=["trajectory_velocity", "trajectory_curvature"])
print(r["embeddings"].shape, r["scores"])
```

Use traces from the **same model and captured layer** for aggregation. The first outputs measure raw hidden states per trace. Grouped `compute_metric` averages each trace's mean, excluding transitions between independent traces. The final `ml_run` example instead measures the **PCA embedding, ungrouped**: its scores include transitions across concatenated trace boundaries and are not a grouped reasoning-geometry measurement. To measure the embedding with boundaries preserved, call `compute_metric` on `r["embeddings"]` with `dataset=_Grouped()`.

The shipped capture experiment uses newline (`delimiter`) segmentation so an unclosed `<think>` does not collapse a multiline response into one step. Hidden-state traces with fewer than two steps are counted in `traces_failed` and excluded from the store; two-step traces permit velocity but not curvature. Text-only traces may still have one step. Longer responses may require `agent.config.max_new_tokens=1024` or more; segmentation and a persisted trace do not establish answer completeness or correctness.

**This release has no answer judge.** Captured traces have `success=None` and `judge="none"`; their summary outcome is always `unjudged`. The summary's `success` and `failure` fields remain zero unless an external producer supplies judged outcomes. `traces_failed` counts extraction/persistence failures separately.

## Tool-calling loop

`manyagents.agent_loop.run_agent_loop` is an async entry point for adapters exposing `chat()` (OpenAI, Ollama, and Claude). Pass a prompt and optional `manyagents.tools.Tool` objects, each pairing a JSON Schema with a trusted sync or async callable. The returned `AgentResult` contains `answer`, `messages`, `steps`, `stopped` (`end_turn` or `max_steps`), and executed `tool_calls`. Pass `messages` back as `history` to continue; `max_steps` bounds model turns. Tool bodies run with the caller's permissions.

## Trusted execution

CellForge and Kosmos execute local subprocesses with the caller's environment. Biomni imports `A1` from `biomni.agent` in-process and runs `agent.go` through `asyncio.to_thread`; a thread is not process isolation, and an async timeout does not terminate the running thread. These integrations execute local code with the caller's permissions and are not for untrusted task configs.

CellForge requires an explicit installation directory via `CellForgeAdapter(cellforge_path="/path/to/CellForge")` or `CELLFORGE_PATH`. Its absolute `main.py` path is bound when the adapter is created; task `working_dir` controls the execution directory.

**Timeout is not a reliable termination bound.** Subprocess cleanup kills only the immediate child, descendants can survive, and cleanup itself can wait without a second deadline. Biomni work can continue running and spending after cancellation. Kosmos environment inheritance, `run_subprocess` process-group cleanup, and Biomni cancellation improvements are deferred to 0.3.0; trusted inputs do not prevent hangs or continued spending.

## Documentation and development

- [Running Experiments](docs/running_experiments.md): local execution, traces, GPU requirements, and cluster prerequisites
- [Config Groups](docs/config_groups.md): Hydra packages and overrides
- [Design Decisions](docs/design_decisions.md): architecture history
- [Contributing](docs/CONTRIBUTING.md): development and adapter conventions
- [Code of conduct](CODE_OF_CONDUCT.md), [security reporting](SECURITY.md), and [citation](CITATION.cff)

```bash
uv sync --locked
uv run --no-sync pytest -q
uv run --no-sync ruff check manyagents/ tests/ scripts/
uv sync --locked --extra traces
uv run --no-sync pytest -q
```

[manylatents](https://github.com/latent-reasoning-works/manylatents) supplies optional DR algorithms and geometric metrics. manyRuns (the run harness) and Shop (cluster infrastructure) are companion repos, not yet public.

MIT License — see [LICENSE](LICENSE).
