<div align="center">

<pre>
                    ◇ ∿∿∿∿∿∿
       ?   ────▶    ◇ ∿∿∿∿        ────▶   κ(·)
                    ◇ ∿∿∿∿∿

              m a n y a g e n t s

          ask many models, keep the trace
</pre>

[![CI](https://github.com/latent-reasoning-works/manyagents/actions/workflows/ci.yml/badge.svg)](https://github.com/latent-reasoning-works/manyagents/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-8B5CF6.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11–3.12-8B5CF6.svg)](https://www.python.org)
[![uv](https://img.shields.io/badge/pkg-uv-8B5CF6.svg)](https://docs.astral.sh/uv/)

</div>

---

Run one prompt through many models, score what comes back, and for local models keep the hidden states that produced it.

```bash
uv sync
manyagents experiment=test_wandb     # two prompts, mock agent, no keys or GPU
```

- **Adapters.** Anthropic, OpenAI, Ollama, Hugging Face transformers and vLLM behind one async `run(config) -> AdapterResult`.
- **Evaluation.** Hydra fans prompts across models, extracts method mentions from free text, and scores them against per-prompt expected and forbidden lists. The shipped vocabulary is single-cell bioinformatics (`metrics/extractor.py`); another domain means editing it in Python, while dispatch and result files carry over.
- **Traces.** For HF and vLLM, the selected-layer hidden states that predicted each emitted token, segmented into steps and stored as JSON plus NPZ. [manylatents](https://github.com/latent-reasoning-works/manylatents) measures their velocity and curvature.
- **Tool loop.** Your tools against any adapter implementing the async `chat()` protocol.

## install

Python **3.11–3.12**, from a source checkout:

```bash
git clone https://github.com/latent-reasoning-works/manyagents.git
cd manyagents && uv sync            # core: API adapters, mock, local HF generation

uv sync --extra traces               # manylatents, segmentation, datasets (GSM8K)
uv sync --extra vllm                 # vLLM generation on a supported GPU
uv sync --extra traces --extra vllm  # vLLM generation + HF hidden-state replay
uv sync --extra full                 # traces + W&B + Biomni; vLLM stays separate
```

Core is already large: `accelerate` pulls in torch. Sync every extra you need in one command, then activate `.venv` or prefix commands with `uv run --no-sync` so the extras stay put. API adapters need `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`; Ollama needs a running server with a pulled model; large HF models and vLLM need a GPU. This page describes 0.2.0, which is this checkout; the previous public release installs with `uv pip install "manyagents @ git+https://github.com/latent-reasoning-works/manyagents@v0.1.1"`.

**Upgrading from 0.1.1:** scores are **incomparable across this upgrade**. Local rejections are now filtered and a failure-indicator match blocks a pass, so re-extract stored `raw_response` text before comparing rates. Details in the [changelog](CHANGELOG.md).

## quickstart

```bash
# no keys, no GPU, no downloads
manyagents experiment=test_wandb

# the same prompts through Claude and OpenAI (needs both keys)
manyagents experiment=test_wandb 'active_agents=[claude,openai]'
```

Each run writes `results.json` and `summary.md` under `output_dir`: per-prompt responses and extracted methods, then aggregate scores per model.

One job per agent, results kept per job:

<!-- example:evaluation-sweep -->
```bash
manyagents --multirun experiment=invariance_full 'active_agents=[claude],[openai],[local_llm]' 'output_dir=${hydra:runtime.output_dir}'
```
<!-- /example:evaluation-sweep -->

The same thing from Python:

```python
from manyagents.api import run

result = run(["experiment=test_wandb"])
result["metrics"]["mock"]   # {"jaccard_similarity_across_prompts": 1.0, "ground_truth_match_rate": 0.5, ...}
```

`run()` composes the same Hydra config as the CLI and keeps its exit semantics: `SystemExit` propagates when nothing succeeded, so catch it when embedding. Bare `manyagents` lists the shipped experiments and exits nonzero. The sweep's `local_llm` runs `Qwen/Qwen3-0.6B`; `agents.local_llm.agent.config.model=<id>` picks another.

**Scoring is a heuristic**, not an answer judge: it matches method mentions against a fixed vocabulary and drops those under a rejection cue. What it catches and what slips past is in [experiment configurations](manyagents/configs/experiment/README.md).

## [adapters](docs/python-api.md)

> 11 classes, 12 registry keys

| key | runs | trace | hidden states |
|---|---|---|---|
| `claude` | Anthropic API | text steps with `build_trace=True` | no |
| `openai`, `ollama` | OpenAI-compatible API; Ollama is a local server | no | no |
| `hf`, `local_llm` | local transformers generation | yes | the predicting position at each decode step of `generate()` |
| `vllm` | vLLM generation | yes | one HF forward pass over the emitted token ids |
| `manylatents` | in-process DR and metrics (`traces`) | n/a | n/a |
| `biomni`, `cellforge`, `kosmos` | external agents: in-process, CLI, CLI | n/a | n/a |
| `mock`, `placeholder` | deterministic responses; development stub | n/a | n/a |

Every adapter takes a config dict and returns `{success, summary, output_files, metadata?, embeddings?}`. Text adapters put the response at `output_files["raw_response"]`; a missing optional dependency comes back as a failed result. Run one directly:

```python
import asyncio
from manyagents.adapters import ADAPTER_REGISTRY
from manyagents.schemas import ReasoningTrace

adapter = ADAPTER_REGISTRY["hf"]()
result = asyncio.run(adapter.run({
    "prompt": "If a train travels 60 km in 1.5 hours, what is its average speed?",
    "system_prompt": "Solve step by step, one step per line.",
    "model": "Qwen/Qwen3-0.6B", "max_new_tokens": 96,
    "build_trace": True, "capture_hidden_states": True, "layers": [-1],
}, {}))
if not result["success"]:
    raise RuntimeError(result["summary"])
text  = result["output_files"]["raw_response"].read_text()
trace = ReasoningTrace.from_json(result["output_files"]["trace"].read_text())
trace.steps, trace.model, trace.task                    # ReasoningStep list, ModelInfo, TaskInfo
result["output_files"]["hidden_states"]                 # Path to the NPZ
```

The tool loop, `agent_loop.run_agent_loop(prompt, agent="openai", tools=[...])`, uses a second, chat-shaped protocol: a registered adapter's `await chat(messages, tools=..., model=...)` returning the assistant message, its text, and normalized tool calls (`claude`, `openai`, `ollama` implement it). The loop executes the calls the model makes and repeats until it stops or `max_steps` runs out. Tool bodies run with your permissions. The result contract, the loop, and DR workflows are in [docs/python-api.md](docs/python-api.md).

## [reasoning traces](docs/running_experiments.md#generation-and-traces)

> capture and measure hidden-state trajectories

Trace extraction is a separate workflow from evaluation: evaluation scores response text, `trace_extraction` generates on a dataset and keeps the states. Neither captures states while the other scores.

A `ReasoningTrace` is one model on one task: `trace.task` (the prompt), `trace.model` (model and generation config), the response text and token counts, and `trace.steps`, segments of the response with a kind (`thinking`, `output`, `tool_call`, `tool_result`) and, for HF and vLLM, a hidden-state tensor each.

```bash
# HF capture: --extra traces; downloads Qwen/Qwen3-0.6B and GSM8K
manyagents experiment=trace_extraction agent=hf agent.config.model=Qwen/Qwen3-0.6B

# vLLM capture: --extra traces --extra vllm, bf16-capable GPU
manyagents experiment=trace_extraction agent=vllm
```

Both paths record the state at the position that predicts each emitted token, so the last token's own position is never captured. HF reads these during `generate()`; vLLM generates first and a teacher-forced HF pass over the emitted ids recovers them, agreeing within tolerance under matching model conditions and holding both models in memory. Text-step alignment stays approximate, because segmentation re-encodes decoded text. Details and the exact caveats: [running experiments](docs/running_experiments.md#generation-and-traces).

Segmentation decides what a step is — `delimiter` (newlines, the default), `tags`, `velocity` (peaks in cosine distance between consecutive token states), or `hybrid`. Token states are mean-pooled per step; the experiment writes a `TraceStore`:

```text
<output_dir>/traces/
├── traces.jsonl               # one ReasoningTrace per line
└── tensors/
    └── <trace_id>.npz         # pooled_steps (n_steps, n_layers, d_model), token_level (n_tokens, n_layers, d_model)
```

Know these before measuring anything:

- **Layer and dtype.** The shipped `layers: [-1]` captures the post-final-norm hidden state, and the Hydra path stores float16. The pre-norm residual stream (`forward_hidden_states(capture_prenorm=True)`) and float32 storage (`extract_trace(state_dtype="float32")`) are direct-Python only; neither adapter forwards `capture_prenorm` or `state_dtype`. Casting saved float16 back to float32 cannot repair overflow, and the runner rejects nonfinite tensors.
- **No answer judge ships.** Every trace has `success=None` and `judge="none"`; the summary counts them as `unjudged` until an external judge fills them in. `traces_failed` counts extraction and persistence failures.
- **Step counts.** Hidden-state traces with fewer than two steps are rejected before the store (velocity needs two, curvature three). Longer answers may need `agent.config.max_new_tokens=1024` or more.
- **Hardware.** `agent/vllm.yaml` defaults to `dtype: bfloat16` with no fallback, so it assumes Ampere or newer. On V100 or RTX 8000, `agent.config.dtype=float16` changes the vLLM engine only; the HF replay model still loads in bfloat16 (`inference.get_model`'s default), and no adapter option changes that.

Below the adapters, `manyagents.inference` is plain functions with a module-level model cache, usable without Hydra: model loading, generation, `forward_hidden_states` (plus batched and recurrent variants), the `segment_*` family, and `extract_trace`. The batched and recurrent forwards, `capture_prenorm`, and `state_dtype` stay outside the Hydra adapters.

## from traces to geometry

`TraceStore.load_tensors(trace_id)` loads that NPZ into a dictionary of arrays, or returns `None` when tensors are unavailable. Select a captured layer by its position in `layers_captured`, cast to float32, and you have the `(steps, d_model)` array manylatents expects. Call **manylatents directly** for trajectory velocity and curvature:

<!-- example:trace-geometry -->
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
<!-- /example:trace-geometry -->

The per-trace prints measure raw hidden states. The grouped `compute_metric` calls average each trace's mean and skip transitions between traces. The final `ml_run` measures the **PCA embedding, ungrouped**, so its scores include the jumps across trace boundaries; to keep them, call `compute_metric` on `r["embeddings"]` with `dataset=_Grouped()`. Aggregate only traces from the same model and captured layer.

`ManyLatentsAdapter` is the wrong door for this: it serves 2-D arrays for DR, its `execute_cached` rejects the stored 3-D trace tensor, and its cached metric registry discovers YAML-backed metrics only, of which manylatents 0.1.7 ships none for trajectories.

## trusted execution

CellForge and Kosmos run local subprocesses with the caller's environment; CellForge needs its install directory via `CellForgeAdapter(cellforge_path=...)` or `CELLFORGE_PATH`. Biomni imports `A1` from `biomni.agent` and runs it in-process through `asyncio.to_thread`: a thread gives no isolation, and an async timeout leaves it running. Trusted task configs only.

**Timeouts are unreliable termination bounds.** Subprocess cleanup kills only the immediate child, descendants can survive, and cleanup itself can wait without a second deadline; Biomni work can keep running and spending after cancellation. Process-group cleanup, Kosmos environment isolation, and Biomni cancellation are deferred to 0.3.0.

## docs & development

- [Running experiments](docs/running_experiments.md): local runs, traces, GPU requirements, cluster prerequisites
- [Config groups](docs/config_groups.md): Hydra packages and override paths
- [Python API](docs/python-api.md): adapter table, result contract, tool loop, DR workflows
- [Design decisions](docs/design_decisions.md): why schema-on-read, why a direct Python API
- [Contributing](docs/CONTRIBUTING.md), [changelog](CHANGELOG.md), [code of conduct](CODE_OF_CONDUCT.md), [security](SECURITY.md), [citation](CITATION.cff)

```bash
uv sync --locked
uv run --no-sync pytest -q
uv run --no-sync ruff check manyagents/ tests/ scripts/
uv sync --locked --extra traces && uv run --no-sync pytest -q
```

The `mila_*` cluster configs are site-specific; only `mila_remote` needs the separately installed Shop launcher. W&B logging covers **evaluation** (`wandb.enabled=true` with the `wandb` or `full` extra); trace extraction returns before the logger is created.

## citing

If manyagents was useful in your research, a citation goes a long way:

```bibtex
@software{manyagents2026,
  title     = {manyagents: Multi-agent evaluation and reasoning trace extraction},
  author    = {Valdez C{\'o}rdova, C{\'e}sar Miguel},
  year      = {2026},
  version   = {0.2.0},
  url       = {https://github.com/latent-reasoning-works/manyagents},
  license   = {MIT}
}
```

<br><br>

<p align="center">
<sub>MIT License &middot; <a href="https://github.com/latent-reasoning-works">Latent Reasoning Works</a></sub>
</p>
