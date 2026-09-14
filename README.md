<div align="center">

<pre>
                    ◇ ∿∿∿∿∿∿
       ?   ────▶    ◇ ∿∿∿∿        ────▶   κ(·)
                    ◇ ∿∿∿∿∿

              m a n y a g e n t s

       test the recommendation, measure the trace
</pre>

[![CI](https://github.com/latent-reasoning-works/manyagents/actions/workflows/ci.yml/badge.svg)](https://github.com/latent-reasoning-works/manyagents/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-8B5CF6.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11–3.12-8B5CF6.svg)](https://www.python.org)
[![uv](https://img.shields.io/badge/pkg-uv-8B5CF6.svg)](https://docs.astral.sh/uv/)

</div>

---

manyagents asks one question of many language models: **does the analysis you recommend fit the geometry of the data?** The shipped suite expects clustering for discrete groups, trajectory inference for branching differentiation, and manifold methods for continuous structure. It scores text descriptions against configured expectations; it does not inspect the underlying biological datasets. The same question is posed with more and less biological context, and the scoring is built to catch the failure that matters: a model that recommends the same pipeline regardless of what the data looks like.

For local models, manyagents also records **hidden-state trajectories**, segmented into text steps and stored beside the response, for velocity and curvature measurements in [manylatents](https://github.com/latent-reasoning-works/manylatents). HF captures states during generation; vLLM generates first and uses a subsequent HF replay. The shipped Qwen `layers: [-1]` captures the post-final-normalization hidden state, not the pre-normalization residual stream.

**Evaluation and trace extraction are currently separate workflows:** the 3×3 evaluation scores biological scenario descriptions, while `trace_extraction` runs GSM8K math tasks. The examples do not measure hidden states while making biological recommendations. manylatents is the compute layer below it (public). manyRuns (run harness) and Shop (cluster launchers) are companion repos, not yet public.

## install

Python **3.11–3.12**, from a source checkout:

```bash
git clone https://github.com/latent-reasoning-works/manyagents.git
cd manyagents && uv sync            # core: API adapters, mock, local HF generation
```

Extras:

```bash
uv sync --extra traces               # geometry, segmentation, datasets (GSM8K)
uv sync --extra vllm                 # vLLM generation on a supported GPU
uv sync --extra traces --extra vllm  # vLLM generation + HF hidden-state replay
uv sync --extra full                 # traces + W&B + Biomni; does not include vLLM
```

Core is already a large install: `accelerate` pulls in **torch**. Plain vLLM generation needs only `vllm` beyond core and loads neither an HF model nor manylatents. `--extra wandb` adds evaluation logging on its own; the default dev group (pytest, ruff) is synced automatically, and `--extra dev` adds pre-commit.

Activate with `source .venv/bin/activate`, or prefix commands with `uv run --no-sync` so the extras you synced stay put.

<details>
<summary>installing the previous public release as a dependency</summary>

```bash
uv pip install "manyagents @ git+https://github.com/latent-reasoning-works/manyagents@v0.1.1"
```

This pins the previous public scorer. For the 0.2.0 changes described here, use this source checkout; no 0.2.0 tag is assumed.

</details>

<details>
<summary>what runs where</summary>

Laptop: Claude/OpenAI API clients, a local Ollama server, the mock agent, and small HF models. GPU: large HF models and vLLM. API adapters need `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`; Ollama needs a running server and a pulled model.

</details>

**Upgrading to 0.2.0:** scoring is **not comparable across this upgrade**. Re-extract stored response text before comparing rates: local rejections are now filtered and failure-indicator matches block a ground-truth pass. The [changelog](CHANGELOG.md) explains the scoring change and earlier 0.1.1 migrations for adapters, exit semantics, and legacy GVector data.

## quickstart

```bash
# no keys, no GPU, no downloads: two mock prompts through the evaluator
manyagents experiment=test_wandb

# the 3 x 3 suite (nine prompts) with the mock agent
manyagents experiment=geometric_reasoning 'active_agents=[mock]'

# the same nine prompts through Claude and OpenAI (needs both keys)
manyagents experiment=geometric_reasoning 'active_agents=[claude,openai]'
```

One job per agent, four prompts each, results kept per job:

<!-- example:evaluation-sweep -->
```bash
manyagents --multirun experiment=invariance_full 'active_agents=[claude],[openai],[local_llm]' 'output_dir=${hydra:runtime.output_dir}'
```
<!-- /example:evaluation-sweep -->

The 3×3 `geometric_reasoning` mock run prints this:

```text
mock:
  Jaccard Similarity: 1.00 (all successful pairs; not an optimization objective)
  Ground Truth Match: 33.3% (higher is better)
  Clustering-for-All: 100.0% (lower is better)
```

That is what failure looks like. The mock answers every prompt with the same three methods, so it scores identically across nine prompts representing three expected geometries (Jaccard 1.0) and recommends clustering for a continuous manifold (clustering-for-all 100%). Read the three together: match rate alone would not show this.

The suite, its scoring vocabulary, and the limits of that scoring are documented in [experiment configurations](manyagents/configs/experiment/README.md).

The same thing from Python:

```python
from manyagents.api import run

result = run(["experiment=test_wandb"])
result["metrics"]["mock"]   # {"jaccard_similarity_across_prompts": 1.0, "ground_truth_match_rate": 0.5, ...}
```

`run()` composes the same Hydra config as the CLI and keeps its exit semantics: `SystemExit` propagates when nothing succeeded, so catch it explicitly when embedding.

<details>
<summary>sweep notes</summary>

The sweep uses agent names defined by `invariance_full`: `claude`, `openai`, `local_llm` (it also defines `biomni`). The local job runs HF with `Qwen/Qwen3-0.6B`; pick another Hub ID or path with `agents.local_llm.agent.config.model=<id>`. The `output_dir` override keeps each job's `results.json` and `summary.md` under Hydra's numbered multirun directory. `--cfg job` prints the composed config without running it and cannot be combined with `--multirun`.

Bare `manyagents` exits with a hint and the list of shipped experiments. Zero successful evaluations exit nonzero; partial failures are recorded alongside successes.

</details>

---

## architecture

```
  CLI                                        API
  manyagents experiment=…                    run(["experiment=…"])
        │                                          │
        └───────────────────┬──────────────────────┘
                            ▼
                   experiment.py  run_experiment()
                            │
           ┌────────────────┴────────────────┐
           ▼                                 ▼
   prompt evaluation                  trace extraction
   ADAPTER_REGISTRY[k]().run()        ADAPTER_REGISTRY[k]().run(build_trace=True)
   metrics/extractor.py               inference.py   generate → segment → pool
   metrics/llm.py                     schemas/reasoning.py   ReasoningTrace, TraceStore
           │                                 │
           ▼                                 ▼
   results.json, summary.md           traces.jsonl + tensors/<id>.npz ──▶ manylatents
```

Every agent sits behind one async interface, `run(task_config, input_files) -> AdapterResult`, where the result is `{success, summary, output_files, metadata?, embeddings?}`. Text adapters put their response at `output_files["raw_response"]`; Claude can add `trace` only; HF and vLLM can add `trace` plus `hidden_states` when capture is enabled. Everything is composed by Hydra config groups under `manyagents/configs/` (`agent/`, `experiment/`, `cluster/`, `logger/`, `prompts/`, `resources/`).

---

## [reasoning traces](docs/running_experiments.md#generation-and-traces)

> capture and measure hidden-state trajectories

A `ReasoningTrace` is one model on one task: the prompt (`trace.task`), the model and generation config (`trace.model`), the response text, token counts, and a list of `ReasoningStep`s, each a segment of the response with a kind (`thinking`, `output`, `tool_call`, `tool_result`) and, for local models, a companion hidden-state tensor.

| adapter | trace | hidden states | how |
|---|---|---|---|
| `claude` | text steps | no | API response blocks, `build_trace=True` |
| `openai`, `ollama` | no | no | generation only |
| `hf` | yes | yes | HF `generate(output_hidden_states=True)`; last position per decode step |
| `vllm` | yes | yes | vLLM generates; one HF forward pass over the emitted token ids recovers states |

vLLM returns the exact prompt and completion token IDs. An HF teacher-forced forward pass over those IDs computes corresponding HF hidden states without a string round-trip for replay. Under matching model conditions, replay agrees with HF generation-time capture within numerical tolerance (the CPU comparison test uses `atol=rtol=1e-3`); this does not recover vLLM's internal activations bit for bit. Replay holds an **HF model alongside the vLLM engine**, so budget memory for both.

The direct Python `inference.extract_traces_batch` batches generation in one vLLM call, then replays each sequence separately. The Hydra command below does **not** call that function: it loops over tasks. Exact replay IDs also do not guarantee exact text-step alignment: segmentation re-encodes decoded text, and stripping leading whitespace or tokenizer round-trips can shift pooling intervals.

```bash
# HF capture: --extra traces; downloads Qwen/Qwen3-0.6B and GSM8K
manyagents experiment=trace_extraction agent=hf agent.config.model=Qwen/Qwen3-0.6B

# vLLM capture: --extra traces --extra vllm, bf16-capable GPU
manyagents experiment=trace_extraction agent=vllm
```

Segmentation decides what a "step" is: `delimiter` (newlines; the shipped default), `tags` (`<think>…</think>` with sentence splits inside), `velocity` (peaks in cosine distance between consecutive token states, so the boundaries come from the geometry itself), or `hybrid`. Token states are mean-pooled per step. The experiment writes a `TraceStore`:

```text
<output_dir>/traces/
├── traces.jsonl               # one ReasoningTrace per line
└── tensors/
    └── <trace_id>.npz         # pooled_steps (n_steps, n_layers, d_model), token_level (n_tokens, n_layers, d_model)
```

With the shipped `layers: [-1]` on Qwen3-0.6B, observed 96-token completions produced `token_level (96, 1, 1024)` with three or four pooled steps, depending on the completion and segmentation. These are examples, not guaranteed step counts or complete answers. The layer is post-final-norm. Hydra stores float16; the adapters forward neither `state_dtype` nor `capture_prenorm`. For float32 storage or pre-final-norm capture, use the direct Python inference API (`extract_trace(state_dtype="float32")` for storage; `forward_hidden_states(capture_prenorm=True)` for pre-norm states). Casting saved float16 arrays to float32 cannot repair overflow; the extraction runner rejects nonfinite tensors.

**Hardware:** `manyagents/configs/agent/vllm.yaml` defaults to `dtype: bfloat16` with no fallback, so it assumes Ampere or newer. On V100 or RTX 8000 pass `agent.config.dtype=float16`.

**No answer judge ships in this release.** Every captured trace has `success=None` and `judge="none"`; the extraction summary counts them as `unjudged`, and its `success`/`failure` fields stay at zero unless an external judge fills them in. `traces_failed` counts extraction and persistence failures. Hidden-state traces with fewer than two steps are rejected before the store (velocity needs two, curvature three); longer answers may need `agent.config.max_new_tokens=1024` or more.

<details>
<summary>below the adapters</summary>

`manyagents.inference` is plain functions with a module-level model cache, usable without Hydra: `get_model`, `get_vllm_engine`, `vllm_generate`, `forward_hidden_states` (with `capture_prenorm=True` for the residual stream before the final norm), `forward_hidden_states_batched` (one padded forward for a panel of sequences), `forward_recurrent_states` (per-recurrence-step states for weight-tied models such as Huginn), the `segment_*` family, and `extract_trace`. The batched and recurrent forwards, `capture_prenorm`, and `state_dtype` controls are not wired into the Hydra adapters.

</details>

---

## from traces to geometry

`TraceStore.load_tensors(trace_id)` loads that NPZ into a dictionary of arrays, or returns `None` when tensors are unavailable. Select a captured layer by its position in `layers_captured`, cast to float32, and you have the `(steps, d_model)` array manylatents expects. Use **manylatents directly** for trajectory velocity and curvature:

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

The per-trace prints measure raw hidden states. The grouped `compute_metric` calls average each trace's mean and exclude transitions between independent traces. The final `ml_run` measures the **PCA embedding, ungrouped**: its scores include the jumps across concatenated trace boundaries and are not a grouped reasoning-geometry measurement — to measure the embedding with boundaries kept, call `compute_metric` on `r["embeddings"]` with `dataset=_Grouped()`. Aggregate only traces from the same model and captured layer.

**`ManyLatentsAdapter` is not this interface.** Its `run(input_data=...)` accepts 2-D arrays for DR, but `execute_cached` rejects the stored 3-D trace tensor and its cached metric registry discovers YAML-backed metrics only — manylatents 0.1.7 ships no trajectory-metric YAMLs.

---

## [adapters](docs/python-api.md)

> 11 classes, 12 registry keys — API (`claude`, `openai`, `ollama`), local (`hf`, `vllm`), compute (`manylatents`), CLI (`cellforge`, `kosmos`), in-process (`biomni`), and `mock`/`placeholder`

Every adapter takes a config dict and returns an `AdapterResult`. Run one directly:

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

Registration does not mean the dependency is installed; a missing one surfaces as a failed result, not an import error. The full table, result contract, tool-calling loop, and DR workflows are in [docs/python-api.md](docs/python-api.md).

---
## trusted execution

CellForge and Kosmos run local subprocesses with the caller's environment. Biomni imports `A1` from `biomni.agent` in-process and runs `agent.go` through `asyncio.to_thread`; a thread is not process isolation, and an async timeout does not stop the running thread. None of these are for untrusted task configs.

CellForge needs an explicit install directory via `CellForgeAdapter(cellforge_path=...)` or `CELLFORGE_PATH`; its `main.py` path is bound at construction and task `working_dir` sets the execution directory.

**Timeout is not a reliable termination bound.** Subprocess cleanup kills only the immediate child, descendants can survive, and cleanup itself can wait without a second deadline. Biomni work can keep running and spending after cancellation. Kosmos environment inheritance, `run_subprocess` process-group cleanup, and Biomni cancellation are deferred to 0.3.0; trusted inputs do not prevent hangs or continued spending.

---

## docs & development

- [Running experiments](docs/running_experiments.md) — local runs, traces, GPU requirements, cluster prerequisites
- [Config groups](docs/config_groups.md) — Hydra packages and override paths
- [Design decisions](docs/design_decisions.md) — why schema-on-read, why a direct Python API
- [Contributing](docs/CONTRIBUTING.md) — development and adapter conventions
- [Changelog](CHANGELOG.md), [code of conduct](CODE_OF_CONDUCT.md), [security](SECURITY.md), [citation](CITATION.cff)

```bash
uv sync --locked
uv run --no-sync pytest -q
uv run --no-sync ruff check manyagents/ tests/ scripts/
uv sync --locked --extra traces && uv run --no-sync pytest -q
```

The `mila_*` cluster configs are site-specific. Only `mila_remote` needs the separately installed Shop launcher; `mila_slurm` and `mila_sweep` use Submitit. W&B logging applies to **evaluation**, enabled with `wandb.enabled=true` and the `wandb` or `full` extra; trace extraction returns before logger creation.

---

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

---

<br><br>

<p align="center">
<sub>MIT License &middot; <a href="https://github.com/latent-reasoning-works">Latent Reasoning Works</a></sub>
</p>
