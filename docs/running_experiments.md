# Running Experiments

Use Python 3.11–3.12 and run `uv sync` from a checkout. Activate `.venv`, or prefix commands with `uv run --no-sync` so your installed extras remain available.

## Local evaluation

```bash
# Two mock prompts: no API keys, GPU, or model download
manyagents experiment=test_wandb

# Nine prompts: 3 domains × 3 information conditions
manyagents experiment=geometric_reasoning 'active_agents=[mock]'

# Real API evaluation: export ANTHROPIC_API_KEY and OPENAI_API_KEY first
manyagents experiment=geometric_reasoning 'active_agents=[claude,openai]'

# Local HF generation with the shipped default model
manyagents experiment=geometric_reasoning 'active_agents=[local_llm]' agents.local_llm.agent.config.model=Qwen/Qwen3-0.6B
```

Each active agent runs every prompt. Evaluation scores text descriptions against configured methods; it does not inspect biological datasets. Evaluation and trace extraction are separate workflows: the 3×3 suite scores descriptions, while `trace_extraction` runs GSM8K.

All three scores use successful prompts; inspect `prompts_evaluated` and `prompts_failed`. Ground-truth match rate is the fraction with at least one extracted expected method and no extracted failure indicator. Local rejections (such as “avoid Leiden” or “UMAP would be wrong”) are filtered; a genuine recommendation mentioning a rejected alternative can still pass. This is a vocabulary heuristic: bare hedges, quoted or hypothetical advice, and complex/distant negation can still count. See the [scoring rules](../README.md#the-experiment) and [0.2.0 migration](../CHANGELOG.md) before comparing old rates.

Jaccard averages **all successful prompt pairs**, including same-geometry pairs. Nine successful prompts yield 36 pairs, nine within a geometry. Consistent, nonempty method sets disjoint across the three geometries score **0.25**; shared methods can raise it and within-geometry inconsistency can lower it. High overlap signals invariance, but lower is not always better. Two empty sets have similarity 1.0. Clustering-for-all is the fraction of successful prompts with an extracted clustering mention: high values indicate broad use across this mixed-geometry suite, not an individual scientific judgment. Undefined measurements appear as `null` in JSON and `n/a` in summaries. Missing ground-truth criteria on any successful prompt make the whole match rate unavailable; Jaccard needs at least two successful prompts. Zero successful evaluations exit nonzero; partial failures remain in the results.

Bare `manyagents` exits with an experiment-selection hint and all available experiment names. `active_agents` must be nonempty and refer to names defined by the selected experiment. A registry key alone does not add an agent to an experiment.

## Sweeps

```bash
manyagents --multirun experiment=invariance_full 'active_agents=[claude],[openai],[local_llm]' 'output_dir=${hydra:runtime.output_dir}'
```

This launches three jobs, each evaluating four prompts with one agent. `invariance_full` defines `claude`, `openai`, `local_llm`, and `biomni`; it does not define `hf` or `mock`. Sweep `active_agents`, not `agent`. The local job defaults to `Qwen/Qwen3-0.6B`; override `agents.local_llm.agent.config.model` to select another accessible model.

The output override keeps separate `results.json` and `summary.md` files in Hydra's numbered job directories. Without it, the experiment's second-resolution output name can collide across fast jobs.

To inspect a single configuration without running it:

```bash
manyagents experiment=geometric_reasoning 'active_agents=[mock]' --cfg job
```

`--cfg job` is not an execution check. Hydra rejects it when combined with `--multirun`. `tests/test_cli_commands.py` executes the documented sweeps in-process with mock adapters and checks every job's responses, scores, and saved files.

## Generation and traces

Core includes local HF generation without manylatents, but it is a large install: accelerate brings in torch. Basic HF hidden-state capture uses `generate(output_hidden_states=True)`. Add `--extra traces` for geometry-based segmentation, downstream geometry, and datasets such as GSM8K. Plain vLLM generation needs `--extra vllm` and uses neither an HF model nor manylatents. vLLM trace replay needs both extras and loads an HF model to recover states. `--extra full` includes traces, W&B, and Biomni, but excludes vLLM.

```bash
# HF hidden-state traces; requires --extra traces and downloads GSM8K/model
manyagents experiment=trace_extraction agent=hf agent.config.model=Qwen/Qwen3-0.6B

# Claude text traces only; requires API key and datasets from --extra traces
manyagents experiment=trace_extraction agent=claude agent.config.capture_hidden_states=false

# GPU traces; requires --extra traces --extra vllm
manyagents experiment=trace_extraction agent=vllm

# V100 / RTX 8000: explicitly choose float16
manyagents experiment=trace_extraction agent=vllm agent.config.dtype=float16
```

**Hardware:** laptops suit API clients, a local Ollama server, mock, and small HF models; large HF models and vLLM need suitable GPUs. `manyagents/configs/agent/vllm.yaml` hardcodes `dtype: bfloat16` with no fallback, assuming bf16-capable hardware (Ampere or newer). Mila's V100 and RTX 8000 are pre-Ampere; use `agent.config.dtype=float16` there. The default remains unchanged to preserve numerics on Ampere+.

With `agent=vllm`, the dtype override is `agent.config.dtype`. When loading vLLM into an experiment package, use the corresponding `agents.<name>.agent.config.dtype` path instead.

vLLM replay was verified finite on an L40S, producing float16 `token_level (n_tokens, 1, d_model)` and `pooled_steps (n_steps, 1, d_model)` arrays with one captured layer. Replay storage dtype is separate from the generation engine dtype. Replay holds an HF model alongside the vLLM engine and uses exact generated token IDs. The HF comparison test checks agreement with generation-time capture within `atol=rtol=1e-3`, not bitwise equality to vLLM activations. Hydra loops over tasks; it does not call `extract_traces_batch`, which batches generation and replays sequences separately. Text-step segmentation re-encodes decoded text, so exact replay IDs do not guarantee exact pooling alignment.

The shipped Qwen `layers: [-1]` is post-final-norm. Adapters forward neither `capture_prenorm` nor `state_dtype`; use direct Python inference for these controls. Hydra stores float16; casting to float32 afterward cannot repair overflow. Ollama and API backends expose no hidden states.

## Output files

Evaluation saves the complete response text, per-prompt results, config, and metrics in `<output_dir>/results.json`, plus a human-readable `<output_dir>/summary.md`. Individual adapter response paths are temporary working outputs and may be overwritten; use the aggregated results as the evaluation record.

Trace extraction instead writes:

```text
<output_dir>/traces/
├── traces.jsonl
└── tensors/
    └── <trace_id>.npz
```

`TraceStore` appends one JSON record per trace. Counts cover only traces newly persisted by the current run. Zero persisted traces exit nonzero; requested hidden states must be present as nonempty, finite float arrays.

## From traces to geometry

Follow the [runnable README bridge](../README.md#from-traces-to-geometry): load each NPZ through `TraceStore`, select the captured layer, cast to float32, then call manyLatents directly. Group by `step_trace_ids` to avoid introducing transitions between separate traces. `ManyLatentsAdapter.execute_cached` does not accept the stored 3-D tensor or expose trajectory velocity/curvature through its YAML metric registry.

The default segmentation is `delimiter` (newlines). Captured traces need at least two steps to enter the store; curvature needs three. Short captured traces count as `traces_failed`. This release has no answer judge: generated traces are always `unjudged`, with `success=None` and `judge="none"`. Extraction success does not mean a correct or complete answer.

## Optional logging and cluster execution

For evaluation, install `uv sync --extra wandb` (or `--extra full`) and configure W&B authentication, then add `wandb.enabled=true`. Trace extraction returns before logger creation. Keep any other required extras in the same sync command.

```bash
manyagents experiment=test_wandb wandb.enabled=true 'wandb.tags=[smoke,mock]'
```

The `mila_*` cluster profiles are site-specific. `mila_slurm` and `mila_sweep` use Submitit. Only `mila_remote` requires the separately installed Shop launcher, along with Mila access and SSH/environment setup; it is not provided by any manyagents extra. With those prerequisites:

```bash
manyagents experiment=geometric_reasoning 'active_agents=[claude,openai]' cluster=mila_remote resources=api
```

Resource profiles select CPU/GPU allocations and launcher settings. The API profile exports local API keys into the job script, so protect that script and its storage. Inspect `manyagents/configs/cluster/` and `manyagents/configs/resources/` for site-specific requirements. Shop is a companion repo, not yet public.

See [Config Groups](config_groups.md) for package paths and [README](../README.md#trusted-execution) for the local-code execution boundary.
