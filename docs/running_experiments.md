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

# Local HF generation, loading directly into the named experiment package
manyagents experiment=geometric_reasoning 'active_agents=[local_llm]' agent@agents.local_llm=hf agents.local_llm.agent.config.model=Qwen/Qwen3-0.6B
```

Each active agent runs every prompt. The runner reads response text, extracts method recommendations, checks expected methods, and computes aggregate scores. Ground-truth match rate up is good; cross-prompt Jaccard and clustering-for-all up are bad in this geometry evaluation. Undefined measurements appear as `null` in JSON and `n/a` in summaries. Zero successful evaluations exit nonzero; partial failures remain in the results.

Bare `manyagents` exits with an experiment-selection hint and all available experiment names. `active_agents` must be nonempty and refer to names defined by the selected experiment. A registry key alone does not add an agent to an experiment.

## Sweeps

```bash
manyagents --multirun experiment=invariance_full 'active_agents=[claude],[openai],[local_llm]' agent@agents.local_llm=hf 'output_dir=${hydra:runtime.output_dir}'
```

This launches three jobs, each evaluating four prompts with one agent. `invariance_full` defines `claude`, `openai`, `local_llm`, and `biomni`; it does not define `hf` or `mock`. Sweep `active_agents`, not `agent`. The named package override loads HF directly for `local_llm`, avoiding the legacy alias's nested-default packaging issue. Outside Mila, append `agents.local_llm.agent.config.model=Qwen/Qwen3-0.6B` (or another accessible model).

The output override keeps separate `results.json` and `summary.md` files in Hydra's numbered job directories. Without it, the experiment's second-resolution output name can collide across fast jobs.

To inspect a single configuration without running it:

```bash
manyagents experiment=geometric_reasoning 'active_agents=[mock]' --cfg job
```

`--cfg job` is not an execution check. Hydra rejects it when combined with `--multirun`. `tests/test_cli_commands.py` executes the documented sweeps in-process with mock adapters and checks every job's responses, scores, and saved files.

## Generation and traces

Core includes local HF generation without manylatents, but it is a large install: accelerate brings in torch. Add `--extra traces` for local hidden-state hooks, segmentation, and datasets such as GSM8K. Plain vLLM generation needs `--extra vllm` and uses neither an HF model nor manylatents. vLLM trace replay needs both extras and loads an HF model to recover states. `--extra full` includes traces, W&B, and Biomni, but excludes vLLM.

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

**Hardware:** laptops suit API clients, a local Ollama server, mock, and small HF models; large HF models and vLLM need suitable GPUs. `configs/agent/vllm.yaml` hardcodes `dtype: bfloat16` with no fallback, assuming bf16-capable hardware (Ampere or newer). Mila's V100 and RTX 8000 are pre-Ampere; use `agent.config.dtype=float16` there. The default remains unchanged to preserve numerics on Ampere+.

With `agent=vllm`, the dtype override is `agent.config.dtype`. When loading vLLM into an experiment package, use the corresponding `agents.<name>.agent.config.dtype` path instead.

vLLM replay was verified finite on an L40S, producing float16 `token_level (n_tokens, 1, d_model)` and `pooled_steps (n_steps, 1, d_model)` arrays with one captured layer. Replay storage dtype is separate from the generation engine dtype. Ollama and API backends expose no hidden states.

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

## Optional logging and cluster execution

Install `uv sync --extra wandb` (or `--extra full`) and configure W&B authentication, then add `wandb.enabled=true`. Keep any other required extras in the same sync command.

```bash
manyagents experiment=test_wandb wandb.enabled=true 'wandb.tags=[smoke,mock]'
```

The `mila_*` cluster profiles are site-specific. Remote execution requires Mila access, SSH/environment setup, and the separately installed Shop launcher; it is not provided by any manyagents extra. With those prerequisites:

```bash
manyagents experiment=geometric_reasoning 'active_agents=[claude,openai]' cluster=mila_remote resources=api
```

Resource profiles select CPU/GPU allocations and launcher settings. The API profile exports local API keys into the job script, so protect that script and its storage. Inspect `manyagents/configs/cluster/` and `manyagents/configs/resources/` for site-specific requirements. Shop is a companion repo, not yet public.

See [Config Groups](config_groups.md) for package paths and [README](../README.md#trusted-execution) for the local-code execution boundary.
