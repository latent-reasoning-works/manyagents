# Hydra Configuration Groups

Configs live under `manyagents/configs/`. The root `main.yaml` loads local execution and an optional experiment; select an experiment to run:

```bash
manyagents experiment=test_wandb
```

## Groups

| Group | Role | Examples |
|-------|------|----------|
| `experiment/` | Prompts, expected methods, active agents, output settings | `test_wandb`, `geometric_reasoning`, `invariance_full`, `trace_extraction` |
| `agent/` | Adapter and model settings | `claude`, `openai`, `hf`, `vllm`, `ollama`, `mock`, `biomni`, legacy `local_llm` aliases |
| `cluster/` | Local or site-specific launchers | `local`, `mila_remote`, `mila_slurm`, `mila_sweep` |
| `resources/` | Launcher allocation/setup overrides | `cpu`, `gpu`, `api`, `local_llm` |
| `logger/` | Logging configuration fragments | `minimal`, `wandb` |
| `prompts/` | Prompt configuration fragments | `discrete`, `trajectory`, `geometric_reasoning/` |

The existence of a group does not mean every experiment loads it. Override a selected group; use an explicit defaults entry when creating an experiment.

## Named evaluation agents

For example, `geometric_reasoning.yaml` contains:

```yaml
defaults:
  - /agent@agents.local_llm: local_llm
  - /agent@agents.claude: claude
  - /agent@agents.openai: openai
  - /agent@agents.mock: mock
  - _self_
```

An agent file itself has an `agent:` key:

```yaml
agent:
  name: claude
  adapter: claude
  config:
    model: claude-opus-5
    temperature: 0.0
    max_tokens: 2000
```

Loading it as `@agents.claude` places settings under **`agents.claude.agent.config`**. `active_agents` selects the named entries to execute:

```bash
manyagents experiment=geometric_reasoning 'active_agents=[claude,openai]'
manyagents experiment=geometric_reasoning 'active_agents=[claude]' agents.claude.agent.config.model=claude-opus-5
```

An evaluation experiment needs `name`, `prompts`, `system_prompt`, `active_agents`, `agents`, and `output_dir`. The runner unwraps the nested `agent` key. `prompts` is a mapping of prompt IDs to `text`, expected `ground_truth_methods`, and optional `failure_indicators` and metadata.

## Sweeping existing names

```bash
manyagents --multirun experiment=invariance_full 'active_agents=[claude],[openai],[local_llm]' agent@agents.local_llm=hf 'output_dir=${hydra:runtime.output_dir}'
```

Each comma-separated list is a separate job. `invariance_full` defines `claude`, `openai`, `local_llm`, and `biomni`; neither `hf` nor `mock` is an active-agent name there. `agent=claude,openai,hf` fails because this experiment loads named packages, not the plain `agent` group.

The legacy `local_llm` config includes `hf` through nested defaults with a global package directive. Under `@agents.local_llm`, its HF settings land outside that named entry. Use `agent@agents.local_llm=hf` to load HF directly into the existing name. This changes only composition for the command; no config-group restructuring is required. Outside Mila, add `agents.local_llm.agent.config.model=Qwen/Qwen3-0.6B`.

`output_dir=${hydra:runtime.output_dir}` stores results under each numbered Hydra job directory, avoiding collisions in the experiment's timestamp-based output directory. Quote list and interpolation overrides to protect them from the shell.

## Single-agent trace extraction

`trace_extraction` loads `/agent: hf`, so it uses the shorter **`agent.config`** path:

```bash
manyagents experiment=trace_extraction agent=hf agent.config.model=Qwen/Qwen3-0.6B
manyagents experiment=trace_extraction agent=vllm agent.config.dtype=float16
manyagents experiment=trace_extraction agent=claude agent.config.capture_hidden_states=false
```

The first command requires `traces`; vLLM replay requires both `traces` and `vllm`; Claude requires its API key and the dataset dependency from `traces`. API adapters cannot capture hidden states.

vLLM defaults assume bf16-capable hardware (Ampere or newer), with no fallback. Use `agent.config.dtype=float16` on V100/RTX 8000; the bfloat16 default is intentionally unchanged.

## Inspecting config and enabling logging

```bash
# Inspect one configuration; does not execute the runner
manyagents experiment=test_wandb --cfg job

# Requires the wandb extra and authentication
manyagents experiment=test_wandb wandb.enabled=true 'wandb.tags=[test,mock]'
```

`--cfg job` cannot be combined with `--multirun`. Use the CLI execution tests to validate runnable examples.

Defaults merge in order; `_self_` determines when the containing file's fields apply, and explicit value overrides apply afterward. Remote launcher configs and resource profiles may add environment interpolations that resolve on the submitting machine. Review the site-specific YAML before using a `mila_*` profile; remote launchers require Shop and cluster access. See [Running Experiments](running_experiments.md).
