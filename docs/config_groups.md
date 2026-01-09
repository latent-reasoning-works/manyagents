# Hydra Configuration Groups

manyAgents uses [Hydra](https://hydra.cc/) for composable configuration. This guide explains how config groups work and how to combine them.

## Overview

Config groups are directories under `manyagents/configs/` that contain related configuration options. When you run manyAgents, you select one option from each group to compose your final configuration.

```bash
# Example: Compose experiment + cluster + resources
manyagents experiment=geometric_reasoning cluster=mila_remote resources=api
```

## Config Groups

### `experiment/`

Defines **what** to run - scenarios, agents, prompts, and expected outputs.

| Config | Purpose |
|--------|---------|
| `geometric_reasoning` | 3×3 matrix: 3 biology domains × 3 information conditions |
| `test_wandb` | Quick test with mock adapter |
| `invariance_golden` | Minimal 4-scenario test |
| `invariance_full` | Complete scenario sweep |

**Key fields**:
```yaml
name: geometric_reasoning_suite
active_agents: [local_llm]           # Which agents to query
scenarios:                            # Prompts with ground truth
  immunology_A:
    text: "..."
    expected_geometry: discrete_clusters
    ground_truth_methods: [leiden, louvain]
```

### `agent/`

Configures **AI systems** - model selection, temperature, API settings.

| Config | Purpose |
|--------|---------|
| `claude` | Anthropic Claude (Sonnet) |
| `openai` | OpenAI GPT-4 |
| `local_llm` | Local Llama 3.1 8B |
| `local_llm_70b` | Local Llama 3.3 70B |
| `mock` | Testing without API calls |

**Key fields**:
```yaml
agent:
  name: claude
  adapter: claude
  config:
    model: claude-sonnet-4-20250514
    temperature: 0.1
    max_tokens: 4000
```

### `cluster/`

Defines **where** code runs - local machine or remote cluster.

| Config | Purpose |
|--------|---------|
| `local` | Default. Runs on current machine |
| `mila_remote` | Submits to Mila SLURM via SSH |

**Key fields** (mila_remote):
```yaml
hydra:
  mode: MULTIRUN
  launcher:
    _target_: shop.hydra.launchers.RemoteSlurmLauncher
    ssh_hostname: login.server.mila.quebec
    remote_dir: $SCRATCH/manyAgents
    conda_env: manyagents
```

### `resources/`

Defines **what hardware** is needed - CPU, memory, GPU.

| Config | Purpose |
|--------|---------|
| `cpu` | 4 CPUs, 16GB RAM, high parallelism (64) |
| `gpu` | 1 GPU, 4 CPUs, 32GB RAM, lower parallelism (16) |
| `api` | Minimal compute, propagates API keys, throttled (50) |

**Key fields** (api):
```yaml
hydra:
  launcher:
    cpus_per_task: 2
    mem: 8G
    array_parallelism: 50  # Prevent API rate limits
    setup_commands:
      - export OPENAI_API_KEY='${oc.env:OPENAI_API_KEY}'
      - export ANTHROPIC_API_KEY='${oc.env:ANTHROPIC_API_KEY}'
```

## How Configs Compose

Hydra merges configs in order, with later configs overriding earlier ones:

```
main.yaml (base)
    ↓
cluster/local.yaml (default)
    ↓
experiment/geometric_reasoning.yaml
    ↓
resources/api.yaml (if specified)
    ↓
CLI overrides
```

### The `defaults` Mechanism

Each config can specify defaults that pull in other configs:

```yaml
# experiment/geometric_reasoning.yaml
defaults:
  - /agent@agents.local_llm: local_llm    # Import agent config
  - /agent@agents.claude: claude
  - /agent@agents.openai: openai
  - /agent@agents.mock: mock
  - _self_                                 # Then apply this file
```

The `@agents.X` syntax places the imported config under the `agents.X` key.

### Override Priority

From lowest to highest priority:

1. `defaults` in config files
2. Config group selections (`experiment=X`)
3. CLI overrides (`wandb.enabled=true`)

## Common Patterns

### Run Locally (Default)

```bash
# Uses cluster=local implicitly
manyagents experiment=geometric_reasoning
```

### Run on Cluster with API Agents

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

manyagents experiment=geometric_reasoning \
    cluster=mila_remote \
    resources=api \
    active_agents=[claude,openai]
```

### Run on Cluster with Local LLMs

```bash
manyagents experiment=geometric_reasoning \
    cluster=mila_remote \
    resources=gpu
```

### Enable WandB Logging

```bash
manyagents experiment=geometric_reasoning wandb.enabled=true
```

### Override Agent Settings

```bash
# Use a different model
manyagents experiment=geometric_reasoning \
    agents.claude.agent.config.model=claude-opus-4-20250514
```

### View Resolved Config

```bash
# See final merged config
manyagents experiment=geometric_reasoning --cfg job

# See Hydra-specific config
manyagents experiment=geometric_reasoning cluster=mila_remote --cfg hydra
```

## Gotchas

### 1. Order of `defaults` Matters

Later defaults override earlier ones. Always put `_self_` last if you want your config to take precedence.

### 2. Nested Agent Configs

Agent configs are nested under `agents.{name}.agent` due to how Hydra's `@` syntax works:

```yaml
# Access with:
cfg.agents.claude.agent.config.model
# Not:
cfg.agent.config.model
```

### 3. Environment Variables in Remote Jobs

Environment variables like `${oc.env:USER}` are resolved at submission time (on your laptop), not at execution time (on the cluster). Use `setup_commands` for cluster-side environment setup.

### 4. `optional` Keyword

Use `optional` for configs that may not be selected:

```yaml
defaults:
  - optional resources: null  # OK if resources not specified
```

## Creating New Configs

### New Experiment

1. Create `configs/experiment/my_experiment.yaml`
2. Define `name`, `scenarios`, `active_agents`
3. Add agent defaults if using agents not in `main.yaml`

### New Resource Profile

1. Create `configs/resources/my_resources.yaml`
2. Use `@package _global_` directive
3. Override `hydra.launcher.*` settings

### New Cluster Target

1. Create `configs/cluster/my_cluster.yaml`
2. Set `hydra.launcher._target_` to appropriate launcher
3. Configure SSH and SLURM settings

## See Also

- [Running Experiments](running_experiments.md) - Execution guide
- [shop/remote-jobs](https://github.com/latent-reasoning-works/shop) - Launcher documentation
