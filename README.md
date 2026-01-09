# manyAgents

Multi-agent orchestration for scientific workflows.

## Install

```bash
uv sync
source .venv/bin/activate
```

## Usage

manyAgents uses [Hydra](https://hydra.cc/) for configuration:

```bash
# Run an experiment
manyagents experiment=geometric_reasoning

# With specific agents
manyagents experiment=geometric_reasoning active_agents=[claude,openai]

# Enable logging
manyagents experiment=geometric_reasoning wandb.enabled=true
```

## Adapters

Adapters provide a unified interface for different AI systems:

| Adapter | Type | Purpose |
|---------|------|---------|
| `claude` | API | Anthropic Claude |
| `openai` | API | OpenAI GPT models |
| `local_llm` | Local | HuggingFace models (Llama, etc.) |
| `manylatents` | Python | Dimensionality reduction algorithms |
| `mock` | Testing | No API calls, configurable responses |

## Cluster Execution

For SLURM clusters, combine `cluster` and `resources` configs:

```bash
manyagents experiment=X cluster=mila_remote resources=api
```

See [docs/running_experiments.md](docs/running_experiments.md) for details.

## Documentation

- [Running Experiments](docs/running_experiments.md) - Execution guide
- [Config Groups](docs/config_groups.md) - Hydra configuration
- [Design Decisions](docs/design_decisions.md) - Architecture rationale

## Related

Part of the [Latent Reasoning Works](https://github.com/latent-reasoning-works) ecosystem:

- **[manyLatents](https://github.com/latent-reasoning-works/manylatents)** - Geometric learning algorithms
- **[Geomancy](https://github.com/latent-reasoning-works/geomancy)** - RL training with geometric rewards
- **[shop](https://github.com/latent-reasoning-works/shop)** - Cluster infrastructure

## License

MIT
