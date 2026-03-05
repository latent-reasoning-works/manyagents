<div align="center">

<pre>
    ∿ · ─ · ∿ · ─
  · ─ · ∿ · ─ · ∿ ·  ──▶  Σ(·)
    ─ · ∿ · ─ · ∿

        m a n y a g e n t s

    coordinate, dispatch, aggregate
</pre>

[![license](https://img.shields.io/badge/license-MIT-8B5CF6.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11+-8B5CF6.svg)](https://www.python.org)
[![uv](https://img.shields.io/badge/pkg-uv-8B5CF6.svg)](https://docs.astral.sh/uv/)

</div>

---

Multi-agent orchestration for scientific workflows. 13 adapters — Claude, OpenAI, HuggingFace, manyLatents, and more — with reasoning trace capture, metric extraction, and cached execution. Part of the [Latent Reasoning Works](https://github.com/latent-reasoning-works) ecosystem.

## install

```bash
git clone https://github.com/latent-reasoning-works/manyagents.git
cd manyagents && uv sync
```

With optional integrations:

```bash
uv sync --extra full    # manylatents + biomni
uv sync --extra dev     # pytest, ruff, pre-commit
```

## quickstart

```bash
# Run an experiment
manyagents experiment=geometric_reasoning

# With specific agents
manyagents experiment=geometric_reasoning active_agents=[claude,openai]

# Extract reasoning traces
manyagents experiment=trace_extraction agent=claude

# Sweep models
manyagents --multirun agent=claude,openai,hf experiment=invariance_full

# Submit to SLURM
manyagents experiment=geometric_reasoning cluster=mila_remote resources=api
```

```python
from manyagents.adapters import ManyLatentsAdapter

adapter = ManyLatentsAdapter()
adapter.setup_metrics(["participation_ratio", "trustworthiness"])

result = await adapter.execute_cached(
    algorithm="UMAP",
    params={"n_neighbors": 15, "min_dist": 0.1},
    data=embeddings,
)
# result.scores: {"participation_ratio": 12.4, "trustworthiness": 0.92}
```

---

## adapters

13 adapters behind a unified `AgentAdapter` interface:

| Adapter | Type | Purpose |
|---------|------|---------|
| `ClaudeAdapter` | API | Anthropic Claude |
| `OpenAIAdapter` | API | OpenAI GPT models |
| `HFAdapter` | Local | HuggingFace models (Llama, etc.) |
| `ManyLatentsAdapter` | Python | DR algorithms + geometric metrics |
| `CellForgeAdapter` | CLI | Cell analysis |
| `KosmosAdapter` | CLI | Kosmos vision model |
| `BiomniAdapter` | CLI | Biomni system |
| `MockAdapter` | Testing | No API calls, configurable responses |

```python
from manyagents.adapters import get_adapter

adapter = get_adapter("claude")
result = await adapter.run(task_config)
# result: {"response": str, "metrics": dict, "trace": ReasoningTrace}
```

---

## reasoning traces

Capture and analyze LLM chain-of-thought:

```python
from manyagents.schemas import ReasoningTrace

# Traces captured automatically during inference
trace: ReasoningTrace = result["trace"]
trace.steps        # list[ReasoningStep] — each CoT step
trace.model_info   # ModelInfo — model, temperature, tokens
trace.task_info    # TaskInfo — prompt, dataset, experiment
```

Trace hidden states feed into [geomancy](https://github.com/latent-reasoning-works/geomancy)'s trajectory geometry pipeline (velocity, curvature, anisotropy).

---

## metrics

LLM response evaluation:

```python
from manyagents.metrics import compute_system_metrics, extract_methods

# Parse recommended methods from LLM text
methods = extract_methods(response_text)

# Aggregate metrics across prompts
metrics = compute_system_metrics(agent_results, prompts)
# {"ground_truth_match_rate": 0.85, "jaccard_similarity": 0.23, ...}
```

---

## layout

```
manyagents/
├── adapters/            # 13 adapters + base + registry
│   ├── claude_adapter.py
│   ├── openai_adapter.py
│   ├── hf_adapter.py
│   ├── manylatents_adapter.py
│   └── ...
├── configs/             # Hydra config groups
│   ├── agent/           # claude, openai, hf, mock, ...
│   ├── experiment/      # geometric_reasoning, trace_extraction, ...
│   ├── cluster/         # local, mila_remote, mila_slurm
│   └── prompts/         # Prompt templates
├── metrics/             # LLM evaluation + method extraction
├── schemas/             # Pydantic models (ReasoningTrace, etc.)
├── workflows/           # Sequence orchestration
├── utils/               # Logging, retry, validation
├── main.py              # Hydra CLI entry point
├── experiment.py        # Experiment runner
└── inference.py         # Inference + trace management
```

---

## documentation

| Doc | Description |
|-----|-------------|
| [Running Experiments](docs/running_experiments.md) | Local and cluster execution |
| [Config Groups](docs/config_groups.md) | Hydra configuration |
| [Design Decisions](docs/design_decisions.md) | Architecture rationale |
| [Adapters vs Utils](docs/adapters_vs_utils.md) | Where code belongs |
| [Contributing](docs/CONTRIBUTING.md) | Development guidelines |

---

## ecosystem

| Library | Role | Responsibility |
|---------|------|----------------|
| **[manyLatents](https://github.com/latent-reasoning-works/manylatents)** | Body | DR algorithms (12), geometric metrics (90+) |
| **[manyAgents](https://github.com/latent-reasoning-works/manyagents)** | Brain | Orchestration, adapters, reasoning traces |
| **[Geomancy](https://github.com/latent-reasoning-works/geomancy)** | Trainer | RL environments, reward, policy training |
| **[Shop](https://github.com/latent-reasoning-works/shop)** | Infra | SLURM launchers, cluster health, log sync |

---

## development

```bash
uv sync --extra dev
uv run pytest tests/ -v
uv run ruff check manyagents/
```

**Dependencies**: pydantic, hydra-core, numpy, anthropic, openai, transformers

---

<p align="center">
<sub>MIT License · <a href="https://github.com/latent-reasoning-works">Latent Reasoning Works</a></sub>
</p>
