# Experiment configurations

These are the experiment configs shipped with manyagents. Select one with `experiment=<name>`; bare `manyagents` lists the available names and exits nonzero. See [Running Experiments](../../../docs/running_experiments.md) for setup and execution details.

## Start here

| Config | Purpose |
| --- | --- |
| `test_wandb.yaml` | Two mock prompts; runs locally without keys, GPU, or model downloads |
| `geometric_reasoning.yaml` | Nine prompts across three domains and three information conditions |
| `invariance_golden.yaml` | Golden invariance evaluation configuration |
| `invariance_full.yaml` | Four-prompt evaluation with Claude, OpenAI, local LLM, and Biomni agents |
| `invariance_compare_models.yaml` | Model comparison configuration |
| `trace_extraction.yaml` | Reasoning trace extraction; defaults to HF and tag segmentation |
| `reasoning_baseline.yaml` | Local reasoning baseline |
| `baseline_sweep.yaml` | Baseline sweep settings |
| `llm_reasoning_sweep.yaml` | Local-model reasoning sweep settings |

Inspect each config's defaults and comments for models, datasets, and dependencies. API experiments require credentials; local models require accessible weights and suitable hardware. Sweeps may include site-specific choices.

```bash
manyagents experiment=test_wandb
manyagents experiment=geometric_reasoning 'active_agents=[mock]'
# Inspect configuration without executing:
manyagents experiment=trace_extraction --cfg job
```

The text-evaluation runner expects text-producing adapters. For dimensionality reduction workflows, use `manyagents.workflows.sequence.execute_sequence` or `ManyLatentsAdapter` with the `traces` extra; there are no shipped single-algorithm or multi-step manylatents experiment configs.

## Add a project experiment

Copy a suitable shipped config into your own `configs/experiment/` directory, retain `# @package _global_`, and configure agents and prompts. For a checkout, add your directory to Hydra's search path:

```bash
manyagents 'hydra.searchpath=[file:///absolute/path/to/your_project/configs]' experiment=my_experiment
```

Keep project-specific data and model choices in project configs. Keep cluster paths and resource allocations in the `cluster=` and `resources=` config groups.
