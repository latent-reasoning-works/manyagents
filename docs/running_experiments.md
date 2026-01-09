# Running Experiments

This guide covers how to run manyAgents experiments, from local debugging to large-scale cluster deployments.

## Quick Start

### Local Execution (Default)

```bash
# Run geometric reasoning experiment with default agent
manyagents experiment=geometric_reasoning

# Use mock adapter for testing (no API calls)
manyagents experiment=geometric_reasoning active_agents=[mock]

# Enable wandb logging
manyagents experiment=geometric_reasoning wandb.enabled=true
```

### What Happens

1. Hydra loads and merges configs
2. For each scenario, each active agent is queried
3. Responses are parsed for method recommendations
4. Metrics are computed (Jaccard similarity, ground truth match)
5. Results saved to `outputs/` directory

## Scaling to Cluster

### When to Use Cluster Deployment

Use cluster deployment when:

- **Large sweeps**: 100+ scenario/agent combinations
- **GPU requirements**: Running local LLMs (Llama 3.1, etc.)
- **Long-running jobs**: Experiments taking hours
- **API rate limits**: Need throttled parallel execution

### Configuration Pattern

Combine `cluster` + `resources` config groups:

```bash
manyagents experiment=X cluster=mila_remote resources=Y
```

| `resources=` | Use Case |
|--------------|----------|
| `cpu` | Data processing, light agents |
| `gpu` | Local LLM inference |
| `api` | Claude/OpenAI (propagates API keys) |

### Example: Run with API Agents on Cluster

```bash
# Set API keys locally (they'll be propagated to cluster)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

manyagents experiment=geometric_reasoning \
    cluster=mila_remote \
    resources=api \
    active_agents=[claude,openai]
```

### Example: Run with Local LLMs on Cluster

```bash
manyagents experiment=geometric_reasoning \
    cluster=mila_remote \
    resources=gpu
```

## WandB Integration

### Mass Experiment Tracking

When running hundreds of jobs, you want them grouped logically in wandb. The `mila_remote` cluster config automatically sets:

```yaml
wandb:
  group: ${oc.env:SLURM_ARRAY_JOB_ID}      # All array jobs grouped
  id: ${oc.env:SLURM_JOB_ID}_${oc.env:SLURM_ARRAY_TASK_ID}
  name: task_${oc.env:SLURM_ARRAY_TASK_ID}  # Unique per task
```

This means a 100-job sweep appears as one collapsible group in wandb, not 100 separate runs.

### Enabling WandB

```bash
# Add to any command
manyagents experiment=X wandb.enabled=true

# With custom tags
manyagents experiment=X wandb.enabled=true wandb.tags=[geometric,icml]
```

### Logged Data

The ExperimentLogger tracks:

| Metric | Description |
|--------|-------------|
| `{agent}/{scenario}/success` | Did the agent respond? |
| `{agent}/{scenario}/method_count` | Methods extracted |
| `{agent}/{scenario}/ground_truth_match` | Correct for geometry? |
| `summary/{agent}/jaccard` | Cross-scenario similarity |
| `results_summary` | Table for paper figures |
| `failure_analysis` | When wrong method recommended |

## API Key Handling

### How Keys Propagate to Remote Jobs

The `resources/api.yaml` config includes setup commands that export your local API keys to the cluster environment:

```yaml
setup_commands:
  - export OPENAI_API_KEY='${oc.env:OPENAI_API_KEY}'
  - export ANTHROPIC_API_KEY='${oc.env:ANTHROPIC_API_KEY}'
```

**Security Note**: Keys are written to the SLURM job script. Ensure your cluster storage has appropriate permissions.

### Local Testing

For local runs, just set the environment variables:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
manyagents experiment=geometric_reasoning active_agents=[claude,openai]
```

## Monitoring and Debugging

### View Resolved Config Before Running

```bash
# Show full config
manyagents experiment=geometric_reasoning cluster=mila_remote --cfg job

# Show Hydra/launcher config
manyagents experiment=geometric_reasoning cluster=mila_remote --cfg hydra
```

### Monitor Cluster Jobs

```bash
# Check job status
ssh mila 'squeue -u $USER'

# Watch job progress
watch -n 10 'ssh mila "squeue -u $USER"'

# View job logs
ssh mila 'cat $SCRATCH/manyAgents/outputs/*/slurm-*.out'
```

### Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| "Unknown adapter" | Agent not in registry | Check `active_agents` matches available agents |
| "API key not found" | Key not exported | Set `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` |
| Cluster jobs fail immediately | SSH/conda issues | Test `ssh mila 'conda activate manyagents'` |
| WandB not logging | Not enabled | Add `wandb.enabled=true` |

## Output Structure

Results are saved to the `output_dir` specified in the experiment config:

```
outputs/geometric_reasoning/
└── geometric_reasoning_suite_2024-01-09_15-39-09/
    ├── results.json          # Full experiment data
    ├── summary.md            # Human-readable table
    └── raw_responses/        # Per-agent responses
        ├── claude/
        │   ├── immunology_A.txt
        │   └── ...
        └── openai/
            └── ...
```

## Reference

For detailed documentation on:

- **Launcher mechanics, SSH config, SLURM debugging**: See [shop/remote-jobs](https://github.com/latent-reasoning-works/shop/blob/main/docs/guide/remote-jobs.md)
- **Cluster-specific setup (Mila, DRAC)**: See [shop/cluster-setup](https://github.com/latent-reasoning-works/shop/blob/main/docs/guide/cluster-setup.md)
- **Config group details**: See [config_groups.md](config_groups.md)

## See Also

- [Config Groups](config_groups.md) - How Hydra configs compose
- [Adapters](adapters_vs_utils.md) - Agent adapter architecture
