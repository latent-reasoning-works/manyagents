# Cluster Scaling Setup Documentation

## Overview
We have implemented a configuration architecture that enables `manyagents` to scale from local debugging to massive cluster deployments (1000+ jobs) using the `shop` infrastructure library. This setup enforces a strict separation of concerns: `manyagents` handles agent logic, while `shop` handles compute orchestration.

## Configuration Structure

### 1. Cluster Profiles (`configs/cluster/`)
Defines *where* the code runs.

*   **`local.yaml`** (Default):
    *   Executes code on the current machine.
    *   Uses standard Hydra basic launcher.
    *   Best for: Debugging, small tests, logic verification.

*   **`mila_remote.yaml`**:
    *   **Target**: Mila/DRAC Slurm clusters.
    *   **Mechanism**: Uses `shop.hydra.launchers.RemoteSlurmLauncher`.
    *   **Action**: Connects via SSH, syncs code to `$SCRATCH`, and submits a SLURM array job.
    *   **Env Vars**: Automatically sets `PROJECT_NAME` and `CONDA_ENV` defaults.

### 2. Resource Profiles (`configs/resources/`)
Defines *what hardware* is needed.

*   **`cpu.yaml`**:
    *   4 CPUs, 16GB RAM.
    *   High parallelism (64 concurrent jobs).
    *   Best for: Data processing, light agents.

*   **`gpu.yaml`**:
    *   1 GPU, 4 CPUs, 32GB RAM.
    *   Lower parallelism (16 concurrent jobs) to respect quotas.
    *   Best for: Local LLM inference (Llama-3, Mistral).

*   **`api.yaml`**:
    *   2 CPUs, 8GB RAM (minimal compute).
    *   **Key Feature**: Propagates `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` from your local environment to the remote cluster node.
    *   **Throttling**: Sets `array_parallelism: 50` to prevent hitting API rate limits (429 errors).
    *   Best for: Claude/OpenAI experiments.

## Usage Examples

**1. Run Locally (Default)**
```bash
python -m manyagents.main experiment=invariance_full
```

**2. Run on Cluster with API Agents**
```bash
# Ensure keys are set locally first
export OPENAI_API_KEY=sk-...
python -m manyagents.main \
    cluster=mila_remote \
    resources=api \
    experiment=invariance_full
```

**3. Run on Cluster with Local LLMs**
```bash
python -m manyagents.main \
    cluster=mila_remote \
    resources=gpu \
    experiment=invariance_full
```

---

## Next Steps: Testing Composable WandB Runs

To ensure mass experiments are trackable, we need to verify WandB integration in distributed mode.

### The Challenge
When running 1000 jobs, we don't want 1000 separate WandB runs cluttering the project. We want them grouped logically.

### Current Setup
The `mila_remote.yaml` config already sets:
```yaml
wandb:
  group: ${oc.env:SLURM_ARRAY_JOB_ID}  # Groups all tasks in the array
  name: task_${oc.env:SLURM_ARRAY_TASK_ID}  # Unique name per task
```

### Testing Plan
1.  **Dry Run**:
    *   Run `python -m manyagents.main cluster=mila_remote resources=api --info` (or similar) to verify the generated SLURM script contains the correct API keys and WandB env vars.
2.  **Mock Experiment**:
    *   Create a `experiment=test_wandb` config that runs a "Placeholder" agent which just logs a random number to WandB.
    *   Run this with `cluster=mila_remote` and `nodes=1` (small scale).
3.  **Verify Grouping**:
    *   Check the WandB dashboard.
    *   Confirm all tasks appear under a single "Group" (the SLURM Job ID).
    *   Confirm logs/artifacts are correctly uploaded from the remote nodes.
4.  **Offline Sync Test** (Optional):
    *   If cluster nodes have no internet, verify `shop` configures `wandb mode=offline` and `RemoteSlurmLauncher` handles syncing the `.wandb` directory back to local after the job finishes.
