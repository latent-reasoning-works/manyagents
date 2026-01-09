# Plan: Enabling Mass Cluster Scaling for ManyAgents

## Objective
Enable `manyagents` to seamlessly execute massive-scale agent workflows (1000+ jobs) on high-performance computing clusters (like Mila/DRAC) by integrating with the `shop` infrastructure library.

## Core Philosophy: Separation of Concerns

### 1. `manyagents` (The Application)
*   **Responsibility**: Defines *what* to run.
*   **Scope**: Contains agent logic, prompts, experiment definitions, and metric calculations.
*   **Independence**: Must remain fully functional for local execution (`python main.py ...`) without requiring `shop` or SLURM. It should not contain complex infrastructure logic.
*   **Interface**: Exposes standard Hydra configurations for experiments and agents.

### 2. `shop` (The Infrastructure Layer)
*   **Responsibility**: Defines *how* and *where* to run.
*   **Scope**: Handles cluster submission (SLURM), remote execution (SSH), environment synchronization (Conda/Pip), and credential propagation.
*   **Role**: Acts as a plugin/launcher for `manyagents`. It wraps the application to lift it from a local process to a distributed cluster workload.
*   **Key Features**:
    *   **Mass Job Orchestration**: Translates Hydra "multiruns" into efficient SLURM job arrays.
    *   **Credential Wrangling**: Securely propagates API keys (OpenAI, Anthropic) and proxy settings to remote compute nodes.
    *   **Resource Abstraction**: precise control over CPU/GPU/Memory requests via `resources` config groups.

## Implementation Strategy

### A. Configuration Structure (in `manyagents`)
We will adopt the `resources` config pattern found in `manylatents` to allow mix-and-match infrastructure selection.

1.  **`configs/resources/`**: New directory for hardware profiles.
    *   `cpu.yaml`: For light logic or local testing.
    *   `gpu.yaml`: For local LLM inference.
    *   `api.yaml`: Specialized for API-based agents (low compute, high concurrency, network-ready).

2.  **`configs/cluster/`**: New directory for execution environments.
    *   `local.yaml`: Standard local execution (default).
    *   `mila_remote.yaml`: Uses `shop.RemoteSlurmLauncher` to dispatch to the cluster.

### B. The Workflow
1.  **Local Dev**:
    ```bash
    python -m manyagents.main experiment=test
    ```
    *Runs locally, uses local env vars.*

2.  **Mass Production**:
    ```bash
    python -m manyagents.main \
      cluster=mila_remote \
      resources=api \
      experiment=invariance_full
    ```
    *   `manyagents` generates the tasks.
    *   `shop` takes over:
        1.  Connects to cluster via SSH.
        2.  Syncs code/env.
        3.  Injects API keys (from local env) into the remote job script.
        4.  Submits a 1000-job array.
    *   Compute nodes execute independent `manyagents` tasks.

## Immediate Action Items
1.  Create `manyagents/configs/resources/` and populate with `cpu.yaml`, `gpu.yaml`.
2.  Create `manyagents/configs/cluster/mila_remote.yaml` utilizing `shop`.
3.  Implement "Credential Wrangling" in `shop` (if not present) or configure `mila_remote.yaml` to pass specific env vars (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) via `setup_commands`.