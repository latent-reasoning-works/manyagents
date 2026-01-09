# Implementation Plan: Biomni and Kosmos Adapters

**Date**: 2026-01-08
**Status**: Planning
**Priority**: High

---

## Executive Summary

This plan addresses the integration of two AI research tools:
1. **Biomni** - Stanford's biomedical AI agent (Python API)
2. **Kosmos** - Autonomous scientist framework (CLI + Python API)

**Core Challenge**: Both tools have complex dependencies that may conflict with manyAgents' uv-managed environment.

**Recommended Strategy**: **Tiered Approach**
1. First attempt direct `uv add` installation
2. Fall back to isolated environment + bridge script pattern if conflicts arise
3. Avoid Docker unless absolutely necessary (too heavyweight)

---

## Tool Analysis

### Biomni (snap-stanford/Biomni)

| Aspect | Details |
|--------|---------|
| **Installation** | `pip install biomni` (also pip-installable from git) |
| **Entry Point** | Python API: `from biomni.agent import A1` |
| **CLI** | None - purely programmatic |
| **API Keys** | `ANTHROPIC_API_KEY` (required), OpenAI/Gemini optional |
| **Data** | Downloads ~11GB datalake on first use (can be disabled) |
| **Weights** | User has weights on Mila cluster - need path configuration |

**Key code pattern**:
```python
from biomni.agent import A1
agent = A1(path='./data', llm='claude-sonnet-4-20250514')
result = agent.go("Analyze gene expression patterns in dataset X")
```

### Kosmos (jimmc414/Kosmos)

| Aspect | Details |
|--------|---------|
| **Installation** | `pip install -e .` from cloned repo |
| **Entry Point** | CLI: `kosmos run "question"` OR Python API |
| **CLI Flags** | `--domain`, `--stream`, `--budget`, `--trace` |
| **API Keys** | `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` |
| **Data** | Minimal - generates artifacts during research |
| **Extras** | Optional Docker for sandboxed code execution |

**Key code patterns**:
```bash
# CLI
kosmos run "What are the latest advances in CRISPR?" --domain biology

# Python API
from kosmos.workflow.research_loop import ResearchWorkflow
workflow = ResearchWorkflow(research_objective="...", artifacts_dir="./out")
result = await workflow.run(num_cycles=5)
```

---

## Dependency Management Strategy

### Option A: Direct uv Installation (TRY FIRST)

```bash
uv add biomni
uv add kosmos  # May need to pip install from git
```

**Pros**:
- Simplest integration
- No subprocess overhead for Python API calls
- Single environment to manage

**Cons**:
- Risk of dependency conflicts (biomni has heavy biomedical stack)
- May pin incompatible versions

**Verdict**: Try this first. If `uv add` succeeds without breaking existing tests, use it.

### Option B: Isolated Environment + Bridge Scripts (FALLBACK)

Each tool gets its own Python environment. Adapter invokes a thin "bridge script" via subprocess.

```
manyagents/
├── adapters/
│   ├── biomni_adapter.py      # Calls bridge via subprocess
│   └── kosmos_adapter.py      # Calls bridge via subprocess
└── bridges/
    ├── biomni_bridge.py       # Lives in biomni's environment
    └── kosmos_bridge.py       # Lives in kosmos's environment
```

**Bridge script pattern**:
```python
#!/usr/bin/env python
"""Bridge script for Biomni - runs in biomni's conda/venv environment."""
import json
import sys

def main():
    config = json.loads(sys.argv[1])
    from biomni.agent import A1
    agent = A1(path=config['data_path'], llm=config.get('llm', 'claude-sonnet-4-20250514'))
    result = agent.go(config['task'])
    print(json.dumps({"success": True, "result": str(result)}))

if __name__ == "__main__":
    main()
```

**Adapter invocation**:
```python
async def run(self, task_config, input_files):
    python_path = self.config.get("python_path", "/path/to/biomni_env/bin/python")
    bridge_script = Path(__file__).parent.parent / "bridges" / "biomni_bridge.py"

    result = await run_subprocess(
        [python_path, str(bridge_script), json.dumps(task_config)],
        timeout=self.config.timeout
    )
    return self._parse_bridge_result(result)
```

**Pros**:
- Complete dependency isolation
- Works with any conda/venv setup
- Matches CellForge pattern

**Cons**:
- Subprocess overhead (~100ms per call)
- More complex setup
- Need to manage multiple environments

### Option C: Docker Containers (AVOID UNLESS NECESSARY)

```bash
docker run --rm -v $(pwd):/data biomni:latest python -c "..."
```

**Verdict**: Too heavyweight. Only consider if:
- Security sandboxing is required
- Reproducibility across systems is critical
- Neither Option A nor B works

---

## Recommended Implementation

### Phase 1: Test Direct Installation

```bash
# In manyAgents directory
cd /network/scratch/c/cesar.valdez/manyAgents
source .venv/bin/activate

# Try adding Biomni
uv add biomni

# Try adding Kosmos (from git since not on PyPI)
uv add git+https://github.com/jimmc414/Kosmos.git

# Run existing tests to check for breakage
pytest manyagents/adapters/test_adapters.py -v
```

**Decision Point**:
- If installation succeeds AND tests pass → Proceed with direct integration
- If conflicts → Use bridge script pattern

### Phase 2a: Direct Integration (if Phase 1 succeeds)

**BiomniAdapter** - Direct Python API calls:
```python
class BiomniAdapter(AgentAdapter):
    def __init__(self, data_path: Optional[str] = None):
        super().__init__("biomni")
        self.data_path = Path(data_path or os.getenv('BIOMNI_DATA_PATH', './data'))

    async def run(self, task_config, input_files) -> AdapterResult:
        from biomni.agent import A1
        agent = A1(path=str(self.data_path), llm=task_config.get('llm'))
        result = await asyncio.to_thread(agent.go, task_config['task'])
        return self.success_response(str(result), metadata={"agent": "A1"})
```

**KosmosAdapter** - CLI invocation (simpler, more stable):
```python
class KosmosAdapter(AgentAdapter):
    async def run(self, task_config, input_files) -> AdapterResult:
        cmd = ["kosmos", "run", task_config["research_question"]]
        if domain := task_config.get("domain"):
            cmd.extend(["--domain", domain])

        result = await run_subprocess(cmd, timeout=task_config.get("timeout", 3600))
        return self._parse_kosmos_output(result)
```

### Phase 2b: Bridge Script Pattern (if Phase 1 fails)

Create isolated environments:
```bash
# Biomni environment
conda create -n biomni_env python=3.11
conda activate biomni_env
pip install biomni

# Kosmos environment
conda create -n kosmos_env python=3.11
conda activate kosmos_env
cd /path/to/Kosmos && pip install -e .
```

Configure adapter paths:
```bash
export BIOMNI_PYTHON=/path/to/biomni_env/bin/python
export KOSMOS_PYTHON=/path/to/kosmos_env/bin/python
```

---

## Configuration Schema

### Environment Variables

```bash
# Biomni
BIOMNI_DATA_PATH=/network/scratch/c/cesar.valdez/biomni_data  # Weights location
BIOMNI_PYTHON=/path/to/biomni/python  # Only if using bridge pattern
ANTHROPIC_API_KEY=sk-...  # Required for Biomni

# Kosmos
KOSMOS_ARTIFACTS_DIR=./kosmos_artifacts
KOSMOS_PYTHON=/path/to/kosmos/python  # Only if using bridge pattern
```

### Adapter Configuration

**BiomniAdapter `task_config`**:
```python
{
    "task": "Analyze single-cell RNA-seq data for cell type markers",  # Required
    "llm": "claude-sonnet-4-20250514",  # Optional, default claude-sonnet
    "data_path": "/path/to/weights",  # Optional, uses BIOMNI_DATA_PATH
    "disable_datalake": True,  # Optional, skip 11GB download
}
```

**KosmosAdapter `task_config`**:
```python
{
    "research_question": "What are optimal hyperparameters for UMAP?",  # Required
    "domain": "machine-learning",  # Optional
    "num_cycles": 5,  # Optional, default 3
    "budget": 10.0,  # Optional, cost limit in $
    "timeout": 3600,  # Optional, default 1 hour
}
```

---

## Mila Cluster Specifics

### Biomni Weights Location

The user mentioned weights are on the Mila cluster. The adapter should:

1. Check for `BIOMNI_DATA_PATH` environment variable
2. Fall back to a sensible cluster default
3. Validate path exists before running

```python
def __init__(self, data_path: Optional[str] = None):
    super().__init__("biomni")
    self.data_path = Path(
        data_path
        or os.getenv('BIOMNI_DATA_PATH')
        or os.getenv('SCRATCH', '/network/scratch') + '/biomni_data'
    )

    if not self.data_path.exists():
        log.warning(f"Biomni data path not found: {self.data_path}")
```

### Conda vs uv on Cluster

Mila clusters often use conda modules. If using bridge pattern:

```bash
# Load conda module on Mila
module load miniconda3

# Activate specific environment
conda activate biomni_env
```

The adapter can handle this via shell commands:
```python
cmd = ["bash", "-c", f"source ~/.bashrc && conda activate biomni_env && python {bridge_script} '{json.dumps(config)}'"]
```

---

## Implementation Timeline

### Step 1: Dependency Test (15 min)
- [ ] Try `uv add biomni`
- [ ] Try `uv add kosmos` (or from git)
- [ ] Run existing tests
- [ ] Document results

### Step 2: Adapter Implementation (parallelizable)

**BiomniAdapter** (if Step 1 succeeds):
- [ ] Create `manyagents/adapters/biomni_adapter.py`
- [ ] Handle data_path configuration
- [ ] Async wrapper for synchronous A1.go()
- [ ] Add to test_adapters.py

**KosmosAdapter**:
- [ ] Create `manyagents/adapters/kosmos_adapter.py`
- [ ] CLI invocation via run_subprocess
- [ ] Parse JSON output
- [ ] Add to test_adapters.py

### Step 3: Testing & Validation
- [ ] Unit tests with mocked API calls
- [ ] Integration test (if API keys available)
- [ ] Run code-simplifier
- [ ] Commit

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Dependency conflicts | High | Medium | Bridge script fallback ready |
| Biomni 11GB download | Medium | Low | Use `expected_data_lake_files=[]` |
| API key costs | Medium | Medium | Use cheaper models for tests |
| Conda module issues on Mila | Low | Medium | Document exact setup steps |
| Kosmos not on PyPI | Certain | Low | Install from git URL |

---

## Success Criteria

- [ ] Both adapters inherit from `AgentAdapter`
- [ ] Both return standardized `AdapterResult`
- [ ] Existing tests still pass (no regressions)
- [ ] New adapter tests pass
- [ ] Added to `ADAPTER_REGISTRY` in main.py
- [ ] Documentation updated
- [ ] Weights path configurable via environment variable

---

## Files to Create/Modify

### New Files
1. `manyagents/adapters/biomni_adapter.py` (~100 lines)
2. `manyagents/adapters/kosmos_adapter.py` (~120 lines)
3. (If bridge needed) `manyagents/bridges/biomni_bridge.py` (~30 lines)
4. (If bridge needed) `manyagents/bridges/kosmos_bridge.py` (~30 lines)

### Modified Files
1. `manyagents/adapters/test_adapters.py` - Add new adapters to parameterized fixture
2. `manyagents/main.py` - Add to ADAPTER_REGISTRY
3. `pyproject.toml` - Add dependencies (if direct install works)

---

## Next Steps

1. **Immediate**: Test `uv add biomni` and observe results
2. **Based on result**: Choose direct integration or bridge pattern
3. **Parallel work**: Both adapters can be implemented simultaneously
4. **Final**: Run code-simplifier, commit, merge
