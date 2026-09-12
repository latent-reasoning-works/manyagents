# Contributing to manyAgents

## Development setup

Use Python **3.11–3.12** and uv. From a source checkout:

```bash
uv sync --locked
uv run --no-sync pytest -q
uv run --no-sync ruff check manyagents/ tests/ scripts/
```

The default development group includes pytest and ruff. `uv sync --extra dev` also installs pre-commit. Enable the shipped Ruff hook with `uv run --no-sync pre-commit install` and check tracked Python files with `uv run --no-sync pre-commit run --all-files`. Optional integration tests require their extras:

```bash
uv sync --locked --extra traces
uv run --no-sync pytest -q
```

Core-only tests skip individual cases requiring manylatents. Keep dependency-free tests runnable in core; use the `requires_manylatents` marker on tests that actually need it. vLLM is a separate GPU extra and is not included in `traces` or `full`. See [Running Experiments](running_experiments.md) for hardware requirements.

## Workflow

1. Create a focused branch, such as `fix/cli-diagnostic` or `docs/trace-contract`.
2. Reproduce a behavior change with a failing test before fixing it. Verify execution and results, not merely config composition.
3. Keep commits focused and document changed public behavior.
4. Run the relevant tests, then the full suite and ruff.
5. Open a pull request explaining the problem, resulting behavior, and validation.

The full suite is `pytest -q`, including tests under both `tests/` and `manyagents/`. For example:

```bash
uv run --no-sync pytest -q tests/test_cli_commands.py
uv run --no-sync pytest -q manyagents/adapters/test_adapters.py
```

CLI tests should invoke the Hydra entry point in-process, with mock adapters, isolated output directories, and assertions on persisted responses and scores. `--cfg job` does not execute the runner, and Hydra rejects it alongside `--multirun`.

## Adapter conventions

Subclass `AgentAdapter` and implement its async `run` method. A minimal text adapter has this shape:

```python
from manyagents.adapters.base import AgentAdapter, AdapterResult


class YourToolAdapter(AgentAdapter):
    def __init__(self):
        super().__init__("yourtool")

    async def run(self, task_config: dict, input_files: dict) -> AdapterResult:
        response = "Use PCA."  # Replace with the external tool call.
        return self.success_response(
            summary="Completed",
            output_files=self.save_response(response),
        )
```

Register the class in `manyagents/adapters/__init__.py`, add an agent YAML, and test success and failure behavior. An agent config uses:

```yaml
# @package _global_
agent:
  name: yourtool
  adapter: yourtool
  config: {}
```

Use `success_response()` / `error_response()` for the result contract and `save_response()` for evaluated text. Compute/artifact adapters must set `PRODUCES_TEXT_RESPONSE = False`; the evaluation runner must not score diagnostic output as an answer. Keep optional dependency imports lazy and name the relevant install extra in errors.

Use type hints at public boundaries, descriptive names, and docstrings explaining arguments and results. Geometric metrics and DR algorithms belong in manylatents; manyagents coordinates them. CellForge, Kosmos, and Biomni execute local code with caller permissions. Biomni runs in-process via `asyncio.to_thread`, which is not a sandbox or a killable subprocess.

## Documentation and review

Update README examples and the relevant `docs/` pages whenever public commands, result shapes, or requirements change. Keep CLI examples executable with their stated dependencies and credentials. Do not promise unavailable files, services, or a hosted documentation site.

Before requesting review, check that tests pass, ruff is clean, the docs match the implementation, and the diff stays within the intended scope. Describe any remaining unverified integration or hardware behavior in the pull request.

## Releases and recognition

Maintainers manage versioning and releases. Release preparation must update both package version locations and the lock, record release notes, and verify the committed source and built wheel before tagging. Those are release tasks, separate from ordinary documentation changes.

Contributors are credited through Git history, pull requests, and release notes. Be respectful and constructive in issues and reviews. Contributions are licensed under the project's [MIT License](../LICENSE).
