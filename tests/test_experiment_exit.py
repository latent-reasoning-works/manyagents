"""Completion accounting and undefined measurements for experiment runs."""

import json
from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf

from manyagents.adapters import MockAdapter
from manyagents.experiment import run_experiment


@pytest.fixture
def experiment_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return OmegaConf.create({
        "name": "accounting", "output_dir": str(tmp_path / "results"),
        "system_prompt": "Recommend methods.", "active_agents": ["mock"],
        "agents": {"mock": {"adapter": "mock", "config": {"delay": 0}}},
        "prompts": {"first": {"text": "First dataset"}, "second": {"text": "Second dataset"}},
    })


@pytest.mark.asyncio
async def test_all_failures_exit_nonzero(experiment_config, caplog):
    from pathlib import Path

    cfg = experiment_config
    cfg.agents.mock.config.fail_rate = 1.0
    with pytest.raises(SystemExit) as exc:
        await run_experiment(cfg)
    assert exc.value.code == 1
    recorded = json.loads((Path(cfg.output_dir) / "results.json").read_text())
    failures = recorded["results"]["mock"]
    assert set(failures) == set(cfg.prompts)
    assert all(not result["success"] for result in failures.values())
    assert all("Simulated failure" in result["error"] for result in failures.values())
    assert recorded["metrics"]["mock"]["prompts_evaluated"] == 0
    assert recorded["metrics"]["mock"]["prompts_failed"] == len(cfg.prompts)
    assert "No successful evaluations" in caplog.text
    assert "mock: 0 succeeded, 2 failed" in caplog.text


@pytest.mark.asyncio
async def test_unknown_agent_rejected(experiment_config, monkeypatch, caplog):
    experiment_config.active_agents = ["mock", "nonexistent"]
    run = AsyncMock(return_value={"success": True, "output_files": {"raw_response": "Use PCA."}})
    monkeypatch.setattr(MockAdapter, "run", run)
    with pytest.raises(SystemExit) as exc:
        await run_experiment(experiment_config)
    assert exc.value.code == 1
    assert "Unknown active agent(s): nonexistent. Configured agents: mock" in caplog.text
    run.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_active_agents_rejected(experiment_config, caplog):
    experiment_config.active_agents = []
    with pytest.raises(SystemExit) as exc:
        await run_experiment(experiment_config)
    assert exc.value.code == 1
    assert "active_agents must not be empty" in caplog.text


@pytest.mark.asyncio
async def test_partial_failure_counts_both(experiment_config):
    cfg = experiment_config
    cfg.agents.failed = {"adapter": "mock", "config": {"delay": 0, "fail_rate": 1.0}}
    cfg.active_agents = ["mock", "failed"]
    result = await run_experiment(cfg)
    assert set(result["results"]) == {"mock", "failed"}
    for agent, succeeded, failed in [("mock", 2, 0), ("failed", 0, 2)]:
        assert set(result["results"][agent]) == set(cfg.prompts)
        assert sum(r["success"] for r in result["results"][agent].values()) == succeeded
        assert result["metrics"][agent]["prompts_evaluated"] == succeeded
        assert result["metrics"][agent]["prompts_failed"] == failed
