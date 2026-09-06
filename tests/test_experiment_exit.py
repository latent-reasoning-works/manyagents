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


@pytest.mark.parametrize("method_sets", [{}, {"only": {"leiden"}}], ids=["zero", "single"])
def test_jaccard_none_with_single_prompt(method_sets):
    from manyagents.metrics.llm import compute_pairwise_jaccard

    assert compute_pairwise_jaccard(method_sets) is None


def test_zero_success_rates_are_none():
    from manyagents.metrics.llm import compute_system_metrics

    metrics = compute_system_metrics({"failed": {"success": False}}, {})
    for name in ["jaccard_similarity_across_prompts", "jaccard_min", "jaccard_max",
                 "ground_truth_match_rate", "clustering_for_all_rate"]:
        assert metrics[name] is None
    assert metrics["prompts_evaluated"] == 0
    assert metrics["prompts_failed"] == 1


def test_undefined_metrics_in_json_console_and_markdown(tmp_path, capsys):
    from manyagents.experiment import _print_summary, _save_results

    result = {
        "timestamp": "2026-09-06", "metrics": {"failed": {
            "jaccard_similarity_across_prompts": None,
            "ground_truth_match_rate": None, "clustering_for_all_rate": None,
            "prompts_evaluated": 0, "prompts_failed": 2,
        }},
    }
    _save_results(result, tmp_path, "undefined")
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["metrics"]["failed"]["ground_truth_match_rate"] is None
    assert (tmp_path / "summary.md").read_text().count("n/a") == 3
    _print_summary(result["metrics"])
    assert capsys.readouterr().out.count("n/a") == 3


def test_wandb_omits_undefined_metrics(monkeypatch):
    from unittest.mock import Mock
    from manyagents.utils import logger as logger_module

    wandb = Mock()
    monkeypatch.setattr(logger_module, "_get_wandb", lambda: wandb)
    logger = logger_module.ExperimentLogger(enabled=False)
    logger.enabled = True
    logger.run = Mock()
    logger.log_system_metrics("failed", {
        "jaccard_similarity_across_prompts": None, "jaccard_min": None, "jaccard_max": None,
        "ground_truth_match_rate": None, "clustering_for_all_rate": None,
        "prompts_evaluated": 0, "prompts_failed": 2,
    })
    assert wandb.log.call_args.args[0] == {
        "summary/failed/prompts_evaluated": 0, "summary/failed/prompts_failed": 2,
    }


def test_wandb_summary_handles_missing_measurements(monkeypatch):
    from unittest.mock import Mock
    from manyagents.utils import logger as logger_module

    wandb = Mock()
    monkeypatch.setattr(logger_module, "_get_wandb", lambda: wandb)
    logger = logger_module.ExperimentLogger(enabled=False)
    logger.enabled = True
    logger.run = Mock()
    logger.log_summary_table({"single": {
        "jaccard_similarity_across_prompts": None, "ground_truth_match_rate": 0.5,
        "clustering_for_all_rate": 1.0, "prompts_evaluated": 1, "prompts_failed": 1,
    }})
    row = wandb.Table.return_value.add_data.call_args.args
    assert row == ("single", None, 0.5, 1.0, 1, 1)


def test_gvector_metric_failure_propagates(monkeypatch):
    import sys
    from types import ModuleType
    from unittest.mock import Mock
    import numpy as np
    from manyagents.workflows.sequence import compute_gvector

    metrics_module = ModuleType("manylatents.metrics")
    metrics_module.compute_metric = Mock(side_effect=RuntimeError("metric exploded"))
    monkeypatch.setitem(sys.modules, "manylatents.metrics", metrics_module)
    with pytest.raises(RuntimeError, match="metric exploded"):
        compute_gvector(np.zeros((3, 2)), ["participation_ratio"])


def test_latex_summary_handles_undefined_metrics():
    from manyagents.metrics.llm import generate_summary_table

    table = generate_summary_table({"metrics": {"failed": {
        "jaccard_similarity_across_prompts": None,
        "ground_truth_match_rate": None, "clustering_for_all_rate": None,
    }}}, format="latex")
    assert table.count("n/a") == 3
