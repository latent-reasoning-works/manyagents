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


def test_gvector_metric_failure_is_recorded(monkeypatch):
    import sys
    from types import ModuleType
    from unittest.mock import Mock
    import numpy as np
    from manyagents.workflows.sequence import compute_gvector

    metrics_module = ModuleType("manylatents.metrics")
    metrics_module.compute_metric = Mock(side_effect=RuntimeError("metric exploded"))
    monkeypatch.setitem(sys.modules, "manylatents.metrics", metrics_module)
    g = compute_gvector(np.zeros((3, 2)), ["participation_ratio"])
    assert g.measurements["participation_ratio"] == {
        "status": "failed", "reason": "RuntimeError: metric exploded",
    }
    with pytest.raises(ValueError, match="participation_ratio.*metric exploded"):
        g.metric_value("participation_ratio")


def test_latex_summary_handles_undefined_metrics():
    from manyagents.metrics.llm import generate_summary_table

    table = generate_summary_table({"metrics": {"failed": {
        "jaccard_similarity_across_prompts": None,
        "ground_truth_match_rate": None, "clustering_for_all_rate": None,
    }}}, format="latex")
    assert table.count("n/a") == 3


@pytest.fixture
def trace_experiment(tmp_path, monkeypatch):
    from manyagents import experiment
    from manyagents.schemas.reasoning import ReasoningTrace, ReasoningStep, TraceStore

    cfg = OmegaConf.create({
        "name": "traces", "output_dir": str(tmp_path),
        "trace_extraction": {"enabled": True, "dataset": "fixture", "n_samples": 2},
        "agent": {"adapter": "mock", "config": {"build_trace": True}},
    })
    tasks = [{"task_id": f"task_{i}", "prompt": f"Problem {i}"} for i in range(2)]
    monkeypatch.setattr(experiment, "_load_dataset_tasks", lambda *args: tasks)
    with TraceStore(tmp_path / "traces") as store:
        store.append(ReasoningTrace(trace_id="previous_run"))
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(ReasoningTrace(trace_id="new_trace", steps=[
        ReasoningStep(index=0, text="First step"), ReasoningStep(index=1, text="Second step"),
    ]).to_json())
    return cfg, tasks, trace_path


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [
    "adapter_failure", "missing_trace", "corrupt_trace", "persist_error", "adapter_error",
    "missing_tensors", "empty_tensors", "zero_size_tensors", "corrupt_tensors", "zero_tasks",
    "string_tensors", "nan_tensors", "inf_tensors", "integer_tensors", "scalar_tensors",
])
async def test_trace_extraction_zero_traces_fails(failure, trace_experiment, monkeypatch, caplog):
    from unittest.mock import Mock
    import numpy as np
    from manyagents.schemas.reasoning import TraceStore

    cfg, tasks, trace_path = trace_experiment
    result = {"success": True, "summary": "Done", "output_files": {"trace": trace_path}}
    if failure == "adapter_failure":
        result = {"success": False, "summary": "Simulated failure", "output_files": {}}
    elif failure == "missing_trace":
        result["output_files"] = {}
    elif failure == "corrupt_trace":
        trace_path.write_text("invalid json")
    elif failure == "persist_error":
        monkeypatch.setattr(TraceStore, "append", Mock(side_effect=OSError("disk full")))
    elif failure.endswith("tensors"):
        cfg.agent.config.capture_hidden_states = True
        if failure != "missing_tensors":
            tensors_path = trace_path.with_suffix(".npz")
            if failure == "empty_tensors":
                np.savez(tensors_path)
            elif failure == "zero_size_tensors":
                np.savez(tensors_path, steps=np.empty((0, 2)))
            elif failure == "string_tensors":
                np.savez(tensors_path, error=np.array("capture failed"))
            elif failure == "nan_tensors":
                np.savez(tensors_path, steps=np.full((2, 3), np.nan))
            elif failure == "inf_tensors":
                np.savez(tensors_path, steps=np.full((2, 3), np.inf))
            elif failure == "integer_tensors":
                np.savez(tensors_path, steps=np.ones((2, 3), dtype=int))
            elif failure == "scalar_tensors":
                np.savez(tensors_path, error=np.array(1.0))
            else:
                tensors_path.write_text("invalid npz")
            result["output_files"]["hidden_states"] = tensors_path
    elif failure == "zero_tasks":
        tasks.clear()
    run = AsyncMock(return_value=result)
    if failure == "adapter_error":
        run.side_effect = RuntimeError("generation failed")
    monkeypatch.setattr(MockAdapter, "run", run)
    summary = Mock(return_value={"total_traces": 99})
    monkeypatch.setattr(TraceStore, "summary", summary)
    with pytest.raises(SystemExit) as exc:
        await run_experiment(cfg)
    assert exc.value.code == 1
    summary.assert_not_called()
    assert run.await_count == len(tasks)
    assert f"0 persisted, {len(tasks)} failed" in caplog.text
    assert len(TraceStore(trace_path.parent / "traces", mode="r")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("with_tensors", [False, True])
async def test_trace_extraction_counts_only_new_persisted_traces(with_tensors, trace_experiment, monkeypatch):
    from unittest.mock import Mock
    import numpy as np
    from manyagents.schemas.reasoning import TraceStore

    cfg, tasks, trace_path = trace_experiment
    cfg.agent.config.capture_hidden_states = with_tensors
    output_files = {"trace": trace_path}
    if with_tensors:
        tensors_path = trace_path.with_suffix(".npz")
        np.savez(tensors_path, steps=np.ones((2, 3)))
        output_files["hidden_states"] = tensors_path
    run = AsyncMock(side_effect=[
        {"success": True, "summary": "Done", "output_files": output_files},
        {"success": False, "summary": "Simulated failure", "output_files": {}},
    ])
    monkeypatch.setattr(MockAdapter, "run", run)
    summary = Mock(return_value={"total_traces": 99})
    monkeypatch.setattr(TraceStore, "summary", summary)
    result = await run_experiment(cfg)
    assert result["summary"]["total_traces"] == 1
    assert result["summary"]["traces_failed"] == 1
    assert result["summary"]["with_tensors"] == int(with_tensors)
    summary.assert_not_called()
    assert run.await_count == len(tasks)
    stored = TraceStore(trace_path.parent / "traces", mode="r")
    assert len(stored) == 2
    if with_tensors:
        np.testing.assert_array_equal(stored.load_tensors("new_trace")["steps"], np.ones((2, 3)))


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [
    {"success": "False"}, {"success": 1}, {"summary": None}, {"output_files": []},
])
async def test_trace_extraction_validates_consumed_result(invalid, trace_experiment, monkeypatch, caplog):
    from manyagents.schemas.reasoning import TraceStore

    cfg, tasks, trace_path = trace_experiment
    result = {"success": True, "summary": "Done", "output_files": {"trace": trace_path}}
    result.update(invalid)
    monkeypatch.setattr(MockAdapter, "run", AsyncMock(return_value=result))
    with pytest.raises(SystemExit) as exc:
        await run_experiment(cfg)
    assert exc.value.code == 1
    assert next(iter(invalid)) in caplog.text
    assert len(TraceStore(trace_path.parent / "traces", mode="r")) == 1


def test_jaccard_pair_ids_do_not_collide():
    from itertools import combinations
    from manyagents.metrics.llm import compute_pairwise_jaccard

    methods = {"a": {"leiden"}, "b_vs_c": {"leiden"}, "a_vs_b": {"pca"}, "c": {"umap"}}
    result = compute_pairwise_jaccard(methods)
    assert len(result["pairwise"]) == 6
    assert result["mean"] == pytest.approx(1 / 6)
    assert result["min"] == 0.0
    assert result["max"] == 1.0
    saved = json.loads(json.dumps(result))
    assert {tuple(json.loads(key)) for key in saved["pairwise"]} == set(combinations(methods, 2))


@pytest.mark.parametrize("n_steps,capture", [(0, True), (1, True), (1, False), (2, True)])
async def test_geometry_requires_two_steps_before_persistence(
    n_steps, capture, trace_experiment, monkeypatch, caplog,
):
    import numpy as np
    from manyagents.schemas.reasoning import TraceStore, ReasoningTrace, ReasoningStep

    cfg, tasks, trace_path = trace_experiment
    tasks[:] = tasks[:1]
    cfg.agent.config.capture_hidden_states = capture
    trace = ReasoningTrace(trace_id="short", steps=[
        ReasoningStep(index=i, text="<think>unfinished", layers_captured=[1])
        for i in range(n_steps)
    ])
    trace_path.write_text(trace.to_json())
    tensors_path = trace_path.with_suffix(".npz")
    np.savez(tensors_path, pooled_steps=np.ones((max(n_steps, 1), 1, 3)))
    monkeypatch.setattr(MockAdapter, "run", AsyncMock(return_value={
        "success": True, "summary": "Done",
        "output_files": {"trace": trace_path, "hidden_states": tensors_path},
    }))
    if n_steps < 2:
        with pytest.raises(SystemExit) as exc:
            await run_experiment(cfg)
        assert exc.value.code == 1
        assert "at least two steps" in caplog.text
        assert len(TraceStore(trace_path.parent / "traces", mode="r")) == 1
        assert not (trace_path.parent / "traces/tensors/short.npz").exists()
    else:
        result = await run_experiment(cfg)
        assert result["summary"]["total_traces"] == 1
        assert result["summary"]["unjudged"] == 1
        assert result["summary"]["success"] == result["summary"]["failure"] == 0
