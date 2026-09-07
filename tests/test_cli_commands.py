"""Execute documented commands through Hydra, with real mock responses and scoring."""

import json
from pathlib import Path
import shlex
import sys

import pytest

from manyagents import experiment
from manyagents.adapters import ADAPTER_REGISTRY, MockAdapter
from manyagents.main import main


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cli(monkeypatch, tmp_path):
    """Keep Hydra, dispatch, scoring and persistence real; isolate all outputs."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HYDRA_FULL_ERROR", "1")
    completed = []
    calls = []
    run_experiment = experiment.run_experiment

    class RecordingMock(MockAdapter):
        async def run(self, task_config, input_files):
            calls.append(dict(task_config))
            return await super().run({**task_config, "delay": 0}, input_files)

    for name in ("mock", "claude", "openai", "hf", "local_llm"):
        monkeypatch.setitem(ADAPTER_REGISTRY, name, RecordingMock)

    async def record(cfg):
        result = await run_experiment(cfg)
        saved = json.loads((Path(cfg.output_dir) / "results.json").read_text())
        assert saved == json.loads(json.dumps(result, default=str))
        assert (Path(cfg.output_dir) / "summary.md").is_file()
        completed.append(saved)
        return result

    monkeypatch.setattr(experiment, "run_experiment", record)

    def invoke(command):
        args = shlex.split(command)
        args += ["wandb.enabled=false"]
        if "--multirun" in args:
            args += ["hydra.sweep.dir=sweep"]
        monkeypatch.setattr(sys, "argv", args)
        main()
        return completed, calls

    return invoke


def assert_evaluations(result, agent, prompt_count):
    assert set(result["results"]) == {agent}
    responses = result["results"][agent]
    assert len(responses) == prompt_count
    assert set(responses) == set(result["prompts"])
    for response in responses.values():
        assert response["success"] is True
        assert response["raw_response"].strip()
        assert set(response["extracted_methods"]) == {"leiden", "umap", "pca"}
        assert response["metadata"]["mock"] is True
    metrics = result["metrics"][agent]
    assert metrics["prompts_evaluated"] == prompt_count
    assert metrics["prompts_failed"] == 0
    assert metrics["jaccard_similarity_across_prompts"] == 1.0
    assert metrics["clustering_for_all_rate"] == 1.0


def test_bare_cli_lists_available_experiments(cli, tmp_path):
    available = sorted(path.stem for path in (ROOT / "manyagents/configs/experiment").glob("*.yaml"))
    with pytest.raises(SystemExit) as exc:
        cli("manyagents")
    assert exc.value.code == (
        "No experiment selected. Run: manyagents experiment=<name>. Available: "
        + ", ".join(available)
    )
    assert not list(tmp_path.rglob("results.json"))


@pytest.mark.parametrize("document", ["README.md", "CLAUDE.md"])
def test_documented_sweep_executes_all_jobs(document, cli, tmp_path):
    commands = [
        line.strip() for line in (ROOT / document).read_text().splitlines()
        if line.startswith("manyagents --multirun ")
    ]
    assert len(commands) == 1, "Keep one runnable canonical sweep in each document"
    completed, calls = cli(commands[0])
    assert len(completed) == 3
    assert len(calls) == 12
    for result, agent in zip(completed, ["claude", "openai", "local_llm"], strict=True):
        assert_evaluations(result, agent, 4)
        assert result["metrics"][agent]["ground_truth_match_rate"] == 0.25
    assert len(list((tmp_path / "sweep").glob("*/results.json"))) == 3


@pytest.mark.parametrize("selection,prompt_count,match_rate", [
    ("experiment=test_wandb", 2, 0.5),
    ("experiment=geometric_reasoning 'active_agents=[mock]'", 9, 2 / 3),
])
def test_mock_quickstarts_execute_and_score(selection, prompt_count, match_rate, cli):
    completed, calls = cli(f"manyagents {selection}")
    assert len(completed) == 1
    assert len(calls) == prompt_count
    assert_evaluations(completed[0], "mock", prompt_count)
    assert completed[0]["metrics"]["mock"]["ground_truth_match_rate"] == pytest.approx(match_rate)
