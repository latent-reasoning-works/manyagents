"""No-key, no-GPU acceptance scores against the shipped mock fixtures."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from manyagents.experiment import run_experiment


@pytest.mark.asyncio
async def test_mock_fixture_scores(tmp_path, monkeypatch):
    config_dir = Path(__file__).resolve().parents[1] / "manyagents" / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="main", overrides=[
            "experiment=test_wandb", "wandb.enabled=false", f"output_dir={tmp_path}",
        ])
    monkeypatch.chdir(tmp_path)
    result = await run_experiment(cfg)
    prompts = result["results"]["mock"]
    assert set(prompts) == {"test_discrete", "test_trajectory"}
    for name, matches in [("test_discrete", ["leiden"]), ("test_trajectory", [])]:
        prompt = prompts[name]
        assert prompt["success"] is True
        assert set(prompt["extracted_methods"]) == {"leiden", "pca", "umap"}
        assert prompt["matches_ground_truth"] is bool(matches)
        assert prompt["ground_truth_details"]["ground_truth_matches"] == matches
    metrics = result["metrics"]["mock"]
    assert metrics["ground_truth_match_rate"] == 0.5
    assert metrics["clustering_for_all_rate"] == 1.0
    assert metrics["jaccard_similarity_across_prompts"] == 1.0
    assert metrics["prompts_evaluated"] == 2
    assert metrics["prompts_failed"] == 0
