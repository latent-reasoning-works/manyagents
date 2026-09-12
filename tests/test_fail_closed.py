"""Missing measurement criteria must never look like successful measurements."""
import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from manyagents.adapters.manylatents_adapter import ManyLatentsAdapter
from manyagents.experiment import _add_ground_truth_matching
from manyagents.metrics.extractor import check_ground_truth_match
from manyagents.metrics.llm import compute_system_metrics
from manyagents.utils.scoring import score_and_rank
from manyagents.utils.validation import validate_and_filter


async def test_missing_scorer_is_an_error():
    with pytest.raises(ValueError, match="scoring_function is required"):
        await score_and_rank(pd.DataFrame({"id": [1, 2]}), "")


async def test_random_scoring_requires_explicit_function_and_labels_rows():
    data = pd.DataFrame({"id": [1, 2, 3]})
    result = await score_and_rank(
        data, "manyagents.utils.scoring:random_scores", {"seed": 7},
        threshold=0.5, max_results=2,
    )
    assert result["score_simulated"].all()
    assert len(result) == 2
    assert result["score"].is_monotonic_decreasing
    assert "score" not in data


@pytest.mark.parametrize("mode", ["strict", "flag"])
async def test_missing_validator_is_an_error(mode):
    with pytest.raises(ValueError, match="filter_function is required"):
        await validate_and_filter(pd.DataFrame({"id": [1]}), "", filter_mode=mode)


@pytest.mark.parametrize("mode", ["strict", "flag"])
async def test_validator_without_flags_is_an_error(monkeypatch, mode):
    module = ModuleType("fixture_validator")
    module.validate = lambda data: data.copy()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(ValueError, match="flag column"):
        await validate_and_filter(
            pd.DataFrame({"id": [1]}), "fixture_validator:validate", filter_mode=mode,
        )


@pytest.mark.parametrize("criterion", [None, []])
def test_missing_ground_truth_is_unavailable(criterion):
    prompt = {} if criterion is None else {"ground_truth_methods": criterion}
    result = {"success": True, "extracted_methods": ["pca"]}
    match, details = check_ground_truth_match(["pca"], criterion, [])
    assert match is None
    assert details["match_ratio"] is None
    _add_ground_truth_matching(result, prompt)
    assert result["matches_ground_truth"] is None
    metrics = compute_system_metrics({"p": result}, {"p": prompt})
    assert metrics["ground_truth_match_rate"] is None
    assert metrics["prompts_evaluated"] == 1
    assert metrics["prompts_failed"] == 0  # Generation succeeded; criterion is absent.


def test_partial_ground_truth_does_not_silently_change_denominator():
    results = {p: {"success": True, "extracted_methods": ["pca"]} for p in ["a", "b"]}
    metrics = compute_system_metrics(results, {"a": {"ground_truth_methods": ["pca"]}})
    assert metrics["ground_truth_match_rate"] is None


@pytest.mark.parametrize("metric", [{"status": "failed"}, {"details": [1, 2]}, {"status": "failed", "scalar": 2}])
async def test_structured_metric_without_measurement_is_not_a_number(monkeypatch, metric):
    module = ModuleType("manylatents.api")
    module.run = lambda **kwargs: {
        "embeddings": np.ones((3, 2)), "scores": {"metric": metric}, "metadata": {},
    }
    monkeypatch.setitem(sys.modules, module.__name__, module)
    result = await ManyLatentsAdapter().run({"algorithm": "pca", "data": "swissroll"}, {})
    assert result["success"], result["summary"]
    assert "metric=n/a" in result["summary"]
    assert result["output_files"]["scores"]["metric"] == metric
