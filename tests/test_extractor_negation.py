"""Scoring contracts: reject explicit refusals without rejecting their alternatives."""

import pytest

from manyagents.experiment import _add_ground_truth_matching, _build_success_result
from manyagents.metrics.extractor import check_ground_truth_match, extract_methods
from manyagents.metrics.llm import compute_pairwise_jaccard, compute_system_metrics


@pytest.mark.parametrize("text", [
    "Do not use UMAP.", "Don't recommend UMAP.", "Avoid UMAP.",
    "UMAP is not appropriate.", "UMAP would be wrong.",
    "It would be wrong to use UMAP.", "UMAP should not be used.",
    "Use PHATE instead of UMAP.", "Use PHATE rather than UMAP.",
    "UMAP is unsuitable here.", "I would not recommend UMAP.",
    "Neither UMAP nor Leiden is appropriate.",
    "Avoid UMAP and Leiden.", "UMAP and Leiden are not appropriate.",
    "Do not use **UMAP**.", "UMAP isn't suitable.",
    "UMAP: not appropriate.", "It is not appropriate to use UMAP.",
    "UMAP should be avoided.", "Use PHATE, not UMAP.",
    "Do not use UMAP or PHATE.", "Do not use UMAP, PHATE, or Leiden.",
])
def test_rejected_method_does_not_count(text):
    assert "umap" not in extract_methods(text)["extracted_methods"]


@pytest.mark.parametrize("text", [
    "Avoid Leiden. Use UMAP.", "Do not use Leiden; use UMAP.",
    "Avoid Leiden and use UMAP.", "Use UMAP rather than Leiden.",
    "Use UMAP instead of Leiden.", "Instead of Leiden, use UMAP.",
    "Rather than Leiden, I recommend UMAP.",
    "Leiden is not appropriate, but UMAP is.",
    "Leiden would be wrong. UMAP is appropriate.",
    "Avoid Leiden, UMAP is appropriate.",
    "Not only UMAP but also PHATE can help.",
    "UMAP is not inappropriate.", "UMAP is not wrong.",
    "Do not avoid UMAP.", "UMAP is useful, but Leiden is not appropriate.",
    "Avoid UMAP for clustering. Use UMAP for visualization.",
    "Use UMAP. Avoid UMAP for clustering.",
    "Use UMAP to avoid losing local structure.",
    "Instead of Leiden or Louvain, UMAP is recommended.",
    "Recommend UMAP rather than use Leiden.",
    "Instead of choosing Leiden, use UMAP.",
    "UMAP is useful and Leiden would be wrong.",
])
def test_affirmed_method_survives_rejected_alternative(text):
    extraction = extract_methods(text)
    assert "umap" in extraction["extracted_methods"]
    assert not extraction["mentions_clustering"]
    assert check_ground_truth_match(extraction["extracted_methods"], ["umap"], ["leiden"])[0] is True


@pytest.mark.parametrize("phrase,flag", [
    ("trajectory inference", "mentions_trajectory"),
    ("cluster the cells", "mentions_clustering"),
    ("cell cycle scoring", "mentions_cell_cycle"),
])
def test_implicit_phrases_respect_negation(phrase, flag):
    assert extract_methods(f"Use {phrase}.")[flag]
    assert not extract_methods(f"Do not use {phrase}.")[flag]


def test_explicit_failure_blocks_match_but_preserves_overlap_diagnostics():
    match, details = check_ground_truth_match(["UMAP", "Leiden"], ["umap", "phate"], ["leiden"])
    assert match is False
    assert details["ground_truth_matches"] == ["umap"]
    assert details["failure_matches"] == ["leiden"]
    assert details["match_ratio"] == 0.5  # vocabulary coverage, not a pass probability
    assert check_ground_truth_match(["leiden"], [], ["leiden"])[0] is None


@pytest.mark.parametrize("text,expected", [
    ("Do not use Leiden. Avoid UMAP. There is no appropriate method here.", False),
    ("Use Leiden and UMAP.", False),
    ("Avoid Leiden. Use UMAP.", True),
    ("UMAP would be wrong; use PHATE instead.", True),
])
def test_per_prompt_and_aggregate_scoring_agree(text, expected):
    prompt = {"ground_truth_methods": ["umap", "phate"], "failure_indicators": ["leiden"]}
    result = _build_success_result(text, {})
    _add_ground_truth_matching(result, prompt)
    assert result["success"] is True  # a refusal is a successful call, but no match
    assert result["matches_ground_truth"] is expected
    metrics = compute_system_metrics({"p": result}, {"p": prompt})
    assert metrics["ground_truth_match_rate"] == float(expected)
    assert metrics["prompts_evaluated"] == 1
    assert metrics["prompts_failed"] == 0


def test_jaccard_includes_same_geometry_pairs():
    methods = {f"{geometry}_{condition}": {geometry}
               for geometry in ("leiden", "slingshot", "phate") for condition in "ABC"}
    stats = compute_pairwise_jaccard(methods)
    assert len(stats["pairwise"]) == 36
    assert stats["mean"] == 0.25
    # Within-geometry inconsistency can lower the score; lower is not an objective.
    inconsistent = {key: {key} for key in methods}
    assert compute_pairwise_jaccard(inconsistent)["mean"] == 0.0
