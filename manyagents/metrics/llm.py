"""Metrics computation for pipeline invariance experiment."""

import json
from itertools import combinations
from statistics import mean
from typing import Any, Dict, Set

from .extractor import check_ground_truth_match


def compute_jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    Compute Jaccard similarity between two sets.

    J(A, B) = |A intersection B| / |A union B|
    Returns 1.0 if both sets are empty (perfect agreement on nothing).
    """
    if not set1 and not set2:
        return 1.0
    union = set1 | set2
    return len(set1 & set2) / len(union) if union else 0.0


def compute_pairwise_jaccard(method_sets: Dict[str, Set[str]]) -> Dict[str, Any] | None:
    """
    Compute pairwise Jaccard similarity between all method sets.

    Args:
        method_sets: Dict mapping prompt_id -> set of methods

    Returns:
        Dict with 'mean', 'min', 'max', and 'pairwise' values, or None without a pair.
        Pairwise keys are JSON-encoded [prompt_id, prompt_id] pairs.
    """
    if len(method_sets) < 2:
        return None

    pairwise = {
        json.dumps([p1, p2]): compute_jaccard_similarity(method_sets[p1], method_sets[p2])
        for p1, p2 in combinations(method_sets.keys(), 2)
    }
    similarities = list(pairwise.values())

    return {
        'mean': mean(similarities),
        'min': min(similarities),
        'max': max(similarities),
        'pairwise': pairwise
    }


def compute_system_metrics(
    system_results: Dict[str, Dict[str, Any]],
    prompts: Dict[str, Dict[str, Any]]
) -> Dict[str, float | None]:
    """
    Compute aggregate metrics for a single AI system across all prompts.

    Args:
        system_results: Dict mapping prompt_id -> result for this system
        prompts: Dict mapping prompt_id -> prompt config with ground_truth

    Returns:
        Dict with aggregate metrics for this system
    """
    method_sets = {}
    ground_truth_matches = []
    clustering_for_all = []

    for prompt_id, result in system_results.items():
        if not result.get('success', False):
            continue

        methods = result.get('extracted_methods', [])
        method_sets[prompt_id] = {m.lower() for m in methods}

        # Check ground truth match
        prompt_config = prompts.get(prompt_id, {})
        is_match, _ = check_ground_truth_match(
            methods,
            prompt_config.get('ground_truth_methods', []),
            prompt_config.get('failure_indicators', [])
        )
        ground_truth_matches.append(float(is_match) if is_match is not None else None)
        clustering_for_all.append(float(result.get('mentions_clustering', False)))

    jaccard_stats = compute_pairwise_jaccard(method_sets)

    return {
        'jaccard_similarity_across_prompts': jaccard_stats['mean'] if jaccard_stats else None,
        'jaccard_min': jaccard_stats['min'] if jaccard_stats else None,
        'jaccard_max': jaccard_stats['max'] if jaccard_stats else None,
        # Do not silently narrow the denominator to prompts with known criteria.
        'ground_truth_match_rate': (
            mean(ground_truth_matches)
            if ground_truth_matches and None not in ground_truth_matches else None
        ),
        'clustering_for_all_rate': mean(clustering_for_all) if clustering_for_all else None,
        'prompts_evaluated': len(method_sets),
        'prompts_failed': len(system_results) - len(method_sets)
    }


def format_metric(value: float | None, format_spec: str | None = '.2f') -> str | float | None:
    """Format unavailable measurements as n/a, or preserve values for numeric sinks."""
    if format_spec is None:
        return value
    return format(value, format_spec) if value is not None else 'n/a'


def generate_summary_table(
    experiment_results: Dict[str, Any],
    format: str = 'markdown'
) -> str:
    """
    Generate a summary table from experiment results.

    Args:
        experiment_results: Full experiment results dict
        format: 'markdown' or 'latex'

    Returns:
        Formatted table string
    """
    metrics = experiment_results.get('metrics', {})

    if format == 'markdown':
        lines = [
            "| System | Jaccard (Invariance) | Ground Truth Match | Clustering-for-All |",
            "|--------|---------------------|-------------------|-------------------|"
        ]
        lines.extend(
            f"| {system} | "
            f"{format_metric(m.get('jaccard_similarity_across_prompts'), '.2f')} | "
            f"{format_metric(m.get('ground_truth_match_rate'), '.1%')} | "
            f"{format_metric(m.get('clustering_for_all_rate'), '.1%')} |"
            for system, m in metrics.items()
        )
        return '\n'.join(lines)

    if format == 'latex':
        def percent(value):
            return format_metric(value, '.0%').replace('%', r'\%')

        lines = [
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"System & Jaccard & GT Match & Cluster-All \\",
            r"\midrule"
        ]
        lines.extend(
            f"{system} & "
            f"{format_metric(m.get('jaccard_similarity_across_prompts'), '.2f')} & "
            f"{percent(m.get('ground_truth_match_rate'))} & "
            f"{percent(m.get('clustering_for_all_rate'))} \\\\"
            for system, m in metrics.items()
        )
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        return '\n'.join(lines)

    return str(metrics)
