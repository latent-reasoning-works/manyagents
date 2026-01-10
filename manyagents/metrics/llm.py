"""Metrics computation for pipeline invariance experiment."""

from itertools import combinations
from statistics import mean
from typing import Any, Dict, List, Set

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


def compute_pairwise_jaccard(method_sets: Dict[str, Set[str]]) -> Dict[str, float]:
    """
    Compute pairwise Jaccard similarity between all method sets.

    Args:
        method_sets: Dict mapping prompt_id -> set of methods

    Returns:
        Dict with 'mean', 'min', 'max', and 'pairwise' (detailed) values
    """
    if len(method_sets) < 2:
        return {'mean': 1.0, 'min': 1.0, 'max': 1.0, 'pairwise': {}}

    pairwise = {
        f"{p1}_vs_{p2}": compute_jaccard_similarity(method_sets[p1], method_sets[p2])
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
) -> Dict[str, float]:
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
        ground_truth_matches.append(float(is_match))
        clustering_for_all.append(float(result.get('mentions_clustering', False)))

    jaccard_stats = compute_pairwise_jaccard(method_sets)

    return {
        'jaccard_similarity_across_prompts': jaccard_stats['mean'],
        'jaccard_min': jaccard_stats['min'],
        'jaccard_max': jaccard_stats['max'],
        'ground_truth_match_rate': mean(ground_truth_matches) if ground_truth_matches else 0.0,
        'clustering_for_all_rate': mean(clustering_for_all) if clustering_for_all else 0.0,
        'prompts_evaluated': len(system_results)
    }


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
            f"{m.get('jaccard_similarity_across_prompts', 0):.2f} | "
            f"{m.get('ground_truth_match_rate', 0):.1%} | "
            f"{m.get('clustering_for_all_rate', 0):.1%} |"
            for system, m in metrics.items()
        )
        return '\n'.join(lines)

    if format == 'latex':
        lines = [
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"System & Jaccard & GT Match & Cluster-All \\",
            r"\midrule"
        ]
        lines.extend(
            f"{system} & "
            f"{m.get('jaccard_similarity_across_prompts', 0):.2f} & "
            f"{m.get('ground_truth_match_rate', 0)*100:.0f}\\% & "
            f"{m.get('clustering_for_all_rate', 0)*100:.0f}\\% \\\\"
            for system, m in metrics.items()
        )
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        return '\n'.join(lines)

    return str(metrics)
