"""Metrics computation for pipeline invariance experiment."""

from typing import Dict, List, Set, Tuple


def compute_jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    Compute Jaccard similarity between two sets.

    J(A, B) = |A ∩ B| / |A ∪ B|

    Returns 1.0 if both sets are empty (perfect agreement on nothing).
    """
    if not set1 and not set2:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def compute_pairwise_jaccard(method_sets: Dict[str, Set[str]]) -> Dict[str, float]:
    """
    Compute pairwise Jaccard similarity between all method sets.

    Args:
        method_sets: Dict mapping prompt_id -> set of methods

    Returns:
        Dict with 'mean', 'min', 'max', and 'pairwise' (detailed) values
    """
    prompt_ids = list(method_sets.keys())
    n = len(prompt_ids)

    if n < 2:
        return {'mean': 1.0, 'min': 1.0, 'max': 1.0, 'pairwise': {}}

    pairwise = {}
    similarities = []

    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = prompt_ids[i], prompt_ids[j]
            sim = compute_jaccard_similarity(method_sets[p1], method_sets[p2])
            pairwise[f"{p1}_vs_{p2}"] = sim
            similarities.append(sim)

    return {
        'mean': sum(similarities) / len(similarities),
        'min': min(similarities),
        'max': max(similarities),
        'pairwise': pairwise
    }


def compute_ground_truth_match(
    extracted_methods: List[str],
    ground_truth_methods: List[str],
    failure_indicators: List[str] = None
) -> Tuple[bool, Dict[str, any]]:
    """
    Check if extracted methods match ground truth criteria.

    Args:
        extracted_methods: Methods extracted from LLM response
        ground_truth_methods: Expected methods for this geometry type
        failure_indicators: Methods that indicate wrong approach (optional)

    Returns:
        Tuple of (is_match, details_dict)
    """
    extracted_set = set(m.lower() for m in extracted_methods)
    ground_truth_set = set(m.lower() for m in ground_truth_methods)
    failure_set = set(m.lower() for m in (failure_indicators or []))

    matches = extracted_set & ground_truth_set
    failures = extracted_set & failure_set

    # Success: at least one ground truth match and failure not primary
    has_correct = len(matches) > 0
    has_failure_as_primary = len(failures) > 0 and len(matches) == 0

    is_match = has_correct and not has_failure_as_primary

    return is_match, {
        'ground_truth_matches': sorted(list(matches)),
        'failure_matches': sorted(list(failures)),
        'match_count': len(matches),
        'total_ground_truth': len(ground_truth_set),
        'match_ratio': len(matches) / len(ground_truth_set) if ground_truth_set else 0
    }


def compute_system_metrics(
    system_results: Dict[str, Dict[str, any]],
    scenarios: Dict[str, Dict[str, any]]
) -> Dict[str, float]:
    """
    Compute aggregate metrics for a single AI system across all scenarios.

    Args:
        system_results: Dict mapping scenario_id -> result for this system
        scenarios: Dict mapping scenario_id -> scenario config with ground_truth

    Returns:
        Dict with aggregate metrics for this system
    """
    method_sets = {}
    ground_truth_matches = []
    clustering_for_all = []

    for scenario_id, result in system_results.items():
        if not result.get('success', False):
            continue

        methods = result.get('extracted_methods', [])
        method_sets[scenario_id] = set(m.lower() for m in methods)

        # Check ground truth match
        scenario = scenarios.get(scenario_id, {})
        gt_methods = scenario.get('ground_truth_methods', [])
        failure_ind = scenario.get('failure_indicators', [])

        is_match, _ = compute_ground_truth_match(methods, gt_methods, failure_ind)
        ground_truth_matches.append(1.0 if is_match else 0.0)

        # Check if clustering recommended regardless of scenario
        recommends_clustering = result.get('mentions_clustering', False)
        clustering_for_all.append(1.0 if recommends_clustering else 0.0)

    # Compute Jaccard across prompts (measures invariance)
    jaccard_stats = compute_pairwise_jaccard(method_sets)

    return {
        'jaccard_similarity_across_prompts': jaccard_stats['mean'],
        'jaccard_min': jaccard_stats['min'],
        'jaccard_max': jaccard_stats['max'],
        'ground_truth_match_rate': sum(ground_truth_matches) / len(ground_truth_matches) if ground_truth_matches else 0.0,
        'clustering_for_all_rate': sum(clustering_for_all) / len(clustering_for_all) if clustering_for_all else 0.0,
        'scenarios_evaluated': len(system_results)
    }


def generate_summary_table(
    experiment_results: Dict[str, any],
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
        for system, m in metrics.items():
            lines.append(
                f"| {system} | "
                f"{m.get('jaccard_similarity_across_prompts', 0):.2f} | "
                f"{m.get('ground_truth_match_rate', 0):.1%} | "
                f"{m.get('clustering_for_all_rate', 0):.1%} |"
            )
        return '\n'.join(lines)

    elif format == 'latex':
        lines = [
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"System & Jaccard & GT Match & Cluster-All \\",
            r"\midrule"
        ]
        for system, m in metrics.items():
            lines.append(
                f"{system} & "
                f"{m.get('jaccard_similarity_across_prompts', 0):.2f} & "
                f"{m.get('ground_truth_match_rate', 0)*100:.0f}\\% & "
                f"{m.get('clustering_for_all_rate', 0)*100:.0f}\\% \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        return '\n'.join(lines)

    return str(metrics)
