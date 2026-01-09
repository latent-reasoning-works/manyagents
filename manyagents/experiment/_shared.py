"""Shared utilities for experiment runners."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .extractor import extract_methods, check_ground_truth_match
from .metrics import generate_summary_table

log = logging.getLogger(__name__)


def extract_raw_response(output_files: Dict[str, Any]) -> Optional[str]:
    """Extract raw response text from adapter output files."""
    if 'raw_response' not in output_files:
        return None
    response_path = output_files['raw_response']
    if isinstance(response_path, Path):
        return response_path.read_text()
    return str(response_path)


def build_success_result(raw_response: Optional[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Build a success result dict from raw response."""
    extraction = extract_methods(raw_response or '') if raw_response else {}
    return {
        'success': True,
        'raw_response': raw_response,
        'extracted_methods': extraction.get('extracted_methods', []),
        'mentions_clustering': extraction.get('mentions_clustering', False),
        'mentions_trajectory': extraction.get('mentions_trajectory', False),
        'mentions_data_inspection': extraction.get('mentions_data_inspection', False),
        'metadata': metadata
    }


def build_error_result(error: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build an error result dict."""
    result = {'success': False, 'error': error, 'raw_response': None}
    if metadata:
        result['metadata'] = metadata
    return result


def add_ground_truth_matching(
    result: Dict[str, Any],
    scenario: Dict[str, Any]
) -> None:
    """Add ground truth matching info to a result dict (mutates in place)."""
    if not result.get('success'):
        return
    is_match, match_details = check_ground_truth_match(
        result.get('extracted_methods', []),
        scenario.get('ground_truth_methods', []),
        scenario.get('failure_indicators', [])
    )
    result['matches_ground_truth'] = is_match
    result['ground_truth_details'] = match_details


def save_experiment_results(
    experiment_results: Dict[str, Any],
    output_dir: Path,
    experiment_id: str,
    include_interpretation: bool = True
) -> None:
    """Save experiment results and summary to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results as JSON
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(experiment_results, f, indent=2, default=str)
    log.info(f"Results saved to {results_path}")

    # Generate and save markdown summary
    summary_md = generate_summary_table(experiment_results, format='markdown')
    summary_path = output_dir / "summary.md"

    with open(summary_path, 'w') as f:
        name = experiment_results.get('config', {}).get('name', 'Pipeline Invariance Experiment')
        f.write(f"# {name}\n\n")
        f.write(f"**Experiment ID:** {experiment_id}\n")
        f.write(f"**Timestamp:** {experiment_results['timestamp']}\n\n")
        f.write("## Results\n\n")
        f.write(summary_md)
        if include_interpretation:
            f.write("\n\n## Interpretation\n\n")
            f.write("- **Jaccard (Invariance):** Higher = same recommendations across prompts (BAD)\n")
            f.write("- **Ground Truth Match:** Higher = geometry-aware recommendations (GOOD)\n")
            f.write("- **Clustering-for-All:** Higher = always recommends clustering (BAD)\n")

    log.info(f"Summary saved to {summary_path}")


def print_metrics_summary(metrics: Dict[str, Dict[str, float]]) -> None:
    """Print metrics summary to console."""
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)
    for agent_name, m in metrics.items():
        print(f"\n{agent_name}:")
        print(f"  Jaccard Similarity: {m.get('jaccard_similarity_across_prompts', 0):.2f}")
        print(f"  Ground Truth Match: {m.get('ground_truth_match_rate', 0):.1%}")
        print(f"  Clustering-for-All: {m.get('clustering_for_all_rate', 0):.1%}")


def generate_experiment_id(name: str = "invariance") -> str:
    """Generate a unique experiment ID."""
    return f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
