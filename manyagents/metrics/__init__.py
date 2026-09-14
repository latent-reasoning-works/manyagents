"""LLM evaluation metrics for manyAgents.

Provides metrics for evaluating LLM reasoning about data geometry.
Available to downstream consumers for G-vector aggregation.
"""

from .llm import compute_jaccard_similarity, compute_system_metrics, generate_summary_table
from .extractor import extract_methods, check_ground_truth_match, METHOD_CATEGORIES

__all__ = [
    "compute_jaccard_similarity",
    "compute_system_metrics",
    "generate_summary_table",
    "extract_methods",
    "check_ground_truth_match",
    "METHOD_CATEGORIES",
]
