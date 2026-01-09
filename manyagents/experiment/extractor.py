"""Extract method recommendations from LLM responses."""

import re
from typing import Any, Dict, List, Set, Tuple

# Comprehensive method categories with aliases
METHOD_CATEGORIES: Dict[str, Set[str]] = {
    "clustering": {
        "leiden", "louvain", "kmeans", "k-means", "hdbscan", "dbscan",
        "spectral_clustering", "agglomerative", "hierarchical_clustering",
        "phenograph", "scanpy_cluster", "seurat_cluster"
    },
    "trajectory": {
        "monocle", "monocle2", "monocle3", "slingshot", "paga", "velocyto",
        "scvelo", "rna_velocity", "cellrank", "wishbone", "dpt",
        "diffusion_pseudotime", "palantir", "tempora", "tradeseq"
    },
    "dim_reduction": {
        "pca", "umap", "tsne", "t-sne", "phate", "diffusion_map",
        "force_atlas", "spring", "mds", "isomap", "lle"
    },
    "differential_expression": {
        "deseq2", "edger", "wilcoxon", "t_test", "mast", "scde",
        "limma", "rank_genes_groups", "findmarkers"
    },
    "cell_cycle": {
        "cyclum", "tricycle", "ccremover", "scran_cyclone", "seurat_cc",
        "cell_cycle_scoring", "recat", "cyclingpheno"
    },
    "spatial": {
        "squidpy", "stlearn", "giotto", "spatialexperiment", "seurat_spatial",
        "tangram", "cell2location", "stereoscope", "rctd"
    },
    "integration": {
        "harmony", "scvi", "scanorama", "bbknn", "mnn", "combat",
        "liger", "seurat_integration", "scgen"
    },
    "annotation": {
        "celltypist", "singlecellnet", "sctype", "garnett", "cellassign",
        "scmap", "scpred"
    }
}

# Flatten all methods for quick lookup
ALL_METHODS: Set[str] = {m for methods in METHOD_CATEGORIES.values() for m in methods}

# Phrase patterns for detecting implicit method mentions
TRAJECTORY_PHRASES = [
    r'pseudotime\s+analysis', r'trajectory\s+inference',
    r'differentiation\s+trajectory', r'developmental\s+trajectory',
    r'lineage\s+tracing', r'cell\s+fate', r'temporal\s+ordering',
    r'ordering\s+cells', r'branching\s+analysis'
]

CLUSTERING_PHRASES = [
    r'cluster\s+(?:the\s+)?cells', r'clustering\s+analysis',
    r'identify\s+(?:cell\s+)?(?:types|populations|clusters)',
    r'find\s+clusters', r'unsupervised\s+clustering'
]

INSPECTION_PHRASES = [
    r'inspect\s+(?:the\s+)?data', r'exploratory\s+analysis',
    r'examine\s+(?:the\s+)?(?:data|distribution)',
    r'look\s+at\s+(?:the\s+)?data', r'understand\s+(?:the\s+)?structure',
    r'visualize\s+first', r'quality\s+control', r'qc\s+metrics'
]


def normalize_method(method: str) -> str:
    """Normalize method name for comparison."""
    normalized = method.lower().strip()
    normalized = re.sub(r'[_\-\s]+', '_', normalized)
    return re.sub(r'[^a-z0-9_]', '', normalized)


def _match_any_pattern(text: str, patterns: List[str]) -> bool:
    """Check if any pattern matches in text."""
    return any(re.search(p, text) for p in patterns)


def _find_method_matches(text_lower: str) -> Dict[str, Set[str]]:
    """Find all method matches organized by category."""
    category_matches = {cat: set() for cat in METHOD_CATEGORIES}

    for category, methods in METHOD_CATEGORIES.items():
        for method in methods:
            # Create flexible pattern that matches variations
            pattern_base = method.replace('_', r'[\s_\-]?')
            if re.search(rf'\b{pattern_base}\b', text_lower) or re.search(rf'\b{method}\b', text_lower):
                category_matches[category].add(method)

    # Add implicit trajectory/clustering mentions from phrases
    if _match_any_pattern(text_lower, TRAJECTORY_PHRASES):
        category_matches['trajectory'].add('trajectory_inference')
    if _match_any_pattern(text_lower, CLUSTERING_PHRASES):
        category_matches['clustering'].add('clustering')

    return category_matches


def extract_methods(text: str, include_categories: bool = False) -> Dict[str, Any]:
    """
    Extract method recommendations from LLM response text.

    Args:
        text: Raw LLM response text
        include_categories: If True, return methods grouped by category

    Returns:
        Dictionary with extracted_methods, mentions_* flags, and optionally by_category
    """
    text_lower = text.lower()
    category_matches = _find_method_matches(text_lower)

    # Flatten all extracted methods
    extracted = {m for methods in category_matches.values() for m in methods}

    result = {
        'extracted_methods': sorted(extracted),
        'mentions_clustering': bool(category_matches['clustering']),
        'mentions_trajectory': bool(category_matches['trajectory']),
        'mentions_cell_cycle': bool(category_matches['cell_cycle']),
        'mentions_spatial': bool(category_matches['spatial']),
        'mentions_data_inspection': _match_any_pattern(text_lower, INSPECTION_PHRASES),
    }

    if include_categories:
        result['by_category'] = {k: sorted(v) for k, v in category_matches.items() if v}

    return result


def check_ground_truth_match(
    extracted: List[str],
    ground_truth: List[str],
    failure_indicators: List[str]
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if extracted methods match ground truth criteria.

    Args:
        extracted: List of extracted methods
        ground_truth: List of expected methods (any match is success)
        failure_indicators: Methods that indicate wrong approach

    Returns:
        Tuple of (matches, details dict)
    """
    extracted_set = {normalize_method(m) for m in extracted}
    ground_truth_set = {normalize_method(m) for m in ground_truth}
    failure_set = {normalize_method(m) for m in failure_indicators}

    matches = extracted_set & ground_truth_set
    failures = extracted_set & failure_set

    has_correct = bool(matches)
    has_failure_as_primary = failures and not matches

    return (has_correct and not has_failure_as_primary), {
        'ground_truth_matches': sorted(matches),
        'failure_matches': sorted(failures),
        'total_extracted': len(extracted_set),
        'match_ratio': len(matches) / len(ground_truth_set) if ground_truth_set else 0
    }
