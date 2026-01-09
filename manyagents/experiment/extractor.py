"""Extract method recommendations from LLM responses."""

import re
from typing import Dict, List, Set, Tuple

# Comprehensive method categories with aliases
METHOD_CATEGORIES = {
    # Clustering methods
    "clustering": {
        "leiden", "louvain", "kmeans", "k-means", "hdbscan", "dbscan",
        "spectral_clustering", "agglomerative", "hierarchical_clustering",
        "phenograph", "scanpy_cluster", "seurat_cluster"
    },

    # Trajectory/pseudotime methods
    "trajectory": {
        "monocle", "monocle2", "monocle3", "slingshot", "paga", "velocyto",
        "scvelo", "rna_velocity", "cellrank", "wishbone", "dpt",
        "diffusion_pseudotime", "palantir", "tempora", "tradeseq"
    },

    # Dimensionality reduction
    "dim_reduction": {
        "pca", "umap", "tsne", "t-sne", "phate", "diffusion_map",
        "force_atlas", "spring", "mds", "isomap", "lle"
    },

    # Differential expression
    "differential_expression": {
        "deseq2", "edger", "wilcoxon", "t_test", "mast", "scde",
        "limma", "rank_genes_groups", "findmarkers"
    },

    # Cell cycle methods
    "cell_cycle": {
        "cyclum", "tricycle", "ccremover", "scran_cyclone", "seurat_cc",
        "cell_cycle_scoring", "recat", "cyclingpheno"
    },

    # Spatial methods
    "spatial": {
        "squidpy", "stlearn", "giotto", "spatialexperiment", "seurat_spatial",
        "tangram", "cell2location", "stereoscope", "rctd"
    },

    # Integration methods
    "integration": {
        "harmony", "scvi", "scanorama", "bbknn", "mnn", "combat",
        "liger", "seurat_integration", "scgen"
    },

    # Annotation methods
    "annotation": {
        "celltypist", "singlecellnet", "sctype", "garnett", "cellassign",
        "scmap", "scpred"
    }
}

# Flatten all methods for quick lookup
ALL_METHODS: Set[str] = set()
for category_methods in METHOD_CATEGORIES.values():
    ALL_METHODS.update(category_methods)


def normalize_method(method: str) -> str:
    """Normalize method name for comparison."""
    method = method.lower().strip()
    method = re.sub(r'[_\-\s]+', '_', method)
    method = re.sub(r'[^a-z0-9_]', '', method)
    return method


def extract_methods(text: str, include_categories: bool = False) -> Dict[str, any]:
    """
    Extract method recommendations from LLM response text.

    Args:
        text: Raw LLM response text
        include_categories: If True, return methods grouped by category

    Returns:
        Dictionary with:
            - extracted_methods: List of identified methods
            - mentions_clustering: Whether clustering is mentioned
            - mentions_trajectory: Whether trajectory/pseudotime is mentioned
            - mentions_data_inspection: Whether data inspection is recommended
            - raw_matches: All pattern matches found
    """
    text_lower = text.lower()

    extracted = set()
    category_matches = {cat: set() for cat in METHOD_CATEGORIES}

    # Pattern 1: Direct method name mentions
    for category, methods in METHOD_CATEGORIES.items():
        for method in methods:
            # Create flexible pattern that matches variations
            pattern_base = method.replace('_', r'[\s_\-]?')
            patterns = [
                rf'\b{pattern_base}\b',
                rf'\b{method}\b',
            ]
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    extracted.add(method)
                    category_matches[category].add(method)
                    break

    # Pattern 2: Look for method-indicating phrases
    trajectory_phrases = [
        r'pseudotime\s+analysis', r'trajectory\s+inference',
        r'differentiation\s+trajectory', r'developmental\s+trajectory',
        r'lineage\s+tracing', r'cell\s+fate', r'temporal\s+ordering',
        r'ordering\s+cells', r'branching\s+analysis'
    ]
    for phrase in trajectory_phrases:
        if re.search(phrase, text_lower):
            extracted.add('trajectory_inference')
            category_matches['trajectory'].add('trajectory_inference')

    clustering_phrases = [
        r'cluster\s+(?:the\s+)?cells', r'clustering\s+analysis',
        r'identify\s+(?:cell\s+)?(?:types|populations|clusters)',
        r'find\s+clusters', r'unsupervised\s+clustering'
    ]
    for phrase in clustering_phrases:
        if re.search(phrase, text_lower):
            extracted.add('clustering')
            category_matches['clustering'].add('clustering')

    # Pattern 3: Check for data inspection recommendations
    inspection_phrases = [
        r'inspect\s+(?:the\s+)?data', r'exploratory\s+analysis',
        r'examine\s+(?:the\s+)?(?:data|distribution)',
        r'look\s+at\s+(?:the\s+)?data', r'understand\s+(?:the\s+)?structure',
        r'visualize\s+first', r'quality\s+control', r'qc\s+metrics'
    ]
    mentions_inspection = any(re.search(p, text_lower) for p in inspection_phrases)

    # Determine primary method category
    has_clustering = bool(category_matches['clustering'])
    has_trajectory = bool(category_matches['trajectory'])
    has_cell_cycle = bool(category_matches['cell_cycle'])
    has_spatial = bool(category_matches['spatial'])

    result = {
        'extracted_methods': sorted(list(extracted)),
        'mentions_clustering': has_clustering,
        'mentions_trajectory': has_trajectory,
        'mentions_cell_cycle': has_cell_cycle,
        'mentions_spatial': has_spatial,
        'mentions_data_inspection': mentions_inspection,
    }

    if include_categories:
        result['by_category'] = {k: sorted(list(v)) for k, v in category_matches.items() if v}

    return result


def check_ground_truth_match(
    extracted: List[str],
    ground_truth: List[str],
    failure_indicators: List[str]
) -> Tuple[bool, Dict[str, any]]:
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

    # Check for ground truth matches
    matches = extracted_set & ground_truth_set

    # Check for failure indicators
    failures = extracted_set & failure_set

    # Determine overall match
    has_correct = len(matches) > 0
    has_failure_as_primary = len(failures) > 0 and len(matches) == 0

    return (has_correct and not has_failure_as_primary), {
        'ground_truth_matches': sorted(list(matches)),
        'failure_matches': sorted(list(failures)),
        'total_extracted': len(extracted_set),
        'match_ratio': len(matches) / len(ground_truth_set) if ground_truth_set else 0
    }
