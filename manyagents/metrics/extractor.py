"""Extract method recommendations from LLM responses."""

import re
from typing import Any, Dict, List, Set, Tuple

# Method categories with aliases
METHOD_CATEGORIES: Dict[str, Set[str]] = {
    "clustering": {
        "leiden", "louvain", "kmeans", "k-means", "hdbscan", "dbscan",
        "spectral_clustering", "agglomerative", "hierarchical_clustering",
        "phenograph", "scanpy_cluster", "seurat_cluster",
        "sc.tl.leiden", "sc.tl.louvain",
        "find_clusters", "identify_clusters", "cluster_analysis",
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
        "cell_cycle_scoring", "recat", "cyclingpheno",
        "cell_cycle_regression", "regress_out_cell_cycle", "regress_cell_cycle",
        "periodic", "cyclical_embedding", "circular_embedding",
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

# Phrase patterns for implicit method detection
_PHRASE_PATTERNS: Dict[str, List[str]] = {
    'trajectory': [
        r'pseudotime\s+analysis', r'trajectory\s+inference',
        r'differentiation\s+trajectory', r'developmental\s+trajectory',
        r'lineage\s+tracing', r'cell\s+fate', r'temporal\s+ordering',
        r'ordering\s+cells', r'branching\s+analysis'
    ],
    'clustering': [
        r'cluster\s+(?:the\s+)?cells', r'clustering\s+analysis',
        r'identify\s+(?:cell\s+)?(?:types|populations|clusters)',
        r'find\s+clusters', r'unsupervised\s+clustering',
        r'differential\s+expression\s+between\s+clusters',
        r'marker\s+genes\s+(?:for|per)\s+cluster',
    ],
    'cell_cycle': [
        r'cell\s+cycle\s+(?:scoring|analysis|phase)',
        r'score\s+(?:for\s+)?cell\s+cycle',
        r'g2m\s+(?:score|phase)', r's\s+phase\s+(?:score|genes)',
        r'mitotic', r'proliferat(?:ion|ing)',
    ],
    'inspection': [
        r'inspect\s+(?:the\s+)?data', r'exploratory\s+analysis',
        r'examine\s+(?:the\s+)?(?:data|distribution)',
        r'look\s+at\s+(?:the\s+)?data', r'understand\s+(?:the\s+)?structure',
        r'visualize\s+first', r'quality\s+control', r'qc\s+metrics'
    ],
}


def normalize_method(method: str) -> str:
    """Normalize method name for comparison."""
    normalized = re.sub(r'[_\-\s]+', '_', method.lower().strip())
    return re.sub(r'[^a-z0-9_]', '', normalized)


def _match_any_pattern(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text) for p in patterns)


def _find_method_matches(text_lower: str) -> Dict[str, Set[str]]:
    """Find all method matches organized by category."""
    matches: Dict[str, Set[str]] = {cat: set() for cat in METHOD_CATEGORIES}

    for category, methods in METHOD_CATEGORIES.items():
        for method in methods:
            pattern_base = method.replace('_', r'[\s_\-]?')
            if re.search(rf'\b{pattern_base}\b', text_lower) or re.search(rf'\b{method}\b', text_lower):
                matches[category].add(method)

    # Add implicit mentions from phrase patterns
    _implicit_map = [('trajectory', 'trajectory_inference'), ('clustering', 'clustering'), ('cell_cycle', 'cell_cycle_scoring')]
    for category, implicit_method in _implicit_map:
        if _match_any_pattern(text_lower, _PHRASE_PATTERNS[category]):
            matches[category].add(implicit_method)

    return matches


def extract_methods(text: str, include_categories: bool = False) -> Dict[str, Any]:
    """Extract method recommendations from LLM response text."""
    text_lower = text.lower()
    matches = _find_method_matches(text_lower)

    result = {
        'extracted_methods': sorted(m for methods in matches.values() for m in methods),
        'mentions_clustering': bool(matches['clustering']),
        'mentions_trajectory': bool(matches['trajectory']),
        'mentions_cell_cycle': bool(matches['cell_cycle']),
        'mentions_spatial': bool(matches['spatial']),
        'mentions_data_inspection': _match_any_pattern(text_lower, _PHRASE_PATTERNS['inspection']),
    }

    if include_categories:
        result['by_category'] = {k: sorted(v) for k, v in matches.items() if v}

    return result


def _normalize_set(items: List[str]) -> Set[str]:
    """Normalize a list of methods to a set for comparison."""
    return {normalize_method(m) for m in items}


def check_ground_truth_match(
    extracted: List[str],
    ground_truth: List[str],
    failure_indicators: List[str]
) -> Tuple[bool, Dict[str, Any]]:
    """Check if extracted methods match ground truth criteria."""
    extracted_set = _normalize_set(extracted)
    ground_truth_set = _normalize_set(ground_truth)
    failure_set = _normalize_set(failure_indicators)

    matches = extracted_set & ground_truth_set
    failures = extracted_set & failure_set

    is_success = bool(matches) and not (failures and not matches)

    return is_success, {
        'ground_truth_matches': sorted(matches),
        'failure_matches': sorted(failures),
        'total_extracted': len(extracted_set),
        'match_ratio': len(matches) / len(ground_truth_set) if ground_truth_set else 0
    }


def check_exclusion_violations(
    extracted: Dict[str, Any],
    expected_beta0: int,
    expected_beta1: int,
) -> Dict[str, Any]:
    """Check if recommendations violate topological constraints (beta0/beta1)."""
    clustering_violation = expected_beta0 == 1 and extracted.get('mentions_clustering', False)
    cell_cycle_violation = expected_beta1 == 0 and extracted.get('mentions_cell_cycle', False)

    violations = []
    if clustering_violation:
        violations.append(f"Clustering recommended but data has beta0={expected_beta0} (single component)")
    if cell_cycle_violation:
        violations.append(f"Cell cycle methods recommended but data has beta1={expected_beta1} (no loops)")

    return {
        'clustering_violation': clustering_violation,
        'cell_cycle_violation': cell_cycle_violation,
        'any_violation': clustering_violation or cell_cycle_violation,
        'violation_details': violations,
    }
