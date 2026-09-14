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


# Deliberately local English rules, not a semantic recommendation judge. Keep
# sentence/contrast boundaries and affirmative cues so rejecting an alternative
# does not suppress a recommendation elsewhere (including another occurrence).
_CLAUSE_BOUNDARY = re.compile(
    r"[;!?\n]|(?<!\w)\.|\.(?!\w)|\b(?:but|however|yet|whereas)\b|\binstead\b(?!\s+of)"
)
_REJECTION = re.compile(
    r"\b(?:do\s+not|don't|should\s+not|shouldn't|would\s+not|wouldn't|"
    r"cannot|can't|must\s+not|mustn't)\s+(?:\w+\s+){0,2}?(?:use|recommend|choose|apply)\b"
    r"|\b(?:avoid|reject|neither)\b"
    r"|\b(?:instead\s+of|rather\s+than)\b(?:\s+(?:use|recommend|choose|apply|prefer|try)\b)?"
    r"|\b(?:wrong|inappropriate|unsuitable|not\s+appropriate|not\s+suitable)\s+to\s+(?:use|recommend|choose|apply)\b"
    r"|\bnot\s+(?:use|recommend|choose|apply)\b|\bnot\s*$"
)
_AFFIRMATION = re.compile(r"\b(?:use|recommend|choose|apply|prefer|try)\b")
_NEGATIVE_PREDICATE = (
    r"(?:(?:(?:is|are|would\s+be|will\s+be)\s+)?"
    r"(?:not\s+(?:appropriate|suitable|recommended|useful)|wrong|inappropriate|unsuitable)"
    r"|(?:isn't|aren't)\s+(?:appropriate|suitable|recommended|useful)"
    r"|(?:should\s+not|shouldn't|must\s+not|mustn't)\s+be\s+(?:used|recommended)"
    r"|(?:should|must)\s+be\s+(?:avoided|rejected))\b"
)


def _is_rejected(text: str, start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    """Check a mention's clause, with an eight-word prefix scope and list suffix.

    Affirmative occurrences win when the same method is also rejected elsewhere.
    Bare hedges (e.g. 'might use') remain mentions; absence of a rejection is not
    proof of endorsement. Report that limitation wherever scores are interpreted.
    """
    left = 0
    right = len(text)
    for boundary in _CLAUSE_BOUNDARY.finditer(text):
        if boundary.end() <= start:
            left = boundary.end()
        elif boundary.start() >= end:
            right = boundary.start()
            break
    before = text[left:start]
    # A comma followed by a new subject is a boundary; a comma-separated method
    # list is not. This protects 'Avoid Leiden, UMAP is appropriate'.
    after = text[end:right]
    if ',' in before and re.match(r"\s+(?:is|can\s+be)\s+(?:appropriate|suitable|useful|recommended|better)\b", after):
        before = before.rsplit(',', 1)[1]
    # Do not turn double negation into rejection.
    before = re.sub(r"\b(?:do\s+not|don't)\s+(?:avoid|reject)\b", "use", before)
    rejections = list(_REJECTION.finditer(before))
    if rejections:
        rejection = rejections[-1]
        tail = before[rejection.end():]
        if len(re.findall(r"\b\w+\b", tail)) <= 8 and not _AFFIRMATION.search(tail):
            return True

    # Allow a shared negative predicate after a coordinated list of known
    # mentions: 'UMAP and PHATE are not appropriate'. No arbitrary prose span.
    for other_start, other_end in sorted(set(spans), reverse=True):
        if end <= other_start < other_end <= right:
            offset_start, offset_end = other_start - end, other_end - end
            after = after[:offset_start] + 'METHOD' + after[offset_end:]
    return bool(re.match(
        r"\s*:?\s*(?:(?:,\s*(?:(?:and|or|nor)\s+)?|(?:and|or|nor)\s+)METHOD\s*)*"
        + _NEGATIVE_PREDICATE, after,
    ))


def _find_method_matches(text_lower: str) -> Dict[str, Set[str]]:
    """Find vocabulary/phrase mentions that have at least one unrejected use."""
    matches: Dict[str, Set[str]] = {cat: set() for cat in METHOD_CATEGORIES}
    mentions = []
    for category, methods in METHOD_CATEGORIES.items():
        for method in sorted(methods):
            pattern = re.escape(method).replace('_', r'[\s_\-]?')
            for match in re.finditer(rf'\b{pattern}\b', text_lower):
                mentions.append((category, method, match.start(), match.end()))

    implicit_map = [('trajectory', 'trajectory_inference'), ('clustering', 'clustering'), ('cell_cycle', 'cell_cycle_scoring')]
    for category, method in implicit_map:
        for pattern in _PHRASE_PATTERNS[category]:
            for match in re.finditer(pattern, text_lower):
                mentions.append((category, method, match.start(), match.end()))

    # Nested aliases/implicit phrases may occupy the same text. Only maximal
    # spans are needed to mask coordinated alternatives in suffix checks.
    spans = sorted({(start, end) for _, _, start, end in mentions})
    spans = [(start, end) for start, end in spans
             if not any(a <= start and end <= b and (a, b) != (start, end) for a, b in spans)]
    for category, method, start, end in mentions:
        if not _is_rejected(text_lower, start, end, spans):
            matches[category].add(method)
    return matches


def extract_methods(text: str, include_categories: bool = False) -> Dict[str, Any]:
    """Extract method recommendations from LLM response text."""
    # Formatting and curly apostrophes should not hide local rejection cues.
    text_lower = re.sub(r"[*`]+", "", text.lower().replace("’", "'"))
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
    ground_truth: List[str] | None,
    failure_indicators: List[str]
) -> Tuple[bool | None, Dict[str, Any]]:
    """Require an expected mention and no failure mentions; absent criteria are unavailable.

    ``match_ratio`` remains vocabulary coverage, even when a failure blocks the
    boolean pass. Callers must supply the negation-filtered extracted methods.
    """
    extracted_set = _normalize_set(extracted)
    ground_truth_set = _normalize_set(ground_truth or [])
    failure_set = _normalize_set(failure_indicators)

    matches = extracted_set & ground_truth_set
    failures = extracted_set & failure_set

    is_success = bool(matches) and not failures if ground_truth_set else None

    return is_success, {
        'ground_truth_matches': sorted(matches),
        'failure_matches': sorted(failures),
        'total_extracted': len(extracted_set),
        'match_ratio': len(matches) / len(ground_truth_set) if ground_truth_set else None
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
