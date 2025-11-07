"""Drug discovery example implementations for manyAgents.

This module demonstrates how to use generic manyAgents adapters for
drug repurposing workflows. All functions are designed to be called
by the generic adapters (DataTransformAdapter, RankingAdapter, FilterAdapter).

Usage:
    In workflow configs, reference these functions via module path:
    
    transform_function: "manyagents.examples.drug_discovery.data:fetch_chembl"
    scoring_function: "manyagents.examples.drug_discovery.scoring:score_repurposing"
    filter_function: "manyagents.examples.drug_discovery.validation:validate_safety"
"""

from . import data, scoring, validation

__all__ = ["data", "scoring", "validation"]
