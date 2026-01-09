"""Experiment infrastructure for pipeline invariance testing."""

from .runner import run_invariance_experiment, run_parallel_agents
from .extractor import extract_methods, check_ground_truth_match, METHOD_CATEGORIES
from .metrics import compute_jaccard_similarity
from .hydra_runner import run_experiment
from .logger import ExperimentLogger, NullLogger
from .prompt_loader import (
    load_scenarios,
    iter_prompts,
    scenarios_to_hydra_config,
    get_scenario_matrix,
)

__all__ = [
    "run_invariance_experiment",
    "run_parallel_agents",
    "run_experiment",
    "extract_methods",
    "check_ground_truth_match",
    "METHOD_CATEGORIES",
    "compute_jaccard_similarity",
    "ExperimentLogger",
    "NullLogger",
    # Prompt loading
    "load_scenarios",
    "iter_prompts",
    "scenarios_to_hydra_config",
    "get_scenario_matrix",
]
