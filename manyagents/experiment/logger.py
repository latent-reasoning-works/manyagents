"""Structured logging for pipeline invariance experiments."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# Lazy import wandb to avoid import errors if not installed
_wandb = None


def _get_wandb():
    """Lazy import wandb."""
    global _wandb
    if _wandb is None:
        try:
            import wandb
            _wandb = wandb
        except ImportError:
            raise ImportError(
                "wandb is required for experiment logging. "
                "Install with: pip install wandb"
            )
    return _wandb


class ExperimentLogger:
    """Structured logging for pipeline invariance experiments.

    Provides wandb integration for:
    - Experiment configuration tracking
    - Per-scenario result logging
    - Aggregate metrics per system
    - Summary tables for paper figures
    - Auto-generated visualizations
    - Artifact storage for raw responses

    Usage:
        logger = ExperimentLogger(
            project="manyagents-invariance",
            experiment_name="trajectory_test",
            config={"scenarios": [...], "systems": [...]}
        )
        logger.log_scenario_result("claude", "trajectory", result)
        logger.log_system_metrics("claude", metrics)
        logger.log_summary_table(all_metrics)
        logger.create_visualizations(experiment_results)
        logger.finish()
    """

    def __init__(
        self,
        project: str = "manyagents",
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        enabled: bool = True,
    ):
        """Initialize wandb run with experiment config.

        Args:
            project: wandb project name
            experiment_name: Name for this run (auto-generated if None)
            config: Experiment configuration dict
            entity: wandb entity (team/user)
            tags: List of tags for the run
            enabled: If False, logging is disabled (for testing)
        """
        self.enabled = enabled
        self.run = None
        self._step = 0

        if not enabled:
            log.info("ExperimentLogger disabled, no wandb logging")
            return

        wandb = _get_wandb()

        self.run = wandb.init(
            project=project,
            name=experiment_name,
            entity=entity,
            tags=tags if tags else None,
            config=config or {},
            reinit=True,
        )

        log.info(f"Initialized wandb run: {self.run.url}")

    def log_config(
        self,
        scenarios: Dict[str, Dict[str, Any]],
        target_systems: List[str],
        system_prompt: Optional[str] = None,
        model_overrides: Optional[Dict[str, str]] = None,
    ) -> None:
        """Log experiment configuration.

        Args:
            scenarios: Dict mapping scenario_id to scenario config
            target_systems: List of AI systems being tested
            system_prompt: Optional system prompt used
            model_overrides: Optional model overrides per system
        """
        if not self.enabled or not self.run:
            return

        wandb = _get_wandb()

        # Use prefixed keys to avoid conflict with Hydra config
        wandb.config.update({
            "exp/scenario_names": list(scenarios.keys()),
            "exp/scenario_count": len(scenarios),
            "exp/target_systems": target_systems,
            "exp/system_count": len(target_systems),
            "exp/system_prompt_preview": system_prompt[:500] if system_prompt else None,
            "exp/model_overrides": model_overrides,
        }, allow_val_change=True)

        # Log scenario details as a table
        scenario_table = wandb.Table(columns=[
            "scenario_id", "expected_geometry", "ground_truth_methods"
        ])
        for sid, scenario in scenarios.items():
            scenario_table.add_data(
                sid,
                scenario.get("expected_geometry", "unknown"),
                ", ".join(scenario.get("ground_truth_methods", []))
            )
        wandb.log({"scenarios": scenario_table})

    def log_scenario_result(
        self,
        system: str,
        scenario_id: str,
        result: Dict[str, Any],
    ) -> None:
        """Log per-scenario results with extracted methods.

        Args:
            system: Name of the AI system
            scenario_id: ID of the scenario
            result: Result dict from run_single_agent
        """
        if not self.enabled or not self.run:
            return

        wandb = _get_wandb()

        success = result.get("success", False)
        prefix = f"{system}/{scenario_id}"

        metrics = {
            f"{prefix}/success": int(success),
        }

        if success:
            methods = result.get("extracted_methods", [])
            metrics.update({
                f"{prefix}/method_count": len(methods),
                f"{prefix}/mentions_clustering": int(result.get("mentions_clustering", False)),
                f"{prefix}/mentions_trajectory": int(result.get("mentions_trajectory", False)),
                f"{prefix}/mentions_data_inspection": int(result.get("mentions_data_inspection", False)),
                f"{prefix}/ground_truth_match": int(result.get("matches_ground_truth", False)),
            })

        wandb.log(metrics, step=self._step)
        self._step += 1

    def log_system_metrics(self, system: str, metrics: Dict[str, float]) -> None:
        """Log aggregate metrics for a system.

        Args:
            system: Name of the AI system
            metrics: Aggregate metrics dict from compute_system_metrics
        """
        if not self.enabled or not self.run:
            return

        wandb = _get_wandb()

        wandb.log({
            f"summary/{system}/jaccard": metrics.get("jaccard_similarity_across_prompts", 0),
            f"summary/{system}/jaccard_min": metrics.get("jaccard_min", 0),
            f"summary/{system}/jaccard_max": metrics.get("jaccard_max", 0),
            f"summary/{system}/ground_truth_match_rate": metrics.get("ground_truth_match_rate", 0),
            f"summary/{system}/clustering_for_all_rate": metrics.get("clustering_for_all_rate", 0),
            f"summary/{system}/scenarios_evaluated": metrics.get("scenarios_evaluated", 0),
        })

    def log_summary_table(self, all_metrics: Dict[str, Dict[str, float]]) -> None:
        """Create wandb.Table for main results (Table 1 in paper).

        Args:
            all_metrics: Dict mapping system name to metrics dict
        """
        if not self.enabled or not self.run:
            return

        wandb = _get_wandb()

        # Main results table
        results_table = wandb.Table(columns=[
            "System",
            "Jaccard (Invariance)",
            "Ground Truth Match",
            "Clustering-for-All",
            "Scenarios"
        ])

        for system, m in all_metrics.items():
            results_table.add_data(
                system,
                round(m.get("jaccard_similarity_across_prompts", 0), 3),
                round(m.get("ground_truth_match_rate", 0), 3),
                round(m.get("clustering_for_all_rate", 0), 3),
                m.get("scenarios_evaluated", 0),
            )

        wandb.log({"results_summary": results_table})

    def log_method_recommendations(
        self,
        experiment_results: Dict[str, Any],
    ) -> None:
        """Log detailed method recommendations table (Figure 3 data).

        Args:
            experiment_results: Full experiment results dict
        """
        if not self.enabled or not self.run:
            return

        wandb = _get_wandb()
        from .extractor import METHOD_CATEGORIES

        # Method recommendations per system/scenario
        methods_table = wandb.Table(columns=[
            "System", "Scenario", "Methods", "Categories"
        ])

        results = experiment_results.get("results", {})
        for system, scenarios in results.items():
            for scenario_id, result in scenarios.items():
                if result.get("success"):
                    methods = result.get("extracted_methods", [])
                    # Determine categories present
                    categories = set()
                    for method in methods:
                        method_lower = method.lower()
                        for cat, cat_methods in METHOD_CATEGORIES.items():
                            if method_lower in cat_methods:
                                categories.add(cat)
                                break

                    methods_table.add_data(
                        system,
                        scenario_id,
                        ", ".join(methods[:10]),  # Limit for display
                        ", ".join(sorted(categories)),
                    )

        wandb.log({"method_recommendations": methods_table})

    def create_visualizations(self, experiment_results: Dict[str, Any]) -> None:
        """Auto-generate visualizations for paper figures.

        Creates:
        - Recommendation heatmap (Figure 1)
        - Jaccard similarity matrix (Figure 2)
        - Failure analysis table (Figure 4)

        Args:
            experiment_results: Full experiment results dict
        """
        if not self.enabled or not self.run:
            return

        wandb = _get_wandb()

        results = experiment_results.get("results", {})
        metrics = experiment_results.get("metrics", {})

        # Figure 4: Failure analysis table
        failure_table = wandb.Table(columns=[
            "System", "Scenario", "Expected", "Recommended Clustering", "GT Match"
        ])

        prompts = experiment_results.get("prompts", {})
        for system, scenarios in results.items():
            for scenario_id, result in scenarios.items():
                if result.get("success"):
                    prompt_info = prompts.get(scenario_id, {})
                    failure_table.add_data(
                        system,
                        scenario_id,
                        prompt_info.get("expected_geometry", "unknown"),
                        "Yes" if result.get("mentions_clustering") else "No",
                        "Yes" if result.get("matches_ground_truth") else "No",
                    )

        wandb.log({"failure_analysis": failure_table})

        # Log method recommendations
        self.log_method_recommendations(experiment_results)

    def save_artifacts(
        self,
        experiment_results: Dict[str, Any],
        output_dir: Path,
        experiment_id: str,
    ) -> None:
        """Save and upload raw response artifacts.

        Args:
            experiment_results: Full experiment results dict
            output_dir: Directory where results are saved
            experiment_id: Unique experiment identifier
        """
        if not self.enabled or not self.run:
            return

        wandb = _get_wandb()

        # Save raw responses to files
        raw_dir = output_dir / "raw_responses"
        raw_dir.mkdir(parents=True, exist_ok=True)

        results = experiment_results.get("results", {})
        for system, scenarios in results.items():
            system_dir = raw_dir / system
            system_dir.mkdir(exist_ok=True)

            for scenario_id, result in scenarios.items():
                if result.get("raw_response"):
                    response_file = system_dir / f"{scenario_id}.txt"
                    response_file.write_text(result["raw_response"])

        # Create and upload artifact
        artifact = wandb.Artifact(
            f"responses_{experiment_id}",
            type="experiment_data",
            description="Raw LLM responses from invariance experiment"
        )
        artifact.add_dir(str(raw_dir))

        # Also add the full results JSON
        results_file = output_dir / "results.json"
        if results_file.exists():
            artifact.add_file(str(results_file))

        wandb.log_artifact(artifact)
        log.info(f"Uploaded artifact: responses_{experiment_id}")

    def finish(self) -> Optional[str]:
        """Close wandb run.

        Returns:
            Run URL if logging was enabled, None otherwise
        """
        if not self.enabled or not self.run:
            return None

        wandb = _get_wandb()

        url = self.run.url
        wandb.finish()
        log.info(f"Finished wandb run: {url}")
        return url


class NullLogger(ExperimentLogger):
    """No-op logger for when wandb is disabled."""

    def __init__(self, **kwargs):
        self.enabled = False
        self.run = None

    def log_config(self, *args, **kwargs):
        pass

    def log_scenario_result(self, *args, **kwargs):
        pass

    def log_system_metrics(self, *args, **kwargs):
        pass

    def log_summary_table(self, *args, **kwargs):
        pass

    def log_method_recommendations(self, *args, **kwargs):
        pass

    def create_visualizations(self, *args, **kwargs):
        pass

    def save_artifacts(self, *args, **kwargs):
        pass

    def finish(self):
        return None
