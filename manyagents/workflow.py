"""Workflow state and helpers."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

log = logging.getLogger(__name__)


@dataclass
class WorkflowState:
    """State container for workflow execution."""
    steps_completed: List[Dict[str, Any]] = field(default_factory=list)
    current_data: Any = None
    output_files: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, name: str, agent: str, result: Dict[str, Any]) -> None:
        """Record completed step."""
        self.steps_completed.append({"name": name, "agent": agent, "result": result})

        # Pass embeddings in-memory to next step
        if embeddings := result.get("output_files", {}).get("embeddings"):
            self.current_data = embeddings

        # Accumulate output files
        self.output_files |= result.get("output_files", {})

        # Store step metadata
        if meta := result.get("metadata"):
            self.metadata[name] = meta

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for result output."""
        return {
            "steps_completed": self.steps_completed,
            "output_files": self.output_files,
            "metadata": self.metadata,
        }


def log_step_to_wandb(step_idx: int, step_name: str, result: Dict[str, Any]) -> None:
    """Log step metrics to WandB if available."""
    try:
        import wandb
        if wandb.run is None:
            return

        metrics = result.get("output_files", {}).get("scores", {})
        embeddings = result.get("output_files", {}).get("embeddings")

        log_dict = {f"step_{step_idx}/{step_name}/success": 1}

        # Add scalar metrics
        if isinstance(metrics, dict):
            log_dict |= {
                f"step_{step_idx}/{step_name}/{k}": v
                for k, v in metrics.items()
                if isinstance(v, (int, float))
            }

        # Add shape information
        if embeddings is not None and hasattr(embeddings, 'shape'):
            log_dict[f"step_{step_idx}/{step_name}/n_samples"] = embeddings.shape[0]
            log_dict[f"step_{step_idx}/{step_name}/n_components"] = embeddings.shape[1]

        wandb.log(log_dict, step=step_idx)
        log.info(f"Logged metrics to WandB for step {step_idx}")
    except ImportError:
        pass
    except Exception as e:
        log.warning(f"Failed to log to WandB: {e}")
