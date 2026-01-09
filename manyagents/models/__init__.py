"""
Data models for manyAgents orchestration.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ExecutionResult:
    """Standardized output from any agent adapter."""
    agent_name: str
    status: str  # 'success' or 'failure'
    summary: str
    output_files: Dict[str, str] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class WorkflowState:
    """A state-conserving object for a single workflow run."""
    goal: str
    history: List[ExecutionResult] = field(default_factory=list)

    def add_result(self, result: ExecutionResult):
        self.history.append(result)

    def last_summary(self) -> Optional[str]:
        if not self.history:
            return None
        return self.history[-1].summary


__all__ = ["ExecutionResult", "WorkflowState"]
