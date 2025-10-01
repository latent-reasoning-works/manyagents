"""Abstract base class for agent adapters."""

from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path


class AgentAdapter(ABC):
    """Abstract base class for wrapping external agents with standardized interface."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def run(self, task_config: Dict[str, Any], input_files: Dict[str, Path]) -> Dict[str, Any]:
        """
        Execute the agent with structured task configuration and input files.

        Args:
            task_config: Dictionary with agent-specific configuration parameters
            input_files: Dictionary mapping file types to file paths that the agent can use

        Returns:
            Dictionary containing:
            - 'summary': Text summary of what the agent accomplished
            - 'output_files': Dict mapping file types to paths of files created by the agent
            - 'success': Boolean indicating if execution was successful
            - 'metadata': Any additional information about the execution
        """
        pass