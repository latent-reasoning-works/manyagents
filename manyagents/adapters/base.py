"""Abstract base class for agent adapters."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, NotRequired, TypedDict


class AdapterResult(TypedDict):
    """Standardized result format for all adapters."""
    success: bool
    summary: str
    output_files: Dict[str, Any]
    metadata: NotRequired[Dict[str, Any]]
    embeddings: NotRequired[Dict[str, Any]]


@dataclass
class AdapterConfig:
    """Common configuration for adapters."""
    output_base_dir: Path = field(default_factory=lambda: Path("outputs"))
    timeout: int = 3600  # Default 1 hour


class AgentAdapter(ABC):
    """Abstract base class for wrapping external agents with standardized interface."""

    PRODUCES_TEXT_RESPONSE: bool = True

    def __init__(self, name: str, config: Optional[AdapterConfig] = None):
        self.name = name
        self.config = config or AdapterConfig()
        self._output_dir: Optional[Path] = None

    @property
    def output_dir(self) -> Path:
        """Get or create the output directory for this adapter."""
        if self._output_dir is None:
            self._output_dir = self.config.output_base_dir / self.name
        self._output_dir.mkdir(parents=True, exist_ok=True)
        return self._output_dir

    def success_response(
        self,
        summary: str,
        output_files: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AdapterResult:
        """
        Create a standardized success response.

        Args:
            summary: Human-readable summary of what was accomplished
            output_files: Dictionary mapping file types to paths
            metadata: Additional execution metadata

        Returns:
            Standardized AdapterResult dictionary
        """
        return {
            "success": True,
            "summary": summary,
            "output_files": output_files or {},
            "metadata": metadata or {},
        }

    def error_response(
        self,
        summary: str,
        error_type: str = "error",
        details: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AdapterResult:
        """
        Create a standardized error response.

        Args:
            summary: Human-readable error summary
            error_type: Error classification (e.g., 'timeout', 'auth_failed')
            details: Additional error details
            metadata: Additional metadata to include

        Returns:
            Standardized AdapterResult dictionary with success=False
        """
        error_metadata = {"error": error_type, **(metadata or {})}
        if details:
            error_metadata["details"] = details

        return {
            "success": False,
            "summary": summary,
            "output_files": {},
            "metadata": error_metadata,
        }

    def save_text_output(self, content: str, filename: str) -> Path:
        """
        Save text content to a file in the adapter's output directory.

        Args:
            content: Text content to save
            filename: Name of the file to create

        Returns:
            Path to the created file
        """
        filepath = self.output_dir / filename
        filepath.write_text(content)
        return filepath

    def save_json_output(self, data: Any, filename: str) -> Path:
        """
        Save JSON content to a file in the adapter's output directory.

        Args:
            data: Data to serialize as JSON
            filename: Name of the file to create

        Returns:
            Path to the created file
        """
        filepath = self.output_dir / filename
        filepath.write_text(json.dumps(data, indent=2))
        return filepath

    @abstractmethod
    async def run(self, task_config: Dict[str, Any], input_files: Dict[str, Path]) -> AdapterResult:
        """
        Execute the agent with structured task configuration and input files.

        Args:
            task_config: Dictionary with agent-specific configuration parameters
            input_files: Dictionary mapping file types to file paths that the agent can use

        Returns:
            AdapterResult containing:
            - 'success': Boolean indicating if execution was successful
            - 'summary': Text summary of what the agent accomplished
            - 'output_files': Dict mapping file types to paths of files created
            - 'metadata': Any additional information about the execution
        """
        pass
