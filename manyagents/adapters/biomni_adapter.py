"""Adapter for Biomni biomedical AI agent."""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .base import AgentAdapter, AdapterResult

log = logging.getLogger(__name__)


class BiomniAdapter(AgentAdapter):
    """
    Adapter for Stanford's Biomni biomedical AI agent.

    Biomni is an LLM-based agent for biomedical research tasks including
    literature review, data analysis, and hypothesis generation.

    Environment Variables:
        BIOMNI_DATA_PATH: Path to Biomni data/weights directory
        ANTHROPIC_API_KEY: Required for LLM calls
    """

    DEFAULT_LLM = "claude-opus-5"
    DEFAULT_TIMEOUT = 3600  # 1 hour

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize Biomni adapter.

        Args:
            data_path: Path to Biomni data directory. Defaults to BIOMNI_DATA_PATH
                      environment variable, then $SCRATCH/biomni_data.
        """
        super().__init__("biomni")
        default_path = os.path.join(os.getenv("SCRATCH", "."), "biomni_data")
        self.data_path = Path(data_path or os.getenv("BIOMNI_DATA_PATH") or default_path)

    def _check_prerequisites(self) -> Optional[str]:
        """Check if prerequisites are met. Returns error message if not."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            return "ANTHROPIC_API_KEY not set. Biomni requires Anthropic API access."
        return None

    async def run(
        self,
        task_config: Dict[str, Any],
        input_files: Dict[str, Path],
    ) -> AdapterResult:
        """
        Execute Biomni agent with given task.

        Args:
            task_config: Configuration including:
                - task or prompt: str (required) - The biomedical task to perform
                - llm: str (optional) - LLM model to use (default: claude-opus-5)
                - data_path: str (optional) - Override data directory path
                - disable_datalake: bool (optional) - Skip datalake download
                - timeout: int (optional) - Timeout in seconds
            input_files: Input files (currently unused by Biomni)

        Returns:
            AdapterResult with task output and metadata
        """
        start_time = time.time()

        # Guard clauses for validation
        if error := self._check_prerequisites():
            return self.error_response(error, error_type="missing_api_key")

        if "task" not in task_config and "prompt" not in task_config:
            return self.error_response(
                "BiomniAdapter requires 'task' or 'prompt' parameter in task_config",
                error_type="missing_parameter"
            )

        # Extract configuration
        task = task_config.get("task", task_config.get("prompt"))
        llm = task_config.get("llm", self.DEFAULT_LLM)
        data_path = Path(task_config.get("data_path", self.data_path))
        timeout = task_config.get("timeout", self.DEFAULT_TIMEOUT)

        log.info(f"Executing Biomni task with LLM: {llm}")
        log.info(f"Data path: {data_path}")

        try:
            from biomni.agent import A1

            # Build agent configuration
            agent_kwargs = {"path": str(data_path), "llm": llm}
            if task_config.get("disable_datalake"):
                agent_kwargs["expected_data_lake_files"] = []

            # Execute agent asynchronously
            agent = A1(**agent_kwargs)
            result = await asyncio.wait_for(
                asyncio.to_thread(agent.go, task),
                timeout=timeout
            )

            # A1.go returns (self.log, message.content); only final content is
            # an answer. The log includes the user prompt and must not be scored.
            content = result[1] if isinstance(result, tuple) and len(result) == 2 else result
            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"Biomni final content must be nonempty text, got {type(content).__name__}"
                )

            execution_time = time.time() - start_time
            output_files = self.save_response(content, "biomni_output.txt")

            return self.success_response(
                summary=f"Biomni completed task in {execution_time:.1f}s",
                output_files=output_files,
                metadata={
                    "llm": llm,
                    "execution_time": execution_time,
                    "data_path": str(data_path),
                }
            )

        except asyncio.TimeoutError:
            return self.error_response(
                f"Biomni task timed out after {timeout}s",
                error_type="timeout",
                metadata={"execution_time": time.time() - start_time}
            )

        except ImportError as e:
            return self.error_response(
                f"Biomni not installed: {e}. Run 'uv sync --extra full'",
                error_type="import_error",
                details=str(e)
            )

        except Exception as e:
            log.error(f"Biomni execution failed: {e}", exc_info=True)
            return self.error_response(
                f"Biomni execution failed: {e}",
                error_type="execution_error",
                details=str(e),
                metadata={"execution_time": time.time() - start_time}
            )
