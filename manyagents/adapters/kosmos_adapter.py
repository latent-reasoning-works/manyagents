"""Adapter for Kosmos autonomous scientist framework."""

import logging
import os
import time
from pathlib import Path
from typing import Any

from .base import AgentAdapter, AdapterResult
from manyagents.utils.helpers import run_subprocess

log = logging.getLogger(__name__)


class KosmosAdapter(AgentAdapter):
    """
    Adapter for Kosmos autonomous scientist framework.

    Kosmos is a multi-agent research system that performs autonomous
    scientific investigation cycles. It requires Python 3.11+, so this
    adapter uses CLI invocation to a separate Kosmos environment.

    Environment Variables:
        KOSMOS_PYTHON: Path to Python interpreter with Kosmos installed
        KOSMOS_PATH: Path to Kosmos installation directory
        ANTHROPIC_API_KEY or OPENAI_API_KEY: Required for LLM calls
    """

    DEFAULT_CYCLES = 3
    DEFAULT_TASKS_PER_CYCLE = 10
    DEFAULT_TIMEOUT = 3600  # 1 hour

    PRODUCES_TEXT_RESPONSE = False

    def __init__(
        self,
        kosmos_python: str | None = None,
        kosmos_path: str | None = None,
    ):
        """
        Initialize Kosmos adapter.

        Args:
            kosmos_python: Path to Python with Kosmos installed.
                          Defaults to KOSMOS_PYTHON env var.
            kosmos_path: Path to Kosmos installation.
                        Defaults to KOSMOS_PATH env var.
        """
        super().__init__("kosmos")
        self.kosmos_python = kosmos_python or os.getenv("KOSMOS_PYTHON")
        self.kosmos_path = Path(kosmos_path or os.getenv("KOSMOS_PATH", "."))

    def _check_prerequisites(self) -> str | None:
        """Check if prerequisites are met. Returns error message if not."""
        if not self.kosmos_python:
            return (
                "KOSMOS_PYTHON not set. Kosmos requires Python 3.11+ in a separate environment. "
                "Set KOSMOS_PYTHON to the path of a Python interpreter with Kosmos installed."
            )

        if not Path(self.kosmos_python).exists():
            return f"Kosmos Python interpreter not found: {self.kosmos_python}"

        if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")):
            return "Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set."

        return None

    def _build_command(self, task_config: dict[str, Any]) -> list[str]:
        """Build CLI command for Kosmos execution."""
        cmd = [self.kosmos_python, "-m", "kosmos.cli", "run", task_config["research_question"]]

        # Optional flags
        if domain := task_config.get("domain"):
            cmd.extend(["--domain", domain])

        if budget := task_config.get("budget"):
            cmd.extend(["--budget", str(budget)])

        if task_config.get("stream"):
            cmd.append("--stream")

        if task_config.get("trace"):
            cmd.append("--trace")

        return cmd

    async def run(
        self,
        task_config: dict[str, Any],
        input_files: dict[str, Path],
    ) -> AdapterResult:
        """
        Execute Kosmos research workflow.

        Args:
            task_config: Configuration including:
                - research_question: str (required) - The research question
                - domain: str (optional) - Research domain (biology, ml, etc.)
                - num_cycles: int (optional) - Number of research cycles
                - budget: float (optional) - Cost limit in dollars
                - timeout: int (optional) - Timeout in seconds
                - artifacts_dir: str (optional) - Directory for outputs
                - stream: bool (optional) - Stream output in real-time
                - trace: bool (optional) - Maximum verbosity
            input_files: Input files (passed to Kosmos working directory)

        Returns:
            AdapterResult with research findings and metadata
        """
        start_time = time.time()

        # Guard: Validate prerequisites
        if error := self._check_prerequisites():
            return self.error_response(error, error_type="configuration_error")

        # Guard: Validate required parameters
        if "research_question" not in task_config:
            return self.error_response(
                "KosmosAdapter requires 'research_question' parameter",
                error_type="missing_parameter"
            )

        # Setup execution environment
        timeout = task_config.get("timeout", self.DEFAULT_TIMEOUT)
        artifacts_dir = Path(task_config.get("artifacts_dir", self.output_dir / "artifacts"))
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Executing Kosmos research: {task_config['research_question'][:100]}...")
        log.info(f"Artifacts directory: {artifacts_dir}")

        # Build and execute command
        cmd = self._build_command(task_config)
        log.info(f"Command: {' '.join(cmd)}")

        env = os.environ.copy()
        env["KOSMOS_ARTIFACTS_DIR"] = str(artifacts_dir)

        result = await run_subprocess(cmd, cwd=self.kosmos_path, timeout=timeout, env=env)
        execution_time = time.time() - start_time

        # Guard: Handle timeout
        if result.timed_out:
            return self.error_response(
                f"Kosmos research timed out after {timeout}s",
                error_type="timeout",
                metadata={
                    "execution_time": execution_time,
                    "partial_stdout": result.stdout[:2000] if result.stdout else "",
                }
            )

        # Collect output files
        output_files = {}
        if result.stdout:
            output_files["stdout"] = self.save_text_output(result.stdout, "kosmos_output.txt")
        if result.stderr:
            output_files["stderr"] = self.save_text_output(result.stderr, "kosmos_stderr.txt")

        # Collect generated artifacts
        if artifact_files := [f for f in artifacts_dir.glob("**/*") if f.is_file()]:
            output_files["artifacts"] = artifact_files

        # Build metadata
        metadata = {
            "exit_code": result.exit_code,
            "execution_time": execution_time,
            "research_question": task_config["research_question"],
            "artifacts_dir": str(artifacts_dir),
        }
        if domain := task_config.get("domain"):
            metadata["domain"] = domain

        # Return success or error based on exit code
        if result.exit_code == 0:
            return self.success_response(
                summary=f"Kosmos completed research in {execution_time:.1f}s",
                output_files=output_files,
                metadata=metadata,
            )

        return self.error_response(
            summary=f"Kosmos failed with exit code {result.exit_code}",
            error_type="execution_failed",
            details=result.stderr[:500] if result.stderr else "",
            metadata=metadata,
        )
