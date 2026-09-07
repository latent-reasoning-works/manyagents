"""Adapter for CellForge multi-agent framework."""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import AgentAdapter, AdapterResult
from manyagents.utils.helpers import run_subprocess

log = logging.getLogger(__name__)


class CellForgeAdapter(AgentAdapter):
    """
    Adapter for CellForge multi-agent computational method design framework.

    CellForge is a CLI-based tool that generates computational methods through
    multi-agent collaboration. It has three main phases:
    1. Task Analysis: Analyze dataset and research objectives
    2. Method Design: Expert discussion to design approach
    3. Code Generation: Generate executable code

    This adapter wraps the CellForge CLI interface, handling:
    - Subprocess invocation with proper error handling
    - Selective phase execution
    - .h5ad input file management
    - stdout/stderr capture
    - Working directory and config file support
    """

    VALID_PHASES = {"task_analysis", "method_design", "code_generation", "all"}

    PRODUCES_TEXT_RESPONSE = False

    def __init__(self, cellforge_path: Optional[str] = None):
        """
        Initialize CellForge adapter.

        Args:
            cellforge_path: Path to CellForge installation directory.
                           Required here or via CELLFORGE_PATH. Relative paths are
                           bound to the caller's directory at construction time.
        """
        super().__init__("cellforge")
        installation = cellforge_path or os.getenv("CELLFORGE_PATH")
        if not installation:
            raise ValueError("Configure cellforge_path or CELLFORGE_PATH explicitly")
        self.cellforge_path = Path(installation).expanduser().resolve()
        # Resolve once, including symlinks, independently of any task working_dir.
        self.main_script = (self.cellforge_path / "main.py").resolve()
        # Keep the active interpreter (including its venv), without PATH/cwd lookup.
        self.python_executable = str(Path(sys.executable).absolute())

    async def run(
        self,
        task_config: Dict[str, Any],
        input_files: Dict[str, Path],
    ) -> AdapterResult:
        """
        Execute CellForge workflow via subprocess.

        Args:
            task_config: Configuration including:
                - phase: Which phase to run ('task_analysis', 'method_design',
                        'code_generation', or 'all'). Defaults to 'all'.
                - data_file: Path to .h5ad dataset file (required if not in input_files)
                - config_file: Optional path to CellForge config JSON
                - working_dir: Optional working directory for execution
                - timeout: Optional timeout in seconds (default: 3600)
            input_files: Input data files. If 'data' key exists, use that as data_file.

        Returns:
            AdapterResult with success status, summary, output_files, and metadata

        Raises:
            ValueError: If required parameters are missing or invalid
            FileNotFoundError: If CellForge installation or data file not found
        """
        start_time = time.time()

        # Validate phase
        phase = task_config.get("phase", "all")
        if phase not in self.VALID_PHASES:
            raise ValueError(f"Invalid phase '{phase}'. Must be one of {self.VALID_PHASES}")

        # Get and validate data file
        data_file_input = task_config.get("data_file") or input_files.get("data")
        if not data_file_input:
            raise ValueError(
                "data_file must be specified in task_config or provided in input_files['data']"
            )

        data_file = Path(data_file_input)
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        # Get and validate optional config file
        config_file = None
        if config_path := task_config.get("config_file"):
            config_file = Path(config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_file}")

        # Setup working directory
        working_dir = Path(task_config.get("working_dir") or Path.cwd())
        working_dir.mkdir(parents=True, exist_ok=True)

        timeout = task_config.get("timeout", self.config.timeout)

        # Check CellForge installation
        if not self.main_script.is_file():
            raise FileNotFoundError(
                f"CellForge main.py is not a file at {self.main_script}. "
                f"Set cellforge_path or CELLFORGE_PATH environment variable."
            )

        # Build and execute CLI command
        cmd = self._build_command(phase=phase, data_file=data_file, config_file=config_file)
        log.info(f"Executing CellForge with command: {' '.join(cmd)}")
        log.info(f"Working directory: {working_dir}")

        # Execute subprocess using shared utility
        result = await run_subprocess(cmd, cwd=working_dir, timeout=timeout)
        execution_time = time.time() - start_time

        # Handle timeout
        if result.timed_out:
            return self._create_error_result(
                phase=phase,
                execution_time=execution_time,
                data_file=data_file,
                working_dir=working_dir,
                error_type="timeout",
                summary=f"CellForge execution timed out after {timeout} seconds"
            )

        # Handle subprocess failure
        if result.exit_code == -1 and not result.timed_out:
            return self._create_error_result(
                phase=phase,
                execution_time=execution_time,
                data_file=data_file,
                working_dir=working_dir,
                error_type="subprocess_error",
                summary=f"CellForge execution failed: {result.stderr}"
            )

        success = result.exit_code == 0

        # Save outputs using base class directory
        cellforge_output_dir = working_dir / "cellforge_outputs"
        cellforge_output_dir.mkdir(parents=True, exist_ok=True)

        (cellforge_output_dir / "stdout.txt").write_text(result.stdout)
        (cellforge_output_dir / "stderr.txt").write_text(result.stderr)

        output_files: Dict[str, Any] = {
            "stdout": cellforge_output_dir / "stdout.txt",
            "stderr": cellforge_output_dir / "stderr.txt",
        }

        # Look for generated outputs
        if (generated_code_dir := working_dir / "generated_code").exists():
            output_files["generated_code"] = list(generated_code_dir.glob("*.py"))
        if (analysis_report := working_dir / "analysis_report.txt").exists():
            output_files["analysis_report"] = analysis_report

        # Generate summary and response
        status = "completed successfully" if success else f"failed with exit code {result.exit_code}"
        summary = f"CellForge {phase} {status} in {execution_time:.1f}s"

        # Build common metadata
        metadata = {
            "phase": phase,
            "exit_code": result.exit_code,
            "execution_time": execution_time,
            "data_file": str(data_file),
            "working_dir": str(working_dir),
        }

        if success:
            metadata["cellforge_version"] = await self._get_cellforge_version()
            return self.success_response(summary=summary, output_files=output_files, metadata=metadata)
        else:
            return self.error_response(summary=summary, error_type="execution_failed", metadata=metadata)

    def _create_error_result(
        self,
        phase: str,
        execution_time: float,
        data_file: Path,
        working_dir: Path,
        error_type: str,
        summary: str
    ) -> AdapterResult:
        """
        Create standardized error result with common metadata.

        Args:
            phase: Execution phase
            execution_time: Time taken for execution
            data_file: Path to data file
            working_dir: Working directory used
            error_type: Type of error that occurred
            summary: Human-readable error summary

        Returns:
            AdapterResult with error information
        """
        return self.error_response(
            summary=summary,
            error_type=error_type,
            metadata={
                "phase": phase,
                "execution_time": execution_time,
                "data_file": str(data_file),
                "working_dir": str(working_dir),
            }
        )

    def _build_command(
        self,
        phase: str,
        data_file: Path,
        config_file: Optional[Path] = None,
    ) -> List[str]:
        """
        Build CLI command for CellForge execution.

        Args:
            phase: Execution phase
            data_file: Path to .h5ad data file
            config_file: Optional path to config JSON

        Returns:
            List of command arguments
        """
        cmd = [self.python_executable, str(self.main_script)]

        # Add phase argument only if not running all phases
        if phase != "all":
            cmd.extend(["--phase", phase])

        # Add required data file
        cmd.extend(["--data", str(data_file)])

        # Add optional config file
        if config_file:
            cmd.extend(["--config", str(config_file)])

        return cmd

    async def _get_cellforge_version(self) -> Optional[str]:
        """
        Attempt to get CellForge version.

        Returns:
            Version string if available, None otherwise
        """
        cmd = [self.python_executable, str(self.main_script), "--version"]
        result = await run_subprocess(cmd, timeout=5.0)
        return result.stdout.strip() if result.exit_code == 0 and result.stdout.strip() else None
