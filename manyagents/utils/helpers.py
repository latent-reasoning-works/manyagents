"""Shared utilities for adapter implementations."""

import asyncio
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)


@dataclass
class SubprocessResult:
    """Result from subprocess execution."""
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


async def run_subprocess(
    cmd: List[str],
    cwd: Optional[Path] = None,
    timeout: Optional[float] = None,
    env: Optional[dict] = None,
) -> SubprocessResult:
    """
    Run a subprocess with timeout support and proper cleanup.

    Args:
        cmd: Command and arguments as a list
        cwd: Working directory for the subprocess
        timeout: Timeout in seconds (None for no timeout)
        env: Environment variables to pass to subprocess

    Returns:
        SubprocessResult with stdout, stderr, exit_code, and timed_out flag

    Example:
        >>> result = await run_subprocess(["python", "--version"], timeout=10)
        >>> print(result.stdout)
        Python 3.10.12
    """
    log.debug(f"Running subprocess: {' '.join(cmd)}")

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd) if cwd else None,
            env=env,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
            return SubprocessResult(
                stdout=stdout_bytes.decode('utf-8', errors='replace'),
                stderr=stderr_bytes.decode('utf-8', errors='replace'),
                exit_code=process.returncode,
                timed_out=False,
            )

        except asyncio.TimeoutError:
            log.warning(f"Subprocess timed out after {timeout}s, killing process")
            process.kill()
            await process.wait()

            # Try to capture any output before timeout
            async def _read_stream(stream):
                try:
                    return (await stream.read()).decode('utf-8', errors='replace')
                except Exception:
                    return ""

            stdout = await _read_stream(process.stdout) if process.stdout else ""
            stderr = await _read_stream(process.stderr) if process.stderr else ""

            return SubprocessResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=-1,
                timed_out=True,
            )

    except FileNotFoundError:
        log.error(f"Command not found: {cmd[0]}")
        return SubprocessResult(
            stdout="",
            stderr=f"Command not found: {cmd[0]}",
            exit_code=-1,
            timed_out=False,
        )

    except Exception as e:
        log.error(f"Subprocess execution failed: {e}")
        return SubprocessResult(
            stdout="",
            stderr=str(e),
            exit_code=-1,
            timed_out=False,
        )


def find_executable(name: str, fallback: Optional[str] = None) -> Optional[str]:
    """
    Find an executable in PATH.

    Args:
        name: Name of the executable to find
        fallback: Fallback value if not found

    Returns:
        Path to executable or fallback value
    """
    return shutil.which(name) or fallback


def get_python_executable() -> str:
    """
    Get the Python executable path.

    Returns:
        Path to python executable, preferring 'python' over 'python3'
    """
    return find_executable("python") or find_executable("python3") or "python"


def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate a string to a maximum length with suffix.

    Args:
        s: String to truncate
        max_length: Maximum length before truncation
        suffix: Suffix to append when truncated

    Returns:
        Truncated string with suffix, or original if short enough
    """
    if len(s) <= max_length:
        return s
    return s[:max_length] + suffix


def parse_json_safe(content: str, logger: Optional[logging.Logger] = None) -> Tuple[Optional[dict], Optional[str]]:
    """
    Safely parse JSON content, returning error message on failure.

    Args:
        content: JSON string to parse
        logger: Optional logger for warnings

    Returns:
        Tuple of (parsed_data, error_message). One will be None.
    """
    try:
        return json.loads(content), None
    except json.JSONDecodeError as e:
        error_msg = str(e)
        if logger:
            logger.warning(f"Failed to parse JSON: {error_msg}")
        return None, error_msg
