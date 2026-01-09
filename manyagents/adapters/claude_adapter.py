"""Claude adapter for LLM-based agent orchestration."""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .base import AgentAdapter, AdapterResult
from manyagents.utils.helpers import truncate_string, parse_json_safe

log = logging.getLogger(__name__)


class ClaudeAdapter(AgentAdapter):
    """
    Adapter for Anthropic Claude API integration.

    Enables LLM-based reasoning and text generation within manyAgents workflows.
    Supports both text and structured JSON output modes.
    """

    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 1.0
    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    DEFAULT_TEMPERATURE = 0.0
    DEFAULT_MAX_TOKENS = 2000

    def __init__(self):
        """Initialize Claude adapter with API client."""
        super().__init__("claude")
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.client = None

        if not self.api_key:
            log.warning(
                "ANTHROPIC_API_KEY not found in environment. "
                "Set it before calling run() to avoid authentication errors."
            )

    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self.client is None and self.api_key:
            try:
                from anthropic import AsyncAnthropic
                self.client = AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                log.error("anthropic package not installed. Run 'uv add anthropic'")
                return None
        return self.client

    def _build_api_params(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build API parameters from task configuration."""
        params = {
            "model": task_config.get("model", self.DEFAULT_MODEL),
            "messages": [{"role": "user", "content": task_config["prompt"]}],
            "temperature": task_config.get("temperature", self.DEFAULT_TEMPERATURE),
            "max_tokens": task_config.get("max_tokens", self.DEFAULT_MAX_TOKENS),
        }

        if system_prompt := task_config.get("system_prompt"):
            params["system"] = system_prompt

        return params

    async def run(
        self,
        task_config: Dict[str, Any],
        input_files: Dict[str, Path],
        input_data: Optional[Any] = None
    ) -> AdapterResult:
        """
        Execute Claude API call with given configuration.

        Args:
            task_config: Dictionary with parameters:
                - prompt (required): The main prompt/question for the LLM
                - model (optional): Model to use (default: "claude-sonnet-4-20250514")
                - temperature (optional): Sampling temperature (default: 0.0)
                - max_tokens (optional): Maximum tokens to generate (default: 2000)
                - response_format (optional): "text" or "json" (default: "text")
                - system_prompt (optional): System message to set context
            input_files: Files from previous workflow steps (currently unused)
            input_data: Data from previous steps (currently unused)

        Returns:
            AdapterResult with success status, summary, output_files, and metadata
        """
        log.info(f"ClaudeAdapter executing with config: {task_config}")

        if "prompt" not in task_config:
            error_msg = "ClaudeAdapter requires 'prompt' parameter in task_config"
            log.error(error_msg)
            return self.error_response(error_msg, error_type="missing_prompt")

        client = self._get_client()
        if not client:
            error_msg = "ANTHROPIC_API_KEY not set or anthropic package not installed."
            log.error(error_msg)
            return self.error_response(error_msg, error_type="missing_api_key")

        api_params = self._build_api_params(task_config)
        response_format = task_config.get("response_format", "text")

        log.info(
            f"Using model: {api_params['model']}, "
            f"temperature: {api_params['temperature']}, "
            f"format: {response_format}"
        )

        start_time = time.time()
        last_error = None

        for attempt in range(self.MAX_RETRIES):
            try:
                log.info(f"Making API call (attempt {attempt + 1}/{self.MAX_RETRIES})")
                response = await client.messages.create(**api_params)
                response_time = time.time() - start_time

                content = response.content[0].text
                usage = response.usage
                total_tokens = usage.input_tokens + usage.output_tokens

                log.info(f"API call completed in {response_time:.2f}s, tokens: {total_tokens}")

                output_files = {"raw_response": self.save_text_output(content, "response.txt")}

                if response_format == "json":
                    parsed_data, parse_error = parse_json_safe(content, log)
                    parsed_output = parsed_data or {"raw": content, "parse_error": parse_error}
                    output_files["parsed_output"] = self.save_json_output(parsed_output, "parsed_output.json")

                return self.success_response(
                    summary=f"Claude API call completed successfully. Response: {truncate_string(content, 100)}",
                    output_files=output_files,
                    metadata={
                        "model": api_params["model"],
                        "tokens_used": {
                            "input_tokens": usage.input_tokens,
                            "output_tokens": usage.output_tokens,
                            "total_tokens": total_tokens,
                        },
                        "response_time": response_time,
                        "stop_reason": response.stop_reason,
                        "response_format": response_format,
                    }
                )

            except Exception as e:
                error_name = type(e).__name__
                last_error = e
                log.warning(f"{error_name}: {e} (attempt {attempt + 1}/{self.MAX_RETRIES})")

                if "AuthenticationError" in error_name:
                    log.error(f"Authentication failed: {e}")
                    return self.error_response(
                        f"Authentication failed: {e}. Please check your ANTHROPIC_API_KEY.",
                        "authentication_failed",
                        str(e)
                    )

                if attempt < self.MAX_RETRIES - 1:
                    wait_time = self.RETRY_BASE_DELAY * (2 ** attempt)
                    log.info(f"Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)

        error_msg = f"Claude API error after {self.MAX_RETRIES} attempts: {last_error}"
        log.error(error_msg)
        return self.error_response(error_msg, "api_error", str(last_error))
