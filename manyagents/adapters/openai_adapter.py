"""OpenAI adapter for LLM-based agent orchestration."""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI, RateLimitError, APIError, AuthenticationError

from .base import AgentAdapter, AdapterResult
from manyagents.utils.helpers import truncate_string, parse_json_safe

log = logging.getLogger(__name__)


class OpenAIAdapter(AgentAdapter):
    """
    Adapter for OpenAI API integration.

    Enables LLM-based reasoning and text generation within manyAgents workflows.
    Supports both text and structured JSON output modes.
    """

    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 1.0
    DEFAULT_MODEL = "gpt-4o"
    DEFAULT_TEMPERATURE = 0.0
    DEFAULT_MAX_TOKENS = 2000

    def __init__(self):
        """Initialize OpenAI adapter with API client."""
        super().__init__("openai")
        api_key = os.getenv('OPENAI_API_KEY')

        if not api_key:
            log.warning(
                "OPENAI_API_KEY not found in environment. "
                "Set it before calling run() to avoid authentication errors."
            )

        self.client = AsyncOpenAI(api_key=api_key) if api_key else None

    def _build_messages(self, prompt: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """Build message list for OpenAI API."""
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_api_params(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build API parameters from task configuration."""
        params = {
            "model": task_config.get("model", self.DEFAULT_MODEL),
            "messages": self._build_messages(
                task_config["prompt"],
                task_config.get("system_prompt")
            ),
            "temperature": task_config.get("temperature", self.DEFAULT_TEMPERATURE),
            "max_tokens": task_config.get("max_tokens", self.DEFAULT_MAX_TOKENS),
        }

        if task_config.get("response_format") == "json_object":
            params["response_format"] = {"type": "json_object"}

        return params

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay for retry."""
        return self.RETRY_BASE_DELAY * (2 ** attempt)

    async def run(
        self,
        task_config: Dict[str, Any],
        input_files: Dict[str, Path],
        input_data: Optional[Any] = None
    ) -> AdapterResult:
        """
        Execute OpenAI API call with given configuration.

        Args:
            task_config: Dictionary with parameters:
                - prompt (required): The main prompt/question for the LLM
                - model (optional): Model to use (default: "gpt-4o")
                - temperature (optional): Sampling temperature (default: 0.0)
                - max_tokens (optional): Maximum tokens to generate (default: 2000)
                - response_format (optional): "text" or "json_object" (default: "text")
                - system_prompt (optional): System message to set context
            input_files: Files from previous workflow steps (currently unused)
            input_data: Data from previous steps (currently unused)

        Returns:
            AdapterResult with success status, summary, output_files, and metadata
        """
        log.info(f"OpenAIAdapter executing with config: {task_config}")

        # Validate inputs
        if "prompt" not in task_config:
            error_msg = "OpenAIAdapter requires 'prompt' parameter in task_config"
            log.error(error_msg)
            return self.error_response(error_msg, error_type=error_msg)

        if not self.client:
            error_msg = "OPENAI_API_KEY not set. Please set the environment variable."
            log.error(error_msg)
            return self.error_response(error_msg, error_type="missing_api_key")

        # Build API parameters
        api_params = self._build_api_params(task_config)
        response_format = task_config.get("response_format", "text")

        log.info(
            f"Using model: {api_params['model']}, "
            f"temperature: {api_params['temperature']}, "
            f"format: {response_format}"
        )

        # Execute with retry logic
        start_time = time.time()
        for attempt in range(self.MAX_RETRIES):
            try:
                log.info(f"Making API call (attempt {attempt + 1}/{self.MAX_RETRIES})")
                response = await self.client.chat.completions.create(**api_params)
                response_time = time.time() - start_time

                # Extract response data
                content = response.choices[0].message.content
                tokens_used = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

                log.info(f"API call completed in {response_time:.2f}s, tokens: {tokens_used['total_tokens']}")

                # Save and parse outputs
                output_files = {"raw_response": self.save_text_output(content, "response.txt")}

                if response_format == "json_object":
                    parsed_data, parse_error = parse_json_safe(content, log)
                    parsed_output = parsed_data or {"raw": content, "parse_error": parse_error}
                    output_files["parsed_output"] = self.save_json_output(parsed_output, "parsed_output.json")

                return self.success_response(
                    summary=f"OpenAI API call completed successfully. Response: {truncate_string(content, 100)}",
                    output_files=output_files,
                    metadata={
                        "model": api_params["model"],
                        "tokens_used": tokens_used,
                        "response_time": response_time,
                        "finish_reason": response.choices[0].finish_reason,
                        "response_format": response_format,
                    }
                )

            except AuthenticationError as e:
                log.error(f"Authentication failed: {e}")
                return self.error_response(
                    f"Authentication failed: {e}. Please check your OPENAI_API_KEY.",
                    "authentication_failed",
                    str(e)
                )

            except (RateLimitError, APIError) as e:
                error_type = "rate_limit" if isinstance(e, RateLimitError) else "api_error"
                log.warning(f"{error_type.replace('_', ' ').title()}: {e} (attempt {attempt + 1}/{self.MAX_RETRIES})")

                if attempt < self.MAX_RETRIES - 1:
                    wait_time = self._calculate_retry_delay(attempt)
                    log.info(f"Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    error_msg = (
                        f"Rate limit exceeded after {self.MAX_RETRIES} attempts"
                        if isinstance(e, RateLimitError)
                        else f"OpenAI API error: {e}"
                    )
                    log.error(error_msg)
                    return self.error_response(error_msg, error_type, str(e))

            except Exception as e:
                log.error(f"Unexpected error calling OpenAI API: {e}", exc_info=True)
                return self.error_response(
                    f"Unexpected error calling OpenAI API: {e}",
                    "unexpected_error",
                    str(e)
                )

        # Fallback (should not reach here due to exception handling above)
        return self.error_response("Failed after all retry attempts", "max_retries_exceeded")
