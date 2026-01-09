"""Claude adapter for LLM-based agent orchestration."""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .base import AgentAdapter, AdapterResult
from manyagents.utils.helpers import truncate_string, parse_json_safe
from manyagents.utils.retry import retry_with_backoff, is_auth_error
from manyagents.utils.constants import (
    PROMPT, SYSTEM_PROMPT, TEMPERATURE, MAX_TOKENS, MODEL, RESPONSE_FORMAT,
    DEFAULT_MAX_RETRIES, DEFAULT_RETRY_BASE_DELAY,
)

log = logging.getLogger(__name__)


class ClaudeAdapter(AgentAdapter):
    """Adapter for Anthropic Claude API integration."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    DEFAULT_TEMPERATURE = 0.0
    DEFAULT_MAX_TOKENS = 2000

    def __init__(self):
        super().__init__("claude")
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.client = None

        if not self.api_key:
            log.warning("ANTHROPIC_API_KEY not found. Set it before calling run().")

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
            "model": task_config.get(MODEL, self.DEFAULT_MODEL),
            "messages": [{"role": "user", "content": task_config[PROMPT]}],
            "temperature": task_config.get(TEMPERATURE, self.DEFAULT_TEMPERATURE),
            "max_tokens": task_config.get(MAX_TOKENS, self.DEFAULT_MAX_TOKENS),
        }
        if system := task_config.get(SYSTEM_PROMPT):
            params["system"] = system
        return params

    async def run(
        self,
        task_config: Dict[str, Any],
        input_files: Dict[str, Path],
        input_data: Optional[Any] = None
    ) -> AdapterResult:
        """Execute Claude API call with given configuration."""
        log.info(f"ClaudeAdapter executing with config: {task_config}")

        # Guard clauses
        if PROMPT not in task_config:
            return self.error_response(
                "ClaudeAdapter requires 'prompt' parameter",
                error_type="missing_prompt"
            )

        if not (client := self._get_client()):
            return self.error_response(
                "ANTHROPIC_API_KEY not set or anthropic package not installed.",
                error_type="missing_api_key"
            )

        api_params = self._build_api_params(task_config)
        response_format = task_config.get(RESPONSE_FORMAT, "text")
        start_time = time.time()

        log.info(f"Using model: {api_params['model']}, temp: {api_params['temperature']}")

        try:
            response = await retry_with_backoff(
                coro_fn=lambda: client.messages.create(**api_params),
                max_retries=DEFAULT_MAX_RETRIES,
                base_delay=DEFAULT_RETRY_BASE_DELAY,
                retryable_check=lambda e: not is_auth_error(e),
                logger=log,
            )
        except Exception as e:
            if is_auth_error(e):
                return self.error_response(
                    f"Authentication failed: {e}",
                    error_type="authentication_failed"
                )
            return self.error_response(
                f"Claude API error: {e}",
                error_type="api_error"
            )

        response_time = time.time() - start_time
        content = response.content[0].text
        usage = response.usage
        total_tokens = usage.input_tokens + usage.output_tokens

        log.info(f"Completed in {response_time:.2f}s, tokens: {total_tokens}")

        output_files = {"raw_response": self.save_text_output(content, "response.txt")}

        if response_format == "json":
            parsed, error = parse_json_safe(content, log)
            output_files["parsed_output"] = self.save_json_output(
                parsed or {"raw": content, "parse_error": error},
                "parsed_output.json"
            )

        return self.success_response(
            summary=f"Claude completed. Response: {truncate_string(content, 100)}",
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
