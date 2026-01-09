"""OpenAI adapter for LLM-based agent orchestration."""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI, AuthenticationError

from .base import AgentAdapter, AdapterResult
from manyagents.utils.helpers import truncate_string, parse_json_safe
from manyagents.utils.retry import retry_with_backoff, is_auth_error
from manyagents.utils.constants import (
    PROMPT, SYSTEM_PROMPT, TEMPERATURE, MAX_TOKENS, MODEL, RESPONSE_FORMAT,
    DEFAULT_MAX_RETRIES, DEFAULT_RETRY_BASE_DELAY,
)

log = logging.getLogger(__name__)


class OpenAIAdapter(AgentAdapter):
    """Adapter for OpenAI API integration."""

    DEFAULT_MODEL = "gpt-4o"
    DEFAULT_TEMPERATURE = 0.0
    DEFAULT_MAX_TOKENS = 2000

    def __init__(self):
        super().__init__("openai")
        api_key = os.getenv('OPENAI_API_KEY')

        if not api_key:
            log.warning("OPENAI_API_KEY not found. Set it before calling run().")

        self.client = AsyncOpenAI(api_key=api_key) if api_key else None

    def _build_messages(self, prompt: str, system_prompt: str | None = None) -> List[Dict[str, str]]:
        """Build message list for OpenAI API."""
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_api_params(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build API parameters from task configuration."""
        params = {
            "model": task_config.get(MODEL, self.DEFAULT_MODEL),
            "messages": self._build_messages(
                task_config[PROMPT],
                task_config.get(SYSTEM_PROMPT)
            ),
            "temperature": task_config.get(TEMPERATURE, self.DEFAULT_TEMPERATURE),
            "max_tokens": task_config.get(MAX_TOKENS, self.DEFAULT_MAX_TOKENS),
        }
        if task_config.get(RESPONSE_FORMAT) == "json_object":
            params["response_format"] = {"type": "json_object"}
        return params

    async def run(
        self,
        task_config: Dict[str, Any],
        input_files: Dict[str, Path],
        input_data: Optional[Any] = None
    ) -> AdapterResult:
        """Execute OpenAI API call with given configuration."""
        log.info(f"OpenAIAdapter executing with config: {task_config}")

        # Guard clauses
        if PROMPT not in task_config:
            return self.error_response(
                "OpenAIAdapter requires 'prompt' parameter",
                error_type="missing_prompt"
            )

        if not self.client:
            return self.error_response(
                "OPENAI_API_KEY not set.",
                error_type="missing_api_key"
            )

        api_params = self._build_api_params(task_config)
        response_format = task_config.get(RESPONSE_FORMAT, "text")
        start_time = time.time()

        log.info(f"Using model: {api_params['model']}, temp: {api_params['temperature']}")

        try:
            response = await retry_with_backoff(
                coro_fn=lambda: self.client.chat.completions.create(**api_params),
                max_retries=DEFAULT_MAX_RETRIES,
                base_delay=DEFAULT_RETRY_BASE_DELAY,
                retryable_check=lambda e: not isinstance(e, AuthenticationError),
                logger=log,
            )
        except AuthenticationError as e:
            return self.error_response(
                f"Authentication failed: {e}",
                error_type="authentication_failed"
            )
        except Exception as e:
            return self.error_response(
                f"OpenAI API error: {e}",
                error_type="api_error"
            )

        response_time = time.time() - start_time
        content = response.choices[0].message.content
        tokens = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        log.info(f"Completed in {response_time:.2f}s, tokens: {tokens['total_tokens']}")

        output_files = {"raw_response": self.save_text_output(content, "response.txt")}

        if response_format == "json_object":
            parsed, error = parse_json_safe(content, log)
            output_files["parsed_output"] = self.save_json_output(
                parsed or {"raw": content, "parse_error": error},
                "parsed_output.json"
            )

        return self.success_response(
            summary=f"OpenAI completed. Response: {truncate_string(content, 100)}",
            output_files=output_files,
            metadata={
                "model": api_params["model"],
                "tokens_used": tokens,
                "response_time": response_time,
                "finish_reason": response.choices[0].finish_reason,
                "response_format": response_format,
            }
        )
