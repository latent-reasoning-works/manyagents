"""Claude adapter for LLM-based agent orchestration."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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

    DEFAULT_MODEL = "claude-opus-4-8"
    DEFAULT_TEMPERATURE = 0.0
    DEFAULT_MAX_TOKENS = 4096

    def __init__(self, api_key: Optional[str] = None):
        super().__init__("claude")
        # Explicit key wins; else env. Lets callers inject a key without env vars.
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
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

    # ---- Agentic loop primitive (mirrors OpenAIAdapter.chat) ----------------
    #
    # The loop is OpenAI-shaped; each adapter translates at its own boundary.
    # OpenAI messages/tools  -> Anthropic messages/tools  (on the way in)
    # Anthropic tool_use blocks -> normalized {id,name,arguments}  (on the way out)

    @staticmethod
    def _to_anthropic_tools(tools: Optional[List[Dict[str, Any]]]):
        if not tools:
            return None
        out = []
        for t in tools:
            fn = t.get("function", t)
            out.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return out

    @staticmethod
    def _to_anthropic_messages(messages: List[Dict[str, Any]]):
        """Translate OpenAI-format messages to (system, anthropic_messages)."""
        system: Optional[str] = None
        out: List[Dict[str, Any]] = []
        for m in messages:
            role = m["role"]
            if role == "system":
                system = m["content"] if system is None else f"{system}\n\n{m['content']}"
            elif role == "tool":
                # OpenAI tool result -> Anthropic user turn with a tool_result block
                out.append({"role": "user", "content": [{
                    "type": "tool_result",
                    "tool_use_id": m["tool_call_id"],
                    "content": m["content"],
                }]})
            elif role == "assistant" and m.get("tool_calls"):
                blocks: List[Dict[str, Any]] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m["tool_calls"]:
                    fn = tc["function"]
                    try:
                        args = json.loads(fn["arguments"]) if fn.get("arguments") else {}
                    except json.JSONDecodeError:
                        args = {}
                    blocks.append({
                        "type": "tool_use", "id": tc["id"],
                        "name": fn["name"], "input": args,
                    })
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": role, "content": m["content"]})
        return system, out

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,  # accepted, ignored (Opus 4.7+ reject it)
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """One Claude turn for the agentic loop. Returns the OpenAI-shaped turn
        the loop expects: {"message", "tool_calls": [{id,name,arguments}], "content"}.
        """
        client = self._get_client()
        if client is None:
            raise RuntimeError(
                "Claude unavailable — set ANTHROPIC_API_KEY (or pass api_key) "
                "and install the anthropic package."
            )

        system, anthropic_messages = self._to_anthropic_messages(messages)
        params: Dict[str, Any] = {
            "model": model or self.DEFAULT_MODEL,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or self.DEFAULT_MAX_TOKENS,
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = self._to_anthropic_tools(tools)

        response = await retry_with_backoff(
            coro_fn=lambda: client.messages.create(**params),
            max_retries=DEFAULT_MAX_RETRIES,
            base_delay=DEFAULT_RETRY_BASE_DELAY,
            retryable_check=lambda e: not is_auth_error(e),
            logger=log,
        )

        text = ""
        tool_calls: List[Dict[str, Any]] = []
        assistant_blocks: List[Dict[str, Any]] = []
        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text = block.text
                assistant_blocks.append({"type": "text", "text": block.text})
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": json.dumps(block.input or {}),  # loop json.loads() this
                })

        # Rebuild the assistant turn in OpenAI shape so the loop can append it and
        # _to_anthropic_messages can round-trip it on the next call.
        assistant_msg: Dict[str, Any] = {"role": "assistant", "content": text}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in tool_calls
            ]
        return {"message": assistant_msg, "tool_calls": tool_calls, "content": text}

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
        # Join text blocks, skipping thinking blocks.
        text_blocks = []
        for block in response.content:
            btype = block.type if hasattr(block, "type") else block.get("type")
            if btype == "text":
                text_blocks.append(block.text if hasattr(block, "text") else block["text"])
        content = "\n".join(text_blocks)
        usage = response.usage
        total_tokens = usage.input_tokens + usage.output_tokens

        log.info(f"Completed in {response_time:.2f}s, tokens: {total_tokens}")

        output_files = self.save_response(content)

        if response_format == "json":
            parsed, error = parse_json_safe(content, log)
            output_files["parsed_output"] = self.save_json_output(
                parsed or {"raw": content, "parse_error": error},
                "parsed_output.json"
            )

        # Optionally build a ReasoningTrace
        if task_config.get("build_trace", False):
            from manyagents.schemas.reasoning import TaskInfo, trace_from_anthropic

            task = TaskInfo(
                dataset=task_config.get("dataset", "unknown"),
                task_id=task_config.get("task_id", f"claude_{int(time.time())}"),
                prompt=task_config[PROMPT],
                expected_answer=task_config.get("expected_answer"),
                domain=task_config.get("domain"),
                logic_type=task_config.get("logic_type"),
            )
            trace = trace_from_anthropic(
                response, task,
                model_name=api_params["model"],
                duration_ms=int(response_time * 1000),
            )
            trace_path = self.output_dir / "trace.json"
            trace_path.write_text(trace.to_json())
            output_files["trace"] = trace_path

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
