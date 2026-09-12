"""Tool contract for the agentic loop.

A ``Tool`` pairs a JSON-schema declaration (what the model sees) with a callable
(what runs when the model calls it). The callable may be sync or async and returns
a string result that is fed back to the model as a ``role: "tool"`` message.

Compute does NOT live here — tools are thin wrappers that call out to the libraries
that own the work (for example, manyLatents for DR). manyAgents
owns the loop; the tool body delegates.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    """A named callable and its JSON Schema arguments, supplied by the caller."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the arguments object
    run: Callable[..., Any]     # (**kwargs) -> str | Awaitable[str]

    def to_openai_schema(self) -> dict[str, Any]:
        """Return the OpenAI function-tool declaration without the callable."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def invoke(self, arguments: dict[str, Any]) -> str:
        """Call the tool with keyword arguments, await if needed, and return text."""
        result = self.run(**arguments)
        if inspect.isawaitable(result):
            result = await result
        return result if isinstance(result, str) else str(result)


def to_openai_schemas(tools: list[Tool]) -> list[dict[str, Any]]:
    """Return an OpenAI function-tool declaration for each supplied tool."""
    return [t.to_openai_schema() for t in tools]


# A no-op tool for CI / loop smoke tests — exercises the call path with no deps.
def _echo(text: str) -> str:
    return f"echo: {text}"


MOCK_TOOL = Tool(
    name="echo",
    description="Echo back the provided text. Use this to test tool calling.",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Text to echo"}},
        "required": ["text"],
    },
    run=_echo,
)
