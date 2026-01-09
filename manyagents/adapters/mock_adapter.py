"""
Mock adapter for testing experiment infrastructure.

Returns predictable responses without making any actual API calls or
requiring GPU resources. Useful for:
- Testing wandb integration
- Testing cluster submission
- CI/CD pipelines
- Debugging experiment logic
"""

import random
import time
from typing import Any, Dict

from manyagents.adapters.base import AgentAdapter


class MockAdapter(AgentAdapter):
    """
    Mock adapter that returns configurable test responses.

    Config options:
        response: str - Fixed response text (default: generates one)
        delay: float - Simulated processing delay in seconds (default: 0.1)
        fail_rate: float - Probability of simulated failure [0-1] (default: 0)
        mock_methods: list - Methods to include in response (default: [leiden, umap])
    """

    def __init__(self):
        super().__init__(name="mock")

    async def run(
        self,
        task_config: Dict[str, Any],
        input_files: Dict[str, Any],
        input_data: Any = None
    ) -> Dict[str, Any]:
        """Generate a mock response."""
        delay = task_config.get("delay", 0.1)
        fail_rate = task_config.get("fail_rate", 0.0)
        mock_methods = task_config.get("mock_methods", ["leiden", "umap", "pca"])

        # Simulated delay
        if delay > 0:
            time.sleep(delay)

        # Simulated failure
        if random.random() < fail_rate:
            return {
                "success": False,
                "summary": "Simulated failure (mock adapter)",
                "output_files": {},
                "metadata": {"mock": True, "simulated_failure": True}
            }

        # Generate response
        prompt = task_config.get("prompt", "")
        response = task_config.get("response")

        if response is None:
            # Generate a response mentioning the mock methods
            methods_text = ", ".join(mock_methods)
            response = f"""Based on your data description, I would recommend the following analysis methods:

1. **{mock_methods[0].upper()}** - This is an excellent choice for your data structure.
2. **{mock_methods[1].upper() if len(mock_methods) > 1 else 'PCA'}** - For dimensionality reduction and visualization.
3. **{mock_methods[2].upper() if len(mock_methods) > 2 else 'TSNE'}** - As an alternative visualization approach.

These methods are well-suited for analyzing the patterns in your dataset.
Mock adapter response (prompt length: {len(prompt)} chars).
"""

        return {
            "success": True,
            "summary": f"Mock response generated ({len(response)} chars)",
            "output_files": {
                "response": response
            },
            "metadata": {
                "mock": True,
                "prompt_length": len(prompt),
                "methods_mentioned": mock_methods,
                "model": "mock-v1"
            }
        }
