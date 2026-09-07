"""HuggingFace adapter for running models via transformers.

Supports both local weight paths and HuggingFace Hub IDs.
"""

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .base import AgentAdapter, AdapterResult
from manyagents.utils.helpers import truncate_string
from manyagents import inference

log = logging.getLogger(__name__)


class HFAdapter(AgentAdapter):
    """Adapter for HuggingFace transformers inference.

    Supports loading models from local paths (e.g., /models/)
    or HuggingFace Hub IDs (e.g., Qwen/Qwen3-4B).
    """

    DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
    DEFAULT_MAX_NEW_TOKENS = 2000
    DEFAULT_TEMPERATURE = 0.1

    def __init__(self, model_name: Optional[str] = None, device_map: str = "auto"):
        super().__init__("hf")
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device_map = device_map

    async def run(
        self,
        task_config: Dict[str, Any],
        input_files: Dict[str, Path],
        input_data: Optional[Any] = None,
    ) -> AdapterResult:
        log.info(f"HFAdapter executing with config: {task_config}")

        if "prompt" not in task_config:
            return self.error_response(
                "HFAdapter requires 'prompt' parameter in task_config",
                error_type="missing_prompt",
            )

        model_name = task_config.get("model", self.model_name)
        max_new_tokens = task_config.get("max_new_tokens", self.DEFAULT_MAX_NEW_TOKENS)
        temperature = task_config.get("temperature", self.DEFAULT_TEMPERATURE)
        system_prompt = task_config.get("system_prompt")
        capture_hidden_states = task_config.get("capture_hidden_states", False)
        build_trace = task_config.get("build_trace", False)
        layers = task_config.get("layers")
        step_delimiter = task_config.get("step_delimiter", "\n")
        segmentation = task_config.get("segmentation", "delimiter")

        try:
            model_path = inference.resolve_model_path(
                model_name, task_config.get("available_models"),
            )

            def _run_inference():
                model, tokenizer, _ = inference.get_model(
                    model_path, device_map=self.device_map,
                )

                if build_trace or capture_hidden_states:
                    from manyagents.schemas.reasoning import TaskInfo

                    task = TaskInfo(
                        dataset=task_config.get("dataset", "unknown"),
                        task_id=task_config.get("task_id", uuid.uuid4().hex[:8]),
                        prompt=task_config["prompt"],
                        expected_answer=task_config.get("expected_answer"),
                        domain=task_config.get("domain"),
                        logic_type=task_config.get("logic_type"),
                    )
                    trace, hidden_states = inference.extract_trace(
                        model, tokenizer,
                        prompt=task_config["prompt"],
                        task=task,
                        model_name=model_name,
                        model_path=model_path,
                        system_prompt=system_prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        layers=layers,
                        step_delimiter=step_delimiter,
                        segmentation=segmentation,
                    )
                    return {
                        "text": trace.response_text,
                        "trace": trace,
                        "hidden_states": hidden_states,
                    }
                else:
                    formatted = inference.build_prompt(
                        tokenizer, task_config["prompt"], system_prompt,
                    )
                    text = inference.generate(
                        model, tokenizer, formatted,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                    return {"text": text, "trace": None, "hidden_states": None}

            start_time = time.time()
            result = await asyncio.to_thread(_run_inference)
            response_time = time.time() - start_time

            content = result["text"]
            log.info(f"Inference completed in {response_time:.2f}s")

            unique_id = uuid.uuid4().hex[:8]
            output_files: Dict[str, Any] = self.save_response(content, f"response_{unique_id}.txt")

            if result["hidden_states"] is not None:
                npz_path = self.output_dir / f"hidden_states_{unique_id}.npz"
                np.savez_compressed(npz_path, **result["hidden_states"])
                output_files["hidden_states"] = npz_path

            if result["trace"] is not None:
                trace_path = self.output_dir / f"trace_{unique_id}.json"
                trace_path.write_text(result["trace"].to_json())
                output_files["trace"] = trace_path

            return self.success_response(
                summary=f"HF inference completed. Response: {truncate_string(content, 100)}",
                output_files=output_files,
                metadata={
                    "model": model_name,
                    "model_path": model_path,
                    "response_time": response_time,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "captured_hidden_states": result["hidden_states"] is not None,
                    "built_trace": result["trace"] is not None,
                },
            )

        except ImportError as e:
            return self.error_response(
                f"Missing dependency: {e}",
                error_type="import_error",
                details=str(e),
            )

        except Exception as e:
            log.error(f"HF inference failed: {e}", exc_info=True)
            return self.error_response(
                f"HF inference failed: {e}",
                error_type="inference_error",
                details=str(e),
            )

    def unload_model(self):
        """Explicitly free GPU memory via the shared cache."""
        inference.clear_model_cache()
        log.info("Model cache cleared and GPU memory freed")
