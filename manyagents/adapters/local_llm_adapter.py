"""Local LLM adapter for running open-source models via HuggingFace transformers."""

import asyncio
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .base import AgentAdapter, AdapterResult
from manyagents.utils.helpers import truncate_string
from manyagents import inference

log = logging.getLogger(__name__)

# Global lock for model loading to prevent race conditions when multiple
# prompts are processed in parallel. Without this, concurrent calls to
# _load_model() can cause "Cannot copy out of meta tensor" errors.
_MODEL_LOAD_LOCK = threading.Lock()


class LocalLLMAdapter(AgentAdapter):
    """
    Adapter for local LLM inference using HuggingFace transformers.

    Supports loading models from local paths (e.g., /network/weights/).
    Designed for use on GPU clusters with pre-downloaded model weights.
    """

    DEFAULT_MODEL = "llama-3.1-8b"
    DEFAULT_MAX_NEW_TOKENS = 2000
    DEFAULT_TEMPERATURE = 0.1

    def __init__(self, model_name: Optional[str] = None, device_map: str = "auto"):
        super().__init__("local_llm")
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device_map = device_map
        self._hf_module = None
        self._model = None
        self._tokenizer = None
        self._loaded_model_path = None

    def _load_model(self, model_path: str):
        """Lazily load model and tokenizer (thread-safe)."""
        if self._model is not None and self._loaded_model_path == model_path:
            return

        with _MODEL_LOAD_LOCK:
            if self._model is not None and self._loaded_model_path == model_path:
                log.debug("Model already loaded by another thread")
                return

            log.info(f"Loading model from {model_path}...")
            load_start = time.time()

            self._model, self._tokenizer, self._hf_module = inference.load_model(
                model_path, device_map=self.device_map
            )

            log.info(f"Model loaded in {time.time() - load_start:.1f}s")
            self._loaded_model_path = model_path

    def _generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        capture_hidden_states: bool = False,
        layer_specs: Optional[List[str]] = None,
    ) -> dict:
        """Generate response, optionally capturing hidden states via hooks."""
        if capture_hidden_states:
            specs = inference.resolve_layer_specs(self.model_name, layer_specs)
            if specs:
                return inference.generate_with_hooks(
                    self._model, self._tokenizer, prompt,
                    layer_specs=specs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
        # Plain generation (no hidden states)
        text = inference.generate(
            self._model, self._tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return {"text": text, "hidden_states": None}

    async def run(
        self,
        task_config: Dict[str, Any],
        input_files: Dict[str, Path],
        input_data: Optional[Any] = None
    ) -> AdapterResult:
        log.info(f"LocalLLMAdapter executing with config: {task_config}")

        if "prompt" not in task_config:
            error_msg = "LocalLLMAdapter requires 'prompt' parameter in task_config"
            log.error(error_msg)
            return self.error_response(error_msg, error_type="missing_prompt")

        model_name = task_config.get("model", self.model_name)
        max_new_tokens = task_config.get("max_new_tokens", self.DEFAULT_MAX_NEW_TOKENS)
        temperature = task_config.get("temperature", self.DEFAULT_TEMPERATURE)
        system_prompt = task_config.get("system_prompt")
        capture_hidden_states = task_config.get("capture_hidden_states", False)
        build_trace = task_config.get("build_trace", False)
        layer_specs = task_config.get("layer_specs")

        try:
            model_path = inference.resolve_model_path(model_name)

            def _run_inference():
                self._load_model(model_path)
                formatted_prompt = inference.build_prompt(
                    self._tokenizer, task_config["prompt"], system_prompt
                )
                return self._generate(
                    formatted_prompt, max_new_tokens, temperature,
                    capture_hidden_states=capture_hidden_states,
                    layer_specs=layer_specs,
                )

            start_time = time.time()
            result = await asyncio.to_thread(_run_inference)
            response_time = time.time() - start_time

            content = result["text"]
            log.info(f"Inference completed in {response_time:.2f}s")

            unique_id = uuid.uuid4().hex[:8]
            response_file = f"response_{unique_id}.txt"

            output_files: Dict[str, Any] = {
                "raw_response": self.save_text_output(content, response_file),
            }

            # Save hidden states as .npz
            if result["hidden_states"] is not None:
                npz_path = self.output_dir / f"hidden_states_{unique_id}.npz"
                np.savez_compressed(npz_path, **result["hidden_states"])
                output_files["hidden_states"] = npz_path

            # Optionally build a ReasoningTrace
            if build_trace and result["hidden_states"] is not None:
                trace_path = self._build_and_save_trace(
                    text=content,
                    model_name=model_name,
                    model_path=model_path,
                    hidden_states=result["hidden_states"],
                    prompt=task_config["prompt"],
                    duration_ms=int(response_time * 1000),
                    unique_id=unique_id,
                )
                output_files["trace"] = trace_path

            return self.success_response(
                summary=f"Local LLM inference completed. Response: {truncate_string(content, 100)}",
                output_files=output_files,
                metadata={
                    "model": model_name,
                    "model_path": model_path,
                    "response_time": response_time,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "captured_hidden_states": result["hidden_states"] is not None,
                }
            )

        except ImportError as e:
            return self.error_response(
                f"Missing dependency: {e}. Run 'uv add transformers accelerate'",
                error_type="import_error",
                details=str(e)
            )

        except Exception as e:
            log.error(f"Local LLM inference failed: {e}", exc_info=True)
            return self.error_response(
                f"Local LLM inference failed: {e}",
                error_type="inference_error",
                details=str(e)
            )

    def _build_and_save_trace(
        self,
        text: str,
        model_name: str,
        model_path: str,
        hidden_states: Dict[str, np.ndarray],
        prompt: str,
        duration_ms: int,
        unique_id: str,
    ) -> Path:
        """Build a ReasoningTrace and save its JSON to disk."""
        from manyagents.schemas.reasoning import (
            ModelBackend, ModelInfo, TaskInfo,
            ReasoningStep, ReasoningTrace, StepKind,
        )

        first_hs = next(iter(hidden_states.values()))
        n_tokens = first_hs.shape[0]

        trace = ReasoningTrace(
            model=ModelInfo(
                name=model_name,
                backend=ModelBackend.LOCAL,
                path=model_path,
            ),
            task=TaskInfo(dataset="local", task_id=unique_id, prompt=prompt),
            steps=[
                ReasoningStep(
                    index=0,
                    text=text,
                    kind=StepKind.OUTPUT,
                    token_count=n_tokens,
                    has_hidden_states=True,
                ),
            ],
            response_text=text,
            has_tensors=True,
            duration_ms=duration_ms,
        )

        trace_path = self.output_dir / f"trace_{unique_id}.json"
        trace_path.write_text(trace.to_json())
        return trace_path

    def unload_model(self):
        """Explicitly unload model to free GPU memory."""
        if self._model is None:
            return

        del self._model
        del self._tokenizer
        self._hf_module = None
        self._model = None
        self._tokenizer = None
        self._loaded_model_path = None

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        log.info("Model unloaded and GPU memory freed")
