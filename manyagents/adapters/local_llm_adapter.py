"""Local LLM adapter for running open-source models via HuggingFace transformers."""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import AgentAdapter, AdapterResult
from manyagents.utils.helpers import truncate_string

log = logging.getLogger(__name__)

# Default model paths on the cluster
DEFAULT_MODEL_PATHS = {
    "llama-3.3-70b": "/network/weights/llama.var/llama_3.3/Llama-3.3-70B-Instruct",
    "llama-3.1-70b": "/network/weights/llama.var/llama_3.1/Meta-Llama-3.1-70B-Instruct",
    "llama-3.1-8b": "/network/weights/llama.var/llama_3.1/Meta-Llama-3.1-8B-Instruct",
    "llama-3.1-405b-fp8": "/network/weights/llama.var/llama_3.1/Meta-Llama-3.1-405B-Instruct-FP8",
    "olmo-7b": "/network/weights/olmo/OLMo-7B-Twin-2T",
    "olmoe-1b-7b": "/network/weights/olmoe/OLMoE-1B-7B-0924",
}


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
        """
        Initialize LocalLLM adapter.

        Args:
            model_name: Model identifier (e.g., "llama-3.1-8b") or full path
            device_map: Device placement strategy ("auto", "cuda:0", etc.)
        """
        super().__init__("local_llm")
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device_map = device_map
        self._model = None
        self._tokenizer = None
        self._loaded_model_path = None

    def _resolve_model_path(self, model_name: str) -> str:
        """Resolve model name to full path."""
        if model_name in DEFAULT_MODEL_PATHS:
            return DEFAULT_MODEL_PATHS[model_name]
        if os.path.exists(model_name):
            return model_name
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(DEFAULT_MODEL_PATHS.keys())} "
            "or provide a full path."
        )

    def _load_model(self, model_path: str):
        """Lazily load model and tokenizer."""
        if self._model is not None and self._loaded_model_path == model_path:
            return  # Already loaded

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "transformers package not installed. Run 'uv add transformers accelerate'"
            )

        log.info(f"Loading model from {model_path}...")
        load_start = time.time()

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            trust_remote_code=True,
        )

        load_time = time.time() - load_start
        log.info(f"Model loaded in {load_time:.1f}s")
        self._loaded_model_path = model_path

    def _build_chat_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Build chat-formatted prompt for instruct models."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Use tokenizer's chat template if available
        if self._tokenizer and hasattr(self._tokenizer, 'apply_chat_template'):
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback for models without chat template
            formatted = ""
            if system_prompt:
                formatted += f"System: {system_prompt}\n\n"
            formatted += f"User: {prompt}\n\nAssistant:"
            return formatted

    def _generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool
    ) -> str:
        """Generate response from model."""
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip()

    async def run(
        self,
        task_config: Dict[str, Any],
        input_files: Dict[str, Path],
        input_data: Optional[Any] = None
    ) -> AdapterResult:
        """
        Execute local LLM inference with given configuration.

        Args:
            task_config: Dictionary with parameters:
                - prompt (required): The main prompt/question
                - model (optional): Model name or path (default: "llama-3.1-8b")
                - max_new_tokens (optional): Max tokens to generate (default: 2000)
                - temperature (optional): Sampling temperature (default: 0.1)
                - system_prompt (optional): System message
            input_files: Files from previous workflow steps (currently unused)
            input_data: Data from previous steps (currently unused)

        Returns:
            AdapterResult with success status, summary, output_files, and metadata
        """
        log.info(f"LocalLLMAdapter executing with config: {task_config}")

        # Validate inputs
        if "prompt" not in task_config:
            error_msg = "LocalLLMAdapter requires 'prompt' parameter in task_config"
            log.error(error_msg)
            return self.error_response(error_msg, error_type="missing_prompt")

        # Get configuration
        model_name = task_config.get("model", self.model_name)
        max_new_tokens = task_config.get("max_new_tokens", self.DEFAULT_MAX_NEW_TOKENS)
        temperature = task_config.get("temperature", self.DEFAULT_TEMPERATURE)
        system_prompt = task_config.get("system_prompt")
        do_sample = temperature > 0

        try:
            # Resolve and load model
            model_path = self._resolve_model_path(model_name)

            # Run model loading and inference in thread to not block event loop
            def _run_inference():
                self._load_model(model_path)
                formatted_prompt = self._build_chat_prompt(
                    task_config["prompt"],
                    system_prompt
                )
                return self._generate(formatted_prompt, max_new_tokens, temperature, do_sample)

            start_time = time.time()
            content = await asyncio.to_thread(_run_inference)
            response_time = time.time() - start_time

            log.info(f"Inference completed in {response_time:.2f}s")

            # Save output
            output_files = {"raw_response": self.save_text_output(content, "response.txt")}

            return self.success_response(
                summary=f"Local LLM inference completed. Response: {truncate_string(content, 100)}",
                output_files=output_files,
                metadata={
                    "model": model_name,
                    "model_path": model_path,
                    "response_time": response_time,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
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

    def unload_model(self):
        """Explicitly unload model to free GPU memory."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._loaded_model_path = None

            # Force CUDA memory cleanup
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            log.info("Model unloaded and GPU memory freed")
