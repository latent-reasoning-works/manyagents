"""vLLM adapter — fast generation via vLLM, hidden states via an HF forward pass.

vLLM is an inference engine: it cannot be trained through and does not expose
clean per-decode-step hidden states. So this adapter splits responsibilities —
vLLM generates the trajectory fast (and returns the exact token ids it saw), and
when hidden states are requested an HF model runs a single teacher-forced forward
pass over ``prompt + completion`` to recover them. For a causal model the state
at each position equals the decode-time state, so the captured trajectory matches
what ``HFAdapter`` would produce had HF generated the same tokens.

Every vLLM knob is settable through ``task_config``:
  - sampling: ``temperature``, ``top_p``, ``top_k``, ``max_new_tokens``, ``seed``,
    ``stop``, plus a free-form ``sampling_params`` dict (min_p, repetition_penalty,
    presence_penalty, n, logprobs, ...).
  - engine: ``max_num_batched_tokens``, ``max_num_seqs``, ``max_model_len``,
    ``gpu_memory_utilization``, ``tensor_parallel_size``, ``enforce_eager``,
    ``dtype``, plus a free-form ``engine_kwargs`` dict (quantization, ...).
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

# Engine-construction knobs pulled out of task_config (see get_vllm_engine).
_ENGINE_KEYS = (
    "dtype",
    "max_num_batched_tokens",
    "max_num_seqs",
    "max_model_len",
    "gpu_memory_utilization",
    "tensor_parallel_size",
    "enforce_eager",
    "trust_remote_code",
    "seed",
)


class VLLMAdapter(AgentAdapter):
    """Adapter for vLLM inference (generation) + HF hidden-state capture.

    Loads models from local paths (e.g. /network/weights/) or HF Hub IDs
    (e.g. Qwen/Qwen3-0.6B), same as ``HFAdapter``.
    """

    DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
    DEFAULT_MAX_NEW_TOKENS = 512
    DEFAULT_TEMPERATURE = 0.7

    def __init__(self, model_name: Optional[str] = None, device_map: str = "auto"):
        super().__init__("vllm")
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device_map = device_map

    def _engine_kwargs(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Collect engine-construction args from task_config (+ free-form dict)."""
        kwargs = {k: task_config[k] for k in _ENGINE_KEYS if k in task_config}
        kwargs.update(task_config.get("engine_kwargs", {}) or {})
        return kwargs

    async def run(
        self,
        task_config: Dict[str, Any],
        input_files: Dict[str, Path],
        input_data: Optional[Any] = None,
    ) -> AdapterResult:
        log.info(f"VLLMAdapter executing with config: {task_config}")

        if "prompt" not in task_config:
            return self.error_response(
                "VLLMAdapter requires 'prompt' parameter in task_config",
                error_type="missing_prompt",
            )

        model_name = task_config.get("model", self.model_name)
        max_new_tokens = task_config.get("max_new_tokens", self.DEFAULT_MAX_NEW_TOKENS)
        temperature = task_config.get("temperature", self.DEFAULT_TEMPERATURE)
        top_p = task_config.get("top_p", 1.0)
        top_k = task_config.get("top_k", -1)
        seed = task_config.get("seed")
        stop = task_config.get("stop")
        sampling_overrides = task_config.get("sampling_params")  # free-form dict
        system_prompt = task_config.get("system_prompt")
        capture_hidden_states = task_config.get("capture_hidden_states", False)
        build_trace = task_config.get("build_trace", False)
        layers = task_config.get("layers")
        step_delimiter = task_config.get("step_delimiter", "\n")
        segmentation = task_config.get("segmentation", "delimiter")

        engine_kwargs = self._engine_kwargs(task_config)

        try:
            model_path = inference.resolve_model_path(model_name)

            def _run_inference():
                engine = inference.get_vllm_engine(model_name, **engine_kwargs)

                if build_trace or capture_hidden_states:
                    # HF model (on GPU) supplies the hidden-state forward pass.
                    hf_model, tokenizer, _ = inference.get_model(
                        model_name, device_map=self.device_map,
                    )
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
                        hf_model, tokenizer,
                        prompt=task_config["prompt"],
                        task=task,
                        model_name=model_name,
                        model_path=model_path,
                        backend="vllm",
                        vllm_engine=engine,
                        sampling_params=sampling_overrides,
                        system_prompt=system_prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
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
                    # Pure generation — no HF model needed; vLLM owns the tokenizer.
                    tokenizer = engine.get_tokenizer()
                    formatted = inference.build_prompt(
                        tokenizer, task_config["prompt"], system_prompt,
                    )
                    out = inference.vllm_generate(
                        formatted,
                        engine=engine,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        seed=seed,
                        stop=stop,
                        sampling_overrides=sampling_overrides,
                    )[0]
                    return {"text": out["text"], "trace": None, "hidden_states": None}

            start_time = time.time()
            result = await asyncio.to_thread(_run_inference)
            response_time = time.time() - start_time

            content = result["text"]
            log.info(f"vLLM inference completed in {response_time:.2f}s")

            unique_id = uuid.uuid4().hex[:8]
            output_files: Dict[str, Any] = {
                "raw_response": self.save_text_output(content, f"response_{unique_id}.txt"),
            }

            if result["hidden_states"] is not None:
                npz_path = self.output_dir / f"hidden_states_{unique_id}.npz"
                np.savez_compressed(npz_path, **result["hidden_states"])
                output_files["hidden_states"] = npz_path

            if result["trace"] is not None:
                trace_path = self.output_dir / f"trace_{unique_id}.json"
                trace_path.write_text(result["trace"].to_json())
                output_files["trace"] = trace_path

            return self.success_response(
                summary=f"vLLM inference completed. Response: {truncate_string(content, 100)}",
                output_files=output_files,
                metadata={
                    "model": model_name,
                    "model_path": model_path,
                    "backend": "vllm",
                    "response_time": response_time,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "engine_kwargs": engine_kwargs,
                    "captured_hidden_states": result["hidden_states"] is not None,
                    "built_trace": result["trace"] is not None,
                },
            )

        except ImportError as e:
            return self.error_response(
                f"Missing dependency: {e}. Run 'uv pip install manyagents[vllm]'",
                error_type="import_error",
                details=str(e),
            )

        except Exception as e:
            log.error(f"vLLM inference failed: {e}", exc_info=True)
            return self.error_response(
                f"vLLM inference failed: {e}",
                error_type="inference_error",
                details=str(e),
            )

    def unload_model(self):
        """Explicitly free GPU memory via the shared caches."""
        inference.clear_model_cache()
        log.info("Model + vLLM engine caches cleared and GPU memory freed")
