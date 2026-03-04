"""ReasoningTrace: Base LLM output schema for reasoning trace capture.

Captures the structured output of an LLM reasoning run — one model on one task.
Designed as the interface between adapters (which produce traces) and downstream
geometry pipelines (which consume them).

Two storage layers via TraceStore:
  - Metadata: JSONL (one ReasoningTrace per line) — text, steps, outcome, timing
  - Tensors:  npz files keyed by trace_id — hidden states from local models only

Three dataclasses, one per concern:
  ModelInfo      →  which model (reused across runs, carries cluster path)
  TaskInfo       →  which task (the experiment's independent variable)
  ReasoningStep  →  one step in the chain (the unit manylatents operates on)

Outcome and token usage are flat fields on ReasoningTrace — no wrapper classes.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ModelBackend(str, Enum):
    """Which inference backend produced this trace."""
    LOCAL = "local"            # HF transformers (OLMo, LLaMA, etc.)
    ANTHROPIC = "anthropic"    # Claude API
    OPENAI = "openai"          # OpenAI API
    VLLM = "vllm"             # vLLM server


class StepKind(str, Enum):
    """What kind of reasoning step this is."""
    THINKING = "thinking"          # internal CoT / Claude extended thinking
    OUTPUT = "output"              # final answer text
    TOOL_CALL = "tool_call"        # agent tool invocation
    TOOL_RESULT = "tool_result"    # tool response


# ---------------------------------------------------------------------------
# Schema components — one class per distinct concern
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    """Which model produced this trace.

    Reused across all traces in a run. Carries the cluster weights path
    for local models and the generation config for reproducibility.
    """
    name: str                                    # "olmo-7b", "claude-sonnet-4-5-20250929"
    backend: ModelBackend
    path: Optional[str] = None                   # local weights path on cluster
    revision: Optional[str] = None               # checkpoint sha / API model version
    generation_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["backend"] = self.backend.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelInfo:
        return cls(**{**d, "backend": ModelBackend(d["backend"])})


@dataclass
class TaskInfo:
    """What task this trace was generated for.

    This is the experiment's independent variable — varies per trace.
    The domain and logic_type labels are used for clustering/evaluation.
    """
    dataset: str                                 # "gsm8k", "math", "arc", "custom_logic"
    task_id: str                                 # unique within dataset
    prompt: str                                  # the full prompt sent to the model
    expected_answer: Optional[str] = None
    domain: Optional[str] = None                 # "math", "code", "science", "logic"
    logic_type: Optional[str] = None             # "deduction", "induction", "abduction"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskInfo:
        return cls(**d)


@dataclass
class ReasoningStep:
    """One step in a reasoning trajectory.

    The unit that manylatents operates on: hidden states are extracted
    per step, velocity = diff between consecutive steps.

    For local models: hidden states live in a companion .npz file.
    The has_hidden_states flag indicates whether tensor data exists.
    """
    index: int
    text: str
    kind: StepKind = StepKind.THINKING
    token_count: int = 0
    has_hidden_states: bool = False
    layers_captured: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReasoningStep:
        return cls(**{**d, "kind": StepKind(d["kind"])})


# ---------------------------------------------------------------------------
# Top-level trace — outcome and token usage are flat fields
# ---------------------------------------------------------------------------

@dataclass
class ReasoningTrace:
    """A single reasoning trace from one model on one task.

    This is the unit of data flowing through the pipeline:
      adapters produce it → TraceStore persists it → geometry pipelines consume it.

    Geometry (velocity, curvature) is NOT stored here — that's downstream output.
    """
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    model: ModelInfo = field(default_factory=lambda: ModelInfo("unknown", ModelBackend.LOCAL))
    task: TaskInfo = field(default_factory=lambda: TaskInfo("unknown", "unknown", ""))
    steps: list[ReasoningStep] = field(default_factory=list)
    response_text: str = ""

    # Outcome — flat, no wrapper
    success: Optional[bool] = None
    judge: str = "none"                          # "self", "exact_match", "human", "none"
    answer_extracted: Optional[str] = None
    score: Optional[float] = None

    # Token usage — flat, no wrapper
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: Optional[int] = None        # Anthropic extended thinking
    total_tokens: int = 0

    # Timing
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    duration_ms: Optional[int] = None

    # Tensor bookkeeping
    has_tensors: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "model": self.model.to_dict(),
            "task": self.task.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "response_text": self.response_text,
            "success": self.success,
            "judge": self.judge,
            "answer_extracted": self.answer_extracted,
            "score": self.score,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "total_tokens": self.total_tokens,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "has_tensors": self.has_tensors,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReasoningTrace:
        return cls(
            trace_id=d["trace_id"],
            model=ModelInfo.from_dict(d["model"]),
            task=TaskInfo.from_dict(d["task"]),
            steps=[ReasoningStep.from_dict(s) for s in d["steps"]],
            response_text=d["response_text"],
            success=d.get("success"),
            judge=d.get("judge", "none"),
            answer_extracted=d.get("answer_extracted"),
            score=d.get("score"),
            input_tokens=d.get("input_tokens", 0),
            output_tokens=d.get("output_tokens", 0),
            thinking_tokens=d.get("thinking_tokens"),
            total_tokens=d.get("total_tokens", 0),
            timestamp=d["timestamp"],
            duration_ms=d.get("duration_ms"),
            has_tensors=d.get("has_tensors", False),
        )

    @classmethod
    def from_json(cls, line: str) -> ReasoningTrace:
        return cls.from_dict(json.loads(line))


# ---------------------------------------------------------------------------
# Adapter response → ReasoningTrace converters
# ---------------------------------------------------------------------------

def steps_from_anthropic_response(response: Any) -> list[ReasoningStep]:
    """Parse an Anthropic Messages API response into ReasoningSteps.

    Handles thinking blocks, text blocks, and tool_use blocks.

    Args:
        response: anthropic.types.Message or dict with "content" list.
    """
    steps: list[ReasoningStep] = []
    idx = 0
    content = response.content if hasattr(response, "content") else response["content"]

    for block in content:
        btype = block.type if hasattr(block, "type") else block["type"]

        if btype == "thinking":
            text = block.thinking if hasattr(block, "thinking") else block["thinking"]
            steps.append(ReasoningStep(index=idx, text=text, kind=StepKind.THINKING))
            idx += 1
        elif btype == "text":
            text = block.text if hasattr(block, "text") else block["text"]
            steps.append(ReasoningStep(index=idx, text=text, kind=StepKind.OUTPUT))
            idx += 1
        elif btype == "tool_use":
            name = block.name if hasattr(block, "name") else block["name"]
            inp = block.input if hasattr(block, "input") else block["input"]
            steps.append(ReasoningStep(
                index=idx,
                text=json.dumps({"tool": name, "input": inp}),
                kind=StepKind.TOOL_CALL,
            ))
            idx += 1

    return steps


def trace_from_anthropic(
    response: Any,
    task: TaskInfo,
    model_name: str = "claude-sonnet-4-5-20250929",
    duration_ms: Optional[int] = None,
) -> ReasoningTrace:
    """Build a ReasoningTrace from an Anthropic API response + task info."""
    usage = response.usage if hasattr(response, "usage") else response.get("usage", {})
    input_tok = getattr(usage, "input_tokens", None) or usage.get("input_tokens", 0)
    output_tok = getattr(usage, "output_tokens", None) or usage.get("output_tokens", 0)

    steps = steps_from_anthropic_response(response)
    response_text = "\n".join(s.text for s in steps if s.kind == StepKind.OUTPUT)

    return ReasoningTrace(
        model=ModelInfo(name=model_name, backend=ModelBackend.ANTHROPIC),
        task=task,
        steps=steps,
        response_text=response_text,
        input_tokens=input_tok,
        output_tokens=output_tok,
        total_tokens=input_tok + output_tok,
        duration_ms=duration_ms,
    )


# ---------------------------------------------------------------------------
# TraceStore: JSONL metadata + companion tensor directory
# ---------------------------------------------------------------------------

class TraceStore:
    """Batch read/write for reasoning traces.

    Directory layout:
        run_dir/
        ├── traces.jsonl          # one ReasoningTrace JSON per line
        └── tensors/              # optional — local models only
            ├── {trace_id}.npz
            └── ...

    Write:
        with TraceStore("path/to/run") as store:
            store.append(trace)
            store.append(trace, hidden_states={"steps": array})

    Read:
        store = TraceStore("path/to/run", mode="r")
        for trace in store:
            tensors = store.load_tensors(trace.trace_id)
    """

    def __init__(self, run_dir: str | Path, mode: str = "w"):
        self.run_dir = Path(run_dir)
        self.mode = mode
        self._jsonl_path = self.run_dir / "traces.jsonl"
        self._tensor_dir = self.run_dir / "tensors"
        self._handle = None

        if mode == "w":
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._tensor_dir.mkdir(exist_ok=True)
            self._handle = open(self._jsonl_path, "a", encoding="utf-8")
        elif mode == "r":
            if not self._jsonl_path.exists():
                raise FileNotFoundError(f"No traces.jsonl in {self.run_dir}")

    def append(
        self,
        trace: ReasoningTrace,
        hidden_states: Optional[dict[str, Any]] = None,
    ) -> None:
        """Append a trace. Optionally save companion tensor data as .npz."""
        if self.mode != "w":
            raise RuntimeError("TraceStore not opened for writing")

        if hidden_states is not None:
            try:
                import numpy as np
                np.savez_compressed(
                    self._tensor_dir / f"{trace.trace_id}.npz",
                    **hidden_states,
                )
                trace.has_tensors = True
                for step in trace.steps:
                    step.has_hidden_states = True
            except ImportError:
                pass

        self._handle.write(trace.to_json() + "\n")
        self._handle.flush()

    def __iter__(self) -> Iterator[ReasoningTrace]:
        with open(self._jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield ReasoningTrace.from_json(line)

    def __len__(self) -> int:
        count = 0
        with open(self._jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def load_tensors(self, trace_id: str) -> Optional[dict[str, Any]]:
        """Load companion .npz for a trace. Returns None if absent."""
        npz_path = self._tensor_dir / f"{trace_id}.npz"
        if not npz_path.exists():
            return None
        try:
            import numpy as np
            return dict(np.load(npz_path, allow_pickle=False))
        except ImportError:
            return None

    def close(self) -> None:
        if self._handle and not self._handle.closed:
            self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def summary(self) -> dict[str, Any]:
        """Quick stats about the store."""
        total = 0
        with_tensors = 0
        backends: dict[str, int] = {}
        datasets: dict[str, int] = {}
        success_count = 0
        failure_count = 0

        for trace in self:
            total += 1
            if trace.has_tensors:
                with_tensors += 1
            b = trace.model.backend.value
            backends[b] = backends.get(b, 0) + 1
            d = trace.task.dataset
            datasets[d] = datasets.get(d, 0) + 1
            if trace.success is True:
                success_count += 1
            elif trace.success is False:
                failure_count += 1

        return {
            "total_traces": total,
            "with_tensors": with_tensors,
            "backends": backends,
            "datasets": datasets,
            "success": success_count,
            "failure": failure_count,
            "unjudged": total - success_count - failure_count,
        }
