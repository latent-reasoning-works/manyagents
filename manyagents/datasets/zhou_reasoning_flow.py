"""Loader for Zhou et al. (2026) carrier-invariant logic dataset.

Dataset: MasterZhou/Reasoning-Flow (HuggingFace Hub)
Paper: "The Geometry of Reasoning" (arXiv:2510.09782, ICLR 2026)

The dataset contains 30 logic structures × (1 symbolic + 80 natural-language)
= 2430 sequences. Each sequence is a list of step-by-step natural deduction
proof steps, marked with [1], [2], etc.

Structure of data.json::

    {
        "logicA": [
            {"steps": ["[1] A→B", "[2] B→C", ...]},                 # symbolic (no topic/lang)
            {"topic": "weather_en", "lang": "en", "steps": [...]},   # natural-language
            ...
        ],
        "logicB": [...],
        ...
    }
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from manyagents.schemas.reasoning import TaskInfo

log = logging.getLogger(__name__)


def _strip_step_marker(text: str) -> str:
    """Remove leading [N] marker from a step string."""
    return re.sub(r"^\[\d+\]\s*", "", text).strip()


def _parse_topic_field(topic_field: str) -> tuple[str, str]:
    """Parse 'weather_en' into ('weather', 'en').

    Falls back to (topic_field, 'unknown') if no language suffix found.
    """
    known_langs = {"en", "zh", "de", "ja"}
    parts = topic_field.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in known_langs:
        return parts[0], parts[1]
    return topic_field, "unknown"


def load_zhou_reasoning_flow(
    n_samples: int | None = None,
    logic_types: list[str] | None = None,
    topics: list[str] | None = None,
    languages: list[str] | None = None,
    include_symbolic: bool = True,
    strip_markers: bool = True,
) -> list[dict]:
    """Load Zhou et al. carrier-invariant logic dataset.

    Downloads from HuggingFace Hub (MasterZhou/Reasoning-Flow) on first call.

    Args:
        n_samples: Limit number of samples (None = all).
        logic_types: Filter to these logic labels (e.g. ["logicA", "logicB"]).
        topics: Filter to these base topic labels (e.g. ["weather", "finance"]).
        languages: Filter to these language codes (e.g. ["en", "zh"]).
        include_symbolic: Include the symbolic-only entry per logic type
            (has no topic/lang). Default True.
        strip_markers: Remove [N] prefix from each step. Default True.

    Returns:
        List of dicts, each with:
            task_info: TaskInfo — for compatibility with manyagents schemas
            steps: list[str] — individual reasoning step texts
            logic_type: str — logic structure label (e.g. "logicA")
            topic: str — semantic carrier label (e.g. "weather"), or "symbolic"
            language: str — language code (e.g. "en"), or "symbolic"
    """
    raw_data = _load_raw()

    entries = []
    for logic_key, items in raw_data.items():
        if logic_types is not None and logic_key not in logic_types:
            continue

        for item in items:
            raw_steps = item["steps"]
            topic_field = item.get("topic")
            lang = item.get("lang")

            if topic_field is None:
                # Symbolic-only entry (no topic/lang)
                if not include_symbolic:
                    continue
                base_topic = "symbolic"
                language = "symbolic"
            else:
                base_topic, lang_from_topic = _parse_topic_field(topic_field)
                language = lang or lang_from_topic

            # Apply filters
            if topics is not None and base_topic not in topics:
                continue
            if languages is not None and language not in languages:
                continue

            # Process steps
            if strip_markers:
                steps = [_strip_step_marker(s) for s in raw_steps]
            else:
                steps = list(raw_steps)

            # Filter out empty steps
            steps = [s for s in steps if s]
            if not steps:
                continue

            task_id = f"{logic_key}_{base_topic}_{language}_{len(entries)}"
            task_info = TaskInfo(
                dataset="zhou_reasoning_flow",
                task_id=task_id,
                prompt=" ".join(raw_steps),
                domain=base_topic,
                logic_type=logic_key,
                metadata={"language": language, "n_steps": len(steps)},
            )

            entries.append({
                "task_info": task_info,
                "steps": steps,
                "logic_type": logic_key,
                "topic": base_topic,
                "language": language,
            })

    # Limit
    if n_samples is not None:
        entries = entries[:n_samples]

    log.info(
        f"Loaded {len(entries)} sequences "
        f"({len(set(e['logic_type'] for e in entries))} logic types, "
        f"{len(set(e['topic'] for e in entries))} topics, "
        f"{len(set(e['language'] for e in entries))} languages)"
    )

    return entries


def _load_raw() -> dict:
    """Load raw data dict from HuggingFace Hub."""
    import json
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "MasterZhou/Reasoning-Flow", "data.json", repo_type="dataset"
    )
    with open(path) as f:
        return json.load(f)
