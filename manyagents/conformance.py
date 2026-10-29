"""Score a model as a manyRuns driver against the seat's conformance cases, offline.

    manyagents-conformance path/to/driver_conformance.json --agent ollama --model qwen3:1.7b

manyRuns ships the cases (``tests/fixtures/driver_conformance.json``): what the scientist
said, the exact offer the seat would send, and an ``intent`` label. This asks the model once
per case, with the offer as one tool, and scores three things:

- ``legal_rate``: calls the seat would accept, over calls made (``seat.check``);
- ``intent_rate``: over cases that ask for a change, a legal call that moves a labelled
  parameter the labelled way (or to the labelled value);
- ``declined_rate``: over cases where no change is right (``none``) or the offer cannot do it
  (``outside``), no call at all.

The repository's contract holds: a case whose call failed is counted in ``errors`` and makes
every rate ``None`` (``n/a``), never a narrowed denominator; a run where no case succeeded
exits nonzero.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from manyagents.drive import TOOL_SYSTEM
from manyagents.seat import PROTOCOLS, check, offer_tool

Chat = Callable[..., Awaitable[dict]]


def load_cases(path: Path | str) -> list[dict]:
    """The cases, refused when their protocol is not one this client speaks."""
    doc = json.loads(Path(path).read_text())
    if doc.get("protocol_version") not in PROTOCOLS:
        raise ValueError(f"fixtures are for protocol {doc.get('protocol_version')!r}; this "
                         f"client speaks {sorted(PROTOCOLS)}")
    cases = doc.get("cases") or []
    missing = [c.get("id") for c in cases if "offer" not in c]
    if missing:
        raise ValueError(f"cases without an `offer` (older fixtures?): {missing[:5]}")
    return cases


def intent_met(case: dict, params: Optional[dict]) -> Optional[bool]:
    """Whether the call does what the case's ``intent`` asks. None when the case is not a move."""
    intent = case.get("intent")
    if not isinstance(intent, dict):
        return None
    if not params or check(case["offer"], params):
        return False
    hit = [k for k in params if k in intent.get("params", ())]
    if not hit:
        return False
    name, value = hit[0], params[hit[0]]
    if intent.get("value") is not None:
        return value == intent["value"]
    if intent.get("direction"):
        current = case["offer"]["params"][name].get("current")
        if not isinstance(current, (int, float)) or isinstance(value, str):
            return False
        return (value > current) == (intent["direction"] == "up")
    return True


async def score_case(case: dict, chat: Chat, model: str, *, knobs: Optional[dict] = None,
                     think: bool = False) -> dict:
    row: dict[str, Any] = {"id": case["id"], "said": case["said"],
                           "kind": case["intent"] if isinstance(case["intent"], str) else "move"}
    try:
        turn = await chat([{"role": "system", "content": TOOL_SYSTEM},
                           {"role": "user", "content": case["said"]}],
                          tools=[offer_tool(case["offer"], knobs)], model=model,
                          temperature=0.0, max_tokens=2048 if think else 256,
                          reasoning_effort=None if think else "none")
    except Exception as error:  # noqa: BLE001 - recorded as an error, never as a score
        row["error"] = f"{type(error).__name__}: {error}"
        return row
    params = None
    for call in turn.get("tool_calls") or []:
        try:
            params = json.loads(call.get("arguments") or "{}")
        except json.JSONDecodeError:
            params = {"_unparsed": call.get("arguments")}
        break
    row["called"] = params is not None
    row["params"] = params
    row["refusal"] = check(case["offer"], params) if params is not None else None
    row["legal"] = None if params is None else not row["refusal"]
    row["intent_met"] = intent_met(case, params)
    row["declined"] = None if row["kind"] == "move" else params is None
    return row


def summarize(rows: list[dict]) -> dict:
    errors = sum(1 for r in rows if "error" in r)
    ok = [r for r in rows if "error" not in r]
    calls = [r for r in ok if r["called"]]
    moves = [r for r in ok if r["kind"] == "move"]
    stays = [r for r in ok if r["kind"] != "move"]

    def rate(hits: int, n: int) -> Optional[float]:
        return None if errors or not n else hits / n

    return {"cases": len(rows), "errors": errors,
            "calls": len(calls), "legal_rate": rate(sum(r["legal"] for r in calls), len(calls)),
            "moves": len(moves), "intent_rate": rate(sum(bool(r["intent_met"]) for r in moves),
                                                     len(moves)),
            "stays": len(stays), "declined_rate": rate(sum(r["declined"] for r in stays),
                                                       len(stays))}


async def run(path: Path | str, chat: Chat, model: str, *, knobs: Optional[dict] = None,
              think: bool = False) -> tuple[list[dict], dict]:
    rows = [await score_case(c, chat, model, knobs=knobs, think=think) for c in load_cases(path)]
    return rows, summarize(rows)


def table(rows: list[dict], summary: dict, model: str) -> str:
    def fmt(v: Optional[float]) -> str:
        return "n/a" if v is None else f"{v:.2f}"

    lines = [f"{model}: legal {fmt(summary['legal_rate'])} ({summary['calls']} calls) · intent "
             f"{fmt(summary['intent_rate'])} ({summary['moves']} moves) · declined "
             f"{fmt(summary['declined_rate'])} ({summary['stays']}) · errors {summary['errors']}"]
    for r in rows:
        if "error" in r:
            lines.append(f"  {r['id']:<24} ERROR {r['error']}")
            continue
        verdict = ("ok" if r["intent_met"] else "miss") if r["kind"] == "move" else \
            ("declined" if r["declined"] else "acted")
        legal = "" if r["legal"] in (None, True) else f"  refused: {'; '.join(r['refusal'])}"
        lines.append(f"  {r['id']:<24} {r['kind']:<8} {verdict:<8} {r['params']}{legal}")
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="manyagents-conformance",
                                     description=__doc__.split("\n")[0])
    parser.add_argument("fixtures", help="manyRuns' driver_conformance.json")
    parser.add_argument("--agent", default="ollama")
    parser.add_argument("--model", required=True)
    parser.add_argument("--knobs", help="JSON file: knob name -> what it means")
    parser.add_argument("--think", action="store_true", help="let a thinking model think")
    parser.add_argument("--json", help="write rows and summary here")
    args = parser.parse_args(argv)

    from manyagents.adapters import ADAPTER_REGISTRY

    if args.agent not in ADAPTER_REGISTRY:
        parser.error(f"unknown agent {args.agent!r}; known: {sorted(ADAPTER_REGISTRY)}")
    adapter = ADAPTER_REGISTRY[args.agent]()
    if not hasattr(adapter, "chat"):
        parser.error(f"adapter {args.agent!r} has no chat(); it cannot be scored as a driver")
    knobs = json.loads(Path(args.knobs).read_text()) if args.knobs else None
    try:
        rows, summary = asyncio.run(run(args.fixtures, adapter.chat, args.model, knobs=knobs,
                                        think=args.think))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(table(rows, summary, args.model))
    if args.json:
        Path(args.json).write_text(json.dumps({"model": args.model, "think": args.think,
                                               "summary": summary, "rows": rows}, indent=1))
    return 0 if summary["errors"] < summary["cases"] else 1


if __name__ == "__main__":
    sys.exit(main())
