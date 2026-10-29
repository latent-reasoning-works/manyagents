"""Drive a manyRuns run from a model: attach to its seat, listen, answer, and act at the level
the scientist allows.

    manyagents-drive outputs/<project> --agent ollama --model qwen3:8b
    manyagents-drive outputs/<project> --agent ollama --model qwen3:8b \\
        --goal "make the trajectory read as clear branches"     # unattended

The level is the scientist's (``ctrl+g`` on the run screen, or ``MANYRUNS_DRIVER_LEVEL`` at
launch); this loop only reads it from ``control`` events and acts within it:

- a message ending in ``?`` gets a short prose answer first;
- then one parameter change from the model, sent as the strongest verb the level allows:
  ``apply`` at drive, ``propose`` at propose, a sentence (``say``) at interpret;
- a refusal is fed back to the model with its reason. With a person watching, it says what it
  would have done and waits; unattended (``goal``), it retries up to ``max_tries`` and then
  accepts what is there, because nobody will step in;
- after ``max_applies`` applied changes it accepts the step.

``chat`` is any adapter's ``chat`` coroutine (``ADAPTER_REGISTRY[agent]().chat``), so the
model behind it is the caller's choice; ``knobs`` is the caller's card of what each knob
means. Nothing here knows any particular method's parameters.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from manyagents.seat import Seat, SeatError, offer_tool

TOOL_SYSTEM = ("You set parameters of one analysis step for a scientist. Call the tool with the "
               "one parameter change that best does what they asked, inside each parameter's "
               "range. If nothing on offer can do it, do not call the tool.")
TALK_SYSTEM = ("You help a scientist read an analysis while it runs. Answer in at most two short "
               "sentences. You can see the step's current parameters.")

Chat = Callable[..., Awaitable[dict]]


async def _choose(chat: Chat, model: str, offer: dict, request: str,
                  knobs: Optional[dict]) -> Optional[dict]:
    """One tool call from the model, as params, or None when it made none."""
    turn = await chat([{"role": "system", "content": TOOL_SYSTEM},
                       {"role": "user", "content": request}],
                      tools=[offer_tool(offer, knobs)], model=model, temperature=0.0,
                      max_tokens=256, reasoning_effort="none")
    for call in turn.get("tool_calls") or []:
        try:
            params = json.loads(call.get("arguments") or "{}")
        except json.JSONDecodeError:
            continue
        if isinstance(params, dict) and params:
            return params
    return None


async def _talk(chat: Chat, model: str, offer: Optional[dict], question: str) -> str:
    current = {n: p.get("current") for n, p in ((offer or {}).get("params") or {}).items()}
    turn = await chat([{"role": "system", "content": TALK_SYSTEM},
                       {"role": "user", "content": f"Current parameters: {current}. {question}"}],
                      model=model, temperature=0.0, max_tokens=160, reasoning_effort="none")
    return (turn.get("content") or "").strip()


async def drive(seat: Seat, chat: Chat, *, model: str, chooser: str, bound: str = "schema",
                goal: str = "", max_applies: int = 1, max_tries: int = 3,
                knobs: Optional[dict] = None,
                log: Callable[[str], Any] = lambda line: None) -> dict:
    """Run one session to its end. Returns the counts of what was done, for the caller's record."""
    hello = seat.hello(chooser, bound)
    level, offer = hello.get("level", "interpret"), hello.get("offer")
    want, applies, tries, reason = goal, 0, 0, ""
    done = {"applied": 0, "proposed": 0, "said": 0, "refused": 0, "accepted": False}
    log(f"attached as {chooser} at {level}")

    def send(request: dict) -> dict:
        reply = seat.ask(request)
        if not reply.get("ok"):
            done["refused"] += 1
            log(f"refused {request.get('verb')}: {reply.get('error')}")
        return reply

    while True:
        # Read every queued event before acting: a refit's offer and a switch to manual
        # can arrive together, and acting on the first would send a move the seat refuses.
        # Unattended at interpret there is nobody to say anything to, so wait for a level.
        if offer and want and (level != "interpret" or not goal) and not seat.drain():
            if applies >= max_applies and level == "drive":
                if send({"verb": "apply", "legal_id": offer["legal_id"],
                         "decision": "accept"}).get("ok"):
                    done["accepted"] = True
                want, offer = "", None
                continue
            if goal and tries >= max_tries:
                # Unattended and still refused: nobody will step in, so keep what is there.
                send({"verb": "say", "text": f"keeping the current values: {reason}"})
                if level == "drive":
                    send({"verb": "apply", "legal_id": offer["legal_id"], "decision": "accept"})
                want, offer = "", None
                continue
            if want.rstrip().endswith("?") and not tries:
                answer = await _talk(chat, model, offer, want)
                if answer and send({"verb": "say", "text": answer}).get("ok"):
                    done["said"] += 1
            tries += 1
            request = want if not reason else f"{want} (your last choice was refused: {reason})"
            params = await _choose(chat, model, offer, request, knobs)
            # The scientist may have moved a key while the model thought: look again, and drop
            # the move if control changed, rather than send one the seat must refuse.
            if any(e.get("event") == "control" for e in seat.pending[:seat.drain()]):
                tries -= 1
                continue
            if params is None:
                reason = "no parameter on offer was chosen"
                if not goal:
                    want = ""
                continue
            base = {"legal_id": offer["legal_id"], "params": params}
            if level == "drive":
                reply = send({"verb": "apply", **base})
                if reply.get("ok"):
                    applies, tries, reason = applies + 1, 0, ""
                    done["applied"] += 1
                    log(f"applied {params}")
                    offer = None  # the next offer arrives after the refit
                    continue
                reason = reply.get("error", "")
                if goal:
                    continue
            elif level == "propose":
                reply = send({"verb": "propose", **base, "rationale": f"for: {want[:80]}"})
                if reply.get("ok"):
                    done["proposed"] += 1
                    log(f"proposed {params}")
            if level == "interpret" or not goal:
                if level == "interpret" and send(
                        {"verb": "say", "text": f"I would try {params}."}).get("ok"):
                    done["said"] += 1
                want, tries, reason = "", 0, ""
        msg = seat.next()
        if msg is None:
            return done
        kind = msg.get("event")
        if kind == "control":
            level = msg.get("level", level)
            log(f"level {level}{' (manual; back to ' + str(msg.get('resume')) + ')' if msg.get('braked') else ''}")
        elif kind == "message":
            want, tries, reason = msg.get("text", ""), 0, ""
        elif kind == "offer":
            offer = msg.get("offer")
            if goal and offer and (applies or tries):
                want = goal  # unattended: keep working toward the goal on each new question
        elif kind == "refused":
            done["refused"] += 1
            reason = msg.get("reason", "")
            log(f"refused on screen: {reason}")
        elif kind == "finished":
            return done


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="manyagents-drive", description=__doc__.split("\n")[0])
    parser.add_argument("folder", help="the run's output folder, which holds driver.json")
    parser.add_argument("--agent", default="ollama", help="adapter key with a chat() (ollama, "
                        "openai, claude, ...)")
    parser.add_argument("--model", required=True, help="model id for that adapter")
    parser.add_argument("--chooser", help="name on the decision rows (default llm:<model>)")
    parser.add_argument("--bound", default="schema", help="how choices are constrained")
    parser.add_argument("--goal", default="", help="unattended: what to work toward")
    parser.add_argument("--max-applies", type=int, default=1)
    parser.add_argument("--max-tries", type=int, default=3)
    parser.add_argument("--knobs", help="JSON file: knob name -> what it means")
    args = parser.parse_args(argv)

    from manyagents.adapters import ADAPTER_REGISTRY

    if args.agent not in ADAPTER_REGISTRY:
        parser.error(f"unknown agent {args.agent!r}; known: {sorted(ADAPTER_REGISTRY)}")
    adapter = ADAPTER_REGISTRY[args.agent]()
    if not hasattr(adapter, "chat"):
        parser.error(f"adapter {args.agent!r} has no chat(); it cannot drive")
    knobs = json.loads(Path(args.knobs).read_text()) if args.knobs else None
    try:
        seat = Seat.open(args.folder)
    except (SeatError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    try:
        done = asyncio.run(drive(seat, adapter.chat, model=args.model,
                                 chooser=args.chooser or f"llm:{args.model}", bound=args.bound,
                                 goal=args.goal, max_applies=args.max_applies,
                                 max_tries=args.max_tries, knobs=knobs,
                                 log=lambda line: print(line, flush=True)))
    except SeatError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    finally:
        seat.close()
    print(json.dumps(done))
    return 0


if __name__ == "__main__":
    sys.exit(main())
