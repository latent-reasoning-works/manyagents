"""A client for a manyRuns driver seat: find it, attach, read events, send verbs.

manyRuns opens a seat for every stepped run: a Unix socket named by
``outputs/<project>/driver.json``. A driver attaches and speaks JSON lines (manyRuns
environment contract §8). This module speaks that protocol and nothing else: it imports
nothing from manyRuns, because the arrow points one way — the environment never knows who
is driving it.

Three pieces, all plain data:

- ``Seat``: the connection. ``ask`` sends one request and returns its reply; events that
  arrive meanwhile are queued for ``next``.
- ``offer_tool``: the seat's step offer as one OpenAI function tool, so any adapter whose
  ``chat`` takes tools can choose from it.
- ``check``: what the seat will refuse, judged from the offer alone, so ``conformance`` can
  score a choice without a live session.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any, Optional

#: The seat protocol versions this client speaks. A seat that names another is refused at
#: ``open`` rather than misread: a driver that guesses at a protocol records wrong rows.
PROTOCOLS = frozenset({1})

#: The tiers a choice's constraint can come from (contract §2.2).
BOUNDS = ("logits", "schema", "prompt", "none")


class SeatError(RuntimeError):
    """The seat cannot be used: missing, a protocol this client does not speak, or closed."""


def pointer(folder: Path | str) -> dict:
    """``driver.json`` from a run's output folder. Raises ``SeatError`` when there is none or
    its protocol is not one this client speaks."""
    path = Path(folder) / "driver.json"
    if not path.is_file():
        raise SeatError(f"no seat at {path}: is a stepped run open in that folder?")
    data = json.loads(path.read_text())
    version = data.get("protocol_version")
    if version not in PROTOCOLS:
        raise SeatError(f"seat protocol {version!r} is not one this client speaks "
                        f"({sorted(PROTOCOLS)}); update manyagents or manyruns")
    return data


class Seat:
    """One connection to a seat. Not thread-safe; one driver loop owns it."""

    def __init__(self, path: str, timeout: Optional[float] = None) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self._timeout = timeout
        self.sock.connect(path)
        self._buf = b""
        self.pending: list[dict] = []

    @classmethod
    def open(cls, folder: Path | str, timeout: Optional[float] = None) -> "Seat":
        return cls(pointer(folder)["socket"], timeout=timeout)

    def _line(self) -> Optional[dict]:
        while b"\n" not in self._buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                return None
            self._buf += chunk
        raw, self._buf = self._buf.split(b"\n", 1)
        return json.loads(raw)

    def ask(self, request: dict) -> dict:
        """Send one request; return its reply. Events that arrive first are queued."""
        self.sock.sendall((json.dumps(request) + "\n").encode("utf-8"))
        while True:
            msg = self._line()
            if msg is None:
                raise SeatError("the seat closed the connection")
            if "event" not in msg:
                return msg
            self.pending.append(msg)

    def hello(self, chooser: str, bound: str) -> dict:
        """Attach. The reply carries the level, the tag, and the offer already waiting."""
        if bound not in BOUNDS:
            raise ValueError(f"bound must be one of {list(BOUNDS)}, not {bound!r}")
        reply = self.ask({"verb": "hello", "chooser": chooser, "bound": bound})
        if not reply.get("ok"):
            raise SeatError(f"hello refused: {reply.get('error')}")
        if reply.get("protocol_version") not in PROTOCOLS:
            raise SeatError(f"seat protocol {reply.get('protocol_version')!r} is not one this "
                            f"client speaks ({sorted(PROTOCOLS)})")
        return reply

    def drain(self) -> int:
        """Queue every event already received, without blocking. Returns how many are queued.

        A driver calls this before it acts, so a ``control`` event (the handbrake) that arrived
        right behind an ``offer`` is seen before a move is sent."""
        self.sock.setblocking(False)
        try:
            while True:
                chunk = self.sock.recv(65536)
                if not chunk:
                    break
                self._buf += chunk
        except (BlockingIOError, InterruptedError):
            pass
        finally:
            self.sock.settimeout(self._timeout)
        while b"\n" in self._buf:
            raw, self._buf = self._buf.split(b"\n", 1)
            self.pending.append(json.loads(raw))
        return len(self.pending)

    def next(self) -> Optional[dict]:
        """The next event (queued first), or None when the seat hung up."""
        return self.pending.pop(0) if self.pending else self._line()

    def close(self) -> None:
        self.sock.close()


def offer_tool(offer: dict, knobs: Optional[dict] = None) -> dict:
    """The step offer as one OpenAI function tool: ``set_<step>`` with one property per knob.

    Only numeric knobs become properties: a string knob has no range a schema can state. A knob
    currently at a word it may take (``also``, e.g. ``auto``) is offered as its number type with
    the word named in the description. ``knobs`` is the caller's card of what each knob MEANS;
    the seat offers names, types and bounds, never meaning, which is the intelligence a driver
    brings."""
    knobs = knobs or {}
    props: dict[str, dict] = {}
    for name, p in (offer.get("params") or {}).items():
        if p.get("type") not in ("integer", "number"):
            continue
        words = f" Or one of: {', '.join(map(str, p['also']))}." if p.get("also") else ""
        prop = {"type": p["type"],
                "description": f"{knobs.get(name, name)} Current {p.get('current')}.{words}"}
        if p.get("minimum") is not None:
            prop["minimum"] = p["minimum"]
        if p.get("maximum_known"):
            prop["maximum"] = p["maximum"]
        if p.get("presets"):
            prop["description"] += f" Presets: {p['presets']}."
        props[name] = prop
    return {"type": "function", "function": {
        "name": f"set_{offer.get('step')}",
        "description": f"Change parameters of the {offer.get('step')} step.",
        "parameters": {"type": "object", "properties": props, "additionalProperties": False}}}


def check(offer: dict, params: Any) -> list[str]:
    """What the seat will refuse in ``params``, from the offer alone. ``[]`` if it may run.

    The seat is the authority (it re-checks everything); this mirrors its stated rules
    (contract §8) so a harness can score a choice offline: unknown names, wrong types, values
    below a minimum or above a known maximum, and a value equal to the current one."""
    if not isinstance(params, dict) or not params:
        return ["no parameters"]
    problems = []
    knobs = offer.get("params") or {}
    for name, value in params.items():
        p = knobs.get(name)
        if p is None:
            problems.append(f"unknown parameter {name!r}")
            continue
        if isinstance(value, str) and value in (p.get("also") or ()):
            if value == p.get("current"):
                problems.append(f"{name}={value} is the current value")
            continue
        # The seat parses a choice as the text a person would type, so "40" is 40.
        if isinstance(value, str):
            for number in (int, float):
                try:
                    value = number(value)
                    break
                except ValueError:
                    pass
        integer = p.get("type") == "integer"
        if isinstance(value, bool) or not isinstance(value, (int, float)) or (
                integer and isinstance(value, float)):
            problems.append(f"invalid value for {name!r}: {value!r}")
            continue
        if p.get("minimum") is not None and value < p["minimum"]:
            problems.append(f"{name}={value} is below the minimum {p['minimum']}")
        if p.get("maximum_known") and value > p["maximum"]:
            problems.append(f"{name}={value} is above the maximum {p['maximum']}")
        if value == p.get("current"):
            problems.append(f"{name}={value} is the current value")
    return problems
