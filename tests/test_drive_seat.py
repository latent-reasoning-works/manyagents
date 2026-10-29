"""Driving a manyRuns seat — CI-safe: a scripted fake seat on a real Unix socket, chat mocked.

The fake seat speaks protocol 1 (manyRuns environment contract §8) from a script: each request
gets a reply, and the script may push events after it. No manyRuns import, no model, no network.
"""

import asyncio
import json
import os
import socket
import tempfile
import threading

import pytest

from manyagents import conformance, drive, seat

OFFER = {"surface": "step", "step": "phate", "question_id": 1, "legal_id": "L1",
         "params": {"knn": {"current": 5, "type": "integer", "minimum": 1, "maximum": None,
                            "maximum_known": False},
                    "t": {"current": "auto", "type": "integer", "also": ["auto"], "minimum": 1,
                          "maximum": None, "maximum_known": False},
                    "min_dist": {"current": 0.1, "type": "number", "minimum": 0.0,
                                 "maximum": 1.0, "maximum_known": True},
                    "metric": {"current": "euclidean", "type": "string", "minimum": None,
                               "maximum": None, "maximum_known": False}}}


def offer(qid):
    return {**OFFER, "question_id": qid, "legal_id": f"L{qid}"}


class FakeSeat:
    """A seat on a real Unix socket. ``on(request) -> (reply, events)`` scripts it."""

    def __init__(self, on, *, level="interpret", hello_offer=OFFER, version=1):
        self.dir = tempfile.mkdtemp(prefix="fs-", dir="/tmp")
        self.path = os.path.join(self.dir, "s")
        self.requests = []
        self.on, self.level, self.hello_offer, self.version = on, level, hello_offer, version
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(self.path)
        self.server.listen(1)
        with open(os.path.join(self.dir, "driver.json"), "w") as f:
            json.dump({"socket": self.path, "protocol_version": version}, f)
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        conn, _ = self.server.accept()
        self.conn = conn
        buf = b""
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                req = json.loads(raw)
                self.requests.append(req)
                if req["verb"] == "hello":
                    reply, events = {"ok": True, "protocol_version": self.version,
                                     "level": self.level, "offer": self.hello_offer}, []
                else:
                    reply, events = self.on(req)
                # One write per batch, so a reply and the events behind it arrive together.
                # The real seat can split them; its own re-check refuses a move that loses the
                # race, and these tests pin the policy, not the race.
                conn.sendall("".join(json.dumps(o) + "\n" for o in [reply, *events]).encode())
                if any(e.get("event") == "finished" for e in events):
                    conn.close()
                    return

    def push(self, event):
        """Send one event now, outside any reply, as the run screen does."""
        self.conn.sendall((json.dumps(event) + "\n").encode())

    def verbs(self):
        return [r["verb"] + (":" + r["decision"] if "decision" in r else "") for r in self.requests]


def chat_returning(*params_or_text):
    """A chat coroutine replaying tool calls (dicts) or prose (strs), in order."""
    script = list(params_or_text)
    seen = []

    async def chat(messages, *, tools=None, **kw):
        seen.append({"messages": messages, "tools": tools, **kw})
        item = script.pop(0)
        if isinstance(item, dict):
            return {"tool_calls": [{"id": "c", "name": "set_phate",
                                    "arguments": json.dumps(item)}], "content": ""}
        return {"tool_calls": [], "content": item or ""}

    chat.seen = seen
    return chat


def run(coro):
    return asyncio.run(coro)


# ── the seat client ──────────────────────────────────────────────────────────────────────────

def test_pointer_refuses_a_missing_seat_and_an_unknown_protocol(tmp_path):
    with pytest.raises(seat.SeatError, match="no seat"):
        seat.pointer(tmp_path)
    (tmp_path / "driver.json").write_text(json.dumps({"socket": "x", "protocol_version": 99}))
    with pytest.raises(seat.SeatError, match="protocol 99"):
        seat.pointer(tmp_path)


def test_hello_refuses_a_bound_the_contract_does_not_have():
    fs = FakeSeat(lambda r: ({"ok": True}, []))
    s = seat.Seat.open(fs.dir)
    with pytest.raises(ValueError, match="bound"):
        s.hello("llm:x", "telepathy")
    s.close()


def test_events_that_arrive_before_a_reply_are_queued_not_lost():
    fs = FakeSeat(lambda r: ({"ok": True}, [{"event": "said", "text": "✓ phate: ok"}]))
    s = seat.Seat.open(fs.dir)
    s.hello("llm:x", "schema")
    assert s.ask({"verb": "say", "text": "hi"})["ok"]
    assert s.next() == {"event": "said", "text": "✓ phate: ok"}
    s.close()


def test_the_offer_becomes_one_function_tool_with_the_seats_bounds():
    tool = seat.offer_tool(OFFER, {"knn": "Neighbours per point."})
    assert tool["type"] == "function" and tool["function"]["name"] == "set_phate"
    props = tool["function"]["parameters"]["properties"]
    assert set(props) == {"knn", "t", "min_dist"}  # a string knob has no range to state
    assert props["knn"] == {"type": "integer", "minimum": 1,
                            "description": "Neighbours per point. Current 5."}
    assert props["min_dist"]["maximum"] == 1.0
    assert "maximum" not in props["knn"]  # unknown is not unlimited, and not invented
    assert "Or one of: auto" in props["t"]["description"]


@pytest.mark.parametrize("params, needle", [
    ({}, "no parameters"), ({"colour": 1}, "unknown"), ({"knn": 0}, "below"),
    ({"knn": 5}, "current"), ({"knn": 2.5}, "invalid"), ({"knn": 10.0}, "invalid"),
    ({"min_dist": 3}, "above"), ({"t": "auto"}, "current"),
])
def test_check_mirrors_what_the_seat_refuses(params, needle):
    assert needle in " ".join(seat.check(OFFER, params))


def test_check_passes_a_legal_choice():
    assert seat.check(OFFER, {"knn": 10, "t": 20}) == []


def test_check_reads_a_numeric_string_as_the_seat_does():
    """The seat parses a choice as typed text, so "40" is 40; a stricter check would score a
    choice the seat runs as refused."""
    assert seat.check(OFFER, {"t": "40"}) == []
    assert seat.check(OFFER, {"knn": "5"}) and "current" in seat.check(OFFER, {"knn": "5"})[0]
    assert "invalid" in seat.check(OFFER, {"knn": "ten"})[0]


def test_it_reads_every_queued_event_before_it_acts():
    """A refit's offer and the handbrake's control event can arrive together; acting on the
    offer before reading the control would send a move the seat must refuse."""
    fs = _scripted_seat("drive", after_apply=lambda q: [{"event": "control", "level": "interpret",
                                                         "braked": True}])
    original = fs.on

    def on(req):
        if req["verb"] == "say":
            return {"ok": True}, [{"event": "finished", "run_id": "r"}]
        return original(req)

    fs.on = on
    # The control event is queued AFTER the offer here (offer first, as a refit sends it).
    fs_after = fs.on

    def reorder(req):
        reply, events = fs_after(req)
        return reply, sorted(events, key=lambda e: e.get("event") != "offer")

    fs.on = reorder
    done = _drive_with(fs, chat_returning({"knn": 3}, {"knn": 2}), "more branches",
                       max_applies=2)
    assert fs.verbs() == ["hello", "apply", "say"]
    assert done["refused"] == 0


# ── the turn policy ──────────────────────────────────────────────────────────────────────────

def _scripted_seat(level, *, after_apply=None):
    """A seat that answers a message, refits after each apply, and finishes on accept."""
    state = {"q": 1}

    def on(req):
        if req["verb"] == "apply" and req.get("decision") == "accept":
            return {"ok": True}, [{"event": "finished", "run_id": "r", "error": None}]
        if req["verb"] == "apply":
            state["q"] += 1
            events = list(after_apply(state["q"]) if after_apply else [])
            return {"ok": True}, events + [{"event": "offer", "offer": offer(state["q"])}]
        return {"ok": True}, []

    return FakeSeat(on, level=level)


def _drive_with(fs, chat, message, **kw):
    s = seat.Seat.open(fs.dir)

    async def go():
        fs_events = [{"event": "message", "text": message}] if message else []
        s.pending.extend(fs_events)
        return await drive.drive(s, chat, model="m", chooser="llm:m", **kw)

    try:
        return run(asyncio.wait_for(go(), 10))
    finally:
        s.close()


def test_at_drive_it_applies_then_accepts_and_the_session_ends():
    fs = _scripted_seat("drive")
    done = _drive_with(fs, chat_returning({"knn": 3}), "make these look like branches")
    assert fs.verbs() == ["hello", "apply", "apply:accept"]
    assert done["applied"] == 1 and done["accepted"] is True


def test_at_propose_it_only_proposes():
    fs = FakeSeat(lambda r: ({"ok": True}, [{"event": "finished", "run_id": "r"}]
                             if r["verb"] == "propose" else []), level="propose")
    done = _drive_with(fs, chat_returning({"knn": 3}), "more branches")
    assert fs.verbs() == ["hello", "propose"]
    assert done["proposed"] == 1 and done["applied"] == 0
    assert fs.requests[1]["params"] == {"knn": 3} and fs.requests[1]["legal_id"] == "L1"


def test_at_interpret_it_answers_a_question_and_says_what_it_would_try():
    fs = FakeSeat(lambda r: ({"ok": True}, [{"event": "finished", "run_id": "r"}]
                             if r["verb"] == "say" and "would" in r["text"] else []))
    chat = chat_returning("The neighbourhood is wide.", {"knn": 3})
    done = _drive_with(fs, chat, "why are these merged?")
    assert fs.verbs() == ["hello", "say", "say"]
    assert fs.requests[1]["text"] == "The neighbourhood is wide."
    assert "knn" in fs.requests[2]["text"] and done["applied"] == 0
    assert chat.seen[1]["tools"][0]["type"] == "function"


def test_after_the_handbrake_it_stops_applying():
    # The scientist pulls the handbrake while the first refit runs.
    fs = _scripted_seat("drive", after_apply=lambda q: [{"event": "control", "level": "interpret",
                                                         "braked": True}])
    fs.on_original = fs.on

    def on(req):
        if req["verb"] == "say":
            return {"ok": True}, [{"event": "finished", "run_id": "r"}]
        return fs.on_original(req)

    fs.on = on
    done = _drive_with(fs, chat_returning({"knn": 3}, {"knn": 2}), "more branches",
                       max_applies=2)
    assert fs.verbs() == ["hello", "apply", "say"]
    assert done["applied"] == 1 and done["accepted"] is False


def test_unattended_it_feeds_the_refusal_back_and_then_keeps_what_is_there():
    def on(req):
        if req["verb"] == "apply" and "params" in req:
            return {"ok": False, "error": "knn=500 is above the maximum 120"}, []
        if req.get("decision") == "accept":
            return {"ok": True}, [{"event": "finished", "run_id": "r"}]
        return {"ok": True}, []

    fs = FakeSeat(on, level="drive")
    chat = chat_returning({"knn": 500}, {"knn": 500}, {"knn": 500})
    s = seat.Seat.open(fs.dir)
    done = run(asyncio.wait_for(drive.drive(s, chat, model="m", chooser="llm:m",
                                            goal="clear branches", max_tries=3), 10))
    s.close()
    assert fs.verbs() == ["hello", "apply", "apply", "apply", "say", "apply:accept"]
    assert "refused: knn=500 is above the maximum 120" in chat.seen[1]["messages"][-1]["content"]
    assert done["refused"] == 3 and done["applied"] == 0


def test_the_cli_refuses_a_folder_without_a_seat_and_an_adapter_without_chat(tmp_path, capsys):
    assert drive.main([str(tmp_path), "--agent", "ollama", "--model", "m"]) == 2
    assert "no seat" in capsys.readouterr().err
    with pytest.raises(SystemExit) as stop:
        drive.main([str(tmp_path), "--agent", "mock", "--model", "m"])
    assert stop.value.code == 2 and "no chat()" in capsys.readouterr().err


# ── conformance ─────────────────────────────────────────────────────────────────────────────

CASES = {"protocol_version": 1, "cases": [
    {"id": "up", "said": "more global", "offer": OFFER,
     "intent": {"params": ["knn"], "direction": "up"}},
    {"id": "value", "said": "knn 20", "offer": OFFER, "intent": {"params": ["knn"], "value": 20}},
    {"id": "outside", "said": "set knn to 0", "offer": OFFER, "intent": "outside"},
    {"id": "none", "said": "thanks", "offer": OFFER, "intent": "none"},
]}


def _fixtures(tmp_path, doc=CASES):
    p = tmp_path / "f.json"
    p.write_text(json.dumps(doc))
    return p


def test_conformance_scores_legality_intent_and_declining(tmp_path):
    chat = chat_returning({"knn": 10}, {"knn": 20}, {"knn": 0}, "")
    rows, summary = run(conformance.run(_fixtures(tmp_path), chat, "m"))
    assert summary == {"cases": 4, "errors": 0, "calls": 3, "legal_rate": 2 / 3,
                       "moves": 2, "intent_rate": 1.0, "stays": 2, "declined_rate": 0.5}
    assert rows[2]["refusal"] and rows[3]["declined"] is True


def test_one_failed_call_makes_every_rate_unavailable(tmp_path):
    async def chat(messages, **kw):
        if "thanks" in messages[-1]["content"]:
            raise ConnectionError("ollama is not running")
        return {"tool_calls": [{"arguments": '{"knn": 10}'}]}

    rows, summary = run(conformance.run(_fixtures(tmp_path), chat, "m"))
    assert summary["errors"] == 1
    assert summary["legal_rate"] is None and summary["intent_rate"] is None
    assert "n/a" in conformance.table(rows, summary, "m")


def test_fixtures_for_another_protocol_or_without_offers_are_refused(tmp_path):
    with pytest.raises(ValueError, match="protocol"):
        conformance.load_cases(_fixtures(tmp_path, {**CASES, "protocol_version": 2}))
    no_offer = {"protocol_version": 1, "cases": [{"id": "x", "said": "", "intent": "none"}]}
    with pytest.raises(ValueError, match="offer"):
        conformance.load_cases(_fixtures(tmp_path, no_offer))


def test_a_handbrake_pulled_while_the_model_thinks_stops_the_move():
    """The scientist brakes after the offer arrived but before the model answered: the move
    the model chose is dropped, never sent."""
    fs = _scripted_seat("drive")
    original = fs.on

    def on(req):
        if req["verb"] == "say":
            return {"ok": True}, [{"event": "finished", "run_id": "r"}]
        return original(req)

    fs.on = on
    calls = {"n": 0}

    async def chat(messages, *, tools=None, **kw):
        calls["n"] += 1
        if calls["n"] == 2:  # the second choice: the brake lands while the model "thinks"
            fs.push({"event": "control", "level": "interpret", "braked": True})
            await asyncio.sleep(0.2)
        return {"tool_calls": [{"arguments": json.dumps({"knn": 3 if calls["n"] == 1 else 2})}]}

    done = _drive_with(fs, chat, "more branches", max_applies=2)
    assert fs.verbs() == ["hello", "apply", "say"]
    assert done["refused"] == 0


def test_unattended_it_resumes_when_handed_back_from_manual():
    """Manual and back (one key each way): with a goal, the driver picks up where it was."""
    state = {"applied": 0}

    def on(req):
        if req.get("decision") == "accept":
            return {"ok": True}, [{"event": "finished", "run_id": "r"}]
        if req["verb"] == "apply":
            state["applied"] += 1
            return {"ok": True}, [{"event": "offer", "offer": offer(2)}]
        return {"ok": True}, []

    fs = FakeSeat(on, level="interpret", hello_offer=offer(1))
    s = seat.Seat.open(fs.dir)
    s.pending.extend([{"event": "control", "level": "interpret", "braked": True,
                       "resume": "drive"},
                      {"event": "control", "level": "drive", "braked": False, "resume": None}])
    done = run(asyncio.wait_for(drive.drive(s, chat_returning({"knn": 3}), model="m",
                                            chooser="llm:m", goal="clear branches"), 10))
    s.close()
    assert fs.verbs() == ["hello", "apply", "apply:accept"]
    assert done["applied"] == 1 and done["accepted"] is True
