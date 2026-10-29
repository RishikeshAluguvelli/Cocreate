"""Parallel requests must never leak between shoots or interleave within one shoot.

    COCREATE_MOCK=1 DATA_DIR=/tmp/cc-conc uv run uvicorn app.main:app --port 8021
    uv run python tests/test_concurrency.py http://127.0.0.1:8021

Runs in mock mode (keyword planner, placeholder edits) so it costs nothing. Fires requests at two shoots at the same
time plus two overlapping requests at one shoot, then checks every shoot's chat holds exactly its own messages in order
and that the second request on a shoot waited for the first.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8021"
FAILS = 0


def check(name: str, ok: bool, detail: str = ""):
    global FAILS
    print(f"  {'OK  ' if ok else 'FAIL'} {name}{(' · ' + detail) if detail else ''}")
    if not ok:
        FAILS += 1


def post(path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode() if body is not None else None, method="POST",
                                 headers={"Content-Type": "application/json"} if body is not None else {})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.loads(r.read())


def chat(sid: str, text: str, out: dict, key: str) -> None:
    """Consume the SSE stream fully; record the event types and the final reply."""
    t0 = time.time()
    req = urllib.request.Request(f"{BASE}/api/shoots/{sid}/chat", data=json.dumps({"text": text, "scope": "shoot"}).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    events, statuses = [], []
    with urllib.request.urlopen(req, timeout=300) as r:
        etype = None
        for raw in r:
            line = raw.decode().rstrip("\r\n")
            if line.startswith("event:"):
                etype = line[6:].strip()
            elif line.startswith("data:") and etype:
                events.append(etype)
                if etype == "status":
                    statuses.append(json.loads(line[5:].strip()).get("message", ""))
    out[key] = {"events": events, "statuses": statuses, "seconds": round(time.time() - t0, 1), "start": t0, "end": time.time()}


def main() -> None:
    a = post("/api/shoots/demo/vehicles"); b = post("/api/shoots/demo/bins")
    print(f"shoots: A={a['id']} ({len(a['photos'])} photos)  B={b['id']} ({len(b['photos'])} photos)")

    print("parallel requests on two different shoots")
    out: dict = {}
    ta = threading.Thread(target=chat, args=(a["id"], "Remove all the vehicles.", out, "A1"))
    tb = threading.Thread(target=chat, args=(b["id"], "Remove the garbage bins", out, "B1"))
    ta.start(); tb.start(); ta.join(); tb.join()
    A, B = get(f"/api/shoots/{a['id']}"), get(f"/api/shoots/{b['id']}")
    check("A has exactly one user + one assistant message", [m["role"] for m in A["chat"]] == ["user", "assistant"], str([m["role"] for m in A["chat"]]))
    check("B has exactly one user + one assistant message", [m["role"] for m in B["chat"]] == ["user", "assistant"])
    check("A's chat contains only A's request", A["chat"][0]["text"] == "Remove all the vehicles." and "garbage" not in json.dumps(A["chat"]).lower())
    check("B's chat contains only B's request", B["chat"][0]["text"] == "Remove the garbage bins" and "vehicle" not in json.dumps(B["chat"]).lower())
    check("both streams ended with done", out["A1"]["events"][-1] == "done" and out["B1"]["events"][-1] == "done")
    check("A's versions are on A's photos only", all(v["url"].split("/shoots/")[1].startswith(a["id"]) for p in A["photos"] for v in p["versions"]))

    print("two overlapping requests on the same shoot")
    out = {}
    t1 = threading.Thread(target=chat, args=(a["id"], "Brighten everything a little.", out, "A2"))
    t2 = threading.Thread(target=chat, args=(a["id"], "Add a watermark saying Cocreate.", out, "A3"))
    t1.start(); time.sleep(0.3); t2.start(); t1.join(); t2.join()
    A = get(f"/api/shoots/{a['id']}")
    roles = [m["role"] for m in A["chat"]]
    texts = [m["text"] for m in A["chat"] if m["role"] == "user"]
    check("chat now has three request/reply pairs in order", roles == ["user", "assistant"] * 3, str(roles))
    check("the two overlapping requests are both recorded, first one first", texts[1:] == ["Brighten everything a little.", "Add a watermark saying Cocreate."], str(texts[1:]))
    overlapped = out["A3"]["start"] < out["A2"]["end"]
    check("the second request reported that it waited (or the first had already finished)",
          (not overlapped) or any("Waiting for the previous request" in s for s in out["A3"]["statuses"]),
          f"overlapped={overlapped} statuses={out['A3']['statuses'][:2]}")
    check("the first request did not wait", not any("Waiting" in s for s in out["A2"]["statuses"]))
    B = get(f"/api/shoots/{b['id']}")
    check("B was untouched by A's later requests", [m["role"] for m in B["chat"]] == ["user", "assistant"] and len(B["chat"]) == 2)
    check("no assistant message mixes tool events from another request",
          all(all(e.get("type") in ("tool_start", "tool_result") for e in m.get("events", [])) for m in A["chat"] if m["role"] == "assistant"))

    print(f"\n{'ALL PASSED' if FAILS == 0 else f'{FAILS} FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
