"""Guest mode end to end: everyone gets a private workspace without signing up; signing up keeps it; the first real
account inherits shoots made before accounts existed.

    # seed one pre-account shoot, then:
    AUTH_MODE=guest DATA_DIR=/tmp/cc-guest COCREATE_MOCK=1 uv run uvicorn app.main:app --port 8022
    uv run python tests/test_guest_mode.py http://127.0.0.1:8022
"""
from __future__ import annotations

import http.cookiejar
import io
import json
import sys
import urllib.error
import urllib.request
import uuid

from PIL import Image

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8022"
FAILS = 0


def client():
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))


def call(op, method, path, body=None, raw=None, ctype=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(BASE + path, data=data, method=method)
    if ctype:
        req.add_header("Content-Type", ctype)
    elif body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with op.open(req, timeout=60) as r:
            txt = r.read()
            try:
                return r.status, json.loads(txt)
            except Exception:
                return r.status, txt
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, None


def check(name, ok, detail=""):
    global FAILS
    print(f"  {'OK  ' if ok else 'FAIL'} {name}{(' · ' + detail) if detail else ''}")
    if not ok:
        FAILS += 1


def upload(op, name, color):
    boundary = "----cc" + uuid.uuid4().hex
    b = io.BytesIO(); Image.new("RGB", (640, 420), color).save(b, "JPEG")
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"name\"\r\n\r\n{name}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"p.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n").encode() + b.getvalue() + f"\r\n--{boundary}--\r\n".encode()
    return call(op, "POST", "/api/shoots", raw=body, ctype=f"multipart/form-data; boundary={boundary}")


def main():
    print("guest workspaces")
    g1, g2 = client(), client()
    st, d1 = call(g1, "GET", "/api/auth/me"); check("first visit gets a guest user", st == 200 and d1["auth_mode"] == "guest" and d1["user"] and d1["user"]["is_guest"])
    st, d2 = call(g2, "GET", "/api/auth/me"); check("another browser gets a different guest", st == 200 and d2["user"]["id"] != d1["user"]["id"])
    st, again = call(g1, "GET", "/api/auth/me"); check("same browser keeps its guest", again["user"]["id"] == d1["user"]["id"])
    st, s1 = upload(g1, "guest one shoot", (200, 60, 60)); check("guest can upload a shoot", st == 200, f"status {st}")
    st, lst = call(g2, "GET", "/api/shoots"); check("other guest does not see it", st == 200 and all(x["id"] != s1["id"] for x in lst))
    st, _ = call(g2, "GET", f"/api/shoots/{s1['id']}"); check("other guest cannot open it", st == 404)
    st, _ = call(g2, "GET", s1["photos"][0]["original_url"]); check("other guest cannot fetch its photo", st == 404)
    st, _ = call(g1, "GET", s1["photos"][0]["original_url"]); check("owner guest can fetch the photo", st == 200)
    st, d = call(g1, "POST", "/api/shoots/demo/bins"); check("guest can open a demo shoot", st == 200)
    anon = urllib.request.build_opener()   # no cookie jar at all
    st, _ = call(anon, "GET", "/api/shoots"); check("a request with no session at all is refused", st == 401)

    print("sign up keeps the guest's work and the first account claims pre-account shoots")
    email = f"first-{uuid.uuid4().hex[:6]}@example.com"
    st, d = call(g1, "POST", "/api/auth/register", {"email": email, "password": "first-pass-123"})
    check("guest upgrades to a real account", st == 200 and d["user"]["email"] == email and not d["user"]["is_guest"])
    check("first account claimed the seeded pre-account shoot", d.get("claimed_shoots", 0) >= 1, f"claimed={d.get('claimed_shoots')}")
    st, lst = call(g1, "GET", "/api/shoots")
    check("upgraded account still sees the guest shoot", any(x["id"] == s1["id"] for x in lst))
    check("and now also the seeded pre-account shoot", any(x["name"] == "pre-account shoot" for x in lst), str([x["name"] for x in lst]))
    st, lst2 = call(g2, "GET", "/api/shoots"); check("other guest still sees nothing of it", all(x["name"] != "pre-account shoot" and x["id"] != s1["id"] for x in lst2))

    print("log in from a guest session switches accounts")
    st, _ = call(g2, "POST", "/api/auth/login", {"email": email, "password": "first-pass-123"}); check("guest browser logs into the real account", st == 200)
    st, lst = call(g2, "GET", "/api/shoots"); check("it now sees the real account's shoots", any(x["id"] == s1["id"] for x in lst))
    st, _ = call(g2, "POST", "/api/auth/logout"); check("logout", st == 200)
    st, d = call(g2, "GET", "/api/auth/me"); check("after logout a fresh guest workspace is issued", st == 200 and d["user"] and d["user"]["is_guest"])
    st, lst = call(g2, "GET", "/api/shoots"); check("the fresh guest sees nothing", st == 200 and lst == [])

    print("report export")
    st, html_ = call(g1, "GET", f"/api/shoots/{s1['id']}/report.html")
    check("owner can download the HTML report", st == 200 and b"edit report" in html_ and b"data:image/jpeg" in html_)
    st, _ = call(g2, "GET", f"/api/shoots/{s1['id']}/report.html"); check("others cannot", st == 404)

    print(f"\n{'ALL PASSED' if FAILS == 0 else f'{FAILS} FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
