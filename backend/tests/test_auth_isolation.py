"""End-to-end check of accounts and tenant isolation against a running server in AUTH_MODE=required.

    AUTH_MODE=required AUTH_SHOW_RESET_LINK=true DATA_DIR=/tmp/cc-auth COCREATE_MOCK=1 uv run uvicorn app.main:app --port 8020
    uv run python tests/test_auth_isolation.py http://127.0.0.1:8020

Uses only the standard library (http.cookiejar) so it can run anywhere.
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

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8020"
FAILS = 0


def client():
    jar = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar)), jar


def call(op, method: str, path: str, body: dict | None = None, raw: bytes | None = None, ctype: str | None = None):
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


def check(name: str, ok: bool, detail: str = ""):
    global FAILS
    print(f"  {'OK  ' if ok else 'FAIL'} {name}{(' · ' + detail) if detail else ''}")
    if not ok:
        FAILS += 1


def multipart(fields: dict[str, str], files: list[tuple[str, str, bytes]]) -> tuple[bytes, str]:
    boundary = "----cc" + uuid.uuid4().hex
    out = io.BytesIO()
    for k, v in fields.items():
        out.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    for field, fname, data in files:
        out.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{fname}\"\r\nContent-Type: image/jpeg\r\n\r\n".encode())
        out.write(data); out.write(b"\r\n")
    out.write(f"--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


def jpeg(color) -> bytes:
    b = io.BytesIO(); Image.new("RGB", (640, 420), color).save(b, "JPEG"); return b.getvalue()


def main() -> None:
    a_op, _ = client(); b_op, _ = client(); anon_op, _ = client()
    ea, eb = f"alice-{uuid.uuid4().hex[:6]}@example.com", f"bob-{uuid.uuid4().hex[:6]}@example.com"

    print("anonymous")
    st, _ = call(anon_op, "GET", "/api/shoots"); check("listing shoots without login is refused", st == 401, f"status {st}")
    st, d = call(anon_op, "GET", "/api/auth/me"); check("/me reports required mode and no user", st == 200 and d["auth_mode"] == "required" and d["user"] is None)

    print("registration")
    st, d = call(a_op, "POST", "/api/auth/register", {"email": ea, "password": "short"}); check("weak password rejected", st == 400)
    st, d = call(a_op, "POST", "/api/auth/register", {"email": "not-an-email", "password": "longenough1"}); check("bad email rejected", st == 400)
    st, d = call(a_op, "POST", "/api/auth/register", {"email": ea, "password": "alice-pass-123"}); check("alice registers", st == 200 and d["user"]["email"] == ea)
    st, d = call(b_op, "POST", "/api/auth/register", {"email": ea, "password": "whatever-123"}); check("duplicate email rejected", st == 400)
    st, d = call(b_op, "POST", "/api/auth/register", {"email": eb, "password": "bob-pass-12345"}); check("bob registers", st == 200)

    print("isolation")
    body, ctype = multipart({"name": "alice shoot"}, [("files", "a.jpg", jpeg((200, 40, 40)))])
    st, sa = call(a_op, "POST", "/api/shoots", raw=body, ctype=ctype); check("alice uploads a shoot", st == 200, f"status {st}")
    sid = sa["id"]; photo_url = sa["photos"][0]["original_url"]
    st, lst = call(b_op, "GET", "/api/shoots"); check("bob's list does not contain alice's shoot", st == 200 and all(x["id"] != sid for x in lst))
    st, _ = call(b_op, "GET", f"/api/shoots/{sid}"); check("bob cannot open alice's shoot", st == 404, f"status {st}")
    st, _ = call(b_op, "DELETE", f"/api/shoots/{sid}"); check("bob cannot delete alice's shoot", st == 404, f"status {st}")
    st, _ = call(b_op, "POST", f"/api/shoots/{sid}/chat/new"); check("bob cannot touch alice's chat", st == 404, f"status {st}")
    st, _ = call(b_op, "GET", photo_url); check("bob cannot fetch alice's photo file", st == 404, f"status {st}")
    st, _ = call(anon_op, "GET", photo_url); check("anonymous cannot fetch the photo file", st == 404, f"status {st}")
    st, _ = call(a_op, "GET", photo_url); check("alice can fetch her own photo file", st == 200, f"status {st}")
    st, lst = call(a_op, "GET", "/api/shoots"); check("alice sees her shoot", st == 200 and any(x["id"] == sid for x in lst))
    st, _ = call(a_op, "GET", "/files/../../etc/passwd"); check("path traversal refused", st in (404, 400), f"status {st}")
    st, d = call(b_op, "POST", "/api/shoots/demo/vehicles"); check("bob can create his own demo shoot", st == 200, f"status {st}")
    if st == 200:
        st, _ = call(a_op, "GET", f"/api/shoots/{d['id']}"); check("alice cannot open bob's demo shoot", st == 404)

    print("guest")
    g_op, _ = client()
    st, d = call(g_op, "POST", "/api/auth/guest"); check("guest session created", st == 200 and d["user"]["is_guest"])
    body, ctype = multipart({"name": "guest shoot"}, [("files", "g.jpg", jpeg((40, 200, 40)))])
    st, sg = call(g_op, "POST", "/api/shoots", raw=body, ctype=ctype); check("guest uploads a shoot", st == 200)
    eg = f"guest-{uuid.uuid4().hex[:6]}@example.com"
    st, d = call(g_op, "POST", "/api/auth/register", {"email": eg, "password": "guest-pass-123"}); check("guest upgrades to an account", st == 200 and not d["user"]["is_guest"])
    st, lst = call(g_op, "GET", "/api/shoots"); check("upgraded account keeps the guest shoot", st == 200 and any(x["id"] == sg["id"] for x in lst))

    print("password change / reset / logout")
    st, _ = call(a_op, "POST", "/api/auth/change-password", {"current_password": "wrong", "new_password": "alice-new-pass-1"}); check("wrong current password rejected", st == 400)
    st, _ = call(a_op, "POST", "/api/auth/change-password", {"current_password": "alice-pass-123", "new_password": "alice-new-pass-1"}); check("password changed", st == 200)
    st, d = call(a_op, "GET", "/api/auth/me"); check("alice still logged in after change (new session)", st == 200 and d["user"] and d["user"]["email"] == ea)
    a2, _ = client()
    st, _ = call(a2, "POST", "/api/auth/login", {"email": ea, "password": "alice-pass-123"}); check("old password no longer works", st == 401)
    st, _ = call(a2, "POST", "/api/auth/login", {"email": ea, "password": "alice-new-pass-1"}); check("new password works", st == 200)
    st, d = call(anon_op, "POST", "/api/auth/forgot", {"email": "nobody-" + uuid.uuid4().hex[:5] + "@example.com"}); check("forgot for unknown email answers ok without leaking", st == 200 and d.get("ok") and "reset_link" not in d)
    st, d = call(anon_op, "POST", "/api/auth/forgot", {"email": ea}); check("forgot for alice returns a link (dev flag on)", st == 200 and "reset_link" in d, str(d)[:80])
    token = d["reset_link"].split("#reset=")[-1] if st == 200 and "reset_link" in d else ""
    r_op, _ = client()
    st, _ = call(r_op, "POST", "/api/auth/reset", {"token": "bogus", "new_password": "alice-reset-pass-1"}); check("bogus reset token rejected", st == 400)
    st, d = call(r_op, "POST", "/api/auth/reset", {"token": token, "new_password": "alice-reset-pass-1"}); check("reset with the link works and logs in", st == 200 and d["user"]["email"] == ea)
    st, _ = call(r_op, "POST", "/api/auth/reset", {"token": token, "new_password": "alice-again-pass-1"}); check("reset token is single-use", st == 400)
    st, d = call(a_op, "GET", "/api/auth/me"); check("alice's older session was signed out by the reset", st == 200 and d["user"] is None)
    st, _ = call(r_op, "GET", f"/api/shoots/{sid}"); check("alice (after reset) still owns her shoot", st == 200)
    st, _ = call(r_op, "POST", "/api/auth/logout"); check("logout", st == 200)
    st, _ = call(r_op, "GET", "/api/shoots"); check("after logout the list is refused again", st == 401)

    print("login rate limit")
    l_op, _ = client()
    codes = [call(l_op, "POST", "/api/auth/login", {"email": eb, "password": "nope-nope-nope"})[0] for _ in range(9)]
    check("repeated bad logins get throttled", codes[-1] == 401 and codes.count(401) == 9)
    st, d = call(l_op, "POST", "/api/auth/login", {"email": eb, "password": "bob-pass-12345"}); check("throttled account refuses even the right password for a while", st == 401 and "Too many" in str(d))

    print(f"\n{'ALL PASSED' if FAILS == 0 else f'{FAILS} FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
