"""Accounts and sessions (tenant isolation).

SQLite from the standard library, scrypt password hashes, opaque session tokens stored hashed and sent in an HttpOnly
cookie, single-use password-reset tokens, and guest accounts (one click, upgradeable to a real account later so the
work is kept). No third-party dependency.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import smtplib
import sqlite3
import threading
import time
from email.message import EmailMessage

from .config import settings

SESSION_TTL = 30 * 24 * 3600          # a session lives 30 days, refreshed on use
RESET_TTL = 60 * 60                   # a reset link is valid for one hour
MAX_FAILURES, FAILURE_WINDOW = 8, 15 * 60   # login attempts per email / per IP before a cool-down
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_lock = threading.RLock()
_db: sqlite3.Connection | None = None


class AuthError(Exception):
    """User-facing error (bad credentials, weak password, too many attempts...)."""


def _conn() -> sqlite3.Connection:
    global _db
    if _db is None:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        _db = sqlite3.connect(settings.data_dir / "users.sqlite", check_same_thread=False)
        _db.row_factory = sqlite3.Row
        _db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY, email TEXT UNIQUE, pw_hash BLOB, salt BLOB, is_guest INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL, last_seen REAL);
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, created_at REAL NOT NULL, expires_at REAL NOT NULL, agent TEXT);
            CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id);
            CREATE TABLE IF NOT EXISTS resets (
                token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, expires_at REAL NOT NULL, used INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS attempts (key TEXT PRIMARY KEY, count INTEGER NOT NULL, first_at REAL NOT NULL);
        """)
    return _db


def _h(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _hash_pw(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)


def _row(r: sqlite3.Row | None) -> dict | None:
    if r is None:
        return None
    return {"id": r["id"], "email": r["email"], "is_guest": bool(r["is_guest"]), "created_at": r["created_at"]}


def normalize_email(email: str) -> str:
    e = (email or "").strip().lower()
    if not EMAIL_RE.match(e) or len(e) > 254:
        raise AuthError("That does not look like an email address.")
    return e


def check_password_policy(pw: str) -> None:
    if len(pw or "") < 8:
        raise AuthError("Use at least 8 characters for the password.")
    if len(pw) > 200:
        raise AuthError("That password is too long.")


# ---------------- rate limiting ----------------
def _check_attempts(key: str) -> None:
    now = time.time()
    with _lock:
        c = _conn()
        r = c.execute("SELECT count, first_at FROM attempts WHERE key=?", (key,)).fetchone()
        if r and now - r["first_at"] < FAILURE_WINDOW and r["count"] >= MAX_FAILURES:
            raise AuthError("Too many attempts. Try again in a few minutes.")


def _record_failure(key: str) -> None:
    now = time.time()
    with _lock:
        c = _conn()
        r = c.execute("SELECT count, first_at FROM attempts WHERE key=?", (key,)).fetchone()
        if r and now - r["first_at"] < FAILURE_WINDOW:
            c.execute("UPDATE attempts SET count=count+1 WHERE key=?", (key,))
        else:
            c.execute("INSERT OR REPLACE INTO attempts(key, count, first_at) VALUES (?, 1, ?)", (key, now))
        c.commit()


def _clear_failures(key: str) -> None:
    with _lock:
        c = _conn(); c.execute("DELETE FROM attempts WHERE key=?", (key,)); c.commit()


# ---------------- users ----------------
def get_user(user_id: str) -> dict | None:
    with _lock:
        return _row(_conn().execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone())


def get_user_by_email(email: str) -> dict | None:
    with _lock:
        return _row(_conn().execute("SELECT * FROM users WHERE email=?", (email,)).fetchone())


def user_count() -> int:
    with _lock:
        return int(_conn().execute("SELECT COUNT(*) FROM users WHERE is_guest=0").fetchone()[0])


def create_guest() -> dict:
    uid = "u_" + secrets.token_hex(6)
    with _lock:
        c = _conn()
        c.execute("INSERT INTO users(id, email, pw_hash, salt, is_guest, created_at, last_seen) VALUES (?, NULL, NULL, NULL, 1, ?, ?)",
                  (uid, time.time(), time.time()))
        c.commit()
    return get_user(uid)  # type: ignore[return-value]


def register(email: str, password: str, upgrade_user_id: str | None = None) -> dict:
    """Create an account, or turn the current guest account into a real one (keeps its shoots)."""
    email = normalize_email(email)
    check_password_policy(password)
    salt = secrets.token_bytes(16)
    pw = _hash_pw(password, salt)
    with _lock:
        c = _conn()
        if c.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
            raise AuthError("An account with this email already exists. Log in instead.")
        if upgrade_user_id:
            g = c.execute("SELECT * FROM users WHERE id=? AND is_guest=1", (upgrade_user_id,)).fetchone()
            if g:
                c.execute("UPDATE users SET email=?, pw_hash=?, salt=?, is_guest=0 WHERE id=?", (email, pw, salt, upgrade_user_id))
                c.commit()
                return get_user(upgrade_user_id)  # type: ignore[return-value]
        uid = "u_" + secrets.token_hex(6)
        c.execute("INSERT INTO users(id, email, pw_hash, salt, is_guest, created_at, last_seen) VALUES (?, ?, ?, ?, 0, ?, ?)",
                  (uid, email, pw, salt, time.time(), time.time()))
        c.commit()
    return get_user(uid)  # type: ignore[return-value]


def authenticate(email: str, password: str, ip: str = "") -> dict:
    email = normalize_email(email)
    _check_attempts("e:" + email)
    if ip:
        _check_attempts("ip:" + ip)
    with _lock:
        r = _conn().execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    ok = bool(r) and r["pw_hash"] is not None and hmac.compare_digest(_hash_pw(password or "", r["salt"]), r["pw_hash"])
    if not ok:
        _record_failure("e:" + email)
        if ip:
            _record_failure("ip:" + ip)
        raise AuthError("Email or password is incorrect.")
    _clear_failures("e:" + email)
    return _row(r)  # type: ignore[return-value]


def change_password(user_id: str, current: str, new: str) -> None:
    check_password_policy(new)
    with _lock:
        c = _conn()
        r = c.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        if not r or r["pw_hash"] is None:
            raise AuthError("This account has no password yet; create an account first.")
        if not hmac.compare_digest(_hash_pw(current or "", r["salt"]), r["pw_hash"]):
            raise AuthError("The current password is incorrect.")
        salt = secrets.token_bytes(16)
        c.execute("UPDATE users SET pw_hash=?, salt=? WHERE id=?", (_hash_pw(new, salt), salt, user_id))
        c.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))   # log out everywhere else
        c.commit()


# ---------------- sessions ----------------
def create_session(user_id: str, agent: str = "") -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    with _lock:
        c = _conn()
        c.execute("INSERT INTO sessions(token_hash, user_id, created_at, expires_at, agent) VALUES (?, ?, ?, ?, ?)",
                  (_h(token), user_id, now, now + SESSION_TTL, agent[:200]))
        c.execute("UPDATE users SET last_seen=? WHERE id=?", (now, user_id))
        c.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
        c.commit()
    return token


def session_user(token: str | None) -> dict | None:
    if not token:
        return None
    now = time.time()
    with _lock:
        c = _conn()
        r = c.execute("SELECT s.expires_at, u.* FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token_hash=?",
                      (_h(token),)).fetchone()
        if not r or r["expires_at"] < now:
            return None
        if r["expires_at"] - now < SESSION_TTL / 2:   # sliding expiry
            c.execute("UPDATE sessions SET expires_at=? WHERE token_hash=?", (now + SESSION_TTL, _h(token)))
            c.execute("UPDATE users SET last_seen=? WHERE id=?", (now, r["id"]))
            c.commit()
        return _row(r)


def delete_session(token: str | None) -> None:
    if not token:
        return
    with _lock:
        c = _conn(); c.execute("DELETE FROM sessions WHERE token_hash=?", (_h(token),)); c.commit()


# ---------------- password reset ----------------
def create_reset_token(email: str) -> tuple[str | None, dict | None]:
    """Returns (token, user) or (None, None) when the email is unknown; callers must answer identically either way."""
    try:
        email = normalize_email(email)
    except AuthError:
        return None, None
    _check_attempts("reset:" + email)
    _record_failure("reset:" + email)          # also throttles how many reset mails one address can trigger
    u = get_user_by_email(email)
    if not u:
        return None, None
    token = secrets.token_urlsafe(32)
    with _lock:
        c = _conn()
        c.execute("INSERT INTO resets(token_hash, user_id, expires_at, used) VALUES (?, ?, ?, 0)", (_h(token), u["id"], time.time() + RESET_TTL))
        c.execute("DELETE FROM resets WHERE expires_at < ?", (time.time(),))
        c.commit()
    return token, u


def reset_password(token: str, new: str) -> dict:
    check_password_policy(new)
    with _lock:
        c = _conn()
        r = c.execute("SELECT * FROM resets WHERE token_hash=?", (_h(token or ""),)).fetchone()
        if not r or r["used"] or r["expires_at"] < time.time():
            raise AuthError("This reset link is invalid or has expired. Request a new one.")
        salt = secrets.token_bytes(16)
        c.execute("UPDATE users SET pw_hash=?, salt=? WHERE id=?", (_hash_pw(new, salt), salt, r["user_id"]))
        c.execute("UPDATE resets SET used=1 WHERE token_hash=?", (r["token_hash"],))
        c.execute("DELETE FROM sessions WHERE user_id=?", (r["user_id"],))
        c.commit()
    return get_user(r["user_id"])  # type: ignore[return-value]


def send_reset_email(to: str, link: str) -> bool:
    """Send the reset link over SMTP when configured; returns False when no mail server is set up."""
    if not settings.smtp_host:
        return False
    msg = EmailMessage()
    msg["Subject"] = "Reset your Cocreate password"
    msg["From"] = settings.smtp_from or settings.smtp_user or "cocreate@localhost"
    msg["To"] = to
    msg.set_content(f"Someone asked to reset the password for this Cocreate account.\n\nOpen this link within one hour:\n{link}\n\n"
                    f"If this was not you, ignore this email; your password stays unchanged.")
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=20) as s:
        if settings.smtp_starttls:
            s.starttls()
        if settings.smtp_user:
            s.login(settings.smtp_user, settings.smtp_password or "")
        s.send_message(msg)
    return True
