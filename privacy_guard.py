"""Privacy routing guard for a litellm smart-router deployment.

Registered as a litellm proxy callback (litellm_settings.callbacks). For any
request whose model is the router (default "smart-router"), it scans the ENTIRE
conversation for sensitive content (PII / secrets / personal-financial /
medical / mail). On a match it rewrites ``data["model"]`` to a LOCAL model so
the sensitive data never reaches a cloud provider. Non-matching requests pass
through unchanged → normal cloud semantic routing.

Design guarantees:
- ONE-WAY FUSE: once a conversation has been routed local (PII seen), it is
  LATCHED local for all future turns — even if later turns are benign and even
  if the original PII has aged out of the context via compression. Cloud→local
  is allowed (PII appearing mid-conversation); local→cloud is impossible.
  The latch is keyed by a stable conversation fingerprint (caller session id if
  present, else a hash of the compression-protected first user message) and
  persisted to disk so it survives proxy restarts.
- STICKY scan: evaluates every message, so PII anywhere trips the fuse.
- FAIL-SAFE: any error → route local (never leak on uncertainty).
- DATA-DRIVEN: rules in privacy_rules.json (admin UI on :8094); hot-reloaded.
- PRIVACY-PRESERVING LOGS: decision log records matched-rule + routed-model +
  text length only — never the content itself.

``evaluate(messages)`` is importable so the admin UI tests the exact same logic.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from litellm.integrations.custom_logger import CustomLogger
except Exception:  # pragma: no cover - allows standalone import by the admin UI
    class CustomLogger:  # type: ignore
        pass

RULES_PATH = os.environ.get(
    "PRIVACY_RULES_PATH", os.path.expanduser("~/.config/litellm/privacy_rules.json")
)
DECISIONS_LOG = os.environ.get(
    "PRIVACY_DECISIONS_LOG",
    os.path.expanduser("~/.local/state/litellm-privacy/decisions.jsonl"),
)
FUSED_PATH = os.environ.get(
    "PRIVACY_FUSED_PATH",
    os.path.expanduser("~/.local/state/litellm-privacy/fused.txt"),
)

_DEFAULT_RULES: Dict[str, Any] = {
    "enabled": True,
    "router_name": "smart-router",
    "local_model": os.environ.get("PRIVACY_GUARD_LOCAL_MODEL", ""),
    "regex_rules": [
        {"id": "email", "label": "Email", "pattern": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "enabled": True},
        {"id": "api_secret_key", "label": "API key", "pattern": r"\b(?:sk|rk|pk)-[A-Za-z0-9_]{16,}\b", "enabled": True},
        {"id": "private_key_block", "label": "Private key", "pattern": r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "enabled": True},
    ],
    "phrase_rules": [],
    "keyword_rules": [],
}


def _luhn_ok(digits: str) -> bool:
    digits = re.sub(r"\D", "", digits)
    if not (13 <= len(digits) <= 19):
        return False
    total, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


class _Rules:
    def __init__(self) -> None:
        self.mtime: float = -1.0
        self.raw: Dict[str, Any] = {}
        self.regex: List[Tuple[str, "re.Pattern[str]", bool]] = []
        self.phrase: List[Tuple[str, "re.Pattern[str]"]] = []
        self.keyword: List[Tuple[str, List[str]]] = []
        self.load(force=True)

    def load(self, force: bool = False) -> None:
        try:
            mtime = os.path.getmtime(RULES_PATH)
        except OSError:
            if not self.raw:
                self._compile(_DEFAULT_RULES)
            return
        if not force and mtime == self.mtime:
            return
        try:
            with open(RULES_PATH, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self._compile(raw)
            self.mtime = mtime
        except Exception:
            if not self.raw:
                self._compile(_DEFAULT_RULES)

    def _compile(self, raw: Dict[str, Any]) -> None:
        self.raw = raw
        self.regex = []
        for r in raw.get("regex_rules", []):
            if not r.get("enabled", True):
                continue
            try:
                self.regex.append((r["id"], re.compile(r["pattern"]), bool(r.get("luhn"))))
            except re.error:
                continue
        self.phrase = []
        for r in raw.get("phrase_rules", []):
            if not r.get("enabled", True):
                continue
            try:
                self.phrase.append((r["id"], re.compile(r["pattern"], re.IGNORECASE)))
            except re.error:
                continue
        self.keyword = []
        for r in raw.get("keyword_rules", []):
            if not r.get("enabled", True):
                continue
            self.keyword.append((r["id"], [t.lower() for t in r.get("terms", [])]))

    @property
    def enabled(self) -> bool:
        return bool(self.raw.get("enabled", True))

    @property
    def router_name(self) -> str:
        return self.raw.get("router_name", "smart-router")

    @property
    def local_model(self) -> str:
        env_default = os.environ.get("PRIVACY_GUARD_LOCAL_MODEL", "")
        return self.raw.get("local_model") or env_default


_RULES = _Rules()

# ── One-way fuse state ───────────────────────────────────────────────────
_FUSED: set = set()


def _load_fused() -> None:
    try:
        with open(FUSED_PATH, "r", encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if ln:
                    _FUSED.add(ln)
    except FileNotFoundError:
        pass
    except Exception:
        pass


_load_fused()


def fused_count() -> int:
    return len(_FUSED)


def _first_user_text(messages: Optional[List[Dict[str, Any]]]) -> str:
    for m in messages or []:
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return " ".join(
                    str(b.get("text", "")) for b in c
                    if isinstance(b, dict) and b.get("type") == "text"
                )
    return ""


def _conversation_key(data: Dict[str, Any], messages: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    """Stable per-conversation key for the fuse latch.

    Prefer a caller-provided session id (if one ever appears in the request);
    otherwise fingerprint the first user message, which Hermes keeps stable
    through context compression (compression.protect_first_n).
    """
    for path in (("user",), ("litellm_session_id",), ("metadata", "session_id"),
                 ("metadata", "litellm_session_id")):
        v: Any = data
        for p in path:
            v = v.get(p) if isinstance(v, dict) else None
            if v is None:
                break
        if isinstance(v, str) and v:
            return "sid:" + v
    ft = _first_user_text(messages).strip()
    if ft:
        return "fp:" + hashlib.sha256(ft.encode("utf-8", "ignore")).hexdigest()[:32]
    return None


def _trip(key: str) -> None:
    if not key or key in _FUSED:
        return
    _FUSED.add(key)
    try:
        os.makedirs(os.path.dirname(FUSED_PATH), exist_ok=True)
        with open(FUSED_PATH, "a", encoding="utf-8") as fh:
            fh.write(key + "\n")
    except Exception:
        pass


def _all_text(messages: Optional[List[Dict[str, Any]]]) -> str:
    parts: List[str] = []
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "text":
                    parts.append(str(b.get("text", "")))
    return "\n".join(parts)


def evaluate(messages: Optional[List[Dict[str, Any]]]) -> Tuple[bool, Optional[str]]:
    """Return (is_sensitive, matched_rule_id). Used by the hook AND the admin UI."""
    _RULES.load()
    if not _RULES.enabled:
        return (False, None)
    text = _all_text(messages)
    if not text:
        return (False, None)
    low = text.lower()
    for rid, pat, luhn in _RULES.regex:
        m = pat.search(text)
        if m and (not luhn or _luhn_ok(m.group(0))):
            return (True, rid)
    for rid, pat in _RULES.phrase:
        if pat.search(low):
            return (True, rid)
    for rid, terms in _RULES.keyword:
        for t in terms:
            if t and t in low:
                return (True, rid)
    return (False, None)


def evaluate_text(text: str) -> Tuple[bool, Optional[str]]:
    return evaluate([{"role": "user", "content": text or ""}])


def _log_decision(rule_id: Optional[str], routed_model: str, text_len: int, note: str = "") -> None:
    try:
        os.makedirs(os.path.dirname(DECISIONS_LOG), exist_ok=True)
        rec = {
            "ts": time.time(),
            "routed_local": True,
            "rule": rule_id,
            "model": routed_model,
            "chars": text_len,
            "note": note,
        }
        with open(DECISIONS_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:
        pass


class PrivacyGuard(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):  # noqa: ANN001
        try:
            _RULES.load()
            if data.get("model") != _RULES.router_name:
                return data  # only intercept the router; direct model calls pass through
            local_model = _RULES.local_model
            messages = data.get("messages")
            key = _conversation_key(data, messages)
            # ── FUSE: a conversation already routed local stays local forever.
            if key and key in _FUSED:
                data["model"] = local_model
                _log_decision("fuse", local_model, len(_all_text(messages)))
                return data
            sensitive, rule_id = evaluate(messages)
            if sensitive:
                if key:
                    _trip(key)  # latch — one-way, can never return to cloud
                data["model"] = local_model
                _log_decision(rule_id, local_model, len(_all_text(messages)))
            return data
        except Exception as e:  # FAIL-SAFE: never leak on uncertainty
            try:
                if data.get("model") == _RULES.router_name:
                    data["model"] = _RULES.local_model
                    _log_decision(None, _RULES.local_model, 0, note=f"failsafe:{type(e).__name__}")
            except Exception:
                pass
            return data


proxy_handler_instance = PrivacyGuard()
