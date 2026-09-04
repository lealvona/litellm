"""Privacy routing guard for a litellm smart-router deployment.

Registered as a litellm proxy callback (litellm_settings.callbacks). For any
request whose model is the router (default "smart-router"), it scans the ENTIRE
conversation for sensitive content (PII / secrets / personal-financial /
medical / mail). On a match it rewrites ``data["model"]`` to a LOCAL model so
the sensitive data never reaches a cloud provider. Non-matching requests pass
through unchanged → normal cloud semantic routing.

Design guarantees:
- CONFIGURABLE STICKINESS: ``fuse_mode=context`` scans the complete payload on
  every request and permits cloud routing again once sensitive text is no longer
  being sent (for example, after context compression). ``fuse_mode=permanent``
  retains the legacy one-way, disk-persisted conversation latch.
- STICKY scan: evaluates every message, so PII anywhere trips the fuse.
- FAIL-SAFE: any error → route local (never leak on uncertainty).
- DATA-DRIVEN: rules in privacy_rules.json (admin UI on :8094); hot-reloaded.
- ROLE-AWARE NUMERIC PII: ambiguous unformatted numbers in machine-generated
  tool output are not treated as phone numbers, while user-authored numbers and
  formatted phone numbers remain protected. Date-stamped technical identifiers
  are not treated as payment-card numbers merely because they pass Luhn.
- ROLE-AWARE EMAIL PII: user-authored email addresses remain protected. Bare
  addresses found only in machine-generated tool output do not force an entire
  documentation/repository turn local; explicit personal-mail intent in the
  user message still trips the separate ``personal_mail`` rule.
- PRIVACY-PRESERVING LOGS: decision log records matched-rule + routed-model +
  text length only — never the content itself.

- CONTEXT COMPACTION: a Hermes compaction request arrives as one giant
  machine-built user message, so scanning it raw would destroy every role-aware
  exemption above and let the guard trip on its own instruction text. The
  compressor fences the material it is summarizing; this module unwraps the
  fence and restores each turn's original role, so a compaction inherits the
  SAME routing decision the conversation itself would get. An unfenced payload
  is scanned whole — which routes local, the fail-safe direction.

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
    "fuse_mode": "permanent",
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
    def router_names(self) -> Tuple[str, ...]:
        """Every model name that should be privacy-routed.

        ``router_names`` (list) is preferred; ``router_name`` (str) stays
        supported so an existing rules file keeps working untouched.
        """
        raw = self.raw.get("router_names")
        if isinstance(raw, list):
            names = tuple(str(n).strip() for n in raw if str(n).strip())
            if names:
                return names
        return (self.router_name,)

    @property
    def compression_router(self) -> str:
        """Model name Hermes calls for context compaction, or "" to disable.

        Empty means the compaction endpoint is not managed here and whatever
        the proxy config points it at (a local model) is used unconditionally.
        """
        return str(self.raw.get("compression_router", "") or "").strip()

    @property
    def compression_cloud_model(self) -> str:
        """Summarizer used when a compaction payload carries nothing sensitive.

        Empty keeps compaction local even on a clean payload.
        """
        return str(self.raw.get("compression_cloud_model", "") or "").strip()

    @property
    def local_model(self) -> str:
        env_default = os.environ.get("PRIVACY_GUARD_LOCAL_MODEL", "")
        return self.raw.get("local_model") or env_default

    @property
    def fuse_mode(self) -> str:
        """Return ``context`` or the fail-safe legacy ``permanent`` mode."""
        mode = str(self.raw.get("fuse_mode", "permanent")).strip().lower()
        return "context" if mode == "context" else "permanent"


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


def _message_text(message: Dict[str, Any]) -> str:
    c = message.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(
            str(b.get("text", ""))
            for b in c
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _all_text(messages: Optional[List[Dict[str, Any]]]) -> str:
    parts: List[str] = []
    for m in messages or []:
        text = _message_text(m)
        if text:
            parts.append(text)
    return "\n".join(parts)


# ── Hermes context-compaction fence ──────────────────────────────────────
# A compaction request is a single machine-built user message: instruction
# scaffolding wrapped around role-labelled conversation material. Scanning it
# raw is wrong twice over — every tool-output address and build identifier
# would read as user-authored, and the guard would trip on its own instruction
# text. The compressor fences the material; the ``[ROLE]:`` labels inside
# restore the roles, so the compaction gets the same decision as the turn.
MATERIAL_BEGIN = "=== BEGIN CONVERSATION MATERIAL"
MATERIAL_END = "=== END CONVERSATION MATERIAL ==="

# Each fence declares what it holds, because the answer changes how it grades:
#   transcript — real turns, keeps its [ROLE]: labels
#   summary    — machine-written and already scrubbed of personal data, so it
#                is graded as tool output. Grading it strictly would pin every
#                later compaction local on nothing worse than a ten-digit
#                build id, which is exactly the kind of value a summary is
#                supposed to carry forward.
#   anything else (memory, focus, or an unlabelled fence) — graded strictly.
_MATERIAL_RE = re.compile(
    re.escape(MATERIAL_BEGIN) + r"(?::[ \t]*([a-z]+))?[ \t]*===\n?(.*?)\n?"
    + re.escape(MATERIAL_END),
    re.DOTALL,
)
_MACHINE_WRITTEN_KINDS = frozenset({"summary"})
# Labels emitted by the compressor's serializer: [USER]:, [ASSISTANT]:,
# [TOOL RESULT <id>]:, [SYSTEM]:, [INTERNAL CONTEXT]:.
_TRANSCRIPT_LABEL_RE = re.compile(
    r"^\[(USER|ASSISTANT|SYSTEM|INTERNAL CONTEXT|TOOL(?:\s+RESULT[^\]]*)?)\]:[ \t]?",
    re.MULTILINE,
)
_LABEL_ROLES = {
    "USER": "user",
    "ASSISTANT": "assistant",
    "SYSTEM": "system",
    # Synthetic turns the agent writes to itself — machine-generated, but
    # graded strictly rather than as tool output because their text is
    # assembled from the conversation.
    "INTERNAL CONTEXT": "assistant",
}


def _transcript_segments(block: str, kind: str = "") -> List[Tuple[str, str]]:
    """Split one fenced block back into ``(role, text)`` turns.

    A block with no ``[ROLE]:`` labels grades as ``user`` — the strict path —
    unless it declares itself machine-written, because derived text otherwise
    carries whatever the conversation carried.
    """
    matches = list(_TRANSCRIPT_LABEL_RE.finditer(block))
    if not matches:
        default_role = "tool" if kind in _MACHINE_WRITTEN_KINDS else "user"
        return [(default_role, block)]
    segments: List[Tuple[str, str]] = []
    head = block[: matches[0].start()]
    if head.strip():
        segments.append(("user", head))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        label = (m.group(1) or "").strip().upper()
        role = "tool" if label.startswith("TOOL") else _LABEL_ROLES.get(label, "user")
        body = block[m.end(): end]
        if body.strip():
            segments.append((role, body))
    return segments


def compaction_messages(text: str) -> Optional[List[Dict[str, Any]]]:
    """Rebuild role-attributed messages from a fenced compaction prompt.

    Returns ``None`` when no fence is present, so the caller falls back to
    scanning the whole request — stricter, never looser.
    """
    blocks = _MATERIAL_RE.findall(text or "")
    if not blocks:
        return None
    messages: List[Dict[str, Any]] = []
    for kind, block in blocks:
        for role, body in _transcript_segments(block, (kind or "").strip().lower()):
            messages.append({"role": role, "content": body})
    return messages


_PHONE_CONTEXT_RE = re.compile(
    r"\b(?:phone|telephone|mobile|cell|tel|sms|call|text(?:ing)?|whatsapp|signal)\b",
    re.IGNORECASE,
)
_TECHNICAL_DATE_ID_RE = re.compile(r"^\d{8}[-_]\d{6}$")
_SECRET_ASSIGNMENT_VALUE_RE = re.compile(
    r"\b(?:password|passwd|passphrase|secret|api[_ ]?key|access[_ ]?token|auth[_ ]?token|private[_ ]?key)\b"
    r"\s*(?:is|are|=|:)\s*(?P<value>\S+)",
    re.IGNORECASE,
)


# Roles whose text a MODEL wrote rather than a person. Only ``tool`` counts by
# default, which is right for a normal turn: the assistant messages there are
# the live conversation. Inside a COMPACTION payload the assistant turns are
# equally machine-generated transcript, and grading them as human authored
# meant a ten-digit build id quoted back by the assistant read as a phone
# number — enough to pin every compaction of a long technical conversation to
# the local model. Callers opt in; nothing changes for ordinary requests.
DEFAULT_MACHINE_ROLES: Tuple[str, ...] = ("tool",)
TRANSCRIPT_MACHINE_ROLES: Tuple[str, ...] = ("tool", "assistant", "system")


def _accept_regex_match(
    rule_id: str,
    match: "re.Match[str]",
    text: str,
    role: str,
    machine_roles: Tuple[str, ...] = DEFAULT_MACHINE_ROLES,
) -> bool:
    """Reject high-confidence numeric false positives without weakening secrets.

    Ten contiguous digits are common in timestamps, counters, ROM metadata, and
    JSON tool output. Treat them as a phone number when a human authored the
    message, when formatting makes the intent clear, or when nearby text says it
    is a phone-like value. Tool output otherwise needs stronger evidence.

    A compact ``YYYYMMDD-HHMMSS`` identifier can accidentally pass Luhn. It is a
    date-stamped technical id, not a payment card; every other Luhn-valid card
    candidate remains protected.
    """
    candidate = match.group(0)
    if rule_id == "email":
        # Documentation, source trees, and command output routinely contain
        # public maintainer/support addresses. Routing a 100K tool result local
        # because of one such address defeats the cloud router and caused a live
        # gpt-5.5 turn to fall back to llm_115. Protect user-authored addresses;
        # for private inbox workflows the user-side ``personal_mail`` phrase
        # rule trips before tool output is sent onward.
        if role in machine_roles:
            return False
    if rule_id == "phone":
        if role not in machine_roles:
            return True
        if re.search(r"[^\d]", candidate):
            return True
        context = text[max(0, match.start() - 48): match.end() + 48]
        return bool(_PHONE_CONTEXT_RE.search(context))
    if rule_id == "credit_card":
        compact = candidate.replace(" ", "")
        if _TECHNICAL_DATE_ID_RE.fullmatch(compact):
            return False
    return True


def _accept_phrase_match(
    rule_id: str,
    match: "re.Match[str]",
    text: str,
    role: str,
    machine_roles: Tuple[str, ...] = DEFAULT_MACHINE_ROLES,
) -> bool:
    """Distinguish secret values from non-secret references in machine text."""
    if rule_id != "secret_assignment" or role not in machine_roles:
        return True
    assignment = _SECRET_ASSIGNMENT_VALUE_RE.match(text, match.start())
    if assignment is None:
        return True
    value = assignment.group("value").strip("'\"`,;)")
    low = value.lower()
    is_path = bool(
        value.startswith(("/", "~/", "./", "../", "\\\\"))
        or re.match(r"^[A-Za-z]:[\\/]", value)
    )
    is_reference = bool(
        value.startswith(("$", "${", "os.environ/", "os.environ["))
        or low.startswith(("env:", "secret://", "vault://"))
        or re.match(r"^\{[A-Za-z_][A-Za-z0-9_]*\}(?:[/\\]|$)", value)
        or re.match(r"^%[A-Za-z_][A-Za-z0-9_]*%(?:[/\\]|$)", value)
    )
    is_placeholder = bool(
        low in {"none", "null", "unset", "missing", "redacted", "true", "false"}
        or value.startswith(("<", "[REDACTED]", "***"))
    )
    return not (is_path or is_reference or is_placeholder)


def evaluate(
    messages: Optional[List[Dict[str, Any]]],
    machine_roles: Tuple[str, ...] = DEFAULT_MACHINE_ROLES,
) -> Tuple[bool, Optional[str]]:
    """Return (is_sensitive, matched_rule_id). Used by the hook AND the admin UI."""
    _RULES.load()
    if not _RULES.enabled:
        return (False, None)
    message_texts = [
        (str(message.get("role") or ""), _message_text(message))
        for message in messages or []
    ]
    message_texts = [(role, text) for role, text in message_texts if text]
    if not message_texts:
        return (False, None)
    for role, text in message_texts:
        low = text.lower()
        for rid, pat, luhn in _RULES.regex:
            for match in pat.finditer(text):
                if luhn and not _luhn_ok(match.group(0)):
                    continue
                if _accept_regex_match(rid, match, text, role, machine_roles):
                    return (True, rid)
        for rid, pat in _RULES.phrase:
            for match in pat.finditer(low):
                if _accept_phrase_match(rid, match, text, role, machine_roles):
                    return (True, rid)
        for rid, terms in _RULES.keyword:
            for term in terms:
                if term and term in low:
                    return (True, rid)
    return (False, None)


def evaluate_text(text: str) -> Tuple[bool, Optional[str]]:
    return evaluate([{"role": "user", "content": text or ""}])


def _log_decision(
    rule_id: Optional[str],
    routed_model: str,
    text_len: int,
    note: str = "",
    routed_local: bool = True,
) -> None:
    try:
        os.makedirs(os.path.dirname(DECISIONS_LOG), exist_ok=True)
        rec = {
            "ts": time.time(),
            "routed_local": routed_local,
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
    def _handle_compaction(self, data: Dict[str, Any], messages: Any) -> Dict[str, Any]:
        """Route a context-compaction call the way its own material grades.

        Compaction is where a conversation's sensitive text would otherwise
        leak wholesale: the summarizer is handed the entire middle of the
        transcript at once. So it gets the same verdict the conversation gets
        — local while sensitive material is still present, the cloud
        summarizer once it is not. Because the compaction prompt instructs the
        summarizer to redact, a context normally clears itself after one local
        pass and later compactions run on the faster model.

        The permanent-fuse mode never promotes: a latched conversation is
        pinned local by definition, and a compaction is not the place to
        relitigate that.
        """
        local_model = _RULES.local_model
        payload_len = len(_all_text(messages))
        if _RULES.fuse_mode == "permanent":
            data["model"] = local_model
            _log_decision("fuse", local_model, payload_len, note="task:compaction")
            return data
        fenced = compaction_messages(_all_text(messages))
        if fenced is not None:
            sensitive, rule_id = evaluate(fenced, TRANSCRIPT_MACHINE_ROLES)
        else:
            sensitive, rule_id = evaluate(messages)
        cloud_model = _RULES.compression_cloud_model
        if sensitive or not cloud_model:
            data["model"] = local_model
            _log_decision(
                rule_id,
                local_model,
                payload_len,
                note="task:compaction fenced:%s" % ("yes" if fenced is not None else "no"),
            )
            return data
        data["model"] = cloud_model
        _log_decision(
            None,
            cloud_model,
            payload_len,
            note="task:compaction clean",
            routed_local=False,
        )
        return data

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):  # noqa: ANN001
        try:
            _RULES.load()
            requested = data.get("model")
            compaction_router = _RULES.compression_router
            if compaction_router and requested == compaction_router:
                return self._handle_compaction(data, data.get("messages"))
            if requested not in _RULES.router_names:
                return data  # only intercept the router; direct model calls pass through
            local_model = _RULES.local_model
            messages = data.get("messages")
            key = _conversation_key(data, messages)
            # The legacy permanent fuse is opt-in. Context mode is still sticky
            # across the entire payload, but does not pin a conversation local
            # after sensitive text has left the request.
            permanent_fuse = _RULES.fuse_mode == "permanent"
            if permanent_fuse and key and key in _FUSED:
                data["model"] = local_model
                _log_decision("fuse", local_model, len(_all_text(messages)))
                return data
            sensitive, rule_id = evaluate(messages)
            if sensitive:
                if permanent_fuse and key:
                    _trip(key)  # latch — one-way, can never return to cloud
                data["model"] = local_model
                _log_decision(
                    rule_id,
                    local_model,
                    len(_all_text(messages)),
                    note=f"fuse_mode:{_RULES.fuse_mode}",
                )
            return data
        except Exception as e:  # FAIL-SAFE: never leak on uncertainty
            try:
                guarded = set(_RULES.router_names)
                if _RULES.compression_router:
                    guarded.add(_RULES.compression_router)
                if data.get("model") in guarded:
                    data["model"] = _RULES.local_model
                    _log_decision(None, _RULES.local_model, 0, note=f"failsafe:{type(e).__name__}")
            except Exception:
                pass
            return data


proxy_handler_instance = PrivacyGuard()
