"""Dunsinane attribution callback for litellm-proxy (phase-2 chokepoint).

Log-only by construction: no async_pre_call_hook, so ZERO pre-call cost.
Both handlers run in litellm's GLOBAL_LOGGING_WORKER off the hot path, and
the blocking POST is pushed through asyncio.to_thread so the proxy's event
loop never waits on the registrar. Failure direction is OPEN twice over:
exceptions here are swallowed by litellm's [Non-Blocking] handler, and the
POST swallows its own.

PRIVACY: standard_logging_object carries FULL message content. This module
reads only scalars from it -- no prompt or completion text ever leaves the
proxy process. That constraint is load-bearing; do not add fields that
quote messages.

Deployed copy: ~/src/litellm-fork/dunsinane_guard.py (litellm resolves
callbacks from its CWD, same as privacy_guard.py -- NOT the uv tool env).
Source of record: ~/src/core-orchestration-unit/guards/. Keep them in sync.
"""
import asyncio
import http.client
import json

from litellm.integrations.custom_logger import CustomLogger


def _post(payload):
    """Blocking; runs only on a to_thread executor thread."""
    try:
        c = http.client.HTTPConnection("127.0.0.1", 8620, timeout=1.0)
        c.request("POST", "/event", json.dumps(payload),
                  {"Content-Type": "application/json"})
        c.getresponse().read()
        c.close()
    except Exception:
        pass


class DunsinaneGuard(CustomLogger):
    async def _record(self, kwargs, status, start_time, end_time):
        slp = kwargs.get("standard_logging_object") or {}
        md = slp.get("metadata") or {}
        try:
            dur_ms = int((end_time - start_time).total_seconds() * 1000)
        except Exception:
            dur_ms = 0
        attrs = {
            "status": status,
            "model": slp.get("model"),
            "model_group": slp.get("model_group"),
            "api_base": slp.get("api_base"),
            "in": slp.get("prompt_tokens"),
            "out": slp.get("completion_tokens"),
            "cost": slp.get("response_cost"),
            "key_alias": md.get("user_api_key_alias"),
            "key_hash": md.get("user_api_key_hash"),
            "user_id": md.get("user_api_key_user_id"),
            "team_id": md.get("user_api_key_team_id"),
            "peer": slp.get("requester_ip_address"),
            "ua": (slp.get("user_agent") or "")[:120] or None,
            "call_id": slp.get("litellm_call_id"),
        }
        payload = {"span": "litellm", "kind": status, "dur_ms": dur_ms,
                   "attrs": {k: v for k, v in attrs.items() if v is not None}}
        await asyncio.to_thread(_post, payload)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        await self._record(kwargs, "success", start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        await self._record(kwargs, "failure", start_time, end_time)


proxy_handler_instance = DunsinaneGuard()
