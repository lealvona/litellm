"""
thinking_control.py — litellm proxy callback (2026-06-23)

Translates Open WebUI's free-text ``reasoning_effort`` request param into each
model's NATIVE thinking-control param, injected via ``extra_body`` so it
SURVIVES the proxy's global ``drop_params: true`` (which otherwise strips
top-level ``reasoning_effort``/``thinking``; see proxy_config.yaml line ~292).

This makes the OWUI "Reasoning Effort" Advanced-Param (Settings → Advanced
Params → Reasoning Effort) a working thinking toggle for DIRECT model picks.

Convention (mirrors Hermes ``parse_reasoning_effort`` vocabulary so the two
agree):
  reasoning_effort absent / ""              -> leave the model's own default
  reasoning_effort in OFF_VALUES            -> thinking OFF
  reasoning_effort in minimal|low|medium|high|xhigh -> thinking ON (+effort where supported)
  any other free-text string                -> thinking ON, default effort

Scope: acts ONLY on DIRECT model picks. Requests whose model is the
smart-router are left untouched here — they are chosen by the auto_router and
handled by the separate Hermes/smart-router post-routing path (Phase B).

Composes with privacy_guard: both mutate ``data`` in place, so when
privacy_guard rewrites smart-router -> llm_115 (PII), this hook (registered
AFTER it) then applies the thinking param to the now-concrete llm_115 model.

Param shapes VERIFIED LIVE through the proxy 2026-06-23 (cache-busted):
  glm-5.2     extra_body.thinking={"type":"enabled"|"disabled"}        -> reasoning_content on/off
  the local vLLM endpoints extra_body.chat_template_kwargs={"enable_thinking":bool} -> reasoning_content on/off
  minimax-m3  reasons inline (no separate reasoning_content); accepts the param without error
  chatgpt-*   top-level reasoning_effort is natively supported (kept, not dropped)
  kimi-*      not toggleable -> the unsupported param is simply stripped
"""
import logging
import os
import json
import time

try:
    from litellm.integrations.custom_logger import CustomLogger
except Exception:  # import-soft so the admin UI can import this module standalone
    class CustomLogger:  # type: ignore
        pass

log = logging.getLogger("thinking_control")

# Content-free observability (parallels privacy_guard's decisions.jsonl): one
# line per applied toggle — which model got which thinking state, no prompt text.
_TLOG = os.path.expanduser("~/.local/state/litellm-thinking/decisions.jsonl")


def _log_thinking(model, state, effort, via):
    try:
        os.makedirs(os.path.dirname(_TLOG), exist_ok=True)
        with open(_TLOG, "a") as f:
            f.write(json.dumps({
                "ts": time.time(), "model": model, "state": state,
                "effort": effort, "via": via,
            }) + "\n")
    except Exception:
        pass

OFF_VALUES = {"none", "off", "false", "0", "no", "disable", "disabled"}
EFFORT_VALUES = {"minimal", "low", "medium", "high", "xhigh"}
ROUTER_NAMES = {"smart-router", "auto_router/smart-router"}


def _norm(effort):
    """Return ('off'|'on', effort_or_None), or (None, None) when there is no signal."""
    if effort is None:
        return None, None
    e = str(effort).strip().lower()
    if e == "":
        return None, None
    if e in OFF_VALUES:
        return "off", None
    if e in EFFORT_VALUES:
        return "on", e
    return "on", "medium"  # unknown free-text -> ON, default effort


def _base(model):
    """Strip any provider prefix litellm may prepend (e.g. 'openai/llm_115')."""
    if not model:
        return ""
    return (model.split("/")[-1] if "/" in model else model).strip().lower()


def apply_thinking(data, state, effort, model=None):
    """Mutate ``data`` in place: set the native thinking param for ``model``.

    ``model`` defaults to ``data['model']`` (direct-pick path). The auto_router
    passes the post-routing CHOSEN model explicitly, since at that point
    ``data['model']`` is still the router name. Pure + dependency-free so the
    admin UI / tests / the auto_router can all exercise it directly.
    """
    via = "router" if model is not None else "direct"
    model = _base(model if model is not None else data.get("model"))
    eb = data.get("extra_body")
    if not isinstance(eb, dict):
        eb = {}
    on = (state == "on")

    if model.startswith("glm"):
        eb["thinking"] = {"type": "enabled" if on else "disabled"}
        data.pop("reasoning_effort", None)
    elif model in ("llm_115", "llm_125") or model.startswith("qwen"):
        ctk = eb.get("chat_template_kwargs")
        if not isinstance(ctk, dict):
            ctk = {}
        ctk["enable_thinking"] = bool(on)
        eb["chat_template_kwargs"] = ctk
        data.pop("reasoning_effort", None)
    elif model.startswith("minimax"):
        eb["thinking"] = {"type": "enabled" if on else "disabled"}
        data.pop("reasoning_effort", None)
    elif model.startswith("chatgpt") or model.startswith("gpt"):
        # gpt reasoning can't be fully disabled; 'minimal' is the floor.
        # reasoning_effort is natively supported here, so keep it top-level.
        data["reasoning_effort"] = "minimal" if not on else (effort or "medium")
    elif model.startswith("kimi"):
        data.pop("reasoning_effort", None)  # not toggleable; strip so it can't error
    else:
        data.pop("reasoning_effort", None)  # unknown model: strip raw param defensively

    if eb:
        data["extra_body"] = eb
    _log_thinking(model, state, effort, via)
    return data


class ThinkingControl(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        try:
            state, effort = _norm(data.get("reasoning_effort"))
            if state is None:
                return data  # no signal -> respect the model's own default
            if _base(data.get("model")) in ROUTER_NAMES:
                return data  # smart-router path is handled post-routing (Phase B)
            apply_thinking(data, state, effort)
            return data
        except Exception as e:  # never break a request over a toggle
            log.warning("thinking_control skipped: %s", e)
            return data


proxy_handler_instance = ThinkingControl()
