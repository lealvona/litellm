#!/usr/bin/env python3
"""Register the GATED cloud chat models (DeepSeek, Qwen via DeepInfra) on the agent-facing proxy.

These are explicitly addressable ONLY:
  - never added to the auto_router (smart-router) routes or to any fallback chain,
    so no agent or router ever reaches them unless it names them;
  - every deployment carries a hard DAILY budget (max_budget + budget_duration)
    and an rpm cap, enforced by the router's budget limiter on the provider's
    price per token (set explicitly below — the models are newer than
    LiteLLM's price map, and without a price the budget never moves).

Idempotent: adds missing models, leaves existing ones alone (use --update to
re-apply the specs to models that already exist).

Needs: the proxy admin key (LITELLM_ADMIN_KEY_FILE, default
~/.config/litellm/admin.key) and DEEPINFRA_API_KEY in the environment. The key is
stored encrypted in the proxy DB; it is never printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

PROXY_URL = os.environ.get("LITELLM_PROXY_URL", "http://127.0.0.1:4000")
ADMIN_KEY_FILE = Path(os.environ.get("LITELLM_ADMIN_KEY_FILE", "~/.config/litellm/admin.key")).expanduser()

# (public name, DeepInfra model id, USD per 1M input, USD per 1M output, USD budget per day, note)
# Prices: DeepInfra model catalog (GET /v1/openai/models metadata.pricing), 2026-10-02.
GATED = [
    ("deepseek-v4-pro", "deepseek-ai/DeepSeek-V4-Pro", 1.30, 2.60, 1.00, "DeepSeek V4 Pro, 1M context"),
    ("deepseek-v4.1-flash", "deepseek-ai/DeepSeek-V4.1-Flash", 0.20, 0.60, 0.50, "DeepSeek V4.1 Flash, 1M context, vision"),
    ("qwen3.8-max", "Qwen/Qwen3.8-Max", 1.65, 4.951, 1.00, "Qwen3.8 Max, 256K context"),
    ("qwen3.8-flash", "Qwen/Qwen3.8-Flash", 0.113, 0.382, 0.50, "Qwen3.8 Flash, 1M context"),
]
RPM = 10


def spec(name: str, model_id: str, cin: float, cout: float, budget: float, note: str, key: str) -> dict:
    return {
        "model_name": name,
        "litellm_params": {
            "model": f"deepinfra/{model_id}",
            "api_key": key,
            "input_cost_per_token": cin / 1e6,
            "output_cost_per_token": cout / 1e6,
            "max_budget": budget,
            "budget_duration": "1d",
            "rpm": RPM,
        },
        "model_info": {
            "mode": "chat",
            "description": f"{note} @ DeepInfra. GATED: explicit use only (not in smart-router or fallbacks); "
                           f"${budget:.2f}/day budget, {RPM} rpm.",
            "gated_cloud": True,
        },
    }


def call(path: str, key: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        f"{PROXY_URL}{path}", method="POST" if body is not None else "GET",
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--update", action="store_true", help="re-apply specs to models that already exist")
    args = ap.parse_args()
    admin = ADMIN_KEY_FILE.read_text().strip()
    di_key = os.environ.get("DEEPINFRA_API_KEY", "")
    if not di_key:
        print("DEEPINFRA_API_KEY is not set", file=sys.stderr)
        return 2
    existing = {m["model_name"]: m for m in call("/model/info", admin).get("data", [])}
    for row in GATED:
        s = spec(*row, key=di_key)
        name = s["model_name"]
        if name in existing:
            if not args.update:
                print(f"skip   {name} (present)")
                continue
            mid = (existing[name].get("model_info") or {}).get("id")
            call("/model/update", admin, {**s, "model_info": {**s["model_info"], "id": mid}})
            print(f"update {name}")
            continue
        r = call("/model/new", admin, s)
        print(f"added  {name} id={str((r.get('model_info') or {}).get('id') or r.get('model_id'))[:12]} "
              f"budget=${s['litellm_params']['max_budget']}/day rpm={RPM}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
