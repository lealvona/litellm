"""Custom LiteLLM image providers for the generation gateway (gen_config.yaml).

Serves three cloud image back ends behind the OpenAI images API
(POST /v1/images/generations), so any OpenAI-compatible client works:

  comfycloud/<graph>         Comfy Cloud (cloud.comfy.org): ComfyUI API-format graphs
  runware/<AIR id>           Runware task API (imageInference)
  deepinfra_native/<model>   DeepInfra native inference route

Why custom handlers instead of LiteLLM's openai/ provider: Comfy Cloud speaks
graphs, Runware is task-based, and DeepInfra's OpenAI-compatible images route
returns no cost while its native route reports the exact USD cost per request.

COST. Every handler sets response._hidden_params["response_cost"] (USD), which
LiteLLM's logging uses instead of its price map, so virtual-key budgets and
per-deployment budgets are enforced on real spend:
  - Runware: includeCost:true -> exact USD per result.
  - DeepInfra native: inference_status.cost -> exact USD.
  - Comfy Cloud: execution seconds x 0.266 credits/s (published GPU rate)
    / 211 credits per USD (published conversion; a 5 s Kling v2.5 turbo pro
    clip = 73.85 credits = $0.35 cross-checks it). The GPU rate is the low end
    of the published band, so this can undercount on premium GPUs.

Graphs are ported from a production Comfy Cloud adapter that validated them
against cloud.comfy.org. API keys arrive as litellm_params.api_key
(os.environ/... in the config) and are never logged or echoed in errors.
"""
from __future__ import annotations

import asyncio
import base64
import os
import time
import uuid
from typing import Any, Optional

import httpx
import litellm
from litellm import CustomLLM
from litellm.llms.custom_llm import CustomLLMError
from litellm.types.utils import ImageObject, ImageResponse

MAX_N = 4

COMFY_BASE = "https://cloud.comfy.org/api"
COMFY_CREDITS_PER_GPU_SEC = float(os.environ.get("GEN_COMFY_CREDITS_PER_GPU_SEC", "0.266"))
COMFY_CREDITS_PER_USD = float(os.environ.get("GEN_COMFY_CREDITS_PER_USD", "211"))
COMFY_TERMINAL = {"success", "error", "non_retryable_error", "lost", "cancelled"}
COMFY_POLL_CAP_S = 600

RUNWARE_URL = "https://api.runware.ai/v1"
DEEPINFRA_NATIVE = "https://api.deepinfra.com/v1/inference/"

# FLUX.1 text encoders / VAE shared by the "flux" graphs
FLUX_CLIP1 = "clip_l.safetensors"
FLUX_CLIP2 = "t5xxl_fp8_e4m3fn_scaled.safetensors"
FLUX_VAE = "ae.safetensors"
DETAIL_LORA = ("flux1-adjust_details_and_photorealism.safetensors", 0.55)

COMFY_MODELS: dict[str, dict[str, Any]] = {
    # photoreal default: Krea dev, guidance 4.5 (its card default), detail LoRA
    "flux-krea": {"graph": "flux", "unet": "flux1-krea-dev_fp8_scaled.safetensors",
                  "steps": 26, "guidance": 4.5, "loras": [DETAIL_LORA]},
    "flux-dev": {"graph": "flux", "ckpt": "flux1-dev-fp8.safetensors",
                 "steps": 25, "guidance": 3.5, "loras": [DETAIL_LORA]},
    # schnell is guidance-free (the official workflow has no FluxGuidance node)
    "flux-schnell": {"graph": "flux", "ckpt": "flux1-schnell-fp8.safetensors",
                     "steps": 4, "no_guidance": True},
    # FLUX.2 klein 9B: 4-step distilled, CFGGuider at cfg 1 with a zeroed negative
    "flux2-klein": {"graph": "flux2", "unet": "flux-2-klein-9b.safetensors",
                    "clip": "qwen_3_8b_fp8mixed.safetensors", "steps": 4, "klein": True},
    "flux2-dev": {"graph": "flux2", "unet": "flux2_dev_fp8mixed.safetensors",
                  "clip": "mistral_3_small_flux2_fp8.safetensors", "steps": 20, "guidance": 4.0},
}


# TTS pricing for the speech models served through LiteLLM's openai/ route. Their
# names are not in LiteLLM's price map, so without this the speech cost path finds
# no input_cost_per_character and books $0 — and budgets would never see TTS spend.
TTS_PRICES_PER_CHAR = {"Qwen/Qwen3-TTS": 20.0 / 1e6, "hexgrad/Kokoro-82M": 0.62 / 1e6}
litellm.register_model({
    key: {"mode": "audio_speech", "litellm_provider": "openai", "input_cost_per_character": price}
    for name, price in TTS_PRICES_PER_CHAR.items() for key in (name, f"openai/{name}")
})

# ---------------------------------------------------------------- helpers

def _params(optional_params: dict, logging_obj: Any) -> dict:
    """LiteLLM strips the OpenAI image params (n, size, quality) before calling a
    custom provider — its image param mapper only passes them to providers it knows.
    Recover them from the original proxy request body, which it keeps on the
    logging object."""
    p = dict(optional_params or {})
    try:
        lp = getattr(logging_obj, "model_call_details", {}).get("litellm_params") or {}
        body = (lp.get("proxy_server_request") or {}).get("body") or {}
        for k in ("n", "size", "quality", "seed"):
            if p.get(k) is None and body.get(k) is not None:
                p[k] = body[k]
    except Exception:
        pass
    return p


def _short(model: str) -> str:
    """LiteLLM may hand us 'provider/name' or just 'name'."""
    for prefix in ("comfycloud/", "runware/", "deepinfra_native/"):
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


def _n(optional_params: dict) -> int:
    try:
        return max(1, min(MAX_N, int(optional_params.get("n") or 1)))
    except (TypeError, ValueError):
        return 1


def _size(optional_params: dict, mult: int) -> tuple[int, int]:
    raw = str(optional_params.get("size") or "1024x1024").lower()
    try:
        w, h = (int(x) for x in raw.split("x", 1))
    except ValueError:
        w = h = 1024
    return tuple(max(256, min(2048, int(round(v / mult)) * mult)) for v in (w, h))  # type: ignore[return-value]


def _seed(optional_params: dict) -> Optional[int]:
    extra = optional_params.get("extra_body") or {}
    s = optional_params.get("seed", extra.get("seed"))
    try:
        return None if s is None else int(s) % (2 ** 31)
    except (TypeError, ValueError):
        return None


def _response(images_b64: list[str], usd: float, provider: str, **extra: Any) -> ImageResponse:
    resp = ImageResponse(created=int(time.time()), data=[ImageObject(b64_json=b) for b in images_b64])
    usd = round(float(usd or 0.0), 6)
    resp._hidden_params["response_cost"] = usd
    resp._hidden_params.setdefault("additional_headers", {})["llm_provider-x-litellm-response-cost"] = usd
    resp._hidden_params["gen_provider"] = provider
    resp._hidden_params.update(extra)
    return resp


def _need_key(api_key: Optional[str], provider: str) -> str:
    if not api_key:
        raise CustomLLMError(status_code=500, message=f"{provider}: api key not configured")
    return api_key


def _run_sync(coro):
    """Sync entry points are rarely used by the proxy (it calls the async ones)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise CustomLLMError(status_code=500, message="sync image_generation called inside a running loop")


# ---------------------------------------------------------------- Comfy Cloud

def _flux_graph(cfg: dict, prompt: str, w: int, h: int, n: int, seed: int) -> dict:
    g: dict = {}
    if cfg.get("unet"):
        g["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": cfg["unet"], "weight_dtype": "default"}}
        g["1c"] = {"class_type": "DualCLIPLoader", "inputs": {
            "clip_name1": FLUX_CLIP1, "clip_name2": FLUX_CLIP2, "type": "flux"}}
        g["1v"] = {"class_type": "VAELoader", "inputs": {"vae_name": FLUX_VAE}}
        model, clip, vae = ["1", 0], ["1c", 0], ["1v", 0]
    else:
        g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": cfg["ckpt"]}}
        model, clip, vae = ["1", 0], ["1", 1], ["1", 2]
    for i, (name, strength) in enumerate(cfg.get("loras") or []):
        g[f"lr{i}"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": model, "lora_name": name, "strength_model": float(strength)}}
        model = [f"lr{i}", 0]
    g["2"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": clip}}
    if cfg.get("no_guidance"):
        pos = ["2", 0]
    else:
        g["3"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": ["2", 0], "guidance": cfg["guidance"]}}
        pos = ["3", 0]
    # at CFG 1.0 a negative text does nothing on flux; ZeroOut is the correct wiring
    g["4"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["2", 0]}}
    g["5"] = {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": n}}
    g["6"] = {"class_type": "KSampler", "inputs": {
        "model": model, "positive": pos, "negative": ["4", 0], "latent_image": ["5", 0],
        "seed": seed, "steps": cfg["steps"], "cfg": 1.0,
        "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}}
    g["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": vae}}
    g["8"] = {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "gen"}}
    return g


def _flux2_graph(cfg: dict, prompt: str, w: int, h: int, n: int, seed: int) -> dict:
    g: dict = {
        "u": {"class_type": "UNETLoader", "inputs": {"unet_name": cfg["unet"], "weight_dtype": "default"}},
        "c": {"class_type": "CLIPLoader", "inputs": {"clip_name": cfg["clip"], "type": "flux2", "device": "default"}},
        "v": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "t": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["c", 0]}},
        "lat": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": w, "height": h, "batch_size": n}},
        "sch": {"class_type": "Flux2Scheduler", "inputs": {"steps": cfg["steps"], "width": w, "height": h}},
        "smp": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "nz": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
    }
    if cfg.get("klein"):
        g["z"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["t", 0]}}
        g["gd"] = {"class_type": "CFGGuider", "inputs": {
            "model": ["u", 0], "positive": ["t", 0], "negative": ["z", 0], "cfg": 1.0}}
    else:
        g["fg"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": ["t", 0], "guidance": cfg["guidance"]}}
        g["gd"] = {"class_type": "BasicGuider", "inputs": {"model": ["u", 0], "conditioning": ["fg", 0]}}
    g["sca"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["nz", 0], "guider": ["gd", 0], "sampler": ["smp", 0],
        "sigmas": ["sch", 0], "latent_image": ["lat", 0]}}
    g["d"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sca", 0], "vae": ["v", 0]}}
    g["sv"] = {"class_type": "SaveImage", "inputs": {"images": ["d", 0], "filename_prefix": "gen"}}
    return g


class ComfyCloudImages(CustomLLM):
    def image_generation(self, model, prompt, api_key, api_base, model_response, optional_params,
                         logging_obj, timeout=None, client=None) -> ImageResponse:
        return _run_sync(self.aimage_generation(model, prompt, model_response, api_key, api_base,
                                                optional_params, logging_obj, timeout))

    async def aimage_generation(self, model, prompt, model_response, api_key, api_base, optional_params,
                                logging_obj, timeout=None, client=None) -> ImageResponse:
        key = _need_key(api_key, "comfycloud")
        optional_params = _params(optional_params, logging_obj)
        name = _short(model)
        cfg = COMFY_MODELS.get(name)
        if not cfg:
            raise CustomLLMError(status_code=400, message=f"comfycloud: unknown graph '{name}' "
                                                          f"(known: {', '.join(sorted(COMFY_MODELS))})")
        w, h = _size(optional_params, 16)
        n = _n(optional_params)
        seed = _seed(optional_params)
        seed = uuid.uuid4().int % (2 ** 31) if seed is None else seed
        build = _flux2_graph if cfg["graph"] == "flux2" else _flux_graph
        graph = build(cfg, prompt, w, h, n, seed)
        headers = {"X-API-Key": key}
        async with httpx.AsyncClient(timeout=120) as http:
            resp = await http.post(f"{COMFY_BASE}/prompt", headers=headers,
                                   json={"prompt": graph, "extra_data": {"api_key_comfy_org": key}})
            if resp.status_code == 402:
                raise CustomLLMError(status_code=402, message="comfycloud: insufficient credits")
            if resp.status_code == 429:
                raise CustomLLMError(status_code=429, message="comfycloud: concurrency limit or inactive plan")
            try:
                body = resp.json()
            except ValueError:
                raise CustomLLMError(status_code=502, message=f"comfycloud: non-JSON reply HTTP {resp.status_code}")
            if resp.status_code >= 400 or body.get("error") or not body.get("prompt_id"):
                detail = str(body.get("error") or body.get("node_errors") or body)[:600]
                raise CustomLLMError(status_code=400 if resp.status_code < 500 else 502,
                                     message=f"comfycloud rejected the graph: {detail}")
            pid = body["prompt_id"]
            delay, waited, status = 2.0, 0.0, ""
            while waited < COMFY_POLL_CAP_S:
                await asyncio.sleep(delay)
                waited += delay
                delay = min(delay * 1.3, 6.0)
                st = await http.get(f"{COMFY_BASE}/job/{pid}/status", headers=headers)
                status = (st.json() or {}).get("status", "") if st.status_code < 400 else ""
                if status in COMFY_TERMINAL:
                    break
            else:
                try:   # stop the provider-side job: a timed-out job must not keep burning credits
                    await http.post(f"{COMFY_BASE}/queue", headers=headers, json={"delete": [pid]})
                except Exception:
                    pass
                raise CustomLLMError(status_code=504, message=f"comfycloud: job {pid} timed out")
            rec = (await http.get(f"{COMFY_BASE}/jobs/{pid}", headers=headers)).json() or {}
            if status != "success":
                detail = str(rec.get("execution_status") or rec.get("error") or status)[:600]
                raise CustomLLMError(status_code=502, message=f"comfycloud job {pid} ended {status}: {detail}")
            files = [it for node in (rec.get("outputs") or {}).values() for val in node.values()
                     if isinstance(val, list) for it in val
                     if isinstance(it, dict) and it.get("filename") and it.get("type") in (None, "output")]
            images = []
            for f in files:
                v = await http.get(f"{COMFY_BASE}/view", headers=headers, follow_redirects=False,
                                   params={"filename": f["filename"], "subfolder": f.get("subfolder") or "",
                                           "type": "output"})
                if v.status_code in (301, 302, 303, 307, 308):
                    # the signed URL is fetched WITHOUT the api key
                    v = await http.get(v.headers["location"], follow_redirects=True)
                if v.status_code != 200:
                    raise CustomLLMError(status_code=502, message=f"comfycloud: download HTTP {v.status_code}")
                images.append(base64.b64encode(v.content).decode())
        if not images:
            raise CustomLLMError(status_code=502, message=f"comfycloud job {pid} produced no images")
        try:
            secs = (rec["execution_end_time"] - rec["execution_start_time"]) / 1000.0
        except (KeyError, TypeError):
            secs = waited
        credits = secs * COMFY_CREDITS_PER_GPU_SEC
        return _response(images, credits / COMFY_CREDITS_PER_USD, "comfycloud",
                         provider_ref=pid, provider_credits=round(credits, 3), gpu_seconds=round(secs, 2))


# ---------------------------------------------------------------- Runware

class RunwareImages(CustomLLM):
    def image_generation(self, model, prompt, api_key, api_base, model_response, optional_params,
                         logging_obj, timeout=None, client=None) -> ImageResponse:
        return _run_sync(self.aimage_generation(model, prompt, model_response, api_key, api_base,
                                                optional_params, logging_obj, timeout))

    async def aimage_generation(self, model, prompt, model_response, api_key, api_base, optional_params,
                                logging_obj, timeout=None, client=None) -> ImageResponse:
        key = _need_key(api_key, "runware")
        optional_params = _params(optional_params, logging_obj)
        w, h = _size(optional_params, 64)
        task = {"taskType": "imageInference", "taskUUID": str(uuid.uuid4()), "model": _short(model),
                "positivePrompt": prompt, "width": w, "height": h, "numberResults": _n(optional_params),
                "outputType": "base64Data", "outputFormat": "PNG", "includeCost": True}
        seed = _seed(optional_params)
        if seed is not None:
            task["seed"] = seed
        async with httpx.AsyncClient(timeout=180) as http:
            resp = await http.post(RUNWARE_URL, json=[task], headers={"Authorization": f"Bearer {key}"})
        try:
            body = resp.json()
        except ValueError:
            raise CustomLLMError(status_code=502, message=f"runware: non-JSON reply HTTP {resp.status_code}")
        errors = body.get("errors") or []
        if errors or resp.status_code >= 400:
            msg = "; ".join(str(e.get("message") or e) for e in errors) or f"HTTP {resp.status_code}"
            raise CustomLLMError(status_code=400 if resp.status_code < 500 else 502, message=f"runware: {msg[:600]}")
        data = body.get("data") or []
        images = [d["imageBase64Data"] for d in data if d.get("imageBase64Data")]
        if not images:
            raise CustomLLMError(status_code=502, message="runware returned no images")
        cost = sum(float(d.get("cost") or 0) for d in data)
        return _response(images, cost, "runware", provider_ref=task["taskUUID"])


# ---------------------------------------------------------------- DeepInfra (native route)

class DeepInfraImages(CustomLLM):
    def image_generation(self, model, prompt, api_key, api_base, model_response, optional_params,
                         logging_obj, timeout=None, client=None) -> ImageResponse:
        return _run_sync(self.aimage_generation(model, prompt, model_response, api_key, api_base,
                                                optional_params, logging_obj, timeout))

    async def aimage_generation(self, model, prompt, model_response, api_key, api_base, optional_params,
                                logging_obj, timeout=None, client=None) -> ImageResponse:
        key = _need_key(api_key, "deepinfra")
        optional_params = _params(optional_params, logging_obj)
        w, h = _size(optional_params, 32)
        payload: dict = {"prompt": prompt, "num_images": _n(optional_params), "width": w, "height": h}
        seed = _seed(optional_params)
        if seed is not None:
            payload["seed"] = seed
        async with httpx.AsyncClient(timeout=180) as http:
            resp = await http.post(DEEPINFRA_NATIVE + _short(model), json=payload,
                                   headers={"Authorization": f"Bearer {key}"})
        if resp.status_code >= 400:
            raise CustomLLMError(status_code=400 if resp.status_code < 500 else 502,
                                 message=f"deepinfra HTTP {resp.status_code}: {resp.text[:400]}")
        body = resp.json()
        # most models answer {"images": [data-URI | URL, ...]}; partner models such as
        # FLUX.2 pro answer {"image_url": "<CDN URL>"} instead
        items = [i for i in (body.get("images") or []) if isinstance(i, str)]
        items += [u for u in [body.get("image_url"), *(body.get("image_urls") or [])] if isinstance(u, str)]
        images = []
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as http:
            for item in items:
                if item.startswith("data:"):
                    images.append(item.split(",", 1)[1])
                elif item.startswith("http"):
                    r = await http.get(item)
                    if r.status_code != 200:
                        raise CustomLLMError(status_code=502, message=f"deepinfra: image download HTTP {r.status_code}")
                    images.append(base64.b64encode(r.content).decode())
        if not images:
            raise CustomLLMError(status_code=502, message="deepinfra returned no images")
        cost = (body.get("inference_status") or {}).get("cost")
        return _response(images, float(cost or 0.0), "deepinfra", cost_reported=cost is not None)


comfy_handler = ComfyCloudImages()
runware_handler = RunwareImages()
deepinfra_handler = DeepInfraImages()
