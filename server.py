#!/usr/bin/env python3
"""
SAE Interpretability Lab — FastAPI server
Replaces the Gradio interface with a clean REST API + static HTML/JS/CSS frontend.
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import asyncio
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

from config import (
    SparseAutoencoder,
    MODEL_NAME, TARGET_LAYER, HOOK_TYPE,
    MEDICAL_OUTPUT_DIR, get_device,
)

_SPECIAL_TOKENS = {
    "<|endoftext|>", "<|begin_of_text|>", "<|end_of_text|>",
    "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>", "",
}

SRC_DIR = Path(__file__).parent / "src"


# =============================================================================
# State
# =============================================================================

class ServerState:
    def __init__(self):
        self.model = None
        self.sae = None
        self.labeled_features: List[Dict] = []
        self.feature_by_index: Dict[int, Dict] = {}
        self.device = get_device()
        self.loaded = False
        self.error: Optional[str] = None

    def load(self):
        if self.loaded or self.error:
            return
        sae_path = MEDICAL_OUTPUT_DIR / "sae.pt"
        if not sae_path.exists():
            self.error = "SAE not found. Run: python main.py"
            return
        try:
            from transformer_lens import HookedTransformer
            print("Loading model...")
            self.model = HookedTransformer.from_pretrained(MODEL_NAME, device=self.device)
            self.model.eval()
            print("Loading SAE...")
            self.sae = SparseAutoencoder.load(sae_path)
            self.sae.to(self.device)
            self.sae.eval()
            labeled_path = MEDICAL_OUTPUT_DIR / "labeled_features.json"
            if labeled_path.exists():
                with open(labeled_path) as f:
                    self.labeled_features = json.load(f)
            features_path = MEDICAL_OUTPUT_DIR / "features.json"
            if features_path.exists():
                with open(features_path) as f:
                    all_feats = json.load(f)
                self.feature_by_index = {feat["index"]: feat for feat in all_feats}
                label_map = {f["index"]: f for f in self.labeled_features}
                for feat in all_feats:
                    if feat["index"] in label_map:
                        lbl = label_map[feat["index"]]
                        self.feature_by_index[feat["index"]]["label"] = lbl.get("label", "")
                        self.feature_by_index[feat["index"]]["confidence"] = lbl.get("confidence", "low")
                        self.feature_by_index[feat["index"]]["reasoning"] = lbl.get("reasoning", "")
            self.loaded = True
            print(f"Ready — {len(self.labeled_features)} labeled features")
        except Exception as e:
            self.error = str(e)
            print(f"Load error: {e}")

    def load_features_only(self):
        """Load just the JSON data (no GPU needed) for browse/search endpoints."""
        if self.feature_by_index:
            return
        labeled_path = MEDICAL_OUTPUT_DIR / "labeled_features.json"
        if labeled_path.exists():
            with open(labeled_path) as f:
                self.labeled_features = json.load(f)
        features_path = MEDICAL_OUTPUT_DIR / "features.json"
        if features_path.exists():
            with open(features_path) as f:
                all_feats = json.load(f)
            self.feature_by_index = {feat["index"]: feat for feat in all_feats}
            # Merge labels
            label_map = {f["index"]: f for f in self.labeled_features}
            for feat in all_feats:
                if feat["index"] in label_map:
                    lbl = label_map[feat["index"]]
                    self.feature_by_index[feat["index"]]["label"] = lbl.get("label", "")
                    self.feature_by_index[feat["index"]]["confidence"] = lbl.get("confidence", "low")
                    self.feature_by_index[feat["index"]]["reasoning"] = lbl.get("reasoning", "")


srv = ServerState()


# =============================================================================
# Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    srv.load_features_only()
    # Defer heavy model loading to first request
    yield


app = FastAPI(lifespan=lifespan, title="SAE Lab API")


# Disable caching for all static assets so frontend changes are always picked up
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        if request.url.path.startswith(("/styles/", "/scripts/")):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

app.add_middleware(NoCacheMiddleware)


# =============================================================================
# Pydantic models
# =============================================================================

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []


class SteerRequest(BaseModel):
    prompt: str
    features: Dict[str, float]   # str keys because JSON objects have string keys
    max_tokens: int = 150


class CircuitRequest(BaseModel):
    text: str


# =============================================================================
# Core helpers
# =============================================================================

def _get_feature_label(idx: int) -> Tuple[str, str]:
    """Return (label, confidence) for a feature index."""
    feat = srv.feature_by_index.get(idx, {})
    label = feat.get("label", f"Feature {idx}")
    conf = feat.get("confidence", "unknown")
    return label, conf


def _encode_text(text: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Tokenize text, run model, return (hidden [seq, n_feat], token_ids, token_strs)."""
    hook_point = f"blocks.{TARGET_LAYER}.hook_{HOOK_TYPE}"
    tokens = srv.model.to_tokens(text)
    token_strs = [srv.model.tokenizer.decode([t]) for t in tokens[0]]
    with torch.no_grad():
        _, cache = srv.model.run_with_cache(tokens)
        activations = cache[hook_point][0]          # [seq, d_model]
        hidden = srv.sae.encode(activations.to(srv.device))  # [seq, n_feat]
    return hidden, tokens, token_strs



def _top_features_full(
    hidden: torch.Tensor,
    top_k: int = 8,
    threshold: float = 0.3,
    include_source: bool = False,
) -> List[Dict]:
    """Like _top_features but also returns token_indices (top positions in the sequence),
    and optionally source_breakdown + max_act_examples for output attribution."""
    max_per_feat = hidden.max(dim=0).values.cpu()
    vals, idxs = max_per_feat.topk(min(top_k * 3, max_per_feat.shape[0]))
    results = []
    for val, idx in zip(vals.tolist(), idxs.tolist()):
        if val < threshold:
            break
        if len(results) >= top_k:
            break
        fdata = srv.feature_by_index.get(idx, {})
        label = fdata.get("label", f"Feature {idx}")
        conf = fdata.get("confidence", "unknown")
        max_act_list = fdata.get("max_activating_tokens", [])
        evidence = [
            e["token"] for e in max_act_list[:4]
            if e.get("token", "").strip() and e.get("token", "").strip() not in _SPECIAL_TOKENS
        ]
        vocab_proj = fdata.get("vocab_projection", [])[:6]

        # Which positions in the sequence activated this feature most?
        feat_acts = hidden[:, idx].cpu()
        n_top = min(3, feat_acts.shape[0])
        top_tok = feat_acts.topk(n_top)
        token_indices = [
            int(i) for i, v in zip(top_tok.indices.tolist(), top_tok.values.tolist())
            if v > threshold * 0.4
        ]

        entry: Dict = {
            "index": idx,
            "label": label,
            "confidence": conf,
            "activation": round(val, 3),
            "evidence": evidence,
            "vocab_proj": vocab_proj,
            "token_indices": token_indices,
        }

        if include_source:
            source_counts: Dict[str, int] = {}
            for tok in max_act_list[:20]:
                sid = tok.get("source_id", "")
                if sid:
                    prefix = sid.split(":")[0]
                    source_counts[prefix] = source_counts.get(prefix, 0) + 1
            entry["source_breakdown"] = source_counts

            examples = []
            for tok in max_act_list[:5]:
                t = tok.get("token", "")
                if t.strip() and t.strip() not in _SPECIAL_TOKENS:
                    sid = tok.get("source_id", "")
                    examples.append({
                        "token": t,
                        "activation": round(tok.get("activation", 0), 3),
                        "context": tok.get("context", "")[:300],
                        "source": sid.split(":")[0] if sid else "unknown",
                    })
            entry["max_act_examples"] = examples

        results.append(entry)
    return results


def _build_prompt(message: str, history: List[Dict[str, str]]) -> str:
    messages = [{"role": t.get("role", "user"), "content": t.get("content", "")} for t in (history or [])]
    messages.append({"role": "user", "content": message})
    return srv.model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# =============================================================================
# API endpoints
# =============================================================================

@app.get("/health")
async def health():
    return {
        "model_loaded": srv.loaded,
        "features_count": len(srv.labeled_features),
        "error": srv.error,
    }



@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    # NOTE: do NOT call srv.load() here — it blocks the event loop.
    # Loading happens inside the executor thread so SSE events can flow immediately.

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _put(obj: dict):
        loop.call_soon_threadsafe(queue.put_nowait, json.dumps(obj))

    def run():
        try:
            # ── 1. Load model (only blocks on first ever request) ────────────
            if not srv.loaded:
                _put({"type": "log", "text": "Loading model…"})
                srv.load()
                if srv.error and not srv.loaded:
                    _put({"type": "error", "text": srv.error or "Failed to load model"})
                    return

            # ── 2. Input attribution ─────────────────────────────────────────
            _put({"type": "log", "text": "Tokenizing input"})
            prompt = _build_prompt(req.message, req.history)

            _put({"type": "log", "text": "Input attribution · SAE encode"})
            input_hidden_full, input_raw_tokens, _ = _encode_text(req.message)
            # Skip BOS at position 0 so token_indices align with input_token_strs
            input_token_strs = [
                srv.model.tokenizer.decode([t])
                for t in input_raw_tokens[0].tolist()[1:]
            ]
            input_features = _top_features_full(input_hidden_full[1:], top_k=8)

            # ── 3. Generate (uses KV cache — fast) ───────────────────────────
            _put({"type": "log", "text": "Generating response"})
            tokens = srv.model.to_tokens(prompt)
            with torch.no_grad():
                generated = srv.model.generate(
                    tokens,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.9,
                    stop_at_eos=True,
                    verbose=False,
                )

            # Stream response tokens to the client
            response_ids = generated[0, tokens.shape[1]:]
            for token_id in response_ids.tolist():
                tok_str = srv.model.tokenizer.decode([token_id])
                if tok_str.strip() and tok_str.strip() not in _SPECIAL_TOKENS:
                    _put({"type": "token", "text": tok_str})

            response = srv.model.tokenizer.decode(response_ids.tolist())

            # ── 4. Output attribution ────────────────────────────────────────
            hook_point = f"blocks.{TARGET_LAYER}.hook_{HOOK_TYPE}"
            prompt_len = tokens.shape[1]
            with torch.no_grad():
                _, cache = srv.model.run_with_cache(generated)
                activations = cache[hook_point][0]
                gen_acts = activations[prompt_len:]
                if gen_acts.shape[0] == 0:
                    gen_acts = activations
                output_hidden = srv.sae.encode(gen_acts.to(srv.device))
            output_features = _top_features_full(output_hidden, top_k=8, include_source=True)

            # Response tokens — keep all (incl. EOS placeholder) so token_indices align
            response_token_strs = [
                srv.model.tokenizer.decode([t]) for t in response_ids.tolist()
            ]

            _put({
                "type": "result",
                "response": response,
                "input_tokens": input_token_strs,
                "response_tokens": response_token_strs,
                "input_features": input_features,
                "output_features": output_features,
            })
        except Exception as e:
            _put({"type": "error", "text": str(e)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    loop.run_in_executor(None, run)

    async def event_stream():
        while True:
            item = await queue.get()
            if item is None:
                break
            yield f"data: {item}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.post("/api/steer")
async def steer(req: SteerRequest):
    srv.load()
    if srv.error and not srv.loaded:
        raise HTTPException(status_code=503, detail=srv.error)

    try:
        features = {int(k): v for k, v in req.features.items()}
        messages = [{"role": "user", "content": req.prompt}]
        prompt = srv.model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        baseline_text, steered_text = _do_steer(prompt, features, req.max_tokens)

        feature_labels = []
        for idx, strength in features.items():
            label, _ = _get_feature_label(idx)
            feature_labels.append({"index": idx, "label": label, "strength": strength})

        return {
            "baseline": baseline_text,
            "steered": steered_text,
            "applied_features": feature_labels,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _do_steer(prompt: str, feature_strengths: Dict[int, float], max_tokens: int) -> Tuple[str, str]:
    steering_vector = torch.zeros(srv.sae.d_hidden, device=srv.device)
    for feat_idx, strength in feature_strengths.items():
        if abs(strength) > 0.1 and feat_idx < srv.sae.d_hidden:
            steering_vector[feat_idx] = strength

    decoder = srv.sae.decoder.weight.detach()
    steer_direction = decoder @ steering_vector

    hook_point = f"blocks.{TARGET_LAYER}.hook_{HOOK_TYPE}"
    tokens_orig = srv.model.to_tokens(prompt)
    input_len = tokens_orig.shape[1]

    with torch.no_grad():
        output_orig = srv.model.generate(
            tokens_orig, max_new_tokens=max_tokens, do_sample=True, temperature=0.7, top_p=0.9,
        )
    text_orig = srv.model.tokenizer.decode(output_orig[0][input_len:], skip_special_tokens=True)

    if steering_vector.abs().sum() < 0.1:
        return text_orig, text_orig

    with torch.no_grad():
        _, cache = srv.model.run_with_cache(tokens_orig)
        resid_norm = cache[hook_point][0].norm(dim=-1).mean().item()

    steer_unit = steer_direction / (steer_direction.norm() + 1e-8)
    total_strength = steering_vector.abs().max().item()
    scaled_steer = steer_unit * total_strength * resid_norm * 0.05

    def steering_hook(activation, hook):
        activation[:, :, :] += scaled_steer.unsqueeze(0).unsqueeze(0)
        return activation

    tokens = srv.model.to_tokens(prompt)
    steer_input_len = tokens.shape[1]
    with torch.no_grad():
        for _ in range(max_tokens):
            srv.model.add_hook(hook_point, steering_hook)
            logits = srv.model(tokens)[:, -1, :]
            srv.model.reset_hooks()
            probs = torch.softmax(logits / 0.7, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
            if next_token.item() == srv.model.tokenizer.eos_token_id:
                break

    text_steer = srv.model.tokenizer.decode(tokens[0][steer_input_len:], skip_special_tokens=True)
    return text_orig, text_steer


@app.post("/api/circuit")
async def circuit(req: CircuitRequest):
    srv.load()
    if srv.error and not srv.loaded:
        raise HTTPException(status_code=503, detail=srv.error)

    try:
        hidden, tokens, token_strs = _encode_text(req.text)
        hidden_np = hidden.cpu().numpy()

        valid = [(i, t) for i, t in enumerate(token_strs) if t.strip() not in _SPECIAL_TOKENS]
        valid_idxs = [i for i, _ in valid]
        valid_toks = [t for _, t in valid]

        if not valid_idxs:
            return {"tokens": [], "features": [], "links": []}

        filtered = hidden_np[valid_idxs]              # [n_valid_tokens, n_features]
        max_per_feat = filtered.max(axis=0)            # [n_features]
        activated = np.where(max_per_feat > 0.5)[0]
        sorted_by_act = activated[np.argsort(max_per_feat[activated])[::-1]]
        top_indices = sorted_by_act[:20].tolist()

        features = []
        for feat_idx in top_indices:
            label, conf = _get_feature_label(feat_idx)
            fdata = srv.feature_by_index.get(feat_idx, {})
            vocab_proj = fdata.get("vocab_projection", [])[:5]
            features.append({
                "index": int(feat_idx),
                "label": label,
                "confidence": conf,
                "activation": round(float(max_per_feat[feat_idx]), 3),
                "vocab_proj": vocab_proj,
            })

        # Build token→feature links
        links = []
        for tok_i, tok_str in enumerate(valid_toks):
            for feat_i, feat_idx in enumerate(top_indices[:15]):
                act = float(filtered[tok_i, feat_idx])
                if act > 0.5:
                    links.append({
                        "source": tok_i,
                        "target": feat_i,
                        "value": round(act, 3),
                    })

        return {
            "tokens": valid_toks,
            "features": features,
            "links": links,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/features")
async def list_features(
    page: int = 0,
    limit: int = 48,
    search: str = "",
    confidence: str = "",
):
    srv.load_features_only()

    feats = [f for f in srv.feature_by_index.values() if "label" in f and f.get("label")]

    if search:
        sl = search.lower()
        feats = [f for f in feats if sl in f.get("label", "").lower() or sl in str(f.get("index", ""))]

    conf_order = {"high": 3, "medium": 2, "low": 1}
    if confidence in conf_order:
        min_val = conf_order[confidence]
        feats = [f for f in feats if conf_order.get(f.get("confidence", "low"), 1) >= min_val]

    feats.sort(key=lambda f: (
        -conf_order.get(f.get("confidence", "low"), 0),
        -f.get("frequency", f.get("activation_frequency", 0)),
    ))

    total = len(feats)
    page_feats = feats[page * limit: (page + 1) * limit]

    return {
        "total": total,
        "page": page,
        "features": [
            {
                "index": f["index"],
                "label": f.get("label", f"Feature {f['index']}"),
                "confidence": f.get("confidence", "unknown"),
                "frequency": round(f.get("frequency", f.get("activation_frequency", 0)) * 100, 2),
                "max_activation": round(f.get("max", f.get("max_activation", 0)), 2),
                "top_tokens": [
                    e.get("token", e[0] if isinstance(e, list) else "")
                    for e in f.get("max_activating_tokens", f.get("max_act_tokens", []))[:3]
                ],
                "vocab_proj": f.get("vocab_projection", f.get("vocab_proj_tokens", []))[:5],
                "reasoning": f.get("reasoning", "")[:120],
            }
            for f in page_feats
        ],
    }


@app.get("/api/feature/{idx}")
async def get_feature(idx: int):
    srv.load_features_only()
    if idx not in srv.feature_by_index:
        raise HTTPException(status_code=404, detail="Feature not found")

    f = srv.feature_by_index[idx]
    label, conf = _get_feature_label(idx)

    max_act_raw = f.get("max_activating_tokens", f.get("max_act_tokens", []))
    examples = []
    for e in max_act_raw[:6]:
        if isinstance(e, dict):
            examples.append({
                "token": e.get("token", ""),
                "activation": round(e.get("activation", 0), 3),
                "context": e.get("context", "")[:200],
                "source_id": e.get("source_id", ""),
            })
        elif isinstance(e, list) and len(e) >= 2:
            examples.append({"token": e[0], "activation": round(e[1], 3), "context": e[2] if len(e) > 2 else "", "source_id": ""})

    return {
        "index": idx,
        "label": label,
        "confidence": conf,
        "reasoning": f.get("reasoning", ""),
        "frequency": round(f.get("frequency", f.get("activation_frequency", 0)) * 100, 3),
        "mean_activation": round(f.get("mean", f.get("mean_activation", 0)), 3),
        "max_activation": round(f.get("max", f.get("max_activation", 0)), 3),
        "examples": examples,
        "vocab_proj": f.get("vocab_projection", f.get("vocab_proj_tokens", []))[:10],
        "vocab_proj_logits": f.get("vocab_projection_logits", [])[:10],
    }


# =============================================================================
# Static files + SPA routing
# =============================================================================

app.mount("/styles", StaticFiles(directory=str(SRC_DIR / "styles")), name="styles")
app.mount("/scripts", StaticFiles(directory=str(SRC_DIR / "scripts")), name="scripts")


NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}


@app.get("/")
async def index():
    return FileResponse(str(SRC_DIR / "index.html"), headers=NO_CACHE_HEADERS)


@app.get("/{path:path}")
async def spa_fallback(path: str):
    file_path = SRC_DIR / path
    if file_path.exists() and file_path.is_file():
        return FileResponse(str(file_path))
    return FileResponse(str(SRC_DIR / "index.html"))


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
