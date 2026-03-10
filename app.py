#!/usr/bin/env python3
"""
SAE Interpretability Lab — Research Interface
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import torch
import numpy as np
import gradio as gr
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from transformer_lens import HookedTransformer

from config import (
    SparseAutoencoder,
    MODEL_NAME, TARGET_LAYER, HOOK_TYPE,
    OUTPUT_DIR, SAE_PATH, MEDICAL_OUTPUT_DIR,
    get_device,
)

_SPECIAL_TOKENS = {"<|endoftext|>", "<|begin_of_text|>", "<|end_of_text|>", "<|eot_id|>", ""}


# =============================================================================
# Helpers
# =============================================================================

def get_feature_choices() -> List[str]:
    try:
        with open(MEDICAL_OUTPUT_DIR / "labeled_features.json") as f:
            features = json.load(f)
        confidence_order = {"high": 0, "medium": 1, "low": 2}
        sorted_features = sorted(
            features,
            key=lambda x: (confidence_order.get(x.get("confidence", "low"), 3), x["index"])
        )
        return [f"{f['index']}: {f['label']} ({f['confidence']})" for f in sorted_features]
    except Exception:
        return ["No features loaded"]


# =============================================================================
# Global State
# =============================================================================

class AppState:
    def __init__(self):
        self.model = None
        self.sae = None
        self.labeled_features = []
        self.all_features = []
        self.feature_by_index = {}
        self.device = get_device()
        self.loaded = False

    def load(self):
        if self.loaded:
            return
        print("Loading models...")
        self.model = HookedTransformer.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        self.sae = SparseAutoencoder.load(MEDICAL_OUTPUT_DIR / "sae.pt")
        self.sae.to(self.device)
        self.sae.eval()
        with open(MEDICAL_OUTPUT_DIR / "labeled_features.json") as f:
            self.labeled_features = json.load(f)
        features_path = MEDICAL_OUTPUT_DIR / "features.json"
        if features_path.exists():
            with open(features_path) as f:
                self.all_features = json.load(f)
            self.feature_by_index = {feat["index"]: feat for feat in self.all_features}
        self.loaded = True
        print(f"Loaded {len(self.labeled_features)} labeled features, {len(self.all_features)} total")


state = AppState()


# =============================================================================
# Core Backend Functions
# =============================================================================

def get_activations(text: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    state.load()
    tokens = state.model.to_tokens(text)
    token_strs = [state.model.tokenizer.decode([t]) for t in tokens[0]]
    hook_point = f"blocks.{TARGET_LAYER}.hook_{HOOK_TYPE}"
    with torch.no_grad():
        _, cache = state.model.run_with_cache(tokens)
        activations = cache[hook_point][0]
        hidden = state.sae.encode(activations.to(state.device))
    return hidden, tokens, token_strs


def generate_with_feature_steering(
    prompt: str, feature_strengths: Dict[int, float], max_tokens: int = 50
) -> Tuple[str, str]:
    state.load()
    steering_vector = torch.zeros(state.sae.d_hidden, device=state.device)
    for feat_idx, strength in feature_strengths.items():
        if abs(strength) > 0.1 and feat_idx < state.sae.d_hidden:
            steering_vector[feat_idx] = strength
    decoder = state.sae.decoder.weight.detach()
    steer_direction = decoder @ steering_vector
    hook_point = f"blocks.{TARGET_LAYER}.hook_{HOOK_TYPE}"
    tokens_orig = state.model.to_tokens(prompt)
    with torch.no_grad():
        output_orig = state.model.generate(
            tokens_orig, max_new_tokens=max_tokens, do_sample=True, temperature=0.8, top_p=0.9,
        )
    text_orig = state.model.tokenizer.decode(output_orig[0])
    if steering_vector.abs().sum() < 0.1:
        return text_orig, text_orig
    with torch.no_grad():
        _, cache = state.model.run_with_cache(state.model.to_tokens(prompt))
        resid_norm = cache[hook_point][0].norm(dim=-1).mean().item()
    steer_unit = steer_direction / (steer_direction.norm() + 1e-8)
    total_strength = steering_vector.abs().max().item()
    scaled_steer = steer_unit * total_strength * resid_norm * 0.1

    def steering_hook(activation, hook):
        activation[:, :, :] += scaled_steer.unsqueeze(0).unsqueeze(0)
        return activation

    tokens = state.model.to_tokens(prompt)
    with torch.no_grad():
        for _ in range(max_tokens):
            state.model.add_hook(hook_point, steering_hook)
            logits = state.model(tokens)[:, -1, :]
            state.model.reset_hooks()
            probs = torch.softmax(logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
            if next_token.item() == state.model.tokenizer.eos_token_id:
                break
    text_steer = state.model.tokenizer.decode(tokens[0])
    return text_orig, text_steer


def create_feature_ranking(text: str) -> go.Figure:
    state.load()
    if not text.strip():
        return go.Figure()
    hidden, _, _ = get_activations(text)
    feature_scores = []
    for f in state.labeled_features:
        idx = f["index"]
        if idx < hidden.shape[1]:
            feature_scores.append({
                "label": f["label"], "index": idx,
                "activation": hidden[:, idx].max().item(),
                "confidence": f.get("confidence", "low"),
            })
    feature_scores.sort(key=lambda x: x["activation"], reverse=True)
    top_features = feature_scores[:15]
    labels = [f"{f['label'][:28]} ({f['index']})" for f in top_features]
    values = [f["activation"] for f in top_features]
    color_map = {"high": "#2dd4a0", "medium": "#f5a623", "low": "#e05c5c"}
    colors = [color_map.get(f["confidence"], "#4a5568") for f in top_features]
    fig = go.Figure(data=go.Bar(x=values, y=labels, orientation='h', marker_color=colors))
    fig.update_layout(
        paper_bgcolor='#0d0f14', plot_bgcolor='#0d0f14',
        font=dict(color='#c8d3e0', family='JetBrains Mono, monospace', size=11),
        xaxis=dict(gridcolor='#1a2133', zerolinecolor='#252d3d', title="Activation"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=280, r=20, t=30, b=30),
        height=460,
    )
    return fig


def create_feature_detail_table(text: str) -> str:
    state.load()
    if not text.strip():
        return "Enter text to see feature activations."
    hidden, _, _ = get_activations(text)
    max_per_feature = hidden.max(dim=0).values.cpu().numpy()
    feature_info = []
    for feat in state.labeled_features:
        idx = feat["index"]
        if idx < len(max_per_feature):
            feature_info.append({
                "index": idx, "label": feat["label"],
                "confidence": feat["confidence"], "max_act": max_per_feature[idx],
            })
    feature_info.sort(key=lambda x: x["max_act"], reverse=True)
    md = "| Feature | Label | Confidence | Activation |\n|---------|-------|------------|------------|\n"
    for f in feature_info[:15]:
        md += f"| {f['index']} | {f['label'][:30]} | {f['confidence']} | {f['max_act']:.3f} |\n"
    return md


def get_feature_label(feature_idx: int) -> str:
    state.load()
    for f in state.labeled_features:
        if f["index"] == feature_idx:
            return f"{feature_idx}: {f['label']}"
    return f"Feature {feature_idx}"


def get_feature_vocab_projection(feature_idx: int, top_k: int = 5) -> List[Tuple[str, float]]:
    state.load()
    decoder_vec = state.sae.decoder.weight[:, feature_idx]
    unembed = state.model.W_U
    logits = decoder_vec @ unembed
    top_values, top_indices = logits.topk(top_k)
    return [(state.model.tokenizer.decode([idx.item()]), val.item()) for val, idx in zip(top_values, top_indices)]


def get_vocab_projections_batched(feature_indices: List[int], top_k: int = 3) -> Dict[int, List[Tuple[str, float]]]:
    state.load()
    indices_tensor = torch.tensor(feature_indices, dtype=torch.long)
    decoder_cols = state.sae.decoder.weight[:, indices_tensor]
    unembed = state.model.W_U
    all_logits = decoder_cols.T @ unembed
    top_values, top_indices = all_logits.topk(top_k, dim=-1)
    return {
        feat_idx: [(state.model.tokenizer.decode([top_indices[i, j].item()]), top_values[i, j].item()) for j in range(top_k)]
        for i, feat_idx in enumerate(feature_indices)
    }


def analyze_circuit(text: str) -> Tuple[go.Figure, str, str]:
    state.load()
    if not text.strip():
        return go.Figure(), "Enter text to analyze.", ""
    hidden, tokens, token_strs = get_activations(text)
    hidden_np = hidden.cpu().numpy()
    valid_indices = [i for i, t in enumerate(token_strs) if t.strip() not in _SPECIAL_TOKENS]
    if not valid_indices:
        return go.Figure(), "No valid tokens found.", ""
    filtered_token_strs = [token_strs[i] for i in valid_indices]
    filtered_hidden = hidden_np[valid_indices]
    n_tokens = len(filtered_token_strs)
    max_per_feature = filtered_hidden.max(axis=0)
    activation_threshold = 0.5
    activated_indices = np.where(max_per_feature > activation_threshold)[0]
    sorted_by_act = activated_indices[np.argsort(max_per_feature[activated_indices])[::-1]]
    top_feature_indices = sorted_by_act[:25]
    total_activated = len(activated_indices)
    with torch.no_grad():
        generated = state.model.generate(
            state.model.to_tokens(text), max_new_tokens=30, temperature=0.8, top_p=0.9, stop_at_eos=True,
        )
        model_output = state.model.tokenizer.decode(generated[0])
    input_labels = [f"IN: {t.strip()[:12]}" for t in filtered_token_strs]
    feature_labels = [get_feature_label(idx)[:25] for idx in top_feature_indices]
    feature_to_outputs = get_vocab_projections_batched([int(i) for i in top_feature_indices], top_k=3)
    output_tokens_set = {}
    for feat_idx in top_feature_indices:
        for tok, logit in feature_to_outputs.get(int(feat_idx), []):
            tok_clean = tok.strip()[:12]
            if tok_clean and tok_clean not in output_tokens_set:
                output_tokens_set[tok_clean] = logit
    sorted_outputs = sorted(output_tokens_set.items(), key=lambda x: x[1], reverse=True)[:20]
    output_labels = [f"OUT: {tok}" for tok, _ in sorted_outputs]
    sankey_labels = input_labels + feature_labels + output_labels
    n_input, n_feature = len(input_labels), len(feature_labels)
    sankey_source, sankey_target, sankey_value, sankey_colors = [], [], [], []
    for tok_idx in range(n_tokens):
        for i, feat_idx in enumerate(top_feature_indices[:15]):
            act = filtered_hidden[tok_idx, feat_idx]
            if act > activation_threshold:
                sankey_source.append(tok_idx)
                sankey_target.append(n_input + i)
                sankey_value.append(float(act))
                sankey_colors.append("rgba(61,142,240,0.45)")
    output_label_set = {label: idx for idx, label in enumerate(output_labels)}
    for i, feat_idx in enumerate(top_feature_indices[:15]):
        for tok, logit in feature_to_outputs.get(int(feat_idx), []):
            out_label = f"OUT: {tok.strip()[:12]}"
            if out_label in output_label_set:
                sankey_source.append(n_input + i)
                sankey_target.append(n_input + n_feature + output_label_set[out_label])
                sankey_value.append(max(0.5, abs(logit)))
                sankey_colors.append("rgba(245,166,35,0.45)")
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=12, thickness=16,
            line=dict(color="#252d3d", width=0.5),
            label=sankey_labels,
            color=["#1e3a5f"] * n_input + ["#1a3d2e"] * n_feature + ["#3d2c0f"] * len(output_labels),
        ),
        link=dict(source=sankey_source, target=sankey_target, value=sankey_value, color=sankey_colors),
    )])
    fig.update_layout(
        title=dict(text=f"Circuit Flow — {total_activated} features activated (showing top 25)", font=dict(size=12, color="#5a6f8a")),
        height=580, font_size=10,
        paper_bgcolor='#0d0f14', plot_bgcolor='#0d0f14',
        font=dict(color='#c8d3e0', family='JetBrains Mono, monospace'),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    md = f"**{total_activated}** features activated at threshold {activation_threshold}\n\n"
    md += "| Rank | Index | Label | Max Act | Promotes |\n|------|-------|-------|---------|----------|\n"
    for rank, feat_idx in enumerate(top_feature_indices[:15], 1):
        feat_idx = int(feat_idx)
        label = get_feature_label(feat_idx)
        max_act = max_per_feature[feat_idx]
        outs = feature_to_outputs.get(feat_idx, [])
        out_str = " · ".join([f"`{t.strip()}`" for t, _ in outs[:3]])
        md += f"| {rank} | {feat_idx} | {label[:28]} | {max_act:.2f} | {out_str} |\n"
    return fig, md, model_output


def analyze_feature_deep(feature_idx: int) -> str:
    state.load()
    label_info = next((f for f in state.labeled_features if f["index"] == feature_idx), None)
    label_text = label_info["label"] if label_info else "Unlabelled"
    confidence = label_info.get("confidence", "unknown") if label_info else None
    outputs = get_feature_vocab_projection(feature_idx, top_k=10)
    decoder_vec = state.sae.decoder.weight[:, feature_idx].detach().cpu().numpy()
    full_data = state.feature_by_index.get(feature_idx, {})
    max_act_examples = full_data.get("max_activating_tokens", [])
    vocab_proj_tokens = full_data.get("vocab_projection", [])
    frequency = full_data.get("frequency", 0)
    md = f"## Feature {feature_idx}\n\n"
    conf_str = f"  ·  **Confidence:** {confidence}" if confidence else ""
    md += f"**Label:** {label_text}{conf_str}  ·  **Frequency:** {frequency:.2%}\n\n"
    md += "### MaxAct Examples\n"
    if max_act_examples:
        for i, ex in enumerate(max_act_examples[:5], 1):
            context = ex.get("context", "")
            token = ex.get("token", "")
            act = ex.get("activation", 0)
            display = context if context else token
            md += f"**{i}.** `{display[:120]}` *(act: {act:.2f})*\n\n"
    else:
        md += "*No examples available*\n\n"
    md += "### VocabProj\n\n| Token | Logit |\n|-------|-------|\n"
    for tok, logit in outputs:
        md += f"| `{tok}` | {logit:.3f} |\n"
    if vocab_proj_tokens:
        md += f"\n*From features.json: {', '.join(vocab_proj_tokens[:8])}*\n"
    md += f"\n### Decoder Vector\n- Norm: `{np.linalg.norm(decoder_vec):.4f}`  ·  Mean: `{decoder_vec.mean():.5f}`  ·  Std: `{decoder_vec.std():.5f}`\n"
    return md


# =============================================================================
# Medical State
# =============================================================================

class MedicalState:
    def __init__(self):
        self.model = None
        self.sae = None
        self.labeled_features = []
        self.feature_by_index = {}
        self.device = get_device()
        self.loaded = False
        self.error = None

    def load(self):
        if self.loaded or self.error:
            return
        med_sae_path = MEDICAL_OUTPUT_DIR / "sae.pt"
        if not med_sae_path.exists():
            self.error = "Medical SAE not found. Run: python main.py"
            return
        print("Loading medical model...")
        try:
            self.model = HookedTransformer.from_pretrained(MODEL_NAME, device=self.device)
            self.model.eval()
            self.sae = SparseAutoencoder.load(med_sae_path)
            self.sae.to(self.device)
            self.sae.eval()
            labeled_path = MEDICAL_OUTPUT_DIR / "labeled_features.json"
            if labeled_path.exists():
                with open(labeled_path) as f:
                    self.labeled_features = json.load(f)
            features_path = MEDICAL_OUTPUT_DIR / "features.json"
            if features_path.exists():
                with open(features_path) as f:
                    all_features = json.load(f)
                self.feature_by_index = {feat["index"]: feat for feat in all_features}
            self.loaded = True
            print(f"  Medical model loaded · {len(self.labeled_features)} labeled features")
        except Exception as e:
            self.error = f"Failed to load: {e}"
            print(f"  ERROR: {self.error}")


med_state = MedicalState()


# =============================================================================
# Attribution Visualization — Custom SVG
# =============================================================================

def build_attribution_html(
    hidden: torch.Tensor,
    response_tokens: Optional[List[str]] = None,
) -> str:
    """
    Output attribution visualization.

    hidden:           [gen_seq_len, d_hidden] — SAE activations for generated tokens
    response_tokens:  list of decoded token strings from the generated response

    Shows: RESPONSE TOKENS → ACTIVE FEATURES → TRAINING EVIDENCE
    Each connection answers: "this response token fired this feature; that feature
    was shaped by this medical text during training."
    """
    max_per_feature = hidden.max(dim=0).values.cpu()
    top_vals, top_idxs = max_per_feature.topk(min(8, max_per_feature.shape[0]))
    active = [(idx.item(), val.item()) for idx, val in zip(top_idxs, top_vals) if val.item() > 0.3]

    if not active:
        return '<div class="attr-empty">No significant feature activations detected.</div>'

    # Confidence color: (stroke, fill)
    CONF_COLORS = {
        "high":    ("#2dd4a0", "#0f2920"),
        "medium":  ("#f5a623", "#2a1f0a"),
        "low":     ("#e05c5c", "#2a0f0f"),
        "unknown": ("#4a6080", "#101820"),
    }

    # Build feature records (with training evidence instead of vocab projections)
    features = []
    for feat_idx, act_val in active:
        lbl = next((f for f in med_state.labeled_features if f["index"] == feat_idx), None)
        label = lbl["label"] if lbl else f"Feature {feat_idx}"
        conf  = lbl.get("confidence", "unknown") if lbl else "unknown"
        fdata = med_state.feature_by_index.get(feat_idx, {})
        # Top max-activating tokens from training data (training evidence)
        maxact_toks = fdata.get("max_activating_tokens", [])
        evidence = [
            e["token"] for e in maxact_toks[:3]
            if e.get("token", "").strip() and e.get("token", "").strip() not in _SPECIAL_TOKENS
        ]
        features.append({"idx": feat_idx, "act": act_val, "label": label, "conf": conf, "evidence": evidence})

    max_act = max(f["act"] for f in features)

    # Build response token records: last 8 non-special tokens from generated response
    tokens_display = []
    if response_tokens:
        valid = [(i, t) for i, t in enumerate(response_tokens)
                 if t.strip() and t.strip() not in _SPECIAL_TOKENS][-8:]
        for ti, (seq_i, tok_str) in enumerate(valid):
            # How strongly did this response token activate each feature?
            acts = [
                float(hidden[seq_i, f["idx"]]) if seq_i < hidden.shape[0] else 0.0
                for f in features
            ]
            tokens_display.append({"ti": ti, "text": tok_str.strip()[:14], "acts": acts})

    # ── Layout constants ──
    has_tokens = len(tokens_display) > 0
    ROW_H     = 52
    MARGIN_T  = 34
    W         = 760
    n_rows    = max(len(features), len(tokens_display) if has_tokens else 0, 1)
    H         = n_rows * ROW_H + MARGIN_T * 2

    COL_TOK   = 100   # center-x of response token column
    COL_FEAT  = has_tokens and 400 or 280   # center-x of feature column
    COL_OUT   = has_tokens and 680 or 560   # center-x of training evidence column

    TOK_W, TOK_H   = 140, 26
    FEAT_W, FEAT_H = 224, 30

    svg_id = f"asvg{abs(hash(str(active)))}"

    parts = []
    parts.append(
        f'<svg id="{svg_id}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;height:{H}px;background:#0a0c10;display:block;overflow:visible;">'
    )

    # ── Column headers ──
    def col_hdr(x, label):
        return (f'<text x="{x}" y="17" text-anchor="middle" '
                f'font-family="JetBrains Mono,monospace" font-size="8" '
                f'fill="#2d3a4a" letter-spacing="1.8" font-weight="600">{label}</text>')

    if has_tokens:
        parts.append(col_hdr(COL_TOK, "RESPONSE TOKENS"))
    parts.append(col_hdr(COL_FEAT, "ACTIVE FEATURES"))
    parts.append(col_hdr(COL_OUT, "TRAINING EVIDENCE"))

    # ── Response Token → Feature paths ──
    if has_tokens:
        for tok in tokens_display:
            ty = MARGIN_T + tok["ti"] * ROW_H + ROW_H / 2
            tx_right = COL_TOK + TOK_W / 2
            for fi, feat in enumerate(features):
                act = tok["acts"][fi]
                if act < 0.15:
                    continue
                fy = MARGIN_T + fi * ROW_H + ROW_H / 2
                fx_left = COL_FEAT - FEAT_W / 2
                opacity = round(max(0.07, min(0.85, act / max_act * 0.9)), 3)
                width   = round(max(0.4, min(2.8, act / max_act * 2.2)), 2)
                cp1x, cp2x = tx_right + 45, fx_left - 45
                parts.append(
                    f'<path class="path ipath" data-fi="{feat["idx"]}" data-ti="{tok["ti"]}" '
                    f'data-bo="{opacity}" '
                    f'd="M{tx_right:.0f},{ty:.0f} C{cp1x},{ty:.0f} {cp2x},{fy:.0f} {fx_left:.0f},{fy:.0f}" '
                    f'fill="none" stroke="#3d8ef0" stroke-width="{width}" opacity="{opacity}" '
                    f'style="transition:opacity .18s,stroke-width .18s;pointer-events:none;"/>'
                )

    # ── Feature → Training Evidence paths ──
    all_evidence = []
    seen_ev = set()
    for feat in features:
        for ev in feat["evidence"][:2]:
            ec = ev.strip()
            if ec and ec not in seen_ev:
                seen_ev.add(ec)
                all_evidence.append(ec)
    all_evidence = all_evidence[:8]
    n_ev = len(all_evidence)
    ev_spacing = H / (n_ev + 1) if n_ev > 0 else ROW_H

    for fi, feat in enumerate(features):
        fy = MARGIN_T + fi * ROW_H + ROW_H / 2
        fx_right = COL_FEAT + FEAT_W / 2
        for ev in feat["evidence"][:2]:
            ec = ev.strip()
            if ec not in all_evidence:
                continue
            ei = all_evidence.index(ec)
            ey = ev_spacing * (ei + 1)
            cp1x = fx_right + 38
            cp2x = COL_OUT - 52
            parts.append(
                f'<path class="path opath" data-fi="{feat["idx"]}" data-bo="0.32" '
                f'd="M{fx_right:.0f},{fy:.0f} C{cp1x},{fy:.0f} {cp2x},{ey:.0f} {COL_OUT - 40:.0f},{ey:.0f}" '
                f'fill="none" stroke="#f5a623" stroke-width="1.0" opacity="0.32" '
                f'style="transition:opacity .18s;pointer-events:none;"/>'
            )

    # ── Response token nodes ──
    if has_tokens:
        for tok in tokens_display:
            ty = MARGIN_T + tok["ti"] * ROW_H + ROW_H / 2
            rx = COL_TOK - TOK_W / 2
            ry = ty - TOK_H / 2
            parts.append(
                f'<g class="tnode" data-ti="{tok["ti"]}" style="cursor:default;">'
                f'<rect x="{rx:.0f}" y="{ry:.0f}" width="{TOK_W}" height="{TOK_H}" rx="3" '
                f'fill="#0e1318" stroke="#1e2a3a" stroke-width="1" '
                f'style="transition:stroke .15s;"/>'
                f'<text x="{COL_TOK}" y="{ty+4:.0f}" text-anchor="middle" '
                f'font-family="JetBrains Mono,monospace" font-size="11" fill="#5a7090">'
                f'{tok["text"]}</text>'
                f'</g>'
            )

    # ── Feature nodes ──
    for fi, feat in enumerate(features):
        fy = MARGIN_T + fi * ROW_H + ROW_H / 2
        rx = COL_FEAT - FEAT_W / 2
        ry = fy - FEAT_H / 2
        stroke, fill = CONF_COLORS.get(feat["conf"], CONF_COLORS["unknown"])
        act_pct = feat["act"] / max_act
        bar_w = max(4, int(act_pct * (FEAT_W - 14)))
        label_short = feat["label"][:27] + ("…" if len(feat["label"]) > 27 else "")
        idx_str = str(feat["idx"])

        parts.append(
            f'<g class="fnode" data-fi="{feat["idx"]}" style="cursor:pointer;">'
            # Main rect
            f'<rect x="{rx:.0f}" y="{ry:.0f}" width="{FEAT_W}" height="{FEAT_H}" rx="3" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="0.8" '
            f'style="transition:filter .15s,stroke-width .15s;"/>'
            # Activation bar (bottom strip)
            f'<rect x="{rx:.0f}" y="{ry+FEAT_H-3:.0f}" width="{bar_w}" height="3" rx="1.5" '
            f'fill="{stroke}" opacity="0.6"/>'
            # Feature index
            f'<text x="{rx+7:.0f}" y="{fy+4:.0f}" '
            f'font-family="JetBrains Mono,monospace" font-size="9" fill="{stroke}" opacity="0.55">'
            f'{idx_str}</text>'
            # Label
            f'<text x="{rx+30:.0f}" y="{fy+4:.0f}" '
            f'font-family="JetBrains Mono,monospace" font-size="10.5" fill="{stroke}">'
            f'{label_short}</text>'
            f'</g>'
        )

    # ── Training evidence labels ──
    for ei, ev_str in enumerate(all_evidence):
        ey = ev_spacing * (ei + 1)
        parts.append(
            f'<text x="{COL_OUT - 32}" y="{ey+4:.0f}" '
            f'font-family="JetBrains Mono,monospace" font-size="10.5" fill="#8a6030">'
            f'{ev_str[:16]}</text>'
        )

    parts.append('</svg>')

    # ── Inline JavaScript for hover interactivity ──
    js = f"""<script>
(function(){{
  var svg=document.getElementById('{svg_id}');
  if(!svg)return;
  function resetAll(){{
    svg.querySelectorAll('.path').forEach(function(p){{
      p.style.opacity=p.dataset.bo;p.style.strokeWidth='';p.style.filter='';
    }});
    svg.querySelectorAll('.fnode rect:first-of-type,.tnode rect').forEach(function(r){{r.style.filter='';r.style.strokeWidth='';r.style.stroke='';r.style.strokeWidth='0.8';
    }});
  }}
  svg.querySelectorAll('.fnode').forEach(function(node){{
    var fi=node.dataset.fi;
    var rect=node.querySelector('rect');
    node.addEventListener('mouseenter',function(){{
      svg.querySelectorAll('.path').forEach(function(p){{p.style.opacity='0.04';}});
      svg.querySelectorAll('.path[data-fi="'+fi+'"]').forEach(function(p){{
        p.style.opacity='0.95';p.style.filter='brightness(1.5)';
      }});
      if(rect){{rect.style.filter='brightness(1.3)';rect.style.strokeWidth='1.4';}}
    }});
    node.addEventListener('mouseleave',resetAll);
  }});
  svg.querySelectorAll('.tnode').forEach(function(node){{
    var ti=node.dataset.ti;
    node.addEventListener('mouseenter',function(){{
      svg.querySelectorAll('.path').forEach(function(p){{p.style.opacity='0.04';}});
      svg.querySelectorAll('.ipath[data-ti="'+ti+'"]').forEach(function(p){{p.style.opacity='0.95';}});
    }});
    node.addEventListener('mouseleave',resetAll);
  }});
  // Entrance animation: fade in paths
  var delay=0;
  svg.querySelectorAll('.path').forEach(function(p){{
    var bo=p.dataset.bo;p.style.opacity='0';
    setTimeout(function(){{p.style.transition='opacity 0.4s';p.style.opacity=bo;}},delay);
    delay+=18;
  }});
}})();
</script>"""

    n_labeled = sum(1 for f in features if f["conf"] in ("high", "medium"))
    summary = (
        f'<div class="attr-summary">'
        f'<span class="attr-count">{len(features)}</span> features active'
        f' &nbsp;·&nbsp; '
        f'<span class="attr-labeled">{n_labeled} labeled</span>'
        f'</div>'
    )
    return f'<div class="attr-container">{summary}{"".join(parts)}{js}</div>'


def generate_medical_response(message: str, history: list) -> Tuple[str, str]:
    """Generate response and build attribution HTML."""
    med_state.load()
    if med_state.error:
        return med_state.error, f'<div class="attr-error">{med_state.error}</div>'

    prompt = ""
    for user_msg, assistant_msg in (history or [])[-3:]:
        prompt += f"User: {user_msg}\nAssistant: {assistant_msg}\n"
    prompt += f"User: {message}\nAssistant:"

    tokens = med_state.model.to_tokens(prompt)

    with torch.no_grad():
        generated = med_state.model.generate(
            tokens, max_new_tokens=150, temperature=0.7, top_p=0.9, stop_at_eos=True,
        )

    response_token_ids = generated[0, tokens.shape[1]:]
    response = med_state.model.tokenizer.decode(response_token_ids.tolist())

    hook_point = f"blocks.{TARGET_LAYER}.hook_{HOOK_TYPE}"
    with torch.no_grad():
        _, cache = med_state.model.run_with_cache(generated)
        activations = cache[hook_point][0]
        # Generated activations (features active in the response)
        gen_acts = activations[tokens.shape[1]:]
        if gen_acts.shape[0] == 0:
            gen_acts = activations
        hidden = med_state.sae.encode(gen_acts.to(med_state.device))

    response_token_strs = [med_state.model.tokenizer.decode([t]) for t in generated[0, tokens.shape[1]:]]
    attribution_html = build_attribution_html(hidden, response_tokens=response_token_strs)
    return response, attribution_html


# =============================================================================
# Design System — CSS
# =============================================================================

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── CSS custom properties (Gradio 4.x theme override) ── */
:root {
    --body-background-fill:              #0a0c10;
    --background-fill-primary:           #0a0c10;
    --background-fill-secondary:         #111520;
    --block-background-fill:             #111520;
    --block-border-color:                #1e2433;
    --block-border-width:                1px;
    --block-label-background-fill:       #111520;
    --block-label-border-color:          #1e2433;
    --block-label-text-color:            #3d4a5e;
    --block-title-text-color:            #e2e8f2;
    --input-background-fill:             #0c0f16;
    --input-background-fill-focus:       #0f1320;
    --input-border-color:                #1e2433;
    --input-border-color-focus:          #3d8ef0;
    --input-placeholder-color:           #2d3a4e;
    --body-text-color:                   #b8c4d4;
    --body-text-color-subdued:           #4a5a70;
    --color-accent:                      #3d8ef0;
    --button-primary-background-fill:    transparent;
    --button-primary-background-fill-hover: rgba(61,142,240,0.10);
    --button-primary-border-color:       #3d8ef0;
    --button-primary-border-color-hover: #5aa3f5;
    --button-primary-text-color:         #3d8ef0;
    --button-secondary-background-fill:  transparent;
    --button-secondary-background-fill-hover: rgba(255,255,255,0.04);
    --button-secondary-border-color:     #1e2433;
    --button-secondary-border-color-hover: #2d3a50;
    --button-secondary-text-color:       #4a5a70;
    --shadow-drop:                       none;
    --shadow-drop-lg:                    none;
    --radius-lg:                         5px;
    --radius-md:                         4px;
    --radius-sm:                         3px;
}

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }
body, html { background: #0a0c10 !important; }

.gradio-container {
    background: #0a0c10 !important;
    font-family: 'Inter', -apple-system, system-ui, sans-serif !important;
    max-width: 100% !important;
    padding: 0 !important;
}

/* ── App header ── */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 24px 13px;
    border-bottom: 1px solid #1a2030;
    background: #0a0c10;
}
.app-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 2.5px;
    color: #c8d3e0;
    text-transform: uppercase;
}
.app-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #2d3a4e;
    letter-spacing: 0.4px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.status-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #2dd4a0;
    animation: pdot 2.8s ease-in-out infinite;
    flex-shrink: 0;
}
@keyframes pdot { 0%,100%{opacity:1;} 50%{opacity:0.25;} }

/* ── Tabs ── */
.tabs > .tab-nav {
    background: transparent !important;
    border-bottom: 1px solid #1a2030 !important;
    padding: 0 20px;
    gap: 0;
}
.tabs > .tab-nav button {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: #2d3a4e !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 13px 20px 12px !important;
    border-radius: 0 !important;
    transition: color .15s, border-color .15s !important;
    margin: 0 !important;
}
.tabs > .tab-nav button:hover { color: #5a6f8a !important; }
.tabs > .tab-nav button.selected {
    color: #3d8ef0 !important;
    border-bottom-color: #3d8ef0 !important;
}

/* ── Panels / blocks ── */
.block {
    border-color: #1e2433 !important;
    border-radius: 5px !important;
    background: #111520 !important;
}
.block > .block-label {
    background: transparent !important;
    border-color: #1a2030 !important;
}
.block > .block-label > span {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: #2d3a4e !important;
    font-weight: 600 !important;
}

/* ── Section label helper ── */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #2d3a4e;
    padding-bottom: 10px;
    margin-bottom: 4px;
    border-bottom: 1px solid #141924;
}

/* ── Inputs & textareas ── */
input, textarea, select {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    color: #c8d3e0 !important;
    background: #0c0f16 !important;
    border-color: #1e2433 !important;
    border-radius: 4px !important;
    caret-color: #3d8ef0;
}
input:focus, textarea:focus {
    border-color: #3d8ef0 !important;
    box-shadow: 0 0 0 1px rgba(61,142,240,0.15) !important;
    outline: none !important;
}
input::placeholder, textarea::placeholder { color: #2d3a4e !important; }

/* ── Buttons ── */
button {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    border-radius: 4px !important;
    transition: all .14s !important;
}
button.primary, .gr-button-primary {
    background: transparent !important;
    border: 1px solid #3d8ef0 !important;
    color: #3d8ef0 !important;
}
button.primary:hover { background: rgba(61,142,240,0.10) !important; }
button.secondary, .gr-button-secondary {
    background: transparent !important;
    border: 1px solid #1e2433 !important;
    color: #3d4a5e !important;
}
button.secondary:hover {
    border-color: #2d3a50 !important;
    color: #5a6f8a !important;
}

/* ── Chatbot ── */
.chatbot {
    background: #0a0c10 !important;
    border-color: #1e2433 !important;
}
.chatbot .message-wrap { padding: 16px !important; gap: 14px !important; }
.chatbot .message {
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    line-height: 1.65 !important;
    border-radius: 4px !important;
    max-width: 92% !important;
}
.chatbot .message.user {
    background: #111d2e !important;
    border: 1px solid #1a2840 !important;
    color: #c8d3e0 !important;
    margin-left: auto !important;
}
.chatbot .message.bot {
    background: #0e1218 !important;
    border: 1px solid #1a2030 !important;
    border-left: 2px solid #1e2d42 !important;
    color: #9ab0c4 !important;
}

/* ── Attribution container ── */
.attr-container {
    background: #0a0c10;
    border: 1px solid #1a2030;
    border-radius: 5px;
    overflow: hidden;
}
.attr-summary {
    padding: 9px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    letter-spacing: 1.2px;
    color: #2d3a4e;
    border-bottom: 1px solid #111824;
    text-transform: uppercase;
}
.attr-count   { color: #3d8ef0; font-weight: 600; }
.attr-labeled { color: #2dd4a0; font-weight: 600; }
.attr-empty {
    padding: 48px 20px;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 1px;
    color: #1e2a3a;
}
.attr-error {
    padding: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #e05c5c;
}

/* ── Markdown / prose ── */
.prose, .markdown-body, .md {
    color: #9ab0c4 !important;
    font-size: 13px !important;
    line-height: 1.65 !important;
}
.prose h2, .markdown-body h2 {
    color: #c8d3e0 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    border-bottom: 1px solid #1a2030 !important;
    padding-bottom: 8px !important;
    margin: 20px 0 12px !important;
}
.prose h3, .markdown-body h3 {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 9px !important;
    font-weight: 600 !important;
    letter-spacing: 1.8px !important;
    text-transform: uppercase !important;
    color: #2d3a4e !important;
    margin: 18px 0 10px !important;
}
.prose code, .markdown-body code, code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    background: #0e1420 !important;
    border: 1px solid #1a2433 !important;
    color: #3d8ef0 !important;
    padding: 1px 5px !important;
    border-radius: 3px !important;
}
.prose table, .markdown-body table {
    border-collapse: collapse !important;
    width: 100% !important;
    font-size: 11px !important;
}
.prose th, .markdown-body th {
    background: #0e1218 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 1.2px !important;
    text-transform: uppercase !important;
    color: #2d3a4e !important;
    padding: 8px 12px !important;
    border: 1px solid #1a2030 !important;
    font-weight: 600 !important;
}
.prose td, .markdown-body td {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    color: #7a90a8 !important;
    padding: 6px 12px !important;
    border: 1px solid #141924 !important;
}
.prose tr:hover td, .markdown-body tr:hover td {
    background: #0e1218 !important;
}
.prose strong, .markdown-body strong { color: #c8d3e0 !important; }
.prose em, .markdown-body em { color: #5a6f8a !important; }

/* ── Dropdown ── */
.gr-dropdown, .multiselect, ul.options {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    background: #0c0f16 !important;
    border-color: #1e2433 !important;
    color: #c8d3e0 !important;
}
ul.options li { background: #0c0f16 !important; color: #9ab0c4 !important; }
ul.options li:hover { background: #111d2e !important; }

/* ── Divider ── */
hr { border: none; border-top: 1px solid #141924 !important; margin: 20px 0 !important; }

/* ── Output monospace boxes ── */
.output-box textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    line-height: 1.7 !important;
    color: #9ab0c4 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0a0c10; }
::-webkit-scrollbar-thumb { background: #1e2433; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #2d3a50; }

/* ── Plotly charts ── */
.js-plotly-plot { background: transparent !important; }

/* ── Gradio footer ── */
.built-with { display: none !important; }
footer { display: none !important; }
"""


# =============================================================================
# UI
# =============================================================================

def build_ui():
    with gr.Blocks(
        title="SAE Interpretability Lab",
        theme=gr.themes.Base(),
        css=CUSTOM_CSS,
    ) as app:

        # ── Header ──
        gr.HTML(f"""
        <div class="app-header">
          <span class="app-title">SAE Interpretability Lab</span>
          <span class="app-meta">
            <span class="status-dot"></span>
            {MODEL_NAME.split('/')[-1]} &nbsp;·&nbsp; Layer {TARGET_LAYER} &nbsp;·&nbsp; {HOOK_TYPE} &nbsp;·&nbsp; TopK SAE
          </span>
        </div>
        """)

        with gr.Tabs():

            # ================================================================
            # TAB 1 — ANALYSIS (flagship: chat + live attribution)
            # ================================================================
            with gr.Tab("ANALYSIS"):
                with gr.Row(equal_height=False):

                    # ── Left: Chat ──
                    with gr.Column(scale=5, min_width=320):
                        gr.HTML('<div class="section-label" style="padding:18px 0 0;">Conversation</div>')
                        chatbot = gr.Chatbot(
                            label="", height=440, show_label=False,
                            bubble_full_width=False, show_copy_button=False,
                        )
                        with gr.Row():
                            chat_input = gr.Textbox(
                                placeholder="Ask a medical question...",
                                show_label=False, lines=2, scale=6,
                            )
                            with gr.Column(scale=1, min_width=74):
                                chat_submit = gr.Button("Send", variant="primary", size="sm")
                                chat_clear  = gr.Button("Clear", size="sm")

                    # ── Right: Attribution canvas ──
                    with gr.Column(scale=5, min_width=320):
                        gr.HTML('<div class="section-label" style="padding:18px 0 0;">Feature Attribution</div>')
                        attribution_out = gr.HTML(
                            '<div class="attr-empty">Send a message to see feature attribution.</div>'
                        )

                def on_chat(message, history):
                    if not message.strip():
                        return history, "", gr.update()
                    response, attr_html = generate_medical_response(message, history)
                    history = (history or []) + [(message, response)]
                    return history, "", attr_html

                def on_clear():
                    return [], "", '<div class="attr-empty">Send a message to see feature attribution.</div>'

                chat_submit.click(on_chat, [chat_input, chatbot], [chatbot, chat_input, attribution_out])
                chat_input.submit(on_chat, [chat_input, chatbot], [chatbot, chat_input, attribution_out])
                chat_clear.click(on_clear, outputs=[chatbot, chat_input, attribution_out])

            # ================================================================
            # TAB 2 — STEERING
            # ================================================================
            with gr.Tab("STEERING"):
                gr.HTML('<div class="section-label" style="padding:18px 0 0;">Feature Steering</div>')
                with gr.Row():
                    with gr.Column(scale=3):
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            value="The scientist announced that",
                            lines=2,
                        )
                        feature_choices = get_feature_choices()
                        feature_select = gr.Dropdown(
                            choices=feature_choices, label="Feature", value=None,
                        )
                        strength_input = gr.Number(value=0.0, label="Steering Strength", precision=1)
                        with gr.Row():
                            generate_btn = gr.Button("Generate", variant="primary")
                            reset_btn    = gr.Button("Reset")
                        steering_info = gr.Markdown(
                            "*Select a feature, set strength, then Generate.*"
                        )

                    with gr.Column(scale=4):
                        with gr.Row():
                            output_orig  = gr.Textbox(label="Baseline", lines=10, elem_classes=["output-box"])
                            output_steer = gr.Textbox(label="Steered",  lines=10, elem_classes=["output-box"])

                def on_generate(prompt, feature, strength):
                    if not feature or abs(float(strength or 0)) < 0.1:
                        orig, steered = generate_with_feature_steering(prompt, {}, 100)
                        info = "*No steering applied.*"
                    else:
                        feat_idx = int(feature.split(":")[0])
                        orig, steered = generate_with_feature_steering(prompt, {feat_idx: float(strength)}, 100)
                        sign = "+" if float(strength) > 0 else ""
                        info = f"**Active:** `{feature}` @ `{sign}{strength}`"
                    return orig, steered, info

                generate_btn.click(
                    on_generate,
                    inputs=[prompt_input, feature_select, strength_input],
                    outputs=[output_orig, output_steer, steering_info],
                )
                reset_btn.click(lambda: (None, 0.0), outputs=[feature_select, strength_input])

            # ================================================================
            # TAB 3 — EXPLORE (circuit + feature inspector)
            # ================================================================
            with gr.Tab("EXPLORE"):
                gr.HTML('<div class="section-label" style="padding:18px 0 0;">Circuit Analysis</div>')
                with gr.Row():
                    circuit_input = gr.Textbox(
                        label="Input text",
                        value="The stock market crashed because investors were worried.",
                        lines=2, scale=6,
                    )
                    circuit_btn = gr.Button("Analyze", variant="primary", scale=1, min_width=90)

                model_output_box = gr.Textbox(
                    label="Model Continuation", lines=2, interactive=False,
                )
                sankey_plot   = gr.Plot(label="")
                circuit_details = gr.Markdown()

                circuit_btn.click(
                    analyze_circuit,
                    inputs=[circuit_input],
                    outputs=[sankey_plot, circuit_details, model_output_box],
                )

                gr.HTML('<hr/>')
                gr.HTML('<div class="section-label">Feature Inspector</div>')
                with gr.Row():
                    feature_index_input = gr.Number(
                        value=0, label="Feature Index", precision=0, scale=1, min_width=120,
                    )
                    analyze_feature_btn = gr.Button("Inspect", variant="primary", scale=1, min_width=90)
                    activation_input = gr.Textbox(label="Activation text", lines=1, scale=4)
                    activation_btn   = gr.Button("Analyze Text", scale=1, min_width=100)

                with gr.Row():
                    with gr.Column(scale=1):
                        feature_analysis_output = gr.Markdown()
                    with gr.Column(scale=1):
                        feature_table = gr.Markdown()

                def on_inspect(idx):
                    if idx is None:
                        return "Enter a feature index."
                    feat_idx = int(idx)
                    if not (0 <= feat_idx < state.sae.d_hidden if state.loaded else 8192):
                        return "Feature index out of range."
                    return analyze_feature_deep(feat_idx)

                analyze_feature_btn.click(on_inspect, [feature_index_input], [feature_analysis_output])
                activation_btn.click(create_feature_detail_table, [activation_input], [feature_table])

            # ================================================================
            # TAB 4 — FEATURES (browser)
            # ================================================================
            with gr.Tab("FEATURES"):
                gr.HTML('<div class="section-label" style="padding:18px 0 0;">Labeled Feature Browser</div>')

                def load_features_table():
                    try:
                        med_state.load()
                        lf = med_state.labeled_features
                    except Exception:
                        lf = []
                    if not lf:
                        return "*No labeled features found. Run `python label_features.py` first.*"
                    conf_order = {"high": 0, "medium": 1, "low": 2}
                    sorted_f = sorted(lf, key=lambda x: (conf_order.get(x.get("confidence", "low"), 3), x["index"]))
                    badges = {"high": "🟢", "medium": "🟡", "low": "🔴"}
                    md = "| # | Label | Conf | MaxAct Tokens | VocabProj |\n|---|-------|------|---------------|----------|\n"
                    for f in sorted_f[:120]:
                        ma = ", ".join(
                            t[0] if isinstance(t, (list, tuple)) else str(t)
                            for t in f.get("max_act_tokens", [])[:3]
                        )
                        vp = ", ".join(str(t) for t in f.get("vocab_proj_tokens", [])[:3])
                        badge = badges.get(f.get("confidence", ""), "⚪")
                        md += f"| {f['index']} | {f['label'][:30]} | {badge} {f['confidence']} | {ma[:28]} | {vp[:28]} |\n"
                    return md

                features_table = gr.Markdown(value=load_features_table)
                refresh_btn = gr.Button("Refresh", size="sm")
                refresh_btn.click(load_features_table, outputs=[features_table])

    return app


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Starting SAE Interpretability Lab...")
    state.load()
    app = build_ui()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
