#!/usr/bin/env python3
"""
Usage:
    python app.py
    # Opens in browser at http://localhost:7860
"""

import os
# Fix for MPS (Apple Silicon) - enable CPU fallback for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import torch
import numpy as np
import gradio as gr
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from transformer_lens import HookedTransformer

from transformers import AutoModelForCausalLM

from config import (
    SparseAutoencoder,
    MODEL_NAME, TARGET_LAYER, HOOK_TYPE,
    OUTPUT_DIR, SAE_PATH, FINETUNE_OUTPUT_DIR, MEDICAL_OUTPUT_DIR,
    get_device,
)

# Special tokens to exclude from circuit analysis (Llama 3.2 + GPT-2 compatible)
_SPECIAL_TOKENS = {"<|endoftext|>", "<|begin_of_text|>", "<|end_of_text|>", "<|eot_id|>", ""}


# =============================================================================
# Helper Functions for Feature Selection
# =============================================================================

def get_feature_choices() -> List[str]:
    """Get list of labeled features for dropdown, sorted by confidence."""
    try:
        with open(OUTPUT_DIR / "labeled_features.json") as f:
            features = json.load(f)
        
        # Sort by confidence (high first)
        confidence_order = {"high": 0, "medium": 1, "low": 2}
        sorted_features = sorted(
            features,
            key=lambda x: (confidence_order.get(x.get("confidence", "low"), 3), x["index"])
        )
        
        # Format as "index: label (confidence)"
        choices = [
            f"{f['index']}: {f['label']} ({f['confidence']})"
            for f in sorted_features
        ]
        return choices
    except Exception:
        return ["No features loaded"]


# =============================================================================
# Global State (loaded once)
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

        # Load model (Llama 3.2 1B requires HF token: huggingface-cli login)
        self.model = HookedTransformer.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        
        # Load SAE
        self.sae = SparseAutoencoder.load(SAE_PATH)
        self.sae.to(self.device)
        self.sae.eval()
        
        # Load labeled features
        with open(OUTPUT_DIR / "labeled_features.json") as f:
            self.labeled_features = json.load(f)
        
        # Load ALL features (for circuit analysis)
        features_path = OUTPUT_DIR / "features.json"
        if features_path.exists():
            with open(features_path) as f:
                self.all_features = json.load(f)
            # Build index lookup
            self.feature_by_index = {f["index"]: f for f in self.all_features}
        else:
            self.all_features = []
            self.feature_by_index = {}
        
        self.loaded = True
        print(f"Loaded {len(self.labeled_features)} labeled features, {len(self.all_features)} total features")
            

state = AppState()


# =============================================================================
# Core Functions
# =============================================================================

def get_activations(text: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Get feature activations for input text."""
    state.load()
    
    tokens = state.model.to_tokens(text)
    token_strs = [state.model.tokenizer.decode([t]) for t in tokens[0]]
    
    hook_point = f"blocks.{TARGET_LAYER}.hook_{HOOK_TYPE}"
    
    with torch.no_grad():
        _, cache = state.model.run_with_cache(tokens)
        activations = cache[hook_point][0]  # [seq_len, d_model]
        
        # Get SAE hidden activations
        hidden = state.sae.encode(activations.to(state.device))  # [seq_len, d_hidden]
    
    return hidden, tokens, token_strs


def generate_with_feature_steering(
    prompt: str,
    feature_strengths: Dict[int, float],
    max_tokens: int = 50,
) -> Tuple[str, str]:
    """
    Generate text with and without steering using individual feature indices.
    
    Args:
        prompt: Input text
        feature_strengths: Dict mapping feature index -> strength
        max_tokens: Max tokens to generate
    """
    state.load()
    
    # Build steering vector from individual features
    steering_vector = torch.zeros(state.sae.d_hidden, device=state.device)
    
    for feat_idx, strength in feature_strengths.items():
        if abs(strength) > 0.1 and feat_idx < state.sae.d_hidden:
            steering_vector[feat_idx] = strength
    
    # Get decoder directions for steering
    decoder = state.sae.decoder.weight.detach()  # [d_model, d_hidden]
    steer_direction = decoder @ steering_vector  # [d_model]

    hook_point = f"blocks.{TARGET_LAYER}.hook_{HOOK_TYPE}"

    # Generate without steering
    tokens_orig = state.model.to_tokens(prompt)
    with torch.no_grad():
        output_orig = state.model.generate(
            tokens_orig,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        )
    text_orig = state.model.tokenizer.decode(output_orig[0])

    # If no steering, return same for both
    if steering_vector.abs().sum() < 0.1:
        return text_orig, text_orig

    # Scale steering direction by residual stream norm (strength=1.0 → 10% of residual norm)
    with torch.no_grad():
        _, cache = state.model.run_with_cache(state.model.to_tokens(prompt))
        resid_norm = cache[hook_point][0].norm(dim=-1).mean().item()  # avg norm across positions

    steer_unit = steer_direction / (steer_direction.norm() + 1e-8)
    total_strength = steering_vector.abs().max().item()  # max strength from all features
    scaled_steer = steer_unit * total_strength * resid_norm * 0.1

    # Generate with steering using manual generation loop
    def steering_hook(activation, hook):
        activation[:, :, :] += scaled_steer.unsqueeze(0).unsqueeze(0)
        return activation
    
    # Manual generation with hooks
    tokens = state.model.to_tokens(prompt)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Add hook for this forward pass
            state.model.add_hook(hook_point, steering_hook)
            
            # Get logits
            logits = state.model(tokens)[:, -1, :]  # [batch, vocab]
            
            # Remove hook after forward pass
            state.model.reset_hooks()
            
            # Sample next token
            probs = torch.softmax(logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Stop at EOS
            if next_token.item() == state.model.tokenizer.eos_token_id:
                break
    
    text_steer = state.model.tokenizer.decode(tokens[0])
    
    return text_orig, text_steer


def create_feature_ranking(text: str) -> go.Figure:
    """Create horizontal bar chart of top feature contributions."""
    state.load()
    
    if not text.strip():
        return go.Figure()
    
    hidden, _, _ = get_activations(text)
    
    # Get max activation per labeled feature
    feature_scores = []
    for f in state.labeled_features:
        idx = f["index"]
        if idx < hidden.shape[1]:
            max_act = hidden[:, idx].max().item()
            feature_scores.append({
                "label": f["label"],
                "index": idx,
                "activation": max_act,
                "confidence": f.get("confidence", "low"),
            })
    
    # Sort and take top 15
    feature_scores.sort(key=lambda x: x["activation"], reverse=True)
    top_features = feature_scores[:15]
    
    labels = [f"{f['label'][:25]} ({f['index']})" for f in top_features]
    values = [f["activation"] for f in top_features]
    
    # Color by confidence
    color_map = {"high": "#27ae60", "medium": "#f39c12", "low": "#e74c3c"}
    colors = [color_map.get(f["confidence"], "#95a5a6") for f in top_features]
    
    fig = go.Figure(data=go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker_color=colors,
    ))
    
    fig.update_layout(
        title="Top 15 Feature Activations (colored by confidence)",
        xaxis_title="Max Activation",
        yaxis_title="Feature",
        height=500,
        margin=dict(l=250),
        yaxis=dict(autorange="reversed"),
    )
    
    return fig


def create_feature_detail_table(text: str) -> str:
    """Create markdown table of top activated features."""
    state.load()
    
    if not text.strip():
        return "Enter text to see feature details."
    
    hidden, _, token_strs = get_activations(text)
    
    # Find top features by max activation
    max_per_feature = hidden.max(dim=0).values.cpu().numpy()
    
    # Map to labeled features
    feature_info = []
    for feat in state.labeled_features:
        idx = feat["index"]
        if idx < len(max_per_feature):
            feature_info.append({
                "index": idx,
                "label": feat["label"],
                "confidence": feat["confidence"],
                "max_act": max_per_feature[idx],
            })
    
    # Sort by activation
    feature_info.sort(key=lambda x: x["max_act"], reverse=True)
    
    # Build markdown table
    md = "### Top Activated Labeled Features\n\n"
    md += "| Feature | Label | Confidence | Max Activation |\n"
    md += "|---------|-------|------------|----------------|\n"
    
    for f in feature_info[:15]:
        md += f"| {f['index']} | {f['label'][:30]} | {f['confidence']} | {f['max_act']:.2f} |\n"
    
    return md


# =============================================================================
# Circuit Analysis Functions (uses ALL features)
# =============================================================================

def get_feature_label(feature_idx: int) -> str:
    """Get label for a feature if it exists, otherwise return index."""
    state.load()
    for f in state.labeled_features:
        if f["index"] == feature_idx:
            return f"{feature_idx}: {f['label']}"
    return f"Feature {feature_idx}"


def get_feature_vocab_projection(feature_idx: int, top_k: int = 5) -> List[Tuple[str, float]]:
    """Get top tokens this feature promotes (VocabProj)."""
    state.load()

    decoder_vec = state.sae.decoder.weight[:, feature_idx]  # [d_model]
    unembed = state.model.W_U  # [d_model, vocab_size]
    logits = decoder_vec @ unembed  # [vocab_size]

    top_values, top_indices = logits.topk(top_k)

    results = []
    for val, idx in zip(top_values, top_indices):
        token = state.model.tokenizer.decode([idx.item()])
        results.append((token, val.item()))

    return results


def get_vocab_projections_batched(feature_indices: List[int], top_k: int = 3) -> Dict[int, List[Tuple[str, float]]]:
    """Get VocabProj for multiple features in one matmul (~25x faster than looping)."""
    state.load()

    indices_tensor = torch.tensor(feature_indices, dtype=torch.long)
    decoder_cols = state.sae.decoder.weight[:, indices_tensor]  # [d_model, n_features]
    unembed = state.model.W_U  # [d_model, vocab_size]
    all_logits = decoder_cols.T @ unembed  # [n_features, vocab_size]

    top_values, top_indices = all_logits.topk(top_k, dim=-1)  # [n_features, top_k]

    results = {}
    for i, feat_idx in enumerate(feature_indices):
        tokens_logits = []
        for j in range(top_k):
            tok = state.model.tokenizer.decode([top_indices[i, j].item()])
            tokens_logits.append((tok, top_values[i, j].item()))
        results[feat_idx] = tokens_logits

    return results


def analyze_circuit(text: str) -> Tuple[go.Figure, str, str]:
    """
    Full circuit analysis: Input tokens → Features → Output tokens
    Uses ALL features, shows labels when available.
    Filters out <|endoftext|> token.
    """
    state.load()
    
    if not text.strip():
        return go.Figure(), "Enter text to analyze.", ""
    
    hidden, tokens, token_strs = get_activations(text)
    hidden_np = hidden.cpu().numpy()  # [seq_len, d_hidden]
    
    # Filter out special/BOS tokens
    valid_indices = [i for i, t in enumerate(token_strs) if t.strip() not in _SPECIAL_TOKENS]
    if not valid_indices:
        return go.Figure(), "No valid tokens found.", ""
    
    filtered_token_strs = [token_strs[i] for i in valid_indices]
    filtered_hidden = hidden_np[valid_indices]  # [filtered_seq_len, d_hidden]
    
    n_tokens = len(filtered_token_strs)
    
    # Find all features with activation > threshold
    max_per_feature = filtered_hidden.max(axis=0)  # [d_hidden]
    activation_threshold = 0.5
    activated_mask = max_per_feature > activation_threshold
    activated_indices = np.where(activated_mask)[0]
    
    # Sort by activation strength, take top 25 for visualization
    sorted_by_activation = activated_indices[np.argsort(max_per_feature[activated_indices])[::-1]]
    top_feature_indices = sorted_by_activation[:25]
    
    total_activated = len(activated_indices)
    
    # =================================================================
    # Generate Model Output
    # =================================================================
    with torch.no_grad():
        input_tokens = state.model.to_tokens(text)
        generated = state.model.generate(
            input_tokens,
            max_new_tokens=30,
            temperature=0.8,
            top_p=0.9,
            stop_at_eos=True,
        )
        model_output = state.model.tokenizer.decode(generated[0])
    
    # =================================================================
    # Input → Feature → Output Sankey Diagram
    # =================================================================
    
    sankey_labels = []
    sankey_source = []
    sankey_target = []
    sankey_value = []
    sankey_colors = []
    
    # Add input token nodes (filtered)
    input_labels = [f"IN: {t.strip()[:12]}" for t in filtered_token_strs]
    sankey_labels.extend(input_labels)
    n_input = len(input_labels)
    
    # Add feature nodes
    feature_labels = [get_feature_label(idx)[:25] for idx in top_feature_indices]
    sankey_labels.extend(feature_labels)
    n_feature = len(feature_labels)
    
    # Add output token nodes (from VocabProj of top features) — batched for speed
    output_tokens_set = {}
    feature_to_outputs = get_vocab_projections_batched([int(i) for i in top_feature_indices], top_k=3)
    for feat_idx in top_feature_indices:
        for tok, logit in feature_to_outputs.get(int(feat_idx), []):
            tok_clean = tok.strip()[:12]
            if tok_clean and tok_clean not in output_tokens_set:
                output_tokens_set[tok_clean] = logit
    
    # Sort output tokens by total logit contribution
    sorted_outputs = sorted(output_tokens_set.items(), key=lambda x: x[1], reverse=True)[:20]
    output_labels = [f"OUT: {tok}" for tok, _ in sorted_outputs]
    sankey_labels.extend(output_labels)
    
    # Links: Input tokens → Features (blue)
    for tok_idx in range(n_tokens):
        for i, feat_idx in enumerate(top_feature_indices[:15]):
            act = filtered_hidden[tok_idx, feat_idx]
            if act > activation_threshold:
                sankey_source.append(tok_idx)
                sankey_target.append(n_input + i)
                sankey_value.append(float(act))
                sankey_colors.append("rgba(31, 119, 180, 0.5)")
    
    # Links: Features → Output tokens (orange)
    output_label_set = {label: idx for idx, label in enumerate(output_labels)}
    for i, feat_idx in enumerate(top_feature_indices[:15]):
        outputs = feature_to_outputs.get(int(feat_idx), [])
        for tok, logit in outputs:
            tok_clean = tok.strip()[:12]
            out_label = f"OUT: {tok_clean}"
            if out_label in output_label_set:
                sankey_source.append(n_input + i)
                sankey_target.append(n_input + n_feature + output_label_set[out_label])
                sankey_value.append(max(0.5, abs(logit)))
                sankey_colors.append("rgba(255, 127, 14, 0.5)")
    
    sankey_fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=sankey_labels,
            color=["#1f77b4"] * n_input + ["#2ca02c"] * n_feature + ["#ff7f0e"] * len(output_labels),
        ),
        link=dict(
            source=sankey_source,
            target=sankey_target,
            value=sankey_value,
            color=sankey_colors,
        )
    )])
    
    sankey_fig.update_layout(
        title=f"Circuit Flow ({total_activated} features activated, showing top 25)",
        height=600,
        font_size=10,
    )
    
    # =================================================================
    # Detailed Path Analysis (Markdown)
    # =================================================================
    
    md = f"## Circuit Analysis Details\n\n"
    md += f"**Total features activated:** {total_activated} (threshold > {activation_threshold})\n\n"
    
    md += "### Top Activated Features\n\n"
    md += "| Rank | Feature | Label | Max Activation | Promotes Tokens |\n"
    md += "|------|---------|-------|----------------|------------------|\n"
    
    for rank, feat_idx in enumerate(top_feature_indices[:15], 1):
        feat_idx = int(feat_idx)
        label = get_feature_label(feat_idx)
        max_act = max_per_feature[feat_idx]
        outputs = feature_to_outputs.get(feat_idx, [])
        out_str = ", ".join([f"'{t.strip()}'" for t, _ in outputs[:3]])
        md += f"| {rank} | {feat_idx} | {label[:30]} | {max_act:.2f} | {out_str} |\n"
    
    md += "\n### Input Token → Feature Mapping\n\n"
    for tok_idx, tok in enumerate(filtered_token_strs[:10]):
        tok_acts = filtered_hidden[tok_idx]
        top_for_token = np.argsort(tok_acts)[-5:][::-1]
        features_str = ", ".join([f"{get_feature_label(int(i))[:20]}({tok_acts[i]:.1f})" for i in top_for_token if tok_acts[i] > 0.1])
        if features_str:
            md += f"**'{tok.strip()}'** → {features_str}\n\n"
    
    return sankey_fig, md, model_output


def analyze_feature_deep(feature_idx: int) -> str:
    """Deep analysis of a single feature - uses full feature data."""
    state.load()
    
    # Check if feature is labeled
    label_info = None
    for f in state.labeled_features:
        if f["index"] == feature_idx:
            label_info = f
            break
    
    if label_info:
        label_text = label_info["label"]
        confidence = label_info.get("confidence", "unknown")
    else:
        label_text = "Unlabelled"
        confidence = None
    
    # Get VocabProj
    outputs = get_feature_vocab_projection(feature_idx, top_k=10)
    
    # Get decoder vector stats
    decoder_vec = state.sae.decoder.weight[:, feature_idx].detach().cpu().numpy()
    
    # Get full feature data (MaxAct examples) if available
    full_data = state.feature_by_index.get(feature_idx, {})
    max_act_examples = full_data.get("max_activating_tokens", [])  # fixed key
    vocab_proj_tokens = full_data.get("vocab_projection", [])       # fixed key
    frequency = full_data.get("frequency", 0)

    md = f"## Feature {feature_idx} Deep Analysis\n\n"
    if confidence:
        md += f"**Label:** {label_text} (confidence: {confidence})\n\n"
    else:
        md += f"**Label:** {label_text}\n\n"
    md += f"**Activation Frequency:** {frequency:.1%} of tokens\n\n"

    md += "### Input Tokens (MaxAct Examples)\n"
    md += "These input contexts maximally activate this feature:\n\n"

    if max_act_examples:
        for i, ex in enumerate(max_act_examples[:5], 1):
            context = ex.get("context", "")
            token = ex.get("token", "")
            act = ex.get("activation", 0)
            if context:
                md += f"**{i}.** `{context}` (activation: {act:.2f})\n\n"
            else:
                md += f"**{i}.** **'{token}'** (activation: {act:.2f})\n\n"
    else:
        md += "*No MaxAct examples available*\n\n"
    
    md += "### Output Tokens (VocabProj)\n"
    md += "These tokens are promoted when this feature activates:\n\n"
    md += "| Token | Logit |\n|-------|-------|\n"
    for tok, logit in outputs:
        md += f"| '{tok}' | {logit:.3f} |\n"
    
    if vocab_proj_tokens:
        md += f"\n*From features.json:* {', '.join(vocab_proj_tokens[:10])}\n"
    
    md += f"\n### Decoder Vector Stats\n"
    md += f"- Norm: {np.linalg.norm(decoder_vec):.3f}\n"
    md += f"- Mean: {decoder_vec.mean():.5f}\n"
    md += f"- Std: {decoder_vec.std():.5f}\n"
    
    return md


# =============================================================================
# Medical State (lazy-loaded separately from base model state)
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
        if not FINETUNE_OUTPUT_DIR.exists() or not med_sae_path.exists():
            self.error = (
                f"Medical model not found. Run:\n"
                f"  1. `python finetune.py` → saves to {FINETUNE_OUTPUT_DIR}\n"
                f"  2. `python main.py --model-path {FINETUNE_OUTPUT_DIR} "
                f"--dataset medical --output-dir {MEDICAL_OUTPUT_DIR}`"
            )
            return

        print("Loading medical model...")
        try:
            hf_model = AutoModelForCausalLM.from_pretrained(
                str(FINETUNE_OUTPUT_DIR), torch_dtype=torch.float32
            )
            self.model = HookedTransformer.from_pretrained(MODEL_NAME, hf_model=hf_model)
            self.model.to(self.device)
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
            print(f"  Medical model loaded | {len(self.labeled_features)} labeled features")
        except Exception as e:
            self.error = f"Failed to load medical model: {e}"
            print(f"  ERROR: {self.error}")


med_state = MedicalState()


# =============================================================================
# Medical Chat & Attribution
# =============================================================================

def _build_attribution_md(hidden: torch.Tensor) -> str:
    """Build the attribution panel markdown from SAE hidden activations."""
    max_per_feature = hidden.max(dim=0).values.cpu()
    top_vals, top_idxs = max_per_feature.topk(min(8, max_per_feature.shape[0]))

    cat_icon = {"semantic": "🔵", "syntactic": "🟡", "positional": "🟣"}

    md = "## Why did the model say this?\n\n"
    max_val = top_vals[0].item() if top_vals[0].item() > 0 else 1.0

    for rank, (feat_idx, feat_val) in enumerate(zip(top_idxs.tolist(), top_vals.tolist()), 1):
        if feat_val <= 0:
            continue

        # Label lookup
        label, category, confidence = f"Feature {feat_idx}", "unknown", ""
        for f in med_state.labeled_features:
            if f["index"] == feat_idx:
                label = f.get("label", label)
                category = f.get("category", "unknown")
                confidence = f.get("confidence", "")
                break

        bar_filled = int(feat_val / max_val * 8)
        bar = "█" * bar_filled + "░" * (8 - bar_filled)
        icon = cat_icon.get(category, "⚪")

        conf_str = f" · confidence: {confidence}" if confidence else ""
        md += f"### {rank}. {icon} {label}\n"
        md += f"`{bar}` **{feat_val:.2f}** · {category}{conf_str}\n\n"

        # Training evidence
        feat_data = med_state.feature_by_index.get(feat_idx, {})
        examples = feat_data.get("max_activating_tokens", [])[:3]
        if examples:
            md += "**Training evidence:**\n\n"
            for ex in examples:
                ctx = ex.get("context", "")
                token = ex.get("token", "")
                source = ex.get("source_id", "")
                ctx_display = ctx.replace(f"[{token}]", f"**[{token}]**") if token else ctx
                md += f"> {ctx_display}\n\n"
                if source:
                    md += f"*Source: {source}*\n\n"

    # VocabProj chips from top 5 features
    top_feat_list = top_idxs[:5].tolist()
    try:
        decoder_cols = med_state.sae.decoder.weight[:, top_feat_list].detach()
        unembed = med_state.model.W_U.detach()
        avg_logits = (decoder_cols.T @ unembed).mean(dim=0)
        top_tok_vals, top_tok_idxs = avg_logits.topk(5)
        chips = []
        for tok_id, logit in zip(top_tok_idxs.tolist(), top_tok_vals.tolist()):
            tok = med_state.model.tokenizer.decode([tok_id]).strip()
            if tok:
                chips.append(f"`{tok}` ({logit:.1f})")
        if chips:
            md += "**Output tokens promoted:** " + " ".join(chips) + "\n"
    except Exception:
        pass

    return md


def generate_medical_response(message: str, history: list) -> Tuple[str, str]:
    """Generate a response from the fine-tuned model and build attribution panel."""
    med_state.load()

    if med_state.error:
        return med_state.error, f"*{med_state.error}*"

    # Build prompt from last 3 history turns
    prompt = ""
    for user_msg, assistant_msg in (history or [])[-3:]:
        prompt += f"User: {user_msg}\nAssistant: {assistant_msg}\n"
    prompt += f"User: {message}\nAssistant:"

    tokens = med_state.model.to_tokens(prompt)

    with torch.no_grad():
        generated = med_state.model.generate(
            tokens,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            stop_at_eos=True,
        )

    response_token_ids = generated[0, tokens.shape[1]:]
    response = med_state.model.tokenizer.decode(response_token_ids.tolist())

    # Get SAE activations for the generated portion
    hook_point = f"blocks.{TARGET_LAYER}.hook_{HOOK_TYPE}"
    with torch.no_grad():
        _, cache = med_state.model.run_with_cache(generated)
        activations = cache[hook_point][0]  # [seq_len, d_model]
        gen_acts = activations[tokens.shape[1]:]
        if gen_acts.shape[0] == 0:
            gen_acts = activations
        hidden = med_state.sae.encode(gen_acts.to(med_state.device))

    attribution_md = _build_attribution_md(hidden)
    return response, attribution_md


# =============================================================================
# Gradio UI
# =============================================================================

def build_ui():
    """Build the Gradio interface."""
    
    with gr.Blocks(
        title="SAE Feature Steering Playground",
        theme=gr.themes.Soft(),
        css="""
        .output-box { font-family: monospace; }
        """
    ) as app:
        
        gr.Markdown("""
        # Playground
        """)
        
        with gr.Tabs():
            # =================================================================
            # Tab 1: Steering Playground
            # =================================================================
            with gr.Tab("Steering Playground"):
                
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="Enter your prompt here...",
                            value="The scientist announced that",
                            lines=2,
                        )
                        
                        gr.Markdown("### Select Feature to Steer")
                        
                        feature_choices = get_feature_choices()
                        
                        feature_select = gr.Dropdown(
                            choices=feature_choices,
                            label="Feature (searchable)",
                            value=None,
                            allow_custom_value=False,
                        )
                        strength_input = gr.Number(
                            value=0,
                            label="Steering Strength",
                        )
                        
                        with gr.Row():
                            generate_btn = gr.Button("🚀 Generate", variant="primary")
                            reset_btn = gr.Button("🔄 Reset")
                    
                    with gr.Column(scale=2):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### Original Output")
                                output_orig = gr.Textbox(
                                    label="",
                                    lines=12,
                                    elem_classes=["output-box"],
                                )
                            with gr.Column():
                                gr.Markdown("### Steered Output")
                                output_steer = gr.Textbox(
                                    label="",
                                    lines=12,
                                    elem_classes=["output-box"],
                                )
                        
                        steering_info = gr.Markdown("*Select a feature and click Generate*")
                
                def on_generate(prompt, feature, strength):
                    if not feature or abs(strength) < 0.1:
                        # No steering
                        orig, steered = generate_with_feature_steering(prompt, {}, 100)
                        info = "*No feature selected or strength is 0 - showing unsteered output*"
                    else:
                        # Parse feature index from dropdown value "123: label"
                        feat_idx = int(feature.split(":")[0])
                        feature_strengths = {feat_idx: strength}
                        orig, steered = generate_with_feature_steering(prompt, feature_strengths, 100)
                        sign = "+" if strength > 0 else ""
                        info = f"**Steering:** {feature} @ {sign}{strength}"
                    
                    return orig, steered, info
                
                def reset_all():
                    return None, 0
                
                generate_btn.click(
                    on_generate,
                    inputs=[prompt_input, feature_select, strength_input],
                    outputs=[output_orig, output_steer, steering_info],
                )
                
                reset_btn.click(
                    reset_all,
                    outputs=[feature_select, strength_input],
                )
            
            # =================================================================
            # Tab 2: Feature Activations
            # =================================================================
            with gr.Tab("📊 Feature Activations"):
                gr.Markdown("""
                ### Feature Activations
                See which SAE features activate on your input text.
                """)
                
                activation_input = gr.Textbox(
                    label="Input Text",
                    placeholder="Enter text to analyze...",
                    value="The president announced new economic policies today.",
                    lines=2,
                )
                
                activation_btn = gr.Button("🔍 Analyze", variant="primary")
                
                feature_ranking_plot = gr.Plot(label="Top Feature Activations")
                feature_table = gr.Markdown()
                
                def on_activation_analyze(text):
                    ranking = create_feature_ranking(text)
                    table = create_feature_detail_table(text)
                    return ranking, table
                
                activation_btn.click(
                    on_activation_analyze,
                    inputs=[activation_input],
                    outputs=[feature_ranking_plot, feature_table],
                )
            
            # =================================================================
            # Tab 3: Circuit Analysis
            # =================================================================
            with gr.Tab("🔬 Circuit Analysis"):
                gr.Markdown("""
                ### Circuit Analysis: Input → Features → Output
                Trace how tokens flow through SAE features to output tokens.
                
                **Legend:** 🔵 Blue = Input tokens → Features | 🟠 Orange = Features → Output tokens
                """)
                
                circuit_input = gr.Textbox(
                    label="Input Text",
                    placeholder="Enter text to trace through the circuit...",
                    value="The stock market crashed because investors were worried.",
                    lines=2,
                )
                
                circuit_analyze_btn = gr.Button("🔬 Analyze Circuit", variant="primary")
                
                model_output_box = gr.Textbox(
                    label="Model Output",
                    lines=3,
                    interactive=False,
                )
                
                with gr.Row():
                    sankey_plot = gr.Plot(label="Input → Features → Output Flow")
                
                circuit_details = gr.Markdown()
                
                circuit_analyze_btn.click(
                    analyze_circuit,
                    inputs=[circuit_input],
                    outputs=[sankey_plot, circuit_details, model_output_box],
                )
            
            # =================================================================
            # Tab 4: Feature Deep Dive
            # =================================================================
            with gr.Tab("🔍 Feature Deep Dive"):
                gr.Markdown("""
                ### Individual Feature Analysis
                Explore any feature in detail - see what inputs activate it and what outputs it promotes.
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        feature_index_input = gr.Number(
                            value=0,
                            label=f"Feature Index (0-{8192-1})",
                            precision=0,
                        )
                        analyze_feature_btn = gr.Button("🔍 Analyze Feature", variant="primary")
                    
                    with gr.Column(scale=2):
                        feature_analysis_output = gr.Markdown()
                
                def on_analyze_feature(index_value):
                    if index_value is None:
                        return "Please enter a feature index."
                    
                    feat_idx = int(index_value)
                    if feat_idx < 0 or feat_idx >= 8192:
                        return "Feature index must be between 0 and 8191."
                    
                    return analyze_feature_deep(feat_idx)
                
                analyze_feature_btn.click(
                    on_analyze_feature,
                    inputs=[feature_index_input],
                    outputs=[feature_analysis_output],
                )
            
            # =================================================================
            # Tab 5: Feature Browser
            # =================================================================
            with gr.Tab("📚 Feature Browser"):
                gr.Markdown("""
                ### Browse Labeled Features
                Explore all labeled features found by the SAE (sorted by confidence).
                """)
                
                def load_features_table():
                    state.load()
                    
                    # Sort by confidence: high > medium > low
                    confidence_order = {"high": 0, "medium": 1, "low": 2}
                    sorted_features = sorted(
                        state.labeled_features,
                        key=lambda x: confidence_order.get(x.get("confidence", "low"), 3)
                    )
                    
                    md = "| Index | Label | Confidence | MaxAct Tokens | VocabProj Tokens |\n"
                    md += "|-------|-------|------------|---------------|------------------|\n"
                    for f in sorted_features[:100]:
                        # max_act_tokens are stored as [token, activation, context] tuples
                        max_act = ", ".join(
                            t[0] if isinstance(t, (list, tuple)) else str(t)
                            for t in f.get("max_act_tokens", [])[:3]
                        )
                        vocab = ", ".join(str(t) for t in f.get("vocab_proj_tokens", [])[:3])
                        md += f"| {f['index']} | {f['label'][:25]} | {f['confidence']} | {max_act[:30]} | {vocab[:30]} |\n"
                    return md
                
                features_table = gr.Markdown(value=load_features_table)
                
                refresh_btn = gr.Button("🔄 Refresh")
                refresh_btn.click(load_features_table, outputs=[features_table])

            # =================================================================
            # Tab 6: Medical Chat + Attribution
            # =================================================================
            with gr.Tab("🏥 Medical Chat + Attribution"):
                gr.Markdown("""
                ### Medical Chat with Training Data Attribution
                Ask medical questions and see **which features activated** and
                **which training passages** influenced the response.

                > Requires: `python finetune.py` then `python main.py --dataset medical`
                """)

                with gr.Row():
                    # Left column — chat
                    with gr.Column(scale=1):
                        chatbot = gr.Chatbot(label="Medical Chat", height=450)
                        chat_input = gr.Textbox(
                            label="Your question",
                            placeholder="e.g. What are the symptoms of myocardial infarction?",
                            lines=2,
                        )
                        with gr.Row():
                            chat_submit = gr.Button("Send", variant="primary")
                            chat_clear = gr.Button("Clear")

                    # Right column — live attribution
                    with gr.Column(scale=1):
                        attribution_out = gr.Markdown(
                            "*Send a message to see training data attribution.*",
                            label="Attribution Panel",
                        )

                def on_chat(message, history):
                    if not message.strip():
                        return history, "", "*Enter a message first.*"
                    response, attribution_md = generate_medical_response(message, history)
                    history = (history or []) + [(message, response)]
                    return history, "", attribution_md

                chat_submit.click(
                    on_chat,
                    inputs=[chat_input, chatbot],
                    outputs=[chatbot, chat_input, attribution_out],
                )
                chat_clear.click(
                    lambda: ([], "", "*Send a message to see training data attribution.*"),
                    outputs=[chatbot, chat_input, attribution_out],
                )

    return app


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Starting SAE Feature Steering Playground...")
    
    # Pre-load models
    state.load()
    
    # Build and launch UI
    app = build_ui()
    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
    )
