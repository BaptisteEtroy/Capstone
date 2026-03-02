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
import plotly.express as px
from transformer_lens import HookedTransformer

from config import (
    SparseAutoencoder,
    MODEL_NAME, TARGET_LAYER, HOOK_TYPE,
    OUTPUT_DIR, SAE_PATH,
    get_device,
)


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


def build_concept_groups(labeled_features: List[Dict]) -> Dict[str, List[int]]:
    """
    Build concept groups directly from actual labeled features.
    Each unique label becomes its own concept group - no hardcoding.
    """
    label_to_indices = {}
    for f in labeled_features:
        label = f["label"]  # Use exact label as-is
        idx = f["index"]
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)
    
    # Return all labels that have features (sorted by number of features)
    sorted_groups = sorted(label_to_indices.items(), key=lambda x: len(x[1]), reverse=True)
    return {k: v for k, v in sorted_groups}


# =============================================================================
# Global State (loaded once)
# =============================================================================

class AppState:
    def __init__(self):
        self.model = None
        self.sae = None
        self.labeled_features = []
        self.concept_features = {}
        self.device = get_device()
        self.loaded = False
    
    def load(self):
        if self.loaded:
            return
        
        print("Loading models...")
        
        # Load GPT-2
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
        
        # Build concept groups from actual labels (for attribution graphs)
        self.concept_features = build_concept_groups(self.labeled_features)
        
        self.loaded = True
        print(f"Loaded {len(self.labeled_features)} labeled features")
            

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
        feature_strengths: Dict mapping feature index -> strength (-5 to +5)
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
    
    # Generate with steering using manual generation loop
    def steering_hook(activation, hook):
        activation[:, :, :] += steer_direction.unsqueeze(0).unsqueeze(0)
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


def create_activation_heatmap(text: str) -> go.Figure:
    """Create heatmap of feature activations for labeled features."""
    state.load()
    
    if not text.strip():
        return go.Figure()
    
    hidden, tokens, token_strs = get_activations(text)
    
    # Get activations for labeled features only (top 30 most active)
    labeled_indices = [f["index"] for f in state.labeled_features]
    labeled_labels = [f["label"] for f in state.labeled_features]
    
    # Get activations for these features
    labeled_acts = hidden[:, labeled_indices].cpu().numpy()  # [seq_len, n_labeled]
    
    # Find top activated features across all tokens
    max_acts = labeled_acts.max(axis=0)
    top_feature_idx = np.argsort(max_acts)[-20:][::-1]  # Top 20
    
    # Build heatmap data
    heatmap_data = labeled_acts[:, top_feature_idx].T  # [n_features, seq_len]
    feature_names = [labeled_labels[i][:25] for i in top_feature_idx]
    
    # Clean token strings for display
    display_tokens = [t.replace('\n', '\\n')[:10] for t in token_strs]
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=display_tokens,
        y=feature_names,
        colorscale='RdBu_r',
        zmid=0,
    ))
    
    fig.update_layout(
        title="Feature Activations (Top 20 Active Features)",
        xaxis_title="Tokens",
        yaxis_title="Features",
        height=500,
        margin=dict(l=150),
    )
    
    return fig


def create_activation_flow(text: str) -> go.Figure:
    """Create line chart showing feature activations across token positions."""
    state.load()
    
    if not text.strip():
        return go.Figure()
    
    hidden, _, token_strs = get_activations(text)
    
    # Get top 8 most active labeled features
    labeled_indices = [f["index"] for f in state.labeled_features]
    labeled_labels = [f["label"] for f in state.labeled_features]
    
    labeled_acts = hidden[:, labeled_indices].cpu().numpy()
    max_acts = labeled_acts.max(axis=0)
    top_idx = np.argsort(max_acts)[-8:][::-1]
    
    # Clean token strings
    display_tokens = [t.replace('\n', '\\n')[:8] for t in token_strs]
    
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    
    for i, idx in enumerate(top_idx):
        fig.add_trace(go.Scatter(
            x=list(range(len(display_tokens))),
            y=labeled_acts[:, idx],
            mode='lines+markers',
            name=labeled_labels[idx][:20],
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=6),
        ))
    
    fig.update_layout(
        title="Feature Activation Flow Across Tokens",
        xaxis_title="Token Position",
        yaxis_title="Activation",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(display_tokens))),
            ticktext=display_tokens,
            tickangle=45,
        ),
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(b=100),
    )
    
    return fig


def create_concept_pie(text: str) -> go.Figure:
    """Create pie chart showing distribution of active concepts."""
    state.load()
    
    if not text.strip():
        return go.Figure()
    
    hidden, _, _ = get_activations(text)
    hidden_mean = hidden.mean(dim=0).cpu().numpy()
    
    concept_scores = {}
    for concept, indices in state.concept_features.items():
        if indices:
            scores = hidden_mean[indices]
            # Only count positive activations
            positive_score = max(0, float(np.mean(scores)))
            if positive_score > 0:
                concept_scores[concept] = positive_score
    
    if not concept_scores:
        return go.Figure()
    
    labels = list(concept_scores.keys())
    values = list(concept_scores.values())
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        textinfo='label+percent',
        textposition='outside',
    )])
    
    fig.update_layout(
        title="Concept Distribution (Active Features)",
        height=400,
        showlegend=False,
    )
    
    return fig


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
            # Tab 2: Attribution Graphs
            # =================================================================
            with gr.Tab("📊 Attribution Graphs"):
                gr.Markdown("""
                ### Feature Attribution & Visualization
                Analyze which features activate on your text and how concepts are distributed.
                """)
                
                heatmap_input = gr.Textbox(
                    label="Input Text",
                    placeholder="Enter text to analyze...",
                    value="The president announced new economic policies today.",
                    lines=2,
                )
                
                analyze_btn = gr.Button("🔍 Analyze", variant="primary")
                
                with gr.Row():
                    with gr.Column():
                        feature_ranking_plot = gr.Plot(label="Top Feature Activations")
                    with gr.Column():
                        concept_pie_plot = gr.Plot(label="Concept Distribution")
                
                with gr.Row():
                    activation_flow_plot = gr.Plot(label="Feature Activation Flow")
                
                with gr.Row():
                    heatmap_plot = gr.Plot(label="Feature × Token Heatmap")
                
                feature_table = gr.Markdown()
                
                def on_analyze(text):
                    ranking = create_feature_ranking(text)
                    pie = create_concept_pie(text)
                    flow = create_activation_flow(text)
                    heatmap = create_activation_heatmap(text)
                    table = create_feature_detail_table(text)
                    return ranking, pie, flow, heatmap, table
                
                analyze_btn.click(
                    on_analyze,
                    inputs=[heatmap_input],
                    outputs=[feature_ranking_plot, concept_pie_plot, activation_flow_plot, 
                             heatmap_plot, feature_table],
                )
            
            # =================================================================
            # Tab 3: Feature Browser
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
                        max_act = ", ".join(f.get("max_act_tokens", [])[:3])
                        vocab = ", ".join(f.get("vocab_proj_tokens", [])[:3])
                        md += f"| {f['index']} | {f['label'][:25]} | {f['confidence']} | {max_act[:30]} | {vocab[:30]} |\n"
                    return md
                
                features_table = gr.Markdown(value=load_features_table)
                
                refresh_btn = gr.Button("🔄 Refresh")
                refresh_btn.click(load_features_table, outputs=[features_table])
    
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
