#!/usr/bin/env python3
"""
Usage:
    python main.py                    # Full pipeline
    python main.py --quick            # Quick test (500 samples, 1 epoch)
    python main.py --skip-collection  # Skip activation collection (use cached)
"""

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm
import argparse
import json

from transformer_lens import HookedTransformer
from datasets import load_dataset

# Import all config and shared classes
from config import (
    MODEL_NAME, D_MODEL, TARGET_LAYER, HOOK_TYPE,
    EXPANSION_FACTOR, TOP_K,
    LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, NUM_SAMPLES,
    OUTPUT_DIR,
    get_device,
    SparseAutoencoder, ActivationData, MaxActExample, FeatureInfo,
)


# =============================================================================
# Model Loading & Activation Collection
# =============================================================================


def load_gpt2(device: Optional[str] = None) -> HookedTransformer:
    device = device or get_device()
    print(f"\nLoading GPT-2 on {device}...")
    
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=device,
        dtype=torch.float32,
    )
    print(f"  d_model: {model.cfg.d_model}, n_layers: {model.cfg.n_layers}")
    return model


@torch.no_grad()
def collect_activations(
    model: HookedTransformer,
    num_samples: int = NUM_SAMPLES,
    batch_size: int = 32,
    max_tokens: int = 128,
) -> ActivationData:
    """
    Collect residual stream activations from GPT-2 layer 6.
    
    Uses OpenWebText dataset for diverse text inputs.
    Returns both activations and corresponding token IDs for MaxAct analysis.
    """
    hook_point = f"blocks.{TARGET_LAYER}.hook_{HOOK_TYPE}"
    print(f"\nCollecting activations from {hook_point}...")
    print(f"  Samples: {num_samples}, Batch size: {batch_size}")
    
    # Load dataset
    dataset = load_dataset("openwebtext", split="train", streaming=True)
    
    all_activations = []
    all_token_ids = []
    batch = []
    count = 0
    
    pbar = tqdm(total=num_samples, desc="Collecting")
    
    for item in dataset:
        text = item.get("text", "")
        if not text.strip():
            continue
        
        batch.append(text[:max_tokens * 4])  # Rough char limit
        count += 1
        pbar.update(1)
        
        if len(batch) >= batch_size:
            # Tokenize and run
            tokens = model.to_tokens(batch)
            if tokens.shape[1] > max_tokens:
                tokens = tokens[:, :max_tokens]
            
            _, cache = model.run_with_cache(tokens, names_filter=[hook_point])
            acts = cache[hook_point].reshape(-1, D_MODEL).cpu()
            all_activations.append(acts)
            
            # Store flattened token IDs (matches activation shape)
            all_token_ids.append(tokens.reshape(-1).cpu())
            
            del cache
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            batch = []
        
        if count >= num_samples:
            break
    
    # Process remaining batch
    if batch:
        tokens = model.to_tokens(batch)
        if tokens.shape[1] > max_tokens:
            tokens = tokens[:, :max_tokens]
        _, cache = model.run_with_cache(tokens, names_filter=[hook_point])
        acts = cache[hook_point].reshape(-1, D_MODEL).cpu()
        all_activations.append(acts)
        all_token_ids.append(tokens.reshape(-1).cpu())
    
    pbar.close()
    
    activations = torch.cat(all_activations, dim=0)
    token_ids = torch.cat(all_token_ids, dim=0)
    
    print(f"  Collected {activations.shape[0]:,} token activations")
    return ActivationData(activations=activations, token_ids=token_ids)


# =============================================================================
# Training
# =============================================================================


def train_sae(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    device: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    Train the TopK SAE. Loss is pure reconstruction MSE — no sparsity penalty needed.
    Sparsity is guaranteed by TopK: exactly K features activate per token.
    """
    device = device or get_device()
    sae = sae.to(device)

    print(f"\nTraining SAE...")
    print(f"  Architecture: {sae.d_model} -> {sae.d_hidden} -> {sae.d_model}")
    print(f"  TopK: K={sae.k} (L0={sae.k} exactly, guaranteed)")
    print(f"  Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}")

    optimizer = Adam(sae.parameters(), lr=learning_rate)
    dataset = torch.utils.data.TensorDataset(activations)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_steps = len(loader) * num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=learning_rate * 0.1)

    history = {"loss": [], "reconstruction": []}
    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for (batch,) in pbar:
            batch = batch.to(device)

            output = sae(batch)

            optimizer.zero_grad()
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            sae._normalize_decoder()

            epoch_loss += output.loss.item()
            global_step += 1

            if global_step % 100 == 0:
                history["loss"].append(output.loss.item())
                history["reconstruction"].append(output.reconstruction_loss.item())

            pbar.set_postfix({"loss": f"{output.loss.item():.4f}"})

        print(f"  Epoch {epoch+1} avg loss: {epoch_loss / len(loader):.4f}")

    print(f"\nTraining complete!")
    return history


# =============================================================================
# Feature Analysis (Input-centric + Output-centric methods from lit review)
# =============================================================================

def build_context_string(global_token_idx: int, token_ids: torch.Tensor, tokenizer, window: int = 5) -> str:
    """Build a ±window token context string with [TOKEN] marker around the trigger token."""
    n = len(token_ids)
    start = max(0, global_token_idx - window)
    end = min(n, global_token_idx + window + 1)

    parts = []
    for pos in range(start, end):
        tok_str = tokenizer.decode([token_ids[pos].item()])
        if pos == global_token_idx:
            parts.append(f"[{tok_str}]")
        else:
            parts.append(tok_str)

    return "".join(parts)


def analyze_features(
    sae: SparseAutoencoder,
    model: HookedTransformer,
    activation_data: ActivationData,
    top_k: int = 20,
    device: Optional[str] = None,
) -> List[FeatureInfo]:
    """
    Analyze learned features using combined input/output-centric methods.
    
    Input-centric (MaxAct): Find the actual input tokens that maximally activate 
    each feature. This tells us what concepts/patterns the feature detects.
    
    Output-centric (VocabProj): Project decoder vectors onto unembedding matrix
    to see what tokens the feature promotes in the output. This tells us
    what effect the feature has on model predictions.
    
    The combination captures both WHAT TRIGGERS features AND THEIR EFFECT on outputs.
    """
    device = device or get_device()
    sae = sae.to(device).eval()
    
    activations = activation_data.activations
    token_ids = activation_data.token_ids
    
    print("\nAnalyzing features...")
    print("  Method 1: MaxAct (input-centric) - what tokens activate each feature")
    print("  Method 2: VocabProj (output-centric) - what tokens each feature promotes")
    
    # Get decoder vectors (feature directions)
    decoder_weights = sae.decoder.weight.detach()  # [d_model, d_hidden]
    
    # =========================================================================
    # Pass 1: Compute basic statistics (batched for memory efficiency)
    # =========================================================================
    print("\n  Pass 1: Computing feature statistics...")
    
    n_tokens = 0
    feature_sum = torch.zeros(sae.d_hidden)
    feature_count = torch.zeros(sae.d_hidden)
    feature_max = torch.full((sae.d_hidden,), float('-inf'))
    
    for i in tqdm(range(0, len(activations), BATCH_SIZE), desc="  Stats"):
        batch = activations[i:i+BATCH_SIZE].to(device)
        with torch.no_grad():
            features = sae.encode(batch).cpu()
        
        feature_sum += features.sum(dim=0)
        feature_count += (features > 0).float().sum(dim=0)
        feature_max = torch.max(feature_max, features.max(dim=0).values)
        n_tokens += features.shape[0]
    
    feature_freq = feature_count / n_tokens
    feature_mean = feature_sum / n_tokens
    
    # =========================================================================
    # Pass 2: MaxAct - Find top activating tokens for TOP features only
    # (Computing for all 6144 features is expensive, so we focus on active ones)
    # =========================================================================
    print("\n  Pass 2: Computing MaxAct for top features...")
    
    # Filter out BOS/EOS token to get semantic features (not positional)
    bos_token_id = model.tokenizer.bos_token_id or model.tokenizer.eos_token_id
    if bos_token_id is None:
        bos_token_id = 50256  # GPT-2's <|endoftext|> token
    print(f"    (Excluding BOS/EOS token {bos_token_id} from MaxAct)")
    
    # Create mask for content tokens (non-BOS positions)
    content_mask = token_ids != bos_token_id
    
    # Analyze ALL features (sorted by frequency for output ordering)
    num_features_to_analyze = sae.d_hidden
    top_feature_indices = feature_freq.topk(num_features_to_analyze).indices.tolist()
    top_feature_set = set(top_feature_indices)
    print(f"    Analyzing all {num_features_to_analyze} features...")
    
    # Track top-k activations per feature — vectorized across ALL features at once
    topk_per_feature = top_k
    n_features = len(top_feature_indices)
    feat_idx_tensor = torch.tensor(top_feature_indices, dtype=torch.long)

    # running_vals[k, f] = k-th largest activation seen so far for feature f
    running_vals = torch.full((topk_per_feature, n_features), float('-inf'))
    running_idxs = torch.zeros(topk_per_feature, n_features, dtype=torch.long)

    for i in tqdm(range(0, len(activations), BATCH_SIZE), desc="  MaxAct"):
        batch = activations[i:i+BATCH_SIZE].to(device)
        batch_start_idx = i
        batch_size_actual = batch.shape[0]

        # Get mask for this batch (exclude BOS tokens)
        batch_mask = content_mask[batch_start_idx:batch_start_idx + batch_size_actual]  # [B]

        with torch.no_grad():
            features_all = sae.encode(batch).cpu()  # [B, d_hidden]

        # Extract only the features we care about: [B, n_features]
        batch_feats = features_all[:, feat_idx_tensor]

        # Mask BOS positions
        batch_feats[~batch_mask] = float('-inf')

        # Global indices for this batch: [B]
        global_idxs = torch.arange(batch_start_idx, batch_start_idx + batch_size_actual)

        # Combine with running top-k and keep top-k
        # combined_vals: [topk + B, n_features], combined_idxs: [topk + B, n_features]
        combined_vals = torch.cat([running_vals, batch_feats], dim=0)              # [topk+B, n_feat]
        combined_idxs_global = torch.cat([
            running_idxs,
            global_idxs.unsqueeze(1).expand(-1, n_features)                        # [B, n_feat]
        ], dim=0)

        # topk over the combined dim=0
        new_vals, new_positions = combined_vals.topk(topk_per_feature, dim=0)     # [topk, n_feat]
        running_vals = new_vals
        running_idxs = combined_idxs_global.gather(0, new_positions)

    # Convert back to per-feature dicts expected by downstream code
    max_act_values = {}
    max_act_indices = {}
    for f_pos, feat_idx in enumerate(top_feature_indices):
        max_act_values[feat_idx] = running_vals[:, f_pos]
        max_act_indices[feat_idx] = running_idxs[:, f_pos]
    
    # =========================================================================
    # Output-centric: VocabProj - Project decoder vectors onto vocabulary
    # =========================================================================
    print("  Computing VocabProj (output-centric)...")
    unembed = model.W_U.detach()  # [d_model, vocab_size]
    vocab_logits = decoder_weights.T @ unembed  # [d_hidden, vocab_size]
    top_vocab_values, top_vocab_indices = vocab_logits.topk(top_k, dim=-1)
    
    tokenizer = model.tokenizer
    
    # =========================================================================
    # Build FeatureInfo for top features
    # =========================================================================
    print("\n  Building feature analysis...")
    
    features_info = []
    for feat_idx in tqdm(top_feature_indices, desc="  Features"):
        # Sort MaxAct results by activation value (descending)
        sorted_order = max_act_values[feat_idx].argsort(descending=True)
        
        # Build MaxAct examples (input-centric: what tokens trigger this feature)
        max_act_examples = []
        seen_tokens = set()
        for sort_pos in sorted_order:
            act_val = max_act_values[feat_idx][sort_pos].item()
            if act_val <= 0:
                continue

            token_global_idx = max_act_indices[feat_idx][sort_pos].item()
            tok_id = token_ids[token_global_idx].item()
            tok_str = tokenizer.decode([tok_id])

            # Deduplicate (same token can appear in multiple positions)
            if tok_str not in seen_tokens:
                seen_tokens.add(tok_str)
                ctx = build_context_string(token_global_idx, token_ids, tokenizer)
                max_act_examples.append(MaxActExample(
                    token=tok_str,
                    token_id=tok_id,
                    activation=act_val,
                    context=ctx,
                    global_token_idx=token_global_idx,
                ))
        
        # VocabProj: tokens this feature promotes (output-centric)
        vocab_tokens = []
        vocab_logit_values = []
        for k in range(min(top_k, 10)):
            tok_id = top_vocab_indices[feat_idx, k].item()
            tok_str = tokenizer.decode([tok_id]).strip()
            logit_val = top_vocab_values[feat_idx, k].item()
            vocab_tokens.append(tok_str)
            vocab_logit_values.append(logit_val)
        
        info = FeatureInfo(
            index=feat_idx,
            activation_frequency=feature_freq[feat_idx].item(),
            mean_activation=feature_mean[feat_idx].item(),
            max_activation=feature_max[feat_idx].item(),
            max_activating_tokens=max_act_examples[:10],
            vocab_projection=vocab_tokens,
            vocab_projection_logits=vocab_logit_values,
        )
        features_info.append(info)
    
    return features_info


def save_results(
    sae: SparseAutoencoder,
    activation_data: ActivationData,
    features_info: List[FeatureInfo],
    history: Dict[str, List[float]],
    output_dir: Path = OUTPUT_DIR,
):
    """Save all results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    activations = activation_data.activations
    token_ids = activation_data.token_ids
    
    print(f"\nSaving results to {output_dir}...")
    
    # Save SAE
    sae.save(output_dir / "sae.pt")
    
    # Save activations and token IDs
    torch.save(activations, output_dir / "activations.pt")
    torch.save(token_ids, output_dir / "token_ids.pt")
    print(f"  Saved activations: {activations.shape}")
    print(f"  Saved token_ids: {token_ids.shape}")
    
    # Save decoder vectors
    decoder_vectors = sae.decoder.weight.detach().cpu()
    torch.save(decoder_vectors, output_dir / "decoder_vectors.pt")
    print(f"  Saved decoder vectors: {decoder_vectors.shape}")
    
    # Compute summary stats in batches (avoid huge tensor in memory)
    device = get_device()
    sae = sae.to(device).eval()
    
    n_tokens = 0
    feature_count = torch.zeros(sae.d_hidden)
    total_l0 = 0.0
    
    print("  Computing summary statistics (batched)...")
    for i in tqdm(range(0, len(activations), BATCH_SIZE), desc="  Stats"):
        batch = activations[i:i+BATCH_SIZE].to(device)
        with torch.no_grad():
            features = sae.encode(batch).cpu()
        
        feature_count += (features > 0).float().sum(dim=0)
        total_l0 += (features > 0).float().sum(dim=-1).sum().item()
        n_tokens += features.shape[0]
    
    feature_freq = feature_count / n_tokens
    dead_features = (feature_freq < 1e-5).sum().item()
    avg_l0 = total_l0 / n_tokens
    
    # Save feature analysis with both MaxAct and VocabProj
    features_data = [
        {
            "index": f.index,
            "frequency": f.activation_frequency,
            "mean": f.mean_activation,
            "max": f.max_activation,
            # Input-centric: what tokens activate this feature
            "max_activating_tokens": [
                {
                    "token": ex.token,
                    "token_id": ex.token_id,
                    "activation": ex.activation,
                    "context": ex.context,
                    "global_token_idx": ex.global_token_idx,
                }
                for ex in f.max_activating_tokens
            ],
            # Output-centric: what tokens this feature promotes
            "vocab_projection": f.vocab_projection,
            "vocab_projection_logits": f.vocab_projection_logits,
        }
        for f in features_info
    ]
    with open(output_dir / "features.json", "w") as f:
        json.dump(features_data, f, indent=2)
    print(f"  Saved feature analysis: {len(features_info)} features")
    print(f"    - MaxAct (input-centric): tokens that activate each feature")
    print(f"    - VocabProj (output-centric): tokens each feature promotes")
    
    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Saved training history")
    
    summary = {
        "model": MODEL_NAME,
        "layer": TARGET_LAYER,
        "hook_type": HOOK_TYPE,
        "d_model": D_MODEL,
        "d_hidden": sae.d_hidden,
        "n_tokens": activations.shape[0],
        "dead_features": dead_features,
        "avg_l0_sparsity": avg_l0,
        "final_loss": history["loss"][-1] if history["loss"] else None,
        "analysis_methods": ["MaxAct (input-centric)", "VocabProj (output-centric)"],
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"  Results Summary")
    print(f"{'='*60}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Layer: {TARGET_LAYER} ({HOOK_TYPE})")
    print(f"  Tokens: {activations.shape[0]:,}")
    print(f"  Features: {sae.d_hidden:,}")
    print(f"  Dead features: {dead_features}")
    print(f"  Avg L0 sparsity: {avg_l0:.1f}")
    print(f"{'='*60}")


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SAE Feature Extraction for GPT-2")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument("--skip-collection", action="store_true", help="Use cached activations")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detected)")
    args = parser.parse_args()
    
    device = args.device or get_device()
    
    # Quick test settings
    num_samples = 500 if args.quick else NUM_SAMPLES
    num_epochs = 1 if args.quick else NUM_EPOCHS
    
    print("\n" + "="*60)
    print("  SAE Feature Extraction Pipeline")
    print(f"  Model: {MODEL_NAME} | Layer: {TARGET_LAYER} | Hook: {HOOK_TYPE}")
    print("="*60)
    
    if args.quick:
        print("  [QUICK TEST MODE]")
    
    # Step 1: Load model
    model = load_gpt2(device)
    
    # Step 2: Collect or load activations (with token IDs for MaxAct analysis)
    activations_path = OUTPUT_DIR / "activations.pt"
    token_ids_path = OUTPUT_DIR / "token_ids.pt"
    
    if args.skip_collection and activations_path.exists() and token_ids_path.exists():
        print(f"\nLoading cached activations from {OUTPUT_DIR}...")
        activations = torch.load(activations_path)
        token_ids = torch.load(token_ids_path)
        activation_data = ActivationData(activations=activations, token_ids=token_ids)
        print(f"  Loaded {activations.shape[0]:,} activations with token IDs")
    elif args.skip_collection and activations_path.exists():
        print(f"\nWARNING: Found activations but no token_ids. Re-collecting for MaxAct analysis...")
        activation_data = collect_activations(model, num_samples=num_samples)
    else:
        activation_data = collect_activations(model, num_samples=num_samples)
    
    # Free model memory for training
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Step 3: Train SAE (uses only activations tensor)
    sae = SparseAutoencoder()
    history = train_sae(sae, activation_data.activations, num_epochs=num_epochs, device=device)
    
    # Step 4: Analyze features (uses both activations and token_ids)
    model = load_gpt2(device)  # Reload for analysis
    print("\n  Feature analysis using dual methods:")
    print("    - MaxAct: Find tokens that maximally activate each feature")
    print("    - VocabProj: Find tokens each feature promotes in output")
    features_info = analyze_features(sae, model, activation_data, device=device)
    
    # Step 5: Save results
    save_results(sae, activation_data, features_info, history)


if __name__ == "__main__":
    main()
