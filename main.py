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
    chunk_size: int = 10_000,  # ~5.2 GB per chunk at 128 tokens × 1024 dims
) -> tuple:
    """
    Collect residual stream activations from GPT-2 layer 12.

    Saves activation chunks to disk to stay within RAM limits.
    Token IDs (~100 MB total) are kept in memory for MaxAct context building.

    Returns:
        token_ids   : torch.Tensor [n_tokens]  — all token IDs in RAM
        chunk_files : List[Path]               — activation chunks on disk
    """
    import gc

    hook_point = f"blocks.{TARGET_LAYER}.hook_{HOOK_TYPE}"
    chunks_dir = OUTPUT_DIR / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCollecting activations from {hook_point}...")
    print(f"  Samples: {num_samples}, Batch size: {batch_size}, Chunk size: {chunk_size}")

    dataset = load_dataset("openwebtext", split="train", streaming=True)

    all_activations = []   # accumulate within current chunk
    all_token_ids = []     # all token IDs (kept in RAM — small)
    batch = []
    count = 0
    chunk_idx = 0
    chunk_files = []

    pbar = tqdm(total=num_samples, desc="Collecting")

    for item in dataset:
        text = item.get("text", "")
        if not text.strip():
            continue

        batch.append(text[:max_tokens * 4])
        count += 1
        pbar.update(1)

        if len(batch) >= batch_size:
            tokens = model.to_tokens(batch)
            if tokens.shape[1] > max_tokens:
                tokens = tokens[:, :max_tokens]

            _, cache = model.run_with_cache(tokens, names_filter=[hook_point])
            acts = cache[hook_point].reshape(-1, D_MODEL).cpu()
            all_activations.append(acts)
            all_token_ids.append(tokens.reshape(-1).cpu())

            del cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch = []

        # Flush chunk to disk — never merge into one giant tensor
        if count % chunk_size == 0 and count > 0 and all_activations:
            chunk_acts = torch.cat(all_activations, dim=0)
            chunk_path = chunks_dir / f"chunk_{chunk_idx}.pt"
            torch.save(chunk_acts, chunk_path)
            chunk_files.append(chunk_path)
            print(f"\n  Saved chunk {chunk_idx} ({chunk_acts.shape[0]:,} tokens) → {chunk_path}")

            del all_activations, chunk_acts
            all_activations = []
            gc.collect()
            chunk_idx += 1

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
        del cache

    pbar.close()

    # Save final (possibly partial) chunk
    if all_activations:
        chunk_acts = torch.cat(all_activations, dim=0)
        chunk_path = chunks_dir / f"chunk_{chunk_idx}.pt"
        torch.save(chunk_acts, chunk_path)
        chunk_files.append(chunk_path)
        print(f"\n  Saved chunk {chunk_idx} ({chunk_acts.shape[0]:,} tokens) → {chunk_path}")
        del all_activations, chunk_acts
        gc.collect()

    token_ids = torch.cat(all_token_ids, dim=0)
    n_tokens = sum(torch.load(p).shape[0] for p in chunk_files)
    print(f"  Collected {n_tokens:,} token activations across {len(chunk_files)} chunks")
    print(f"  Token IDs in RAM: {token_ids.shape[0]:,} ({token_ids.nbytes / 1e6:.1f} MB)")
    return token_ids, chunk_files


# =============================================================================
# Training
# =============================================================================


def train_sae(
    sae: SparseAutoencoder,
    chunk_files: List[Path],
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    device: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    Train the TopK SAE on chunked activations stored on disk.

    Loads one chunk at a time — peak RAM = one chunk + SAE weights.
    Chunks are shuffled each epoch for better training diversity.
    """
    import gc
    import random

    device = device or get_device()
    sae = sae.to(device)

    # Estimate total steps for scheduler (use first chunk size as proxy)
    first_chunk = torch.load(chunk_files[0])
    steps_per_chunk = (first_chunk.shape[0] + batch_size - 1) // batch_size
    del first_chunk
    total_steps = steps_per_chunk * len(chunk_files) * num_epochs

    print(f"\nTraining SAE...")
    print(f"  Architecture: {sae.d_model} -> {sae.d_hidden} -> {sae.d_model}")
    print(f"  TopK: K={sae.k} (L0={sae.k} exactly, guaranteed)")
    print(f"  Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    print(f"  Chunks: {len(chunk_files)} (loaded one at a time)")

    optimizer = Adam(sae.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=learning_rate * 0.1)

    history = {"loss": [], "reconstruction": []}
    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        # Shuffle chunk order each epoch
        shuffled_chunks = chunk_files.copy()
        random.shuffle(shuffled_chunks)

        for chunk_path in shuffled_chunks:
            activations = torch.load(chunk_path)
            dataset = torch.utils.data.TensorDataset(activations)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} [{chunk_path.name}]")
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
                epoch_steps += 1
                global_step += 1

                if global_step % 100 == 0:
                    history["loss"].append(output.loss.item())
                    history["reconstruction"].append(output.reconstruction_loss.item())

                pbar.set_postfix({"loss": f"{output.loss.item():.4f}"})

            del activations, dataset, loader
            gc.collect()

        print(f"  Epoch {epoch+1} avg loss: {epoch_loss / epoch_steps:.4f}")

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
    token_ids: torch.Tensor,
    chunk_files: List[Path],
    top_k: int = 20,
    max_tokens: int = 128,
    device: Optional[str] = None,
) -> List[FeatureInfo]:
    """
    Analyze learned features using combined input/output-centric methods.

    Processes activation chunks from disk one at a time — no full tensor in RAM.
    Token IDs stay in RAM (small) for context string building.
    """
    import gc

    device = device or get_device()
    sae = sae.to(device).eval()

    print("\nAnalyzing features...")
    print("  Method 1: MaxAct (input-centric) - what tokens activate each feature")
    print("  Method 2: VocabProj (output-centric) - what tokens each feature promotes")

    decoder_weights = sae.decoder.weight.detach()  # [d_model, d_hidden]

    # =========================================================================
    # Pass 1: Compute basic statistics — one chunk at a time
    # =========================================================================
    print("\n  Pass 1: Computing feature statistics...")

    n_tokens = 0
    feature_sum = torch.zeros(sae.d_hidden)
    feature_count = torch.zeros(sae.d_hidden)
    feature_max = torch.full((sae.d_hidden,), float('-inf'))

    for chunk_path in tqdm(chunk_files, desc="  Stats (chunks)"):
        chunk = torch.load(chunk_path)
        for i in range(0, len(chunk), BATCH_SIZE):
            batch = chunk[i:i+BATCH_SIZE].to(device)
            with torch.no_grad():
                features = sae.encode(batch).cpu()
            feature_sum += features.sum(dim=0)
            feature_count += (features > 0).float().sum(dim=0)
            feature_max = torch.max(feature_max, features.max(dim=0).values)
            n_tokens += features.shape[0]
        del chunk
        gc.collect()

    feature_freq = feature_count / n_tokens
    feature_mean = feature_sum / n_tokens

    # =========================================================================
    # Pass 2: MaxAct — running top-k across all chunks
    # =========================================================================
    print("\n  Pass 2: Computing MaxAct for top features...")

    bos_token_id = model.tokenizer.bos_token_id or model.tokenizer.eos_token_id
    if bos_token_id is None:
        bos_token_id = 50256
    print(f"    (Excluding BOS/EOS token {bos_token_id} from MaxAct)")

    content_mask = token_ids != bos_token_id

    num_features_to_analyze = sae.d_hidden
    top_feature_indices = feature_freq.topk(num_features_to_analyze).indices.tolist()
    print(f"    Analyzing all {num_features_to_analyze} features...")

    topk_per_feature = top_k
    n_features = len(top_feature_indices)
    feat_idx_tensor = torch.tensor(top_feature_indices, dtype=torch.long)

    running_vals = torch.full((topk_per_feature, n_features), float('-inf'))
    running_idxs = torch.zeros(topk_per_feature, n_features, dtype=torch.long)

    global_offset = 0  # track absolute token index across chunks
    for chunk_path in tqdm(chunk_files, desc="  MaxAct (chunks)"):
        chunk = torch.load(chunk_path)

        for i in range(0, len(chunk), BATCH_SIZE):
            batch = chunk[i:i+BATCH_SIZE].to(device)
            batch_start_idx = global_offset + i
            batch_size_actual = batch.shape[0]

            batch_mask = content_mask[batch_start_idx:batch_start_idx + batch_size_actual]

            with torch.no_grad():
                features_all = sae.encode(batch).cpu()

            batch_feats = features_all[:, feat_idx_tensor]
            batch_feats[~batch_mask] = float('-inf')

            global_idxs = torch.arange(batch_start_idx, batch_start_idx + batch_size_actual)

            combined_vals = torch.cat([running_vals, batch_feats], dim=0)
            combined_idxs_global = torch.cat([
                running_idxs,
                global_idxs.unsqueeze(1).expand(-1, n_features)
            ], dim=0)

            new_vals, new_positions = combined_vals.topk(topk_per_feature, dim=0)
            running_vals = new_vals
            running_idxs = combined_idxs_global.gather(0, new_positions)

        global_offset += len(chunk)
        del chunk
        gc.collect()

    max_act_values = {}
    max_act_indices = {}
    max_act_positions = {}  # (mean, std) of token position-in-sequence
    for f_pos, feat_idx in enumerate(top_feature_indices):
        max_act_values[feat_idx] = running_vals[:, f_pos]
        max_act_indices[feat_idx] = running_idxs[:, f_pos]
        # Position within each sequence: global_idx % max_tokens
        # (valid since all sequences are truncated/padded to exactly max_tokens)
        valid_mask = running_vals[:, f_pos] > 0
        if valid_mask.any():
            positions = (running_idxs[:, f_pos][valid_mask].float() % max_tokens)
            pos_mean = positions.mean().item()
            pos_std = positions.std().item() if valid_mask.sum() > 1 else 0.0
        else:
            pos_mean, pos_std = 0.0, 0.0
        max_act_positions[feat_idx] = (pos_mean, pos_std)
    
    # =========================================================================
    # Output-centric: VocabProj - Project decoder vectors onto vocabulary
    # =========================================================================
    print("  Computing VocabProj (output-centric)...")
    unembed = model.W_U.detach()  # [d_model, vocab_size]
    vocab_logits = decoder_weights.T @ unembed  # [d_hidden, vocab_size]
    # Subtract per-feature mean so we see what each feature promotes ABOVE its own
    # baseline. Without this, raw logits are dominated by common subword tokens the
    # model predicts everywhere (e.g. 'etts', 'arted'), masking the actual signal.
    vocab_logits = vocab_logits - vocab_logits.mean(dim=-1, keepdim=True)
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
        
        pos_mean, pos_std = max_act_positions.get(feat_idx, (0.0, 0.0))
        info = FeatureInfo(
            index=feat_idx,
            activation_frequency=feature_freq[feat_idx].item(),
            mean_activation=feature_mean[feat_idx].item(),
            max_activation=feature_max[feat_idx].item(),
            max_activating_tokens=max_act_examples[:10],
            vocab_projection=vocab_tokens,
            vocab_projection_logits=vocab_logit_values,
            position_mean=pos_mean,
            position_std=pos_std,
        )
        features_info.append(info)
    
    return features_info


def save_results(
    sae: SparseAutoencoder,
    token_ids: torch.Tensor,
    chunk_files: List[Path],
    features_info: List[FeatureInfo],
    history: Dict[str, List[float]],
    output_dir: Path = OUTPUT_DIR,
):
    """Save all results to disk. Processes chunks one at a time — no OOM."""
    import gc

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to {output_dir}...")

    sae.save(output_dir / "sae.pt")

    # Token IDs are small — save directly
    torch.save(token_ids, output_dir / "token_ids.pt")
    print(f"  Saved token_ids: {token_ids.shape}")

    # Chunk file manifest (so --skip-collection can find them)
    chunk_manifest = [str(p) for p in chunk_files]
    with open(output_dir / "activations.json", "w") as f:
        json.dump(chunk_manifest, f)
    print(f"  Saved chunk manifest: {len(chunk_files)} chunks")

    decoder_vectors = sae.decoder.weight.detach().cpu()
    torch.save(decoder_vectors, output_dir / "decoder_vectors.pt")
    print(f"  Saved decoder vectors: {decoder_vectors.shape}")

    # Summary stats — one chunk at a time
    device = get_device()
    sae = sae.to(device).eval()

    n_tokens = 0
    feature_count = torch.zeros(sae.d_hidden)
    total_l0 = 0.0

    print("  Computing summary statistics (chunked)...")
    for chunk_path in tqdm(chunk_files, desc="  Stats"):
        chunk = torch.load(chunk_path)
        for i in range(0, len(chunk), BATCH_SIZE):
            batch = chunk[i:i+BATCH_SIZE].to(device)
            with torch.no_grad():
                features = sae.encode(batch).cpu()
            feature_count += (features > 0).float().sum(dim=0)
            total_l0 += (features > 0).float().sum(dim=-1).sum().item()
            n_tokens += features.shape[0]
        del chunk
        gc.collect()

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
            # Positional: where in the sequence this feature tends to fire
            "position_mean": round(f.position_mean, 2),
            "position_std": round(f.position_std, 2),
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
        "n_tokens": n_tokens,
        "n_chunks": len(chunk_files),
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
    print(f"  Tokens: {n_tokens:,} ({len(chunk_files)} chunks on disk)")
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
    parser.add_argument("--skip-collection", action="store_true", help="Use cached chunks")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detected)")
    args = parser.parse_args()

    device = args.device or get_device()

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

    # Step 2: Collect activations as on-disk chunks (never load all into RAM)
    manifest_path = OUTPUT_DIR / "activations.json"
    token_ids_path = OUTPUT_DIR / "token_ids.pt"

    if args.skip_collection and manifest_path.exists() and token_ids_path.exists():
        print(f"\nLoading cached chunks from {OUTPUT_DIR}...")
        with open(manifest_path) as f:
            chunk_files = [Path(p) for p in json.load(f)]
        token_ids = torch.load(token_ids_path)
        print(f"  Found {len(chunk_files)} chunk files, {token_ids.shape[0]:,} token IDs")
    else:
        token_ids, chunk_files = collect_activations(model, num_samples=num_samples)

    # Free GPT-2 memory before training
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 3: Train SAE — loads one chunk at a time
    sae = SparseAutoencoder()
    history = train_sae(sae, chunk_files, num_epochs=num_epochs, device=device)

    # Step 4: Analyze features — loads one chunk at a time
    model = load_gpt2(device)
    print("\n  Feature analysis using dual methods:")
    print("    - MaxAct: Find tokens that maximally activate each feature")
    print("    - VocabProj: Find tokens each feature promotes in output")
    features_info = analyze_features(sae, model, token_ids, chunk_files, device=device)

    # Step 5: Save results
    save_results(sae, token_ids, chunk_files, features_info, history)


if __name__ == "__main__":
    main()
