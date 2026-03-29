#!/usr/bin/env python3
"""
Medical SAE pipeline for Llama 3.2 1B.

Usage:
    python main.py                          # Full pipeline, layer 8 (default)
    python main.py --layers 4 8 12         # Train SAEs on layers 4, 8, and 12
    python main.py --quick                  # Quick test (500 samples, 1 epoch)
    python main.py --skip-collection        # Reuse cached activations
    python main.py --device mps             # Force device

Output layout:
    Single layer (default)  → medical_outputs/
    Multi-layer             → medical_outputs/layer_N/  per layer
                              medical_outputs/cross_layer_analysis.json
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm
import argparse
import json
from collections import Counter

from transformer_lens import HookedTransformer

# Import all config and shared classes
from config import (
    MODEL_NAME, D_MODEL, TARGET_LAYER, HOOK_TYPE,
    LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, NUM_SAMPLES,
    MEDICAL_OUTPUT_DIR,
    get_device,
    SparseAutoencoder, MaxActExample, FeatureInfo,
)
from dataset import stream_medical_texts


# =============================================================================
# Model Loading & Activation Collection
# =============================================================================


def load_model(device: Optional[str] = None) -> HookedTransformer:
    """Load Llama 3.2 1B into TransformerLens."""
    device = device or get_device()
    print(f"\nLoading {MODEL_NAME} on {device}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.float32)
    print(f"  d_model: {model.cfg.d_model}, n_layers: {model.cfg.n_layers}")
    return model


# Llama 3.2 special token IDs — excluded from MaxAct analysis.
# These structural tokens fire based on format, not semantics, and contaminate labeling.
# 128000=<|begin_of_text|>  128001=<|end_of_text|>  128006=<|start_header_id|>
# 128007=<|end_header_id|>  128008=<|eot_id|> (some configs)  128009=<|eot_id|>
_LLAMA_SPECIAL_IDS = {128000, 128001, 128006, 128007, 128008, 128009}




@torch.no_grad()
def collect_activations(
    model: HookedTransformer,
    num_samples: int = NUM_SAMPLES,
    batch_size: int = 32,
    max_tokens: int = 128,
    chunk_size: int = 10_000,
    output_dir: Path = MEDICAL_OUTPUT_DIR,
    layer: int = TARGET_LAYER,
) -> tuple:
    """
    Collect residual stream activations from the given layer using medical datasets.

    Saves activation chunks to disk to stay within RAM limits.
    Token IDs and source IDs are kept in memory (small) for MaxAct context building.

    Returns:
        token_ids   : torch.Tensor [n_tokens]  — all token IDs in RAM
        chunk_files : List[Path]               — activation chunk paths on disk
        source_ids  : torch.Tensor [n_tokens]  — index into source_list per token
        source_list : List[str]                — source ID strings
    """
    import gc

    hook_point = f"blocks.{layer}.hook_{HOOK_TYPE}"
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCollecting activations from {hook_point} (layer {layer})...")
    print(f"  Dataset: medmcqa + pubmed_qa + pubmed_abs (instruction-formatted) | Samples: {num_samples}")

    all_activations = []
    all_token_ids = []
    all_source_idxs = []   # int index per token → source_list
    source_list = []
    source_name_to_idx: Dict[str, int] = {}

    batch_texts: List[str] = []
    batch_sources: List[str] = []
    count = 0
    chunk_idx = 0
    chunk_files = []

    pbar = tqdm(total=num_samples, desc="Collecting")

    for text, source_id in stream_medical_texts(num_samples, max_tokens):
        batch_texts.append(text)
        batch_sources.append(source_id)
        count += 1
        pbar.update(1)

        if len(batch_texts) >= batch_size:
            tokens = model.to_tokens(batch_texts)
            if tokens.shape[1] > max_tokens:
                tokens = tokens[:, :max_tokens]

            _, cache = model.run_with_cache(tokens, names_filter=[hook_point])
            acts_batch = cache[hook_point].cpu()  # [batch, seq_len, d_model]

            # Filter out padding positions per sequence (Llama uses <|eot_id|> as pad)
            pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
            for i, src_id in enumerate(batch_sources):
                real_mask = tokens[i].cpu() != pad_id
                real_toks = tokens[i].cpu()[real_mask]
                real_acts = acts_batch[i][real_mask]
                if real_toks.shape[0] == 0:
                    continue
                all_activations.append(real_acts)
                all_token_ids.append(real_toks)
                if src_id not in source_name_to_idx:
                    source_name_to_idx[src_id] = len(source_list)
                    source_list.append(src_id)
                all_source_idxs.extend([source_name_to_idx[src_id]] * real_toks.shape[0])

            del cache, acts_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch_texts = []
            batch_sources = []

        # Flush chunk to disk
        if count % chunk_size == 0 and count > 0 and all_activations:
            chunk_acts = torch.cat(all_activations, dim=0).half()  # float16 to save disk space
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
    if batch_texts:
        tokens = model.to_tokens(batch_texts)
        if tokens.shape[1] > max_tokens:
            tokens = tokens[:, :max_tokens]
        _, cache = model.run_with_cache(tokens, names_filter=[hook_point])
        acts_batch = cache[hook_point].cpu()
        pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
        for i, src_id in enumerate(batch_sources):
            real_mask = tokens[i].cpu() != pad_id
            real_toks = tokens[i].cpu()[real_mask]
            real_acts = acts_batch[i][real_mask]
            if real_toks.shape[0] == 0:
                continue
            all_activations.append(real_acts)
            all_token_ids.append(real_toks)
            if src_id not in source_name_to_idx:
                source_name_to_idx[src_id] = len(source_list)
                source_list.append(src_id)
            all_source_idxs.extend([source_name_to_idx[src_id]] * real_toks.shape[0])
        del cache, acts_batch

    pbar.close()

    # Save final chunk
    if all_activations:
        chunk_acts = torch.cat(all_activations, dim=0).half()  # float16 to save disk space
        chunk_path = chunks_dir / f"chunk_{chunk_idx}.pt"
        torch.save(chunk_acts, chunk_path)
        chunk_files.append(chunk_path)
        print(f"\n  Saved chunk {chunk_idx} ({chunk_acts.shape[0]:,} tokens) → {chunk_path}")
        del all_activations, chunk_acts
        gc.collect()

    token_ids = torch.cat(all_token_ids, dim=0)
    source_ids = torch.tensor(all_source_idxs, dtype=torch.long)

    # Trim to match actual token count (source_idxs may be slightly longer due to padding)
    source_ids = source_ids[:token_ids.shape[0]]

    print(f"  Collected {token_ids.shape[0]:,} tokens across {len(chunk_files)} chunks | Sources: {len(source_list)} unique")
    return token_ids, chunk_files, source_ids, source_list


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
    first_chunk = torch.load(chunk_files[0])  # No .float() needed, just checking shape
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

    history = {"loss": [], "reconstruction": [], "dead_features_per_epoch": [], "epoch_losses": []}
    global_step = 0

    # EMA of per-feature firing rate — tracks which features are alive over time.
    # alpha=0.99: slow decay so the estimate reflects hundreds of batches, not just the last one.
    # A feature is "dead" when its EMA drops below 1e-5 (fires < 0.001% of tokens on average).
    EMA_ALPHA = 0.99
    ema_freq = torch.zeros(sae.d_hidden, device=device)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        # Shuffle chunk order each epoch
        shuffled_chunks = chunk_files.copy()
        random.shuffle(shuffled_chunks)

        for chunk_path in shuffled_chunks:
            activations = torch.load(chunk_path).float()
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

                # Update EMA firing rate: fraction of tokens in this batch that fired each feature
                with torch.no_grad():
                    batch_freq = (output.hidden > 0).float().mean(dim=0)
                    ema_freq = EMA_ALPHA * ema_freq + (1 - EMA_ALPHA) * batch_freq

                epoch_loss += output.loss.item()
                epoch_steps += 1
                global_step += 1

                if global_step % 100 == 0:
                    history["loss"].append(output.loss.item())
                    history["reconstruction"].append(output.reconstruction_loss.item())

                pbar.set_postfix({"loss": f"{output.loss.item():.4f}"})

            del activations, dataset, loader
            gc.collect()

        # End-of-epoch metrics
        epoch_dead = int((ema_freq < 1e-5).sum().item())
        history["dead_features_per_epoch"].append(epoch_dead)
        history["epoch_losses"].append(epoch_loss / epoch_steps)
        print(f"  Epoch {epoch+1} avg loss: {epoch_loss / epoch_steps:.4f}  |  dead features (EMA): {epoch_dead}/{sae.d_hidden}")

    print(f"\nTraining complete!")
    return history


# =============================================================================
# Feature Analysis (Input-centric + Output-centric methods from lit review)
# =============================================================================

def token_entropy(max_act_examples: List[MaxActExample]) -> float:
    """
    Shannon entropy (bits) of the MaxAct token distribution for one feature.

    Low entropy  → monosemantic: feature fires on a tight cluster of similar tokens.
    High entropy → polysemantic: feature fires broadly across unrelated tokens.

    Tokens are lowercased and stripped so surface variants ("The"/"the") don't
    artificially inflate diversity.  Returns 0.0 for features with ≤1 example.
    """
    import math

    tokens = [ex.token.strip().lower() for ex in max_act_examples if ex.token.strip()]
    if len(tokens) <= 1:
        return 0.0
    counts = Counter(tokens)
    total = sum(counts.values())
    return round(-sum((c / total) * math.log2(c / total) for c in counts.values()), 4)


def source_breakdown(max_act_examples: List[MaxActExample], all_sources: List[str]) -> Dict[str, float]:
    """
    Fraction of a feature's MaxAct examples that come from each source dataset.

    source_id strings are in the form "dataset:doc_index" (e.g. "medmcqa:54").
    This function aggregates by the dataset prefix (the part before the colon)
    so the result is always a 3-key dict over {medmcqa, pubmed_qa, pubmed_abs}.

    Returns a sparse dict — only datasets that actually appear are included.
    An empty dict means source tracking was unavailable.
    """
    # Aggregate by dataset prefix, ignoring the per-document index
    counts: Dict[str, int] = Counter()
    for ex in max_act_examples:
        if ex.source_id:
            dataset = ex.source_id.split(":")[0]
            counts[dataset] += 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {dataset: round(count / total, 4) for dataset, count in sorted(counts.items())}


def build_context_string(global_token_idx: int, token_ids: torch.Tensor, tokenizer, window: int = 50) -> str:
    """
    Build a ±window token context string with [TOKEN] marker around the trigger token.

    Respects document boundaries: the window is clipped at the nearest
    <|begin_of_text|> token (ID 128000) on either side so contexts never
    bleed across unrelated documents.
    """
    n = len(token_ids)
    bos_id = 128000  # <|begin_of_text|> for Llama 3.2

    # Scan left for the nearest document boundary
    start = max(0, global_token_idx - window)
    for pos in range(global_token_idx - 1, start - 1, -1):
        if token_ids[pos].item() == bos_id:
            start = pos + 1  # start after the BOS, not at it
            break

    # Scan right for the nearest document boundary
    end = min(n, global_token_idx + window + 1)
    for pos in range(global_token_idx + 1, end):
        if token_ids[pos].item() == bos_id:
            end = pos  # stop before the next BOS
            break

    parts = []
    for pos in range(start, end):
        tok_str = tokenizer.decode([token_ids[pos].item()])
        if pos == global_token_idx:
            parts.append(f"[{tok_str}]")
        else:
            parts.append(tok_str)

    return "".join(parts)


def _context_tokens_for_example(
    global_token_idx: int,
    token_ids: torch.Tensor,
    window: int = 50,
) -> tuple:
    """
    Return (context_token_ids [seq], within_context_position) for a MaxAct example.
    Mirrors build_context_string boundary logic exactly — respects BOS (ID 128000).
    Used by TokenChange to reconstruct the forward-pass context without re-tokenizing.
    """
    n = len(token_ids)
    bos_id = 128000

    start = max(0, global_token_idx - window)
    for pos in range(global_token_idx - 1, start - 1, -1):
        if token_ids[pos].item() == bos_id:
            start = pos + 1
            break

    end = min(n, global_token_idx + window + 1)
    for pos in range(global_token_idx + 1, end):
        if token_ids[pos].item() == bos_id:
            end = pos
            break

    context = token_ids[start:end]
    position = global_token_idx - start
    return context, position


def compute_token_change_for_feature(
    model: HookedTransformer,
    sae: "SparseAutoencoder",
    feat_idx: int,
    max_act_examples: list,
    token_ids: torch.Tensor,
    layer: int,
    device: str,
    mean_activation: float,
    scale_multiplier: float = 5.0,
    top_k: int = 10,
) -> Optional[Dict]:
    """
    Causal output-centric analysis. Inject feat_idx's decoder direction at `layer`
    via a TransformerLens hook and measure how the next-token distribution shifts at
    the active token position — averaged across up to 5 MaxAct contexts.

    Returns None if no valid examples could be processed.

    Why this is better than VocabProj at layer 12:
    - VocabProj: decoder_col @ W_U (single linear step, ignores 4 downstream layers)
    - TokenChange: real forward pass through all remaining layers + RMSNorm + W_U
    Dead features are naturally filtered: they produce low, inconsistent KL across
    contexts, so token_change_kl_std / token_change_kl will be high.
    """
    examples = [e for e in max_act_examples[:5] if e.global_token_idx >= 0]
    if not examples:
        return None

    decoder_dir = sae.decoder.weight[:, feat_idx].detach().to(device)  # [d_model]
    hook_point = f"blocks.{layer}.hook_resid_post"
    scale = max(scale_multiplier * mean_activation, 1.0)

    kl_list: list = []
    deltas: list = []

    for ex in examples:
        context, position = _context_tokens_for_example(ex.global_token_idx, token_ids)
        if position < 0 or position >= len(context):
            continue

        ctx_tensor = context.unsqueeze(0).to(device)  # [1, seq]

        # Closure captures current `position` correctly via default arg
        def inject_fn(activation, hook, _pos=position):
            activation[:, _pos, :] = activation[:, _pos, :] + scale * decoder_dir
            return activation

        try:
            with torch.no_grad():
                baseline_logits = model(ctx_tensor)[0, position, :].float()
                steered_logits = model.run_with_hooks(
                    ctx_tensor, fwd_hooks=[(hook_point, inject_fn)]
                )[0, position, :].float()
        except Exception:
            continue

        bp = F.softmax(baseline_logits, dim=-1)
        sp = F.softmax(steered_logits, dim=-1)
        kl = F.kl_div(bp.log(), sp, reduction="sum").item()
        kl_list.append(kl)
        deltas.append((sp - bp).cpu())

    if not deltas:
        return None

    mean_delta = torch.stack(deltas).mean(0)          # [vocab_size]
    mean_kl = sum(kl_list) / len(kl_list)
    kl_std = float(torch.tensor(kl_list).std()) if len(kl_list) > 1 else 0.0

    top_promoted   = mean_delta.topk(top_k).indices.tolist()
    top_suppressed = mean_delta.topk(top_k, largest=False).indices.tolist()
    tok = model.tokenizer

    return {
        "top_promoted":   [tok.decode([i]) for i in top_promoted],
        "top_suppressed": [tok.decode([i]) for i in top_suppressed],
        "mean_kl":        mean_kl,
        "kl_std":         kl_std,
        "n_contexts":     len(deltas),
    }


def analyze_features(
    sae: SparseAutoencoder,
    model: HookedTransformer,
    token_ids: torch.Tensor,
    chunk_files: List[Path],
    top_k: int = 20,
    max_tokens: int = 128,
    device: Optional[str] = None,
    source_ids: Optional[torch.Tensor] = None,
    source_list: Optional[List[str]] = None,
    layer: int = TARGET_LAYER,
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
        chunk = torch.load(chunk_path).float()  # Convert float16 back to float32
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

    # Exclude all Llama 3.2 structural special tokens from MaxAct.
    # These fire based on chat-template position, not semantics.
    special_ids_set = _LLAMA_SPECIAL_IDS.copy()
    # Also add any additional special tokens reported by the tokenizer
    if model.tokenizer.all_special_ids:
        special_ids_set.update(model.tokenizer.all_special_ids)
    print(f"    (Excluding {len(special_ids_set)} special token IDs from MaxAct)")

    content_mask = torch.ones(len(token_ids), dtype=torch.bool)
    for sid in special_ids_set:
        content_mask &= (token_ids != sid)

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
        chunk = torch.load(chunk_path).float()  # Convert float16 back to float32

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

    # Compute source-start positions: for each source index, what is the global
    # token offset where that source begins? Used for document-relative positions.
    source_start_positions: Dict[int, int] = {}
    if source_ids is not None:
        prev_sid = -1
        for i in range(len(source_ids)):
            sid = source_ids[i].item()
            if sid != prev_sid:
                source_start_positions[sid] = i
                prev_sid = sid

    max_act_values = {}
    max_act_indices = {}
    max_act_positions = {}  # (mean, std) of token position-within-source-document
    for f_pos, feat_idx in enumerate(top_feature_indices):
        max_act_values[feat_idx] = running_vals[:, f_pos]
        max_act_indices[feat_idx] = running_idxs[:, f_pos]

        valid_mask = running_vals[:, f_pos] > 0
        if valid_mask.any():
            global_idxs = running_idxs[:, f_pos][valid_mask]
            if source_ids is not None and source_start_positions:
                # Position = global_idx - start_of_that_token's_source_document
                positions = []
                for gidx in global_idxs.tolist():
                    sid = source_ids[gidx].item() if gidx < len(source_ids) else 0
                    src_start = source_start_positions.get(sid, 0)
                    positions.append(float(gidx - src_start))
                positions_t = torch.tensor(positions)
            else:
                # Fallback: position within the max_tokens window (approximate)
                positions_t = global_idxs.float() % max_tokens
            pos_mean = positions_t.mean().item()
            pos_std  = positions_t.std().item() if valid_mask.sum() > 1 else 0.0
        else:
            pos_mean, pos_std = 0.0, 0.0
        max_act_positions[feat_idx] = (pos_mean, pos_std)
    
    # =========================================================================
    # Output-centric: VocabProj - Project decoder vectors onto vocabulary
    # =========================================================================
    print("  Computing VocabProj (output-centric)...")
    # Compute in CPU chunks to avoid OOM: [d_hidden, vocab_size] at fp32 is ~8 GB for d_hidden=16384.
    # Process FEAT_CHUNK features at a time instead of the full matrix at once.
    unembed_cpu = model.W_U.detach().cpu()       # [d_model, vocab_size]
    decoder_cpu = decoder_weights.cpu()           # [d_model, d_hidden]
    FEAT_CHUNK = 512
    all_top_values = []
    all_top_indices = []
    for chunk_start in tqdm(range(0, sae.d_hidden, FEAT_CHUNK), desc="  VocabProj (chunks)"):
        chunk_end = min(chunk_start + FEAT_CHUNK, sae.d_hidden)
        # [chunk_size, d_model] @ [d_model, vocab_size] → [chunk_size, vocab_size]
        chunk_logits = decoder_cpu[:, chunk_start:chunk_end].T @ unembed_cpu
        # Mean-subtract: removes common tokens predicted everywhere (Gao et al. 2024)
        chunk_logits = chunk_logits - chunk_logits.mean(dim=-1, keepdim=True)
        chunk_top_vals, chunk_top_idx = chunk_logits.topk(top_k, dim=-1)
        all_top_values.append(chunk_top_vals)
        all_top_indices.append(chunk_top_idx)
        del chunk_logits
    top_vocab_values  = torch.cat(all_top_values,  dim=0)  # [d_hidden, top_k]
    top_vocab_indices = torch.cat(all_top_indices, dim=0)  # [d_hidden, top_k]
    del unembed_cpu, decoder_cpu, all_top_values, all_top_indices

    tokenizer = model.tokenizer

    # =========================================================================
    # Pass 4: TokenChange (causal output-centric)
    # Inject each feature's decoder direction at `layer` via a hook and measure
    # how the next-token distribution shifts at the active token position.
    # Uses up to 5 MaxAct contexts per feature; skips dead features (no MaxAct).
    # =========================================================================
    print("\n  Pass 4: Computing TokenChange (causal output-centric)...")
    token_change_results: Dict[int, Optional[Dict]] = {}
    for feat_idx in tqdm(top_feature_indices, desc="  TokenChange"):
        if max_act_values[feat_idx].max().item() <= 0:
            token_change_results[feat_idx] = None
            continue

        # Build lightweight stubs — only global_token_idx and activation are used
        stubs = []
        for sp in max_act_values[feat_idx].argsort(descending=True):
            act_val = max_act_values[feat_idx][sp].item()
            if act_val <= 0:
                break
            stubs.append(MaxActExample(
                token="", token_id=0, activation=act_val,
                global_token_idx=max_act_indices[feat_idx][sp].item(),
            ))
            if len(stubs) >= 5:
                break

        token_change_results[feat_idx] = compute_token_change_for_feature(
            model, sae, feat_idx, stubs, token_ids,
            layer=layer, device=device,
            mean_activation=feature_mean[feat_idx].item(),
        )

    # =========================================================================
    # Build FeatureInfo for top features
    # =========================================================================
    print("\n  Building feature analysis...")
    
    features_info = []
    for feat_idx in tqdm(top_feature_indices, desc="  Features"):
        # Sort MaxAct results by activation value (descending)
        sorted_order = max_act_values[feat_idx].argsort(descending=True)
        
        # Build MaxAct examples (input-centric: what tokens trigger this feature).
        # Three deduplication layers:
        #   1. Skip Llama special tokens (structural noise)
        #   2. Skip duplicate token strings (same surface form)
        #   3. Skip duplicate context prefixes (same document flooding multiple slots)
        #   4. Cap examples per source document (max 2) to prevent outlier docs
        #      from consuming all top-k slots.
        max_act_examples = []
        seen_tokens: set = set()
        seen_ctx_keys: set = set()
        source_example_count: Dict[str, int] = {}
        MAX_EXAMPLES_PER_SOURCE = 2

        for sort_pos in sorted_order:
            act_val = max_act_values[feat_idx][sort_pos].item()
            if act_val <= 0:
                continue

            token_global_idx = max_act_indices[feat_idx][sort_pos].item()
            tok_id = token_ids[token_global_idx].item()

            # Skip all structural special tokens
            if tok_id in special_ids_set:
                continue

            tok_str = tokenizer.decode([tok_id])

            # Skip duplicate token strings
            if tok_str in seen_tokens:
                continue

            src_id = ""
            if source_ids is not None and source_list is not None:
                src_idx = source_ids[token_global_idx].item()
                src_id = source_list[src_idx] if src_idx < len(source_list) else ""

            # Cap per-source examples to prevent outlier documents dominating
            if source_example_count.get(src_id, 0) >= MAX_EXAMPLES_PER_SOURCE:
                continue

            ctx = build_context_string(token_global_idx, token_ids, tokenizer)

            # Skip duplicate context windows (same document section seen before)
            ctx_key = ctx[:100]
            if ctx_key in seen_ctx_keys:
                continue

            seen_tokens.add(tok_str)
            seen_ctx_keys.add(ctx_key)
            source_example_count[src_id] = source_example_count.get(src_id, 0) + 1

            max_act_examples.append(MaxActExample(
                token=tok_str,
                token_id=tok_id,
                activation=act_val,
                context=ctx,
                global_token_idx=token_global_idx,
                source_id=src_id,
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
        tc = token_change_results.get(feat_idx) or {}
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
            maxact_entropy=token_entropy(max_act_examples[:10]),
            source_breakdown=source_breakdown(max_act_examples[:10], source_list or []),
            token_change_promoted=tc.get("top_promoted", []),
            token_change_suppressed=tc.get("top_suppressed", []),
            token_change_kl=tc.get("mean_kl", 0.0),
            token_change_kl_std=tc.get("kl_std", 0.0),
            token_change_n_contexts=tc.get("n_contexts", 0),
        )
        features_info.append(info)
    
    return features_info


def save_results(
    sae: SparseAutoencoder,
    token_ids: torch.Tensor,
    chunk_files: List[Path],
    features_info: List[FeatureInfo],
    history: Dict[str, List[float]],
    output_dir: Path = MEDICAL_OUTPUT_DIR,
    source_ids: Optional[torch.Tensor] = None,
    source_list: Optional[List[str]] = None,
    layer: int = TARGET_LAYER,
):
    """Save all results to disk. Processes chunks one at a time — no OOM."""
    import gc

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to {output_dir}...")

    sae.save(output_dir / "sae.pt")

    # Token IDs are small — save directly
    torch.save(token_ids, output_dir / "token_ids.pt")
    print(f"  Saved token_ids: {token_ids.shape}")

    # Source IDs for training data attribution
    if source_ids is not None and source_list is not None:
        torch.save(source_ids, output_dir / "source_ids.pt")
        with open(output_dir / "source_list.json", "w") as f:
            json.dump(source_list, f)
        print(f"  Saved source_ids: {source_ids.shape[0]:,} tokens, {len(source_list)} sources")

    # Chunk file manifest (so --skip-collection can find them)
    chunk_manifest = [str(p) for p in chunk_files]
    with open(output_dir / "activations.json", "w") as f:
        json.dump(chunk_manifest, f)
    print(f"  Saved chunk manifest: {len(chunk_files)} chunks")

    # Summary stats — one chunk at a time
    device = get_device()
    sae = sae.to(device).eval()

    n_tokens = 0
    feature_count = torch.zeros(sae.d_hidden)
    total_l0 = 0.0
    # Explained variance accumulators (single-pass via E[x²] - E[x]²).
    # Summed over tokens, averaged at the end.  All in float64 to avoid
    # catastrophic cancellation when subtracting two large numbers.
    sum_x   = torch.zeros(sae.d_model, dtype=torch.float64)
    sum_x2  = torch.zeros(sae.d_model, dtype=torch.float64)
    sum_res2 = 0.0   # sum of ||x - x̂||² over all tokens

    print("  Computing summary statistics (chunked)...")
    for chunk_path in tqdm(chunk_files, desc="  Stats"):
        chunk = torch.load(chunk_path).float()  # Convert float16 back to float32
        for i in range(0, len(chunk), BATCH_SIZE):
            batch = chunk[i:i+BATCH_SIZE].to(device)
            with torch.no_grad():
                features = sae.encode(batch)
                recon    = sae.decode(features)
                residual = batch - recon
            features = features.cpu()
            feature_count += (features > 0).float().sum(dim=0)
            total_l0 += (features > 0).float().sum(dim=-1).sum().item()
            n_tokens += batch.shape[0]
            # Accumulate for explained variance (move to CPU before float64 cast —
            # MPS does not support float64, so the cast must happen on CPU)
            b = batch.cpu().double()
            sum_x   += b.sum(dim=0)
            sum_x2  += (b ** 2).sum(dim=0)
            sum_res2 += residual.cpu().double().pow(2).sum().item()
        del chunk
        gc.collect()

    feature_freq = feature_count / n_tokens
    dead_features = (feature_freq < 1e-5).sum().item()
    avg_l0 = total_l0 / n_tokens

    # Explained variance = 1 - Var(residual) / Var(x)
    # Var(x) per dimension = E[x²] - E[x]²; total = sum over dimensions
    mean_x      = sum_x / n_tokens
    var_x_total = float((sum_x2 / n_tokens - mean_x ** 2).sum().item())
    var_res     = sum_res2 / n_tokens          # mean per-token residual variance
    explained_variance = 1.0 - (var_res / var_x_total) if var_x_total > 0 else 0.0
    explained_variance = round(float(explained_variance), 6)
    
    # Save feature analysis with both MaxAct and VocabProj
    features_data = [
        {
            "index": f.index,
            "frequency": f.activation_frequency,
            "mean": f.mean_activation,
            "max": f.max_activation,
            "max_activating_tokens": [
                {
                    "token": ex.token,
                    "token_id": ex.token_id,
                    "activation": ex.activation,
                    "context": ex.context,
                    "global_token_idx": ex.global_token_idx,
                    "source_id": ex.source_id,
                }
                for ex in f.max_activating_tokens
            ],
            "vocab_projection": f.vocab_projection,
            "vocab_projection_logits": f.vocab_projection_logits,
            # Positional: where in the sequence this feature tends to fire
            "position_mean": round(f.position_mean, 2),
            "position_std": round(f.position_std, 2),
            # Monosemanticity proxy: precomputed in analyze_features()
            "maxact_entropy": f.maxact_entropy,
            # Source attribution: precomputed in analyze_features()
            "source_breakdown": f.source_breakdown,
            # TokenChange (causal output-centric): empty lists if not computed
            "token_change_promoted": f.token_change_promoted,
            "token_change_suppressed": f.token_change_suppressed,
            "token_change_kl": round(f.token_change_kl, 4),
            "token_change_kl_std": round(f.token_change_kl_std, 4),
            "token_change_n_contexts": f.token_change_n_contexts,
        }
        for f in features_info
    ]
    with open(output_dir / "features.json", "w") as f:
        json.dump(features_data, f, indent=2)
    print(f"  Saved feature analysis: {len(features_info)} features")
    
    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Saved training history")
    
    summary = {
        "model": MODEL_NAME,
        "layer": layer,
        "hook_type": HOOK_TYPE,
        "d_model": D_MODEL,
        "d_hidden": sae.d_hidden,
        "n_tokens": n_tokens,
        "n_chunks": len(chunk_files),
        "dead_features": dead_features,
        "avg_l0_sparsity": avg_l0,
        "explained_variance": explained_variance,
        "final_loss": history["loss"][-1] if history["loss"] else None,
        "analysis_methods": ["MaxAct (input-centric)", "VocabProj (output-centric)"],
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Results Summary")
    print(f"{'='*60}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Layer: {layer} ({HOOK_TYPE})")
    print(f"  Tokens: {n_tokens:,} ({len(chunk_files)} chunks on disk)")
    print(f"  Features: {sae.d_hidden:,}")
    print(f"  Dead features: {dead_features}")
    print(f"  Avg L0 sparsity: {avg_l0:.1f}")
    print(f"  Explained variance: {explained_variance:.4f}")
    print(f"{'='*60}")


# =============================================================================
# Cross-Layer Analysis
# =============================================================================

def cross_layer_analysis(layer_dirs: Dict[int, Path]):
    """
    Compare decoder weight directions across layers to quantify feature evolution.

    For each pair of trained layers, computes the distribution of maximum cosine
    similarities between decoder feature directions.  High similarity = the same
    concept is represented in both layers; low = a layer-specific specialisation.

    Saves medical_outputs/cross_layer_analysis.json with:
      - per-pair statistics (mean/median max-sim, shared-feature counts)
      - similarity histograms for plotting
      - top shared features (cosine sim > 0.7) with their indices in each layer
    """
    from itertools import combinations

    print(f"\n{'='*60}")
    print("  Cross-Layer Feature Analysis")
    print(f"  Layers: {sorted(layer_dirs)}")
    print(f"{'='*60}")

    # Load all available SAEs
    saes: Dict[int, SparseAutoencoder] = {}
    for layer, layer_dir in sorted(layer_dirs.items()):
        sae_path = layer_dir / "sae.pt"
        if not sae_path.exists():
            print(f"  WARNING: No SAE at {sae_path} — skipping layer {layer}.")
            continue
        sae = SparseAutoencoder.load(sae_path)
        sae.eval()
        saes[layer] = sae
        print(f"  Loaded layer {layer}: {sae.d_hidden} features")

    if len(saes) < 2:
        print("  Need at least 2 trained layers for cross-layer analysis. Skipping.")
        return

    HIGH_SIM_THRESHOLD = 0.7
    CHUNK = 512   # rows per matmul chunk — keeps memory bounded
    BIN_EDGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    results: Dict = {"layers": sorted(saes.keys()), "pairs": {}}

    for layer_a, layer_b in combinations(sorted(saes.keys()), 2):
        sae_a = saes[layer_a]
        sae_b = saes[layer_b]

        # Decoder weight: [d_model, d_hidden].  Transpose → [d_hidden, d_model]
        # so each ROW is one feature's direction in model space.
        W_a = F.normalize(sae_a.decoder.weight.T.float(), dim=-1)  # [n_feat, d_model]
        W_b = F.normalize(sae_b.decoder.weight.T.float(), dim=-1)

        print(f"\n  L{layer_a} → L{layer_b} "
              f"({W_a.shape[0]} × {W_b.shape[0]} features) …")

        def _best_matches(src: torch.Tensor, tgt: torch.Tensor):
            """For each row in src, find max cosine sim to any row in tgt."""
            max_sims, max_idxs = [], []
            for i in range(0, src.shape[0], CHUNK):
                block = src[i:i + CHUNK]          # [chunk, d_model]
                sim = block @ tgt.T               # [chunk, n_tgt]
                ms, mi = sim.max(dim=-1)
                max_sims.extend(ms.tolist())
                max_idxs.extend(mi.tolist())
            return torch.tensor(max_sims), max_idxs

        sims_a2b, idx_a2b = _best_matches(W_a, W_b)
        sims_b2a, _       = _best_matches(W_b, W_a)

        def _histogram(sims: torch.Tensor):
            counts = []
            for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
                hi_ = hi + 0.01 if hi == 1.0 else hi   # include 1.0 in last bin
                counts.append(int(((sims >= lo) & (sims < hi_)).sum().item()))
            return {"bins": BIN_EDGES, "counts": counts}

        shared_a2b_mask = sims_a2b > HIGH_SIM_THRESHOLD
        shared_count_a2b = int(shared_a2b_mask.sum().item())
        shared_count_b2a = int((sims_b2a > HIGH_SIM_THRESHOLD).sum().item())

        # Collect top shared feature pairs (idx_in_A, idx_in_B, similarity)
        shared_pairs = []
        if shared_count_a2b > 0:
            shared_indices_a = shared_a2b_mask.nonzero(as_tuple=True)[0].tolist()
            for idx_a in sorted(shared_indices_a, key=lambda i: -sims_a2b[i].item())[:50]:
                shared_pairs.append({
                    "layer_a_feature": int(idx_a),
                    "layer_b_feature": int(idx_a2b[idx_a]),
                    "cosine_similarity": round(float(sims_a2b[idx_a].item()), 4),
                })

        pair_key = f"{layer_a}_{layer_b}"
        results["pairs"][pair_key] = {
            "layer_a": layer_a,
            "layer_b": layer_b,
            "n_features_a": int(W_a.shape[0]),
            "n_features_b": int(W_b.shape[0]),
            "similarity_threshold": HIGH_SIM_THRESHOLD,
            "a_to_b": {
                "mean_max_similarity":   round(float(sims_a2b.mean()), 4),
                "median_max_similarity": round(float(sims_a2b.median()), 4),
                "std_max_similarity":    round(float(sims_a2b.std()), 4),
                "shared_features_count": shared_count_a2b,
                "shared_features_pct":   round(shared_count_a2b / W_a.shape[0] * 100, 2),
                "histogram": _histogram(sims_a2b),
            },
            "b_to_a": {
                "mean_max_similarity":   round(float(sims_b2a.mean()), 4),
                "median_max_similarity": round(float(sims_b2a.median()), 4),
                "shared_features_count": shared_count_b2a,
                "shared_features_pct":   round(shared_count_b2a / W_b.shape[0] * 100, 2),
            },
            "top_shared_pairs": shared_pairs,
        }

        print(f"    mean_max_sim={results['pairs'][pair_key]['a_to_b']['mean_max_similarity']:.4f}  "
              f"shared (>{HIGH_SIM_THRESHOLD}): "
              f"{shared_count_a2b}/{W_a.shape[0]} "
              f"({results['pairs'][pair_key]['a_to_b']['shared_features_pct']:.1f}%)")

    out_path = MEDICAL_OUTPUT_DIR / "cross_layer_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Cross-layer analysis saved → {out_path}")


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Medical SAE pipeline for Llama 3.2 1B")
    parser.add_argument("--quick", action="store_true", help="Quick test (500 samples, 1 epoch)")
    parser.add_argument("--skip-collection", action="store_true", help="Reuse cached activations")
    parser.add_argument("--device", type=str, default=None, help="Device override (auto-detected)")
    parser.add_argument(
        "--validate-token-change", action="store_true",
        help=(
            "Sanity-check TokenChange on the 10 most coherent features (low MaxAct "
            "entropy, sufficient examples) and print a comparison table, then exit. "
            "Requires trained SAE + features.json in the output directory."
        ),
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, default=[TARGET_LAYER],
        help=(
            "One or more residual-stream layers to train SAEs on "
            "(default: [8]).  E.g. --layers 4 8 12.  "
            "When more than one layer is given, cross_layer_analysis() is run "
            "afterwards and the decoder cosine-similarity results are written to "
            "medical_outputs/cross_layer_analysis.json."
        ),
    )
    args = parser.parse_args()

    device = args.device or get_device()

    # ------------------------------------------------------------------
    # --validate-token-change: sanity-check on 10 coherent features
    # ------------------------------------------------------------------
    if args.validate_token_change:
        layers_val: List[int] = args.layers
        layer_val = layers_val[0]

        def _val_dir(l: int) -> Path:
            if len(layers_val) == 1 and l == TARGET_LAYER:
                return MEDICAL_OUTPUT_DIR
            return MEDICAL_OUTPUT_DIR / f"layer_{l}"

        out_dir = _val_dir(layer_val)
        feat_path = out_dir / "features.json"
        sae_path  = out_dir / "sae.pt"
        if not feat_path.exists() or not sae_path.exists():
            print(f"[validate] Missing {feat_path} or {sae_path} — run full pipeline first.")
            return

        print(f"\n[validate-token-change] Layer {layer_val}, dir: {out_dir}")
        with open(feat_path) as fh:
            all_feats = json.load(fh)

        # Load labeled labels if available
        label_map: Dict[int, str] = {}
        lbl_path = out_dir / "labeled_features.json"
        if lbl_path.exists():
            with open(lbl_path) as fh:
                for lf in json.load(fh):
                    label_map[lf["index"]] = lf.get("label", "")

        # Pick top-10 by coherence: enough MaxAct examples + low entropy + alive
        candidates = [
            f for f in all_feats
            if len(f.get("max_activating_tokens", [])) >= 3
            and f.get("frequency", 0) > 0.0001
            and f.get("maxact_entropy", 999) < 3.0
        ]
        candidates.sort(key=lambda f: f.get("maxact_entropy", 999))
        candidates = candidates[:10]

        if not candidates:
            print("[validate] No suitable features found — check features.json.")
            return

        sae = SparseAutoencoder.load(sae_path, device=device)
        model = load_model(device)

        # Load token_ids for context reconstruction
        tok_path = out_dir / "token_ids.pt"
        if not tok_path.exists():
            print("[validate] token_ids.pt not found — cannot reconstruct contexts.")
            return
        token_ids_val = torch.load(tok_path)

        print(f"\n{'─'*100}")
        print(f"{'Idx':>6}  {'Label':<35}  {'MaxAct top-3':<30}  {'VocabProj top-3':<30}  {'TC promoted top-3':<30}  {'KL':>6}")
        print(f"{'─'*100}")

        for feat in candidates:
            fidx = feat["index"]
            stubs = [
                MaxActExample(token="", token_id=0, activation=e["activation"],
                              global_token_idx=e["global_token_idx"])
                for e in feat.get("max_activating_tokens", [])[:5]
            ]
            tc = compute_token_change_for_feature(
                model, sae, fidx, stubs, token_ids_val,
                layer=layer_val, device=device,
                mean_activation=feat.get("mean", 1.0),
            )
            maxact_top  = [e["token"] for e in feat.get("max_activating_tokens", [])[:3]]
            vocab_top   = feat.get("vocab_projection", [])[:3]
            tc_top      = (tc or {}).get("top_promoted", [])[:3]
            kl_val      = (tc or {}).get("mean_kl", 0.0)
            label       = label_map.get(fidx, "")[:34]
            print(f"{fidx:>6}  {label:<35}  {str(maxact_top):<30}  {str(vocab_top):<30}  {str(tc_top):<30}  {kl_val:>6.3f}")

        print(f"{'─'*100}")
        print("\n[validate] Done. High KL + TC tokens matching MaxAct = method working correctly.")
        return

    num_samples = 500 if args.quick else NUM_SAMPLES
    num_epochs = 1 if args.quick else NUM_EPOCHS
    layers: List[int] = args.layers

    # Per-layer output directory.
    # Single-layer default (layer 8) → medical_outputs/  (backward-compatible).
    # Any other configuration       → medical_outputs/layer_N/
    def _layer_dir(layer: int) -> Path:
        if len(layers) == 1 and layer == TARGET_LAYER:
            return MEDICAL_OUTPUT_DIR
        return MEDICAL_OUTPUT_DIR / f"layer_{layer}"

    MEDICAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  Medical SAE Pipeline")
    layers_str = str(layers[0]) if len(layers) == 1 else str(layers)
    print(f"  Model: {MODEL_NAME} | Layers: {layers_str} | Hook: {HOOK_TYPE}")
    print(f"  Dataset: medmcqa + pubmed_qa + pubmed_abs (chat-formatted)")
    print("="*60)
    if args.quick:
        print("  [QUICK TEST MODE]")
    if len(layers) > 1:
        print(f"  Multi-layer mode: will train {len(layers)} SAEs, then run cross-layer analysis.")

    # -------------------------------------------------------------------------
    # Phase 1 — Collect activations for every requested layer (model loaded once)
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Phase 1: Activation Collection")
    print(f"{'='*60}")

    model = load_model(device)
    layer_data: Dict[int, tuple] = {}   # layer → (token_ids, chunk_files, source_ids, source_list)

    for layer in layers:
        layer_dir = _layer_dir(layer)
        layer_dir.mkdir(parents=True, exist_ok=True)

        manifest_path   = layer_dir / "activations.json"
        token_ids_path  = layer_dir / "token_ids.pt"
        source_ids_path = layer_dir / "source_ids.pt"
        source_list_path = layer_dir / "source_list.json"

        if args.skip_collection and manifest_path.exists() and token_ids_path.exists():
            print(f"\n  [Layer {layer}] Loading cached chunks from {layer_dir}…")
            with open(manifest_path) as f:
                chunk_files = [Path(p) for p in json.load(f)]
            token_ids = torch.load(token_ids_path)
            source_ids = torch.load(source_ids_path) if source_ids_path.exists() else None
            source_list = json.load(open(source_list_path)) if source_list_path.exists() else None
            print(f"  Found {len(chunk_files)} chunks, {token_ids.shape[0]:,} tokens")
        else:
            print(f"\n  [Layer {layer}] Collecting from layer {layer}…")
            token_ids, chunk_files, source_ids, source_list = collect_activations(
                model, num_samples=num_samples, output_dir=layer_dir, layer=layer,
            )

        layer_data[layer] = (token_ids, chunk_files, source_ids, source_list)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Phase 2 — Train one SAE per layer (no GPU model needed)
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Phase 2: SAE Training")
    print(f"{'='*60}")

    trained: Dict[int, tuple] = {}   # layer → (sae, history)

    for layer in layers:
        token_ids, chunk_files, source_ids, source_list = layer_data[layer]
        print(f"\n  [Layer {layer}] Training SAE…")

        sae = SparseAutoencoder()

        # Initialise b_pre from a sample of real activations (geometric median approx).
        # This centres the encoder input so pre-activations start near zero — critical
        # for fast convergence and reducing dead features (Gao et al. 2024).
        print("  Initialising SAE bias from data sample…")
        _init_chunk = torch.load(chunk_files[0])
        _init_sample = _init_chunk[:min(2048, len(_init_chunk))]
        sae.init_bias_from_data(_init_sample)
        del _init_chunk, _init_sample
        print(f"  b_pre initialised (mean norm: {sae.b_pre.data.norm().item():.3f})")

        history = train_sae(sae, chunk_files, num_epochs=num_epochs, device=device)
        trained[layer] = (sae, history)

    # -------------------------------------------------------------------------
    # Phase 3 — Feature analysis & save (model reloaded once for VocabProj)
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Phase 3: Feature Analysis & Save")
    print(f"{'='*60}")

    model = load_model(device)

    for layer in layers:
        layer_dir = _layer_dir(layer)
        token_ids, chunk_files, source_ids, source_list = layer_data[layer]
        sae, history = trained[layer]

        print(f"\n  [Layer {layer}] Analysing features…")
        print("    MaxAct (input-centric)  + VocabProj (output-centric)")

        features_info = analyze_features(
            sae, model, token_ids, chunk_files, device=device,
            source_ids=source_ids, source_list=source_list,
            layer=layer,
        )
        save_results(
            sae, token_ids, chunk_files, features_info, history,
            output_dir=layer_dir, source_ids=source_ids, source_list=source_list,
            layer=layer,
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Phase 4 — Cross-layer analysis (only when >1 layer trained)
    # -------------------------------------------------------------------------
    if len(layers) > 1:
        layer_dirs = {layer: _layer_dir(layer) for layer in layers}
        cross_layer_analysis(layer_dirs)


if __name__ == "__main__":
    main()
