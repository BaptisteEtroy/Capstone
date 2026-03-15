#!/usr/bin/env python3
"""
Medical SAE pipeline for Llama 3.2 1B.

Usage:
    python main.py              # Full pipeline (collect activations → train SAE → analyze)
    python main.py --quick      # Quick test (500 samples, 1 epoch)
    python main.py --skip-collection    # Skip activation collection (reuse cached)
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
) -> tuple:
    """
    Collect residual stream activations from layer TARGET_LAYER using medical datasets.

    Saves activation chunks to disk to stay within RAM limits.
    Token IDs and source IDs are kept in memory (small) for MaxAct context building.

    Returns:
        token_ids   : torch.Tensor [n_tokens]  — all token IDs in RAM
        chunk_files : List[Path]               — activation chunk paths on disk
        source_ids  : torch.Tensor [n_tokens]  — index into source_list per token
        source_list : List[str]                — source ID strings
    """
    import gc

    hook_point = f"blocks.{TARGET_LAYER}.hook_{HOOK_TYPE}"
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCollecting activations from {hook_point}...")
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

    history = {"loss": [], "reconstruction": []}
    global_step = 0

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
    output_dir: Path = MEDICAL_OUTPUT_DIR,
    source_ids: Optional[torch.Tensor] = None,
    source_list: Optional[List[str]] = None,
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

    print("  Computing summary statistics (chunked)...")
    for chunk_path in tqdm(chunk_files, desc="  Stats"):
        chunk = torch.load(chunk_path).float()  # Convert float16 back to float32
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
    parser = argparse.ArgumentParser(description="Medical SAE pipeline for Llama 3.2 1B")
    parser.add_argument("--quick", action="store_true", help="Quick test (500 samples, 1 epoch)")
    parser.add_argument("--skip-collection", action="store_true", help="Reuse cached activations")
    parser.add_argument("--device", type=str, default=None, help="Device override (auto-detected)")
    args = parser.parse_args()

    device = args.device or get_device()
    num_samples = 500 if args.quick else NUM_SAMPLES
    num_epochs = 1 if args.quick else NUM_EPOCHS
    output_dir = MEDICAL_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  Medical SAE Pipeline")
    print(f"  Model: {MODEL_NAME} | Layer: {TARGET_LAYER} | Hook: {HOOK_TYPE}")
    print(f"  Dataset: medmcqa + pubmed_qa + pubmed_abs (chat-formatted) | Output: {output_dir}")
    print("="*60)
    if args.quick:
        print("  [QUICK TEST MODE]")

    # Step 1: Load model
    model = load_model(device)

    # Step 2: Collect activations as on-disk chunks (never load all into RAM)
    manifest_path = output_dir / "activations.json"
    token_ids_path = output_dir / "token_ids.pt"
    source_ids_path = output_dir / "source_ids.pt"
    source_list_path = output_dir / "source_list.json"

    if args.skip_collection and manifest_path.exists() and token_ids_path.exists():
        print(f"\nLoading cached chunks from {output_dir}...")
        with open(manifest_path) as f:
            chunk_files = [Path(p) for p in json.load(f)]
        token_ids = torch.load(token_ids_path)
        source_ids = torch.load(source_ids_path) if source_ids_path.exists() else None
        source_list = json.load(open(source_list_path)) if source_list_path.exists() else None
        print(f"  Found {len(chunk_files)} chunk files, {token_ids.shape[0]:,} token IDs")
    else:
        token_ids, chunk_files, source_ids, source_list = collect_activations(
            model, num_samples=num_samples, output_dir=output_dir,
        )

    # Free model memory before SAE training
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 3: Train SAE — loads one chunk at a time
    sae = SparseAutoencoder()

    # Initialise b_pre from a sample of real activations (geometric median approx).
    # This centres the encoder input so pre-activations start near zero — critical
    # for fast convergence and reducing dead features (Gao et al. 2024).
    print("\nInitialising SAE bias from data sample...")
    _init_chunk = torch.load(chunk_files[0])
    _init_sample = _init_chunk[:min(2048, len(_init_chunk))]
    sae.init_bias_from_data(_init_sample)
    del _init_chunk, _init_sample
    print(f"  b_pre initialised (mean norm: {sae.b_pre.data.norm().item():.3f})")

    history = train_sae(sae, chunk_files, num_epochs=num_epochs, device=device)

    # Step 4: Analyze features — loads one chunk at a time
    model = load_model(device)
    print("\n  Feature analysis using dual methods:")
    print("    - MaxAct: Find tokens that maximally activate each feature (±50 token context)")
    print("    - VocabProj: Find tokens each feature promotes in output")
    features_info = analyze_features(
        sae, model, token_ids, chunk_files, device=device,
        source_ids=source_ids, source_list=source_list,
    )

    # Step 5: Save results
    save_results(sae, token_ids, chunk_files, features_info, history,
                 output_dir=output_dir, source_ids=source_ids, source_list=source_list)


if __name__ == "__main__":
    main()
