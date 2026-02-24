#!/usr/bin/env python3
"""
Sparse Autoencoder for Mechanistic Interpretability
====================================================
A simplified pipeline for extracting interpretable features from GPT-2 using SAEs.

Based on:
- Anthropic's "Towards Monosemanticity" research
- Literature review recommendations for SAE-based feature extraction

Usage:
    python main.py                    # Full pipeline
    python main.py --quick            # Quick test (500 samples, 1 epoch)
    python main.py --skip-collection  # Skip activation collection (use cached)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
from tqdm import tqdm
import argparse
import json

from transformer_lens import HookedTransformer
from datasets import load_dataset

# =============================================================================
# Configuration - Hardcoded based on literature review recommendations
# =============================================================================

# Model: GPT-2 (well-studied, no auth required, good for interpretability)
MODEL_NAME = "gpt2"
D_MODEL = 768
N_LAYERS = 12

# Layer: Middle layer for semantic features (lit review: features are most interpretable here)
TARGET_LAYER = 6
HOOK_TYPE = "resid_post"

# SAE: Standard architecture with L1 regularization (lit review: most effective method)
EXPANSION_FACTOR = 8  # Hidden dim = 768 * 8 = 6144
L1_COEFFICIENT = 10

# Training
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
BATCH_SIZE = 4096
NUM_SAMPLES = 10000

# Output
OUTPUT_DIR = Path("outputs")


# =============================================================================
# Sparse Autoencoder
# =============================================================================

@dataclass
class SAEOutput:
    """Output from SAE forward pass."""
    reconstructed: torch.Tensor
    hidden: torch.Tensor
    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    sparsity_loss: torch.Tensor
    l0_sparsity: torch.Tensor


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for extracting interpretable features.
    
    Architecture: Input -> Encoder -> ReLU -> Hidden (sparse) -> Decoder -> Output
    
    The L1 penalty encourages sparse activations, leading to monosemantic features
    where each hidden unit responds to a single interpretable concept.
    """
    
    def __init__(
        self,
        d_model: int = D_MODEL,
        expansion_factor: int = EXPANSION_FACTOR,
        l1_coeff: float = L1_COEFFICIENT,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_model * expansion_factor
        self.l1_coeff = l1_coeff
        
        # Encoder and decoder
        self.encoder = nn.Linear(d_model, self.d_hidden, bias=True)
        self.decoder = nn.Linear(self.d_hidden, d_model, bias=True)
        self.b_pre = nn.Parameter(torch.zeros(d_model))  # Pre-encoder bias
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.decoder.weight, nonlinearity="linear")
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
        self._normalize_decoder()
        
        # Track neuron activity for ghost gradients
        self.register_buffer("neuron_activity", torch.zeros(self.d_hidden))
        self.register_buffer("steps", torch.tensor(0))
    
    def _normalize_decoder(self):
        """Keep decoder columns at unit norm for interpretability."""
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=0, keepdim=True)
            self.decoder.weight.data = self.decoder.weight.data / (norms + 1e-8)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations to sparse feature space."""
        return F.relu(self.encoder(x - self.b_pre))
    
    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to activation space."""
        return self.decoder(hidden) + self.b_pre
    
    def forward(self, x: torch.Tensor) -> SAEOutput:
        """Forward pass with loss computation."""
        hidden = self.encode(x)
        reconstructed = self.decode(hidden)
        
        # Losses
        reconstruction_loss = F.mse_loss(reconstructed, x)
        sparsity_loss = hidden.abs().mean()  # L1 penalty on activations
        loss = reconstruction_loss + self.l1_coeff * sparsity_loss
        l0_sparsity = (hidden > 0).float().sum(dim=-1).mean()
        
        # Track activity for ghost gradients
        if self.training:
            with torch.no_grad():
                batch_activity = (hidden > 0).float().mean(dim=0)
                self.neuron_activity = 0.999 * self.neuron_activity + 0.001 * batch_activity
                self.steps += 1
        
        return SAEOutput(
            reconstructed=reconstructed,
            hidden=hidden,
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            sparsity_loss=sparsity_loss,
            l0_sparsity=l0_sparsity,
        )
    
    def get_dead_neurons(self, threshold: float = 1e-5) -> torch.Tensor:
        """Identify neurons that rarely activate."""
        return self.neuron_activity < threshold
    
    def save(self, path: Path):
        """Save model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "d_model": self.d_model,
            "d_hidden": self.d_hidden,
            "l1_coeff": self.l1_coeff,
        }, path)
        print(f"Saved SAE to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "SparseAutoencoder":
        """Load model from disk."""
        checkpoint = torch.load(path, weights_only=False)
        sae = cls(
            d_model=checkpoint["d_model"],
            expansion_factor=checkpoint["d_hidden"] // checkpoint["d_model"],
            l1_coeff=checkpoint["l1_coeff"],
        )
        sae.load_state_dict(checkpoint["state_dict"])
        return sae


# =============================================================================
# Model Loading & Activation Collection
# =============================================================================

def get_device() -> str:
    """Detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


@dataclass
class ActivationData:
    """Container for activations and their corresponding token IDs."""
    activations: torch.Tensor  # [n_tokens, d_model]
    token_ids: torch.Tensor    # [n_tokens] - the actual input tokens


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
    Train the SAE with ghost gradients for dead neuron revival.
    """
    device = device or get_device()
    sae = sae.to(device)
    
    print(f"\nTraining SAE...")
    print(f"  Architecture: {sae.d_model} -> {sae.d_hidden} -> {sae.d_model}")
    print(f"  Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    
    # Setup
    optimizer = Adam(sae.parameters(), lr=learning_rate)
    dataset = torch.utils.data.TensorDataset(activations)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    total_steps = len(loader) * num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=learning_rate * 0.1)
    
    # L1 warmup (gradually increase sparsity penalty)
    l1_warmup_steps = 2000
    
    history = {"loss": [], "reconstruction": [], "sparsity": [], "l0": [], "dead": []}
    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for (batch,) in pbar:
            batch = batch.to(device)
            
            # L1 warmup
            warmup_factor = min(1.0, global_step / l1_warmup_steps)
            sae.l1_coeff = L1_COEFFICIENT * warmup_factor
            
            # Forward pass
            output = sae(batch)
            
            # Ghost gradient loss for dead neurons
            ghost_loss = torch.tensor(0.0, device=device)
            dead_mask = sae.get_dead_neurons()
            n_dead = dead_mask.sum().item()
            
            if n_dead > 0:
                residual = batch - output.reconstructed
                residual_norm = residual.norm(dim=-1, keepdim=True) + 1e-8
                dead_encoder = sae.encoder.weight[dead_mask]
                ghost_acts = torch.einsum("bd,nd->bn", residual / residual_norm, dead_encoder)
                ghost_loss = -ghost_acts.mean() * 0.1
            
            total_loss = output.loss + ghost_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Normalize decoder
            sae._normalize_decoder()
            
            epoch_loss += output.loss.item()
            global_step += 1
            
            # Log every 100 steps
            if global_step % 100 == 0:
                history["loss"].append(output.loss.item())
                history["reconstruction"].append(output.reconstruction_loss.item())
                history["sparsity"].append(output.sparsity_loss.item())
                history["l0"].append(output.l0_sparsity.item())
                history["dead"].append(n_dead)
            
            pbar.set_postfix({
                "loss": f"{output.loss.item():.4f}",
                "L0": f"{output.l0_sparsity.item():.1f}",
                "dead": n_dead,
            })
        
        print(f"  Epoch {epoch+1} avg loss: {epoch_loss / len(loader):.4f}")
    
    print(f"\nTraining complete!")
    print(f"  Final dead neurons: {sae.get_dead_neurons().sum().item()}")
    return history


# =============================================================================
# Feature Analysis (Input-centric + Output-centric methods from lit review)
# =============================================================================

@dataclass
class MaxActExample:
    """A single example of a token that strongly activates a feature."""
    token: str
    token_id: int
    activation: float


@dataclass
class FeatureInfo:
    """
    Information about a learned feature using both interpretability methods.
    
    Input-centric (MaxAct): What inputs trigger this feature?
    Output-centric (VocabProj): What outputs does this feature promote?
    """
    index: int
    activation_frequency: float
    mean_activation: float
    max_activation: float
    
    # Input-centric: tokens that maximally ACTIVATE this feature (what triggers it)
    max_activating_tokens: List[MaxActExample]
    
    # Output-centric: tokens this feature PROMOTES in output (what it does)
    vocab_projection: List[str]
    vocab_projection_logits: List[float]


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
    
    # Select top features by frequency (these are the interesting ones)
    num_features_to_analyze = min(100, sae.d_hidden)
    top_feature_indices = feature_freq.topk(num_features_to_analyze).indices.tolist()
    top_feature_set = set(top_feature_indices)
    
    # Track top-k activations per feature
    topk_per_feature = top_k
    max_act_values = {idx: torch.full((topk_per_feature,), float('-inf')) for idx in top_feature_indices}
    max_act_indices = {idx: torch.zeros(topk_per_feature, dtype=torch.long) for idx in top_feature_indices}
    
    for i in tqdm(range(0, len(activations), BATCH_SIZE), desc="  MaxAct"):
        batch = activations[i:i+BATCH_SIZE].to(device)
        batch_start_idx = i
        
        with torch.no_grad():
            features = sae.encode(batch).cpu()  # [batch_size, d_hidden]
        
        # For each feature we're tracking, update its top-k
        for feat_idx in top_feature_indices:
            feat_acts = features[:, feat_idx]  # [batch_size]
            
            # Combine current batch with existing top-k and keep overall top-k
            combined_vals = torch.cat([max_act_values[feat_idx], feat_acts])
            combined_idxs = torch.cat([
                max_act_indices[feat_idx],
                torch.arange(batch_start_idx, batch_start_idx + len(feat_acts))
            ])
            
            # Get top-k from combined
            topk_vals, topk_positions = combined_vals.topk(topk_per_feature)
            max_act_values[feat_idx] = topk_vals
            max_act_indices[feat_idx] = combined_idxs[topk_positions]
    
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
                max_act_examples.append(MaxActExample(
                    token=tok_str,
                    token_id=tok_id,
                    activation=act_val,
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
                {"token": ex.token, "token_id": ex.token_id, "activation": ex.activation}
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
    print(f"  Analysis: MaxAct + VocabProj")
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
    print("  Model: GPT-2 | Layer: 6 | Hook: resid_post")
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
