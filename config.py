"""
Configuration & Shared Components
==================================
Central configuration and reusable classes for the SAE interpretability pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
from typing import List


# =============================================================================
# Configuration Constants
# =============================================================================

# Model: Llama 3.2 1B (instruction-tuned available at meta-llama/Llama-3.2-1B-Instruct)
MODEL_NAME = "meta-llama/Llama-3.2-1B"
D_MODEL = 2048
N_LAYERS = 16

# Layer: Middle layer for semantic features (lit review: features are most interpretable here)
TARGET_LAYER = 8  # Middle of 16 layers
HOOK_TYPE = "resid_post"

# SAE: TopK architecture (Gao et al. 2024) — activates exactly K features per token.
# Eliminates L1 tuning, ghost gradients, and dead neurons by construction.
EXPANSION_FACTOR = 4  # Hidden dim = 2048 * 4 = 8192 features (same count as before)
TOP_K = 100           # Exactly 100 features active per token (L0 = K, always)
AUX_COEFF = 1 / 32   # Auxiliary loss weight (Gao et al. 2024): gives dead features gradient signal

# Training
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
BATCH_SIZE = 4096
NUM_SAMPLES = 10_000

# Output
OUTPUT_DIR = Path("outputs")
SAE_PATH = OUTPUT_DIR / "sae.pt"
FEATURES_PATH = OUTPUT_DIR / "features.json"


# =============================================================================
# Utility Functions
# =============================================================================

def get_device() -> str:
    """Detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =============================================================================
# Data Classes
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


@dataclass
class ActivationData:
    """Container for activations and their corresponding token IDs."""
    activations: torch.Tensor  # [n_tokens, d_model]
    token_ids: torch.Tensor    # [n_tokens] - the actual input tokens


@dataclass
class MaxActExample:
    """A single example of a token that strongly activates a feature."""
    token: str
    token_id: int
    activation: float
    context: str = ""           # ±5 token window with [TOKEN] marker
    global_token_idx: int = -1  # position in flat token_ids tensor


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


# =============================================================================
# Sparse Autoencoder Model
# =============================================================================

class SparseAutoencoder(nn.Module):
    """
    TopK Sparse Autoencoder (Gao et al. 2024).

    Architecture: Input -> Encoder -> TopK -> Hidden (exactly K active) -> Decoder -> Output

    Sparsity is enforced structurally: exactly K features activate per token.
    No L1 penalty, no ghost gradients, no dead neuron problem.
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        expansion_factor: int = EXPANSION_FACTOR,
        k: int = TOP_K,
        aux_coeff: float = AUX_COEFF,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_model * expansion_factor
        self.k = k
        self.aux_coeff = aux_coeff

        self.encoder = nn.Linear(d_model, self.d_hidden, bias=True)
        self.decoder = nn.Linear(self.d_hidden, d_model, bias=True)
        self.b_pre = nn.Parameter(torch.zeros(d_model))

        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.decoder.weight, nonlinearity="linear")
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
        self._normalize_decoder()

    def _normalize_decoder(self):
        """Keep decoder columns at unit norm for interpretability."""
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=0, keepdim=True)
            self.decoder.weight.data = self.decoder.weight.data / (norms + 1e-8)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """TopK encoding: activate exactly k features per token."""
        pre = self.encoder(x - self.b_pre)              # [B, d_hidden]
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)  # [B, k]
        hidden = torch.zeros_like(pre)
        hidden.scatter_(-1, topk_idx, topk_vals.clamp(min=0))
        return hidden

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.decoder(hidden) + self.b_pre

    def forward(self, x: torch.Tensor) -> SAEOutput:
        pre = self.encoder(x - self.b_pre)              # [B, d_hidden]
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)  # [B, k]
        hidden = torch.zeros_like(pre)
        hidden.scatter_(-1, topk_idx, topk_vals.clamp(min=0))
        reconstructed = self.decode(hidden)
        reconstruction_loss = F.mse_loss(reconstructed, x)
        l0_sparsity = (hidden > 0).float().sum(dim=-1).mean()

        # Auxiliary loss (Gao et al. 2024): give non-top-K features a gradient by
        # having them reconstruct the residual. Prevents dead features without
        # affecting the quality of top-K features (coeff is small, residual detached).
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.training and self.aux_coeff > 0:
            pre_aux = pre.clone()
            pre_aux.scatter_(-1, topk_idx, -1e9)          # mask out main top-K
            aux_vals, aux_idx = pre_aux.topk(self.k, dim=-1)
            aux_hidden = torch.zeros_like(pre)
            aux_hidden.scatter_(-1, aux_idx, aux_vals.clamp(min=0))
            residual = (x - reconstructed).detach()       # no gradient to main path
            aux_recon = self.decode(aux_hidden)
            aux_loss = F.mse_loss(aux_recon, residual) * self.aux_coeff

        return SAEOutput(
            reconstructed=reconstructed,
            hidden=hidden,
            loss=reconstruction_loss + aux_loss,
            reconstruction_loss=reconstruction_loss,
            sparsity_loss=aux_loss,
            l0_sparsity=l0_sparsity,
        )

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "d_model": self.d_model,
            "d_hidden": self.d_hidden,
            "k": self.k,
            "aux_coeff": self.aux_coeff,
        }, path)
        print(f"Saved SAE to {path}")

    @classmethod
    def load(cls, path: Path) -> "SparseAutoencoder":
        checkpoint = torch.load(path, weights_only=False)
        sae = cls(
            d_model=checkpoint["d_model"],
            expansion_factor=checkpoint["d_hidden"] // checkpoint["d_model"],
            k=checkpoint["k"],
            aux_coeff=checkpoint.get("aux_coeff", AUX_COEFF),
        )
        sae.load_state_dict(checkpoint["state_dict"])
        return sae
