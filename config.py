"""
Configuration & Shared Components
==================================
Central configuration and reusable classes for the SAE interpretability pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict


# =============================================================================
# Configuration Constants
# =============================================================================

# Model: Llama 3.2 1B (instruction-tuned)
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
D_MODEL = 2048

# Layer: Middle layer for semantic features
TARGET_LAYER = 12
HOOK_TYPE = "resid_post"

# SAE: BatchTopK architecture (Bussmann et al. 2024) — batch-level sparsity allocation.
# Expansion 4x gives 8192 features — proven stable at this token budget (~15M tokens).
# 8x (16384) requires ~50M+ tokens to avoid 80%+ dead features; reverted.
# TOP_K 64: avg features active per token (canonical 4x value; 96 was for 8x).
# AUX_COEFF 1/32 per OpenAI scaling paper.
# AUX_K 256: larger with 4x dict — more gradient signal to dead features (was 128 at 8x).
EXPANSION_FACTOR = 4    # d_hidden = 2048 * 4 = 8192 features
TOP_K = 64              # avg features active per token
AUX_COEFF = 1 / 32     # auxiliary loss coefficient (Gao et al. 2024)
AUX_K = 256            # features used in auxiliary loss (scaled up from 128 at 8x)

# Training
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
BATCH_SIZE = 4096
NUM_SAMPLES = 500_000   

# Output
MEDICAL_OUTPUT_DIR = Path("medical_outputs")


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
class MaxActExample:
    """A single example of a token that strongly activates a feature."""
    token: str
    token_id: int
    activation: float
    context: str = ""
    global_token_idx: int = -1
    source_id: str = ""


@dataclass
class FeatureInfo:
    """
    Information about a learned feature using both interpretability methods.

    Input-centric (MaxAct): What inputs trigger this feature?
    Output-centric (VocabProj): What outputs does this feature promote?
    Positional: Does this feature fire at specific positions in the sequence?
    """
    index: int
    activation_frequency: float
    mean_activation: float
    max_activation: float
    max_activating_tokens: List[MaxActExample]
    vocab_projection: List[str]
    vocab_projection_logits: List[float]
    position_mean: float = 0.0
    position_std: float = 0.0
    # Monosemanticity proxy: Shannon entropy of MaxAct token distribution (bits).
    # Low entropy = feature fires on a tight token cluster = more monosemantic.
    maxact_entropy: float = 0.0
    # Fraction of MaxAct examples that came from each training dataset.
    # Keys are source_id strings (e.g. "medmcqa", "pubmed_qa", "pubmed_abs").
    source_breakdown: Dict[str, float] = field(default_factory=dict)
    # TokenChange (causal output-centric): inject decoder direction via hook, measure
    # how next-token distribution shifts. More rigorous than VocabProj at layer 12
    # because it runs the real forward pass through all downstream transformer layers.
    token_change_promoted: List[str] = field(default_factory=list)
    token_change_suppressed: List[str] = field(default_factory=list)
    token_change_kl: float = 0.0       # mean KL(steered || baseline) across contexts
    token_change_kl_std: float = 0.0   # std of KL across contexts (consistency signal)
    token_change_n_contexts: int = 0   # how many MaxAct examples were used


# =============================================================================
# Sparse Autoencoder Model
# =============================================================================

class SparseAutoencoder(nn.Module):
    """
    BatchTopK Sparse Autoencoder (Bussmann et al. 2024 + Gao et al. 2024).

    Improvement over standard TopK: selects the top batch_size*k activations
    across the ENTIRE BATCH rather than per-sample. This allows variable sparsity
    per sample (some examples naturally need more features) and provides more
    gradient signal to each feature, reducing dead features.

    During inference (eval mode) falls back to per-sample TopK for determinism.

    Encoder initialisation follows Gao et al. 2024:
      - encoder.weight initialised as decoder.weight^T  (parallel directions)
      - b_pre initialised from a sample of training data (geometric median approx)
      - decoder columns at unit norm throughout training
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        expansion_factor: int = EXPANSION_FACTOR,
        k: int = TOP_K,
        aux_coeff: float = AUX_COEFF,
        aux_k: int = AUX_K,
        use_batch_topk: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_model * expansion_factor
        self.k = k
        self.aux_coeff = aux_coeff
        self.aux_k = aux_k
        self.use_batch_topk = use_batch_topk

        self.encoder = nn.Linear(d_model, self.d_hidden, bias=True)
        self.decoder = nn.Linear(self.d_hidden, d_model, bias=True)
        self.b_pre = nn.Parameter(torch.zeros(d_model))

        # Decoder^T initialisation: encoder directions parallel to decoder directions.
        # This is the recommended init from the OpenAI TopK SAE paper.
        nn.init.kaiming_uniform_(self.decoder.weight, nonlinearity="linear")
        self._normalize_decoder()
        self.encoder.weight.data = self.decoder.weight.data.T.clone()
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def init_bias_from_data(self, sample: torch.Tensor):
        """
        Initialise b_pre to the mean of a data sample — an efficient approximation
        of the geometric median (Gao et al. 2024). Centres the encoder input so that
        pre-activations start near zero rather than being dominated by the data mean.

        Call this ONCE before training begins, with a representative batch of
        model activations.
        """
        with torch.no_grad():
            self.b_pre.data = sample.float().mean(dim=0).to(self.b_pre.device)
        # Re-sync encoder init after bias is set
        self.encoder.weight.data = self.decoder.weight.data.T.clone()

    def _normalize_decoder(self):
        """Keep decoder columns at unit norm for interpretability."""
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=0, keepdim=True)
            self.decoder.weight.data = self.decoder.weight.data / (norms + 1e-8)

    def _batch_topk(self, pre: torch.Tensor):
        """
        BatchTopK: select top batch_size*k activations across the whole batch.

        Returns (hidden, fired_mask) where hidden has exactly batch_size*k
        non-zero values total (not per-sample), and fired_mask is a bool tensor
        marking which (sample, feature) pairs fired.
        """
        B = pre.shape[0]
        flat = pre.reshape(-1)
        total_k = B * self.k
        # Use kth-largest as threshold (handles ties conservatively)
        threshold = flat.topk(total_k).values[-1]
        fired_mask = pre >= threshold
        hidden = (pre * fired_mask.float()).clamp(min=0)
        return hidden, fired_mask

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode: activate features via BatchTopK (training) or per-sample TopK (inference).
        Inference always uses per-sample TopK for deterministic, consistent outputs.
        """
        pre = self.encoder(x - self.b_pre)
        if self.use_batch_topk and self.training and x.dim() == 2 and x.shape[0] > 1:
            hidden, _ = self._batch_topk(pre)
        else:
            topk_vals, topk_idx = pre.topk(self.k, dim=-1)
            hidden = torch.zeros_like(pre)
            hidden.scatter_(-1, topk_idx, topk_vals.clamp(min=0))
        return hidden

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.decoder(hidden) + self.b_pre

    def forward(self, x: torch.Tensor) -> SAEOutput:
        pre = self.encoder(x - self.b_pre)

        # Main activation
        if self.use_batch_topk and self.training and x.shape[0] > 1:
            hidden, fired_mask = self._batch_topk(pre)
        else:
            topk_vals, topk_idx = pre.topk(self.k, dim=-1)
            hidden = torch.zeros_like(pre)
            hidden.scatter_(-1, topk_idx, topk_vals.clamp(min=0))
            fired_mask = hidden > 0

        reconstructed = self.decode(hidden)
        reconstruction_loss = F.mse_loss(reconstructed, x)
        l0_sparsity = (hidden > 0).float().sum(dim=-1).mean()

        # Auxiliary loss (Gao et al. 2024): give unfired features gradient by
        # having them reconstruct the residual error.
        # Uses aux_k=512 (much larger than main k) to give more dead features signal.
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.training and self.aux_coeff > 0:
            pre_aux = pre.detach().clone()
            pre_aux.masked_fill_(fired_mask, -1e9)
            k_aux = min(self.aux_k, self.d_hidden - int(fired_mask.long().sum(dim=-1).min().item()))
            if k_aux > 0:
                aux_vals, aux_idx = pre_aux.topk(k_aux, dim=-1)
                aux_hidden = torch.zeros_like(pre)
                aux_hidden.scatter_(-1, aux_idx, aux_vals.clamp(min=0))
                residual = (x - reconstructed).detach()
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
            "aux_k": self.aux_k,
            "use_batch_topk": self.use_batch_topk,
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
            aux_k=checkpoint.get("aux_k", AUX_K),
            use_batch_topk=checkpoint.get("use_batch_topk", True),
        )
        sae.load_state_dict(checkpoint["state_dict"])
        return sae
