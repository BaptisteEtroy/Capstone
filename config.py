"""
configuration and shared components for the SAE interpretability pipeline.
all hyperparameters live here — change things here, not in other files.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict


# configuration constants

# llama 3.2 1b instruct-tuned
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
D_MODEL = 2048

# layer 12 sits in the middle of the network and has rich semantic features
TARGET_LAYER = 12
HOOK_TYPE = "resid_post"

# batchTopK SAE (Bussmann et al. 2024), batch-level sparsity allocation.
# 4x expansion = 8192 features, stable at ~15M tokens.
# tried 8x (16384) but needed 50M+ tokens to avoid 80%+ dead features, so reverted.
# k=64 is the canonical value for 4x; aux_k=256 gives more gradient to dead features.
EXPANSION_FACTOR = 4    # d_hidden = 2048 * 4 = 8192 features
TOP_K = 64              # avg features active per token
AUX_COEFF = 1 / 32     # auxiliary loss coefficient (Gao et al. 2024)
AUX_K = 256            # features used in auxiliary loss (scaled up from 128 at 8x)

# training
LEARNING_RATE = 1e-4
NUM_EPOCHS = 8
BATCH_SIZE = 4096
NUM_SAMPLES = 500_000

# output
MEDICAL_OUTPUT_DIR = Path("medical_outputs")


# utility functions

def get_device() -> str:
    """picks the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# data classes

@dataclass
class SAEOutput:
    """bundles everything the SAE forward pass returns."""
    reconstructed: torch.Tensor
    hidden: torch.Tensor
    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    sparsity_loss: torch.Tensor
    l0_sparsity: torch.Tensor


@dataclass
class MaxActExample:
    """one token example that strongly activates a given feature."""
    token: str
    token_id: int
    activation: float
    context: str = ""
    global_token_idx: int = -1
    source_id: str = ""


@dataclass
class FeatureInfo:
    """
    holds everything we know about a learned feature after analysis.

    maxact tells us what inputs trigger it, vocabproj tells us what
    outputs it promotes, and position stats tell us if it's positional.
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
    # shannon entropy of the maxact token distribution (bits).
    # low entropy means the feature fires on a tight cluster of similar tokens (monosemantic).
    maxact_entropy: float = 0.0
    # fraction of maxact examples from each training dataset
    source_breakdown: Dict[str, float] = field(default_factory=dict)
    # tokenchange: inject the decoder direction and measure how next-token probs shift.
    # more reliable than vocabproj since it goes through the actual downstream layers.
    token_change_promoted: List[str] = field(default_factory=list)
    token_change_suppressed: List[str] = field(default_factory=list)
    token_change_kl: float = 0.0       # mean kl(steered || baseline) across contexts
    token_change_kl_std: float = 0.0   # std of kl across contexts (consistency signal)
    token_change_n_contexts: int = 0   # how many maxact examples were used


# sparse autoencoder model

class SparseAutoencoder(nn.Module):
    """
    batchTopK sparse autoencoder (Bussmann et al. 2024 + Gao et al. 2024).

    the key difference from standard topk: instead of selecting k features
    per sample, we select batch_size*k activations across the whole batch.
    this lets some samples use more features than others and gives more
    gradient signal to each feature, which really helps with dead features.

    falls back to per-sample topk at inference for deterministic outputs.

    init follows Gao et al. 2024:
      - encoder.weight = decoder.weight^T (parallel directions)
      - b_pre = mean of a data sample (approximates geometric median)
      - decoder columns kept at unit norm throughout training
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

        # init encoder as transpose of decoder so their directions are aligned.
        # this is the recommended init from the openai topk sae paper.
        nn.init.kaiming_uniform_(self.decoder.weight, nonlinearity="linear")
        self._normalize_decoder()
        self.encoder.weight.data = self.decoder.weight.data.T.clone()
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def init_bias_from_data(self, sample: torch.Tensor):
        """
        sets b_pre to the mean of a data sample, which approximates the geometric
        median (Gao et al. 2024). this centres the encoder input so pre-activations
        start near zero instead of being dominated by the data mean.

        call this once before training with a representative batch of activations.
        """
        with torch.no_grad():
            self.b_pre.data = sample.float().mean(dim=0).to(self.b_pre.device)
        # re-sync encoder weights after b_pre is set
        self.encoder.weight.data = self.decoder.weight.data.T.clone()

    def _normalize_decoder(self):
        """keeps decoder columns at unit norm, which is important for interpretability."""
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=0, keepdim=True)
            self.decoder.weight.data = self.decoder.weight.data / (norms + 1e-8)

    def _batch_topk(self, pre: torch.Tensor):
        """
        selects the top batch_size*k activations across the whole batch (not per sample).

        returns (hidden, fired_mask) where hidden has batch_size*k non-zero values
        total, and fired_mask marks which (sample, feature) pairs fired.
        """
        B = pre.shape[0]
        flat = pre.reshape(-1)
        total_k = B * self.k
        # use kth-largest as threshold, handles ties conservatively
        threshold = flat.topk(total_k).values[-1]
        fired_mask = pre >= threshold
        hidden = (pre * fired_mask.float()).clamp(min=0)
        return hidden, fired_mask

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        encodes x using batchTopK during training, per-sample topk during inference.
        inference uses per-sample topk so outputs are deterministic regardless of batch size.
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

        # auxiliary loss (Gao et al. 2024): unfired features get gradient by trying
        # to reconstruct the residual error. aux_k=256 is much larger than k=64
        # so more dead features get signal each step.
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
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
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
