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

# Model: GPT-2 Medium (355M params, richer representations)
MODEL_NAME = "gpt2-medium"
D_MODEL = 1024
N_LAYERS = 24

# Layer: Middle layer for semantic features (lit review: features are most interpretable here)
TARGET_LAYER = 12  # Middle of 24 layers
HOOK_TYPE = "resid_post"

# SAE: Standard architecture with L1 regularization (lit review: most effective method)
EXPANSION_FACTOR = 8  # Hidden dim = 1024 * 8 = 8192 features
L1_COEFFICIENT = 40   # Compromise: 30 gave L0=354 (too high), 50 gave 36% dead (too many)

# Training
LEARNING_RATE = 1e-4  # convergence
NUM_EPOCHS = 5        # Extra epochs give ghost grads more time to revive dead neurons
BATCH_SIZE = 4096
NUM_SAMPLES = 50_000  # 50k samples (fits in 24GB RAM)

# Output
OUTPUT_DIR = Path("outputs")
SAE_PATH = OUTPUT_DIR / "sae.pt"
FEATURES_PATH = OUTPUT_DIR / "features.json"

# Ghost gradients
GHOST_GRAD_COEFFICIENT = 0.25  # Increased from 0.1: stronger revival signal for dead neurons
GHOST_DEAD_THRESHOLD = 1e-5   # Consistent with summary dead-neuron counting (feature_freq < 1e-5)


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
    
    def get_dead_neurons(self, threshold: float = GHOST_DEAD_THRESHOLD) -> torch.Tensor:
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
