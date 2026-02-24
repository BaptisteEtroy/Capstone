#!/usr/bin/env python3
"""
Feature Analysis & Steering
============================
Analyze learned SAE features and steer GPT-2 behavior by intervening on features.

Usage:
    python steer.py --analyze              # Analyze top features
    python steer.py --steer "Hello"        # Generate with feature steering
    python steer.py --interactive          # Interactive steering mode
"""

import os
# Fix MPS compatibility issue with TransformerLens generate
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass
import argparse
import json

from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

# Import SAE class from main
from main import SparseAutoencoder, D_MODEL, TARGET_LAYER, HOOK_TYPE, get_device

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = Path("outputs")
SAE_PATH = OUTPUT_DIR / "sae.pt"
FEATURES_PATH = OUTPUT_DIR / "features.json"


# =============================================================================
# Feature Analysis
# =============================================================================

def load_sae_and_model(device: Optional[str] = None) -> Tuple[SparseAutoencoder, HookedTransformer]:
    """Load trained SAE and GPT-2."""
    device = device or get_device()
    
    print(f"Loading SAE from {SAE_PATH}...")
    sae = SparseAutoencoder.load(SAE_PATH).to(device).eval()
    print(f"  Features: {sae.d_hidden}")
    
    print("Loading GPT-2...")
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    
    return sae, model


def analyze_feature(
    feature_idx: int,
    sae: SparseAutoencoder,
    model: HookedTransformer,
    top_k: int = 10,
) -> dict:
    """
    Analyze a single feature using VocabProj method.
    
    Returns the tokens this feature most strongly promotes in the output.
    """
    device = next(sae.parameters()).device
    
    # Get decoder vector for this feature
    decoder_vec = sae.decoder.weight[:, feature_idx]  # [d_model]
    
    # Project onto vocabulary (VocabProj method)
    unembed = model.W_U  # [d_model, vocab_size]
    logits = decoder_vec @ unembed  # [vocab_size]
    
    # Get top tokens
    top_values, top_indices = logits.topk(top_k)
    
    top_tokens = []
    for val, idx in zip(top_values, top_indices):
        token = model.tokenizer.decode([idx.item()])
        top_tokens.append({
            "token": repr(token),
            "logit": val.item(),
        })
    
    # Get bottom tokens (what it suppresses)
    bottom_values, bottom_indices = logits.topk(top_k, largest=False)
    bottom_tokens = []
    for val, idx in zip(bottom_values, bottom_indices):
        token = model.tokenizer.decode([idx.item()])
        bottom_tokens.append({
            "token": repr(token),
            "logit": val.item(),
        })
    
    return {
        "feature_idx": feature_idx,
        "promotes": top_tokens,
        "suppresses": bottom_tokens,
    }


def print_feature_analysis(analysis: dict):
    """Pretty print feature analysis (VocabProj only - computed on-the-fly)."""
    print(f"\n{'='*60}")
    print(f"  Feature {analysis['feature_idx']}")
    print(f"{'='*60}")
    
    print("\n  PROMOTES (top tokens this feature pushes toward):")
    for t in analysis["promotes"][:5]:
        print(f"    {t['token']:20s}  logit: {t['logit']:+.2f}")
    
    print("\n  SUPPRESSES (tokens this feature pushes away from):")
    for t in analysis["suppresses"][:5]:
        print(f"    {t['token']:20s}  logit: {t['logit']:+.2f}")


def print_saved_feature(feature_data: dict):
    """
    Pretty print feature analysis from saved features.json.
    Shows both MaxAct (input-centric) and VocabProj (output-centric).
    """
    print(f"\n{'='*60}")
    print(f"  Feature {feature_data['index']}")
    print(f"  Frequency: {feature_data['frequency']*100:.1f}%  |  Max activation: {feature_data['max']:.1f}")
    print(f"{'='*60}")
    
    # Input-centric: MaxAct - what triggers this feature
    print("\n  INPUT-CENTRIC (MaxAct): Tokens that ACTIVATE this feature")
    print("  (What concepts/patterns does this feature detect?)")
    if "max_activating_tokens" in feature_data and feature_data["max_activating_tokens"]:
        for t in feature_data["max_activating_tokens"][:5]:
            tok = repr(t["token"]) if isinstance(t, dict) else repr(t)
            act = t.get("activation", 0) if isinstance(t, dict) else 0
            print(f"    {tok:25s}  activation: {act:.1f}")
    else:
        print("    (No MaxAct data available)")
    
    # Output-centric: VocabProj - what this feature does
    print("\n  OUTPUT-CENTRIC (VocabProj): Tokens this feature PROMOTES")
    print("  (What effect does this feature have on predictions?)")
    if "vocab_projection" in feature_data:
        logits = feature_data.get("vocab_projection_logits", [0] * len(feature_data["vocab_projection"]))
        for tok, logit in zip(feature_data["vocab_projection"][:5], logits[:5]):
            print(f"    {repr(tok):25s}  logit: {logit:+.2f}")


def find_features_for_concept(
    concept_tokens: List[str],
    sae: SparseAutoencoder,
    model: HookedTransformer,
    top_k: int = 10,
) -> List[Tuple[int, float]]:
    """
    Find features that most strongly promote given tokens.
    
    Useful for finding features related to a concept (e.g., "love", "hate").
    """
    device = next(sae.parameters()).device
    
    # Get token IDs
    token_ids = [model.tokenizer.encode(t)[0] for t in concept_tokens]
    
    # Get decoder weights and unembed
    decoder = sae.decoder.weight  # [d_model, d_hidden]
    unembed = model.W_U  # [d_model, vocab_size]
    
    # Project all features onto vocabulary
    all_logits = decoder.T @ unembed  # [d_hidden, vocab_size]
    
    # Sum logits for target tokens
    target_logits = all_logits[:, token_ids].sum(dim=1)  # [d_hidden]
    
    # Get top features
    top_values, top_indices = target_logits.topk(top_k)
    
    return [(idx.item(), val.item()) for idx, val in zip(top_indices, top_values)]


# =============================================================================
# Feature Steering
# =============================================================================

@dataclass
class SteeringConfig:
    """Configuration for feature steering."""
    feature_idx: int
    strength: float = 5.0  # Multiplier for feature activation (negative to suppress)
    

def create_steering_hook(
    sae: SparseAutoencoder,
    configs: List[SteeringConfig],
) -> Callable:
    """
    Create a hook function that modifies activations based on SAE features.
    
    The hook:
    1. Encodes activations to feature space
    2. Amplifies/suppresses specified features
    3. Decodes back to activation space
    """
    device = next(sae.parameters()).device
    
    def steering_hook(activations: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        # activations: [batch, seq_len, d_model]
        original_shape = activations.shape
        flat_acts = activations.reshape(-1, activations.shape[-1])  # [batch*seq, d_model]
        
        # Encode to features
        with torch.no_grad():
            features = sae.encode(flat_acts)  # [batch*seq, d_hidden]
        
        # Modify specified features
        for config in configs:
            if config.strength > 0:
                # Amplify: multiply existing activation
                features[:, config.feature_idx] *= config.strength
            else:
                # Suppress: set to zero (or negative multiplier)
                features[:, config.feature_idx] *= abs(config.strength)
        
        # Decode back
        with torch.no_grad():
            modified_acts = sae.decode(features)  # [batch*seq, d_model]
        
        return modified_acts.reshape(original_shape)
    
    return steering_hook


def generate_with_steering(
    model: HookedTransformer,
    sae: SparseAutoencoder,
    prompt: str,
    steering_configs: List[SteeringConfig],
    max_tokens: int = 50,
    temperature: float = 0.7,
) -> str:
    """
    Generate text with feature steering applied.
    
    Args:
        model: GPT-2 model
        sae: Trained SAE
        prompt: Input prompt
        steering_configs: List of features to amplify/suppress
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated text
    """
    device = next(sae.parameters()).device
    hook_point = f"blocks.{TARGET_LAYER}.hook_{HOOK_TYPE}"
    
    # Create steering hook
    steering_hook = create_steering_hook(sae, steering_configs)
    
    # Generate with hook
    with model.hooks([(hook_point, steering_hook)]):
        output = model.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
        )
    
    # Handle both tensor and string returns
    if isinstance(output, str):
        return output
    return model.tokenizer.decode(output[0].tolist())


def compare_generations(
    model: HookedTransformer,
    sae: SparseAutoencoder,
    prompt: str,
    feature_idx: int,
    strength: float = 5.0,
    max_tokens: int = 50,
):
    """Compare generation with and without feature steering."""
    
    print(f"\n{'='*60}")
    print(f"  Steering Comparison: Feature {feature_idx}")
    print(f"  Strength: {strength:+.1f}")
    print(f"{'='*60}")
    
    print(f"\nPrompt: {repr(prompt)}")
    
    # Baseline (no steering)
    print(f"\n--- BASELINE (no steering) ---")
    baseline = model.generate(
        prompt,
        max_new_tokens=max_tokens,
        temperature=0.7,
        do_sample=True,
    )
    # Handle both tensor and string returns
    if isinstance(baseline, str):
        print(baseline)
    else:
        print(model.tokenizer.decode(baseline[0].tolist()))
    
    # With steering
    print(f"\n--- WITH STEERING (feature {feature_idx} × {strength}) ---")
    config = SteeringConfig(feature_idx=feature_idx, strength=strength)
    steered = generate_with_steering(model, sae, prompt, [config], max_tokens)
    print(steered)


# =============================================================================
# Interactive Mode
# =============================================================================

def interactive_mode(sae: SparseAutoencoder, model: HookedTransformer):
    """Interactive feature exploration and steering."""
    
    print("\n" + "="*60)
    print("  Interactive Feature Steering")
    print("="*60)
    print("\nCommands:")
    print("  analyze <idx>          - Analyze feature")
    print("  find <token1> <token2> - Find features for tokens")
    print("  steer <idx> <strength> - Set steering (e.g., 'steer 42 5.0')")
    print("  clear                  - Clear all steering")
    print("  gen <prompt>           - Generate with current steering")
    print("  compare <idx> <prompt> - Compare with/without feature")
    print("  quit                   - Exit")
    
    active_steering: List[SteeringConfig] = []
    
    while True:
        try:
            cmd = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
            
        if not cmd:
            continue
            
        parts = cmd.split(maxsplit=1)
        action = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if action == "quit":
            print("Goodbye!")
            break
            
        elif action == "analyze":
            try:
                idx = int(args)
                analysis = analyze_feature(idx, sae, model)
                print_feature_analysis(analysis)
            except ValueError:
                print("Usage: analyze <feature_idx>")
                
        elif action == "find":
            tokens = args.split()
            if tokens:
                results = find_features_for_concept(tokens, sae, model)
                print(f"\nTop features for {tokens}:")
                for idx, score in results:
                    print(f"  Feature {idx:4d}: score {score:.2f}")
            else:
                print("Usage: find <token1> <token2> ...")
                
        elif action == "steer":
            try:
                steer_parts = args.split()
                idx = int(steer_parts[0])
                strength = float(steer_parts[1]) if len(steer_parts) > 1 else 5.0
                active_steering.append(SteeringConfig(idx, strength))
                print(f"Added: Feature {idx} × {strength}")
                print(f"Active steering: {[(c.feature_idx, c.strength) for c in active_steering]}")
            except (ValueError, IndexError):
                print("Usage: steer <feature_idx> [strength]")
                
        elif action == "clear":
            active_steering.clear()
            print("Cleared all steering")
            
        elif action == "gen":
            if not args:
                print("Usage: gen <prompt>")
            else:
                if active_steering:
                    output = generate_with_steering(model, sae, args, active_steering)
                else:
                    output = model.generate(args, max_new_tokens=50, temperature=0.7, do_sample=True)
                    if not isinstance(output, str):
                        output = model.tokenizer.decode(output[0].tolist())
                print(f"\n{output}")
                
        elif action == "compare":
            try:
                compare_parts = args.split(maxsplit=1)
                idx = int(compare_parts[0])
                prompt = compare_parts[1] if len(compare_parts) > 1 else "The meaning of life is"
                compare_generations(model, sae, prompt, idx)
            except (ValueError, IndexError):
                print("Usage: compare <feature_idx> <prompt>")
                
        else:
            print(f"Unknown command: {action}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Feature Analysis & Steering")
    parser.add_argument("--analyze", action="store_true", help="Analyze top features")
    parser.add_argument("--feature", type=int, help="Analyze specific feature")
    parser.add_argument("--find", nargs="+", help="Find features for tokens")
    parser.add_argument("--steer", type=str, help="Prompt to steer")
    parser.add_argument("--feature-idx", type=int, default=0, help="Feature to steer")
    parser.add_argument("--strength", type=float, default=5.0, help="Steering strength")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--device", type=str, help="Device")
    args = parser.parse_args()
    
    device = args.device or get_device()
    sae, model = load_sae_and_model(device)
    
    if args.interactive:
        interactive_mode(sae, model)
        
    elif args.analyze:
        # Load saved feature analysis (with both MaxAct and VocabProj)
        if FEATURES_PATH.exists():
            with open(FEATURES_PATH) as f:
                features = json.load(f)
            print(f"\nDisplaying top {min(10, len(features))} features (MaxAct + VocabProj)...")
            print("="*60)
            print("  MaxAct (input-centric): What tokens TRIGGER each feature")
            print("  VocabProj (output-centric): What tokens each feature PROMOTES")
            print("="*60)
            for feat in features[:10]:
                print_saved_feature(feat)
        else:
            print("No features.json found. Computing VocabProj for first 10 features...")
            for i in range(10):
                analysis = analyze_feature(i, sae, model)
                print_feature_analysis(analysis)
                
    elif args.feature is not None:
        analysis = analyze_feature(args.feature, sae, model)
        print_feature_analysis(analysis)
        
    elif args.find:
        results = find_features_for_concept(args.find, sae, model)
        print(f"\nTop features for {args.find}:")
        for idx, score in results:
            print(f"  Feature {idx:4d}: score {score:.2f}")
            
    elif args.steer:
        compare_generations(model, sae, args.steer, args.feature_idx, args.strength)
        
    else:
        # Default: interactive mode
        interactive_mode(sae, model)


if __name__ == "__main__":
    main()
