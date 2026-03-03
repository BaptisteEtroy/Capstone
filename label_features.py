#!/usr/bin/env python3
"""
Features are only labeled if:
1. MaxAct and VocabProj tokens are semantically coherent
2. The pattern is interpretable/meaningful

Usage:
    python label_features.py              # Label features
    python label_features.py --dry-run    # Preview without API calls
"""

import json
import os
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from config import OUTPUT_DIR, FEATURES_PATH


@dataclass
class LabeledFeature:
    """A feature with its auto-generated label."""
    index: int
    label: str
    confidence: str  # "high", "medium", "low"
    max_act_tokens: List[str]
    vocab_proj_tokens: List[str]
    reasoning: str


# =============================================================================
# OpenAI API
# =============================================================================

def call_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call OpenAI API (using v1.x SDK)."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in mechanistic interpretability of neural networks. "
                        "Your task is to label features extracted from a Sparse Autoencoder (SAE) trained on GPT-2 Medium. "
                        "MaxAct tokens are real input tokens that triggered the feature (what it detects). "
                        "VocabProj tokens are output tokens the feature promotes (what it causes). "
                        "Give specific, concrete labels — avoid vague terms like 'language patterns', 'text features', 'linguistic', or 'general concepts'."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        return response.choices[0].message.content.strip()
    except ImportError:
        raise ImportError("Please install openai: pip install 'openai>=1.0.0'")
    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {e}")


# =============================================================================
# Feature Labeling Logic (Batched for efficiency)
# =============================================================================

BATCH_SIZE = 50  # Features per API call (can go up to ~100)


def extract_feature_tokens(feature: Dict[str, Any]) -> tuple:
    """Extract MaxAct and VocabProj tokens from a feature."""
    max_act_tokens = []
    if "max_activating_tokens" in feature:
        for item in feature["max_activating_tokens"][:5]:
            if isinstance(item, dict):
                max_act_tokens.append((item.get("token", ""), item.get("activation", 0.0), item.get("context", "")))
            else:
                max_act_tokens.append((str(item), 0.0, ""))

    vocab_proj_tokens = feature.get("vocab_projection", [])[:5]
    return max_act_tokens, vocab_proj_tokens


def build_batch_prompt(features_batch: List[Dict[str, Any]]) -> str:
    """Build a prompt for labeling multiple features at once."""

    features_text = ""
    for feature in features_batch:
        max_act_tokens, vocab_proj_tokens = extract_feature_tokens(feature)

        # Format MaxAct entries with activation values and optional context
        max_act_parts = []
        for tok, act, ctx in max_act_tokens:
            entry = f"'{tok}' (act={act:.1f})"
            if ctx:
                entry += f" ctx: {ctx}"
            max_act_parts.append(entry)

        features_text += f"""
---
FEATURE {feature['index']} (freq: {feature['frequency']*100:.1f}%)
MaxAct (triggers): {', '.join(max_act_parts) if max_act_parts else 'No data'}
VocabProj (promotes): {', '.join(repr(t) for t in vocab_proj_tokens) if vocab_proj_tokens else 'No data'}
---
"""

    prompt = f"""Analyze these neural network features from a Sparse Autoencoder trained on GPT-2 Medium.

For each feature:
- MaxAct = tokens (with activation strength) that TRIGGER the feature (input-centric)
- VocabProj = tokens the feature PROMOTES in output (output-centric)

Rules:
1. Label ONLY if MaxAct and VocabProj are semantically coherent (monosemantic feature).
2. Use "UNLABELED" if tokens seem random, incoherent, or too mixed.
3. Be SPECIFIC: use concrete labels like "Python function definitions", "past-tense verbs", "US state names", "sports scores".
4. AVOID vague labels like "language patterns", "text features", "linguistic structures", "general concepts".
5. High confidence = both MaxAct and VocabProj clearly share the same theme.
6. Medium confidence = mostly coherent but some noise.
7. Low confidence = weak signal, leaning toward UNLABELED.

{features_text}

Respond with EXACTLY one line per feature in this format:
FEATURE <id>: <LABEL or UNLABELED> | <high/medium/low> | <short reasoning>

Examples:
FEATURE 123: Python function definitions | high | MaxAct has 'def', 'return', 'class'; VocabProj promotes 'def', 'import'
FEATURE 456: UNLABELED | low | Tokens appear random with no clear pattern
FEATURE 789: US state names | medium | MaxAct has 'Texas', 'Ohio', VocabProj partially related
FEATURE 101: past-tense irregular verbs | high | MaxAct has 'went', 'knew', 'said'; VocabProj promotes similar past-tense forms"""

    return prompt


def parse_batch_response(response: str, features_batch: List[Dict]) -> List[tuple]:
    """Parse the batched LLM response."""
    results = []
    feature_ids = {f["index"] for f in features_batch}
    
    for line in response.split("\n"):
        line = line.strip()
        if not line.startswith("FEATURE"):
            continue
        
        try:
            # Parse: FEATURE 123: label | confidence | reasoning
            parts = line.split(":", 1)
            if len(parts) < 2:
                continue
            
            feature_id = int(parts[0].replace("FEATURE", "").strip())
            if feature_id not in feature_ids:
                continue
            
            rest = parts[1].strip()
            segments = rest.split("|")
            
            label = segments[0].strip() if len(segments) > 0 else "UNLABELED"
            confidence = segments[1].strip().lower() if len(segments) > 1 else "low"
            reasoning = segments[2].strip() if len(segments) > 2 else ""
            
            results.append((feature_id, label, confidence, reasoning))
        except (ValueError, IndexError):
            continue
    
    return results


def label_features(
    features: List[Dict],
    model: str = "gpt-4o-mini",
    dry_run: bool = False,
    max_features: int = 1000,
    batch_size: int = BATCH_SIZE,
) -> List[LabeledFeature]:
    """Label features using OpenAI with batched API calls."""
    
    features_to_label = features[:max_features]
    num_batches = (len(features_to_label) + batch_size - 1) // batch_size
    
    print(f"\nLabeling {len(features_to_label)} features using OpenAI ({model})...")
    print(f"  Batch size: {batch_size} features per API call")
    print(f"  Total API calls: {num_batches}")
    
    labeled_features = []
    unlabeled_count = 0
    
    # Create feature lookup for token extraction
    feature_lookup = {f["index"]: f for f in features_to_label}
    
    for i in tqdm(range(0, len(features_to_label), batch_size), desc="Batches"):
        batch = features_to_label[i:i+batch_size]
        
        if dry_run:
            for feature in batch:
                max_act, vocab_proj = extract_feature_tokens(feature)
                print(f"\n--- Feature {feature['index']} ---")
                print(f"MaxAct: {max_act[:3]}")
                print(f"VocabProj: {vocab_proj[:3]}")
            continue
        
        try:
            prompt = build_batch_prompt(batch)
            response = call_openai(prompt, model)
            results = parse_batch_response(response, batch)
            
            # Process results
            labeled_ids = set()
            for feature_id, label, confidence, reasoning in results:
                if label.upper() != "UNLABELED":
                    max_act, vocab_proj = extract_feature_tokens(feature_lookup[feature_id])
                    labeled_features.append(LabeledFeature(
                        index=feature_id,
                        label=label,
                        confidence=confidence,
                        max_act_tokens=max_act,
                        vocab_proj_tokens=vocab_proj,
                        reasoning=reasoning,
                    ))
                    labeled_ids.add(feature_id)
                else:
                    unlabeled_count += 1
                    labeled_ids.add(feature_id)
            
            # Count features that weren't in response as unlabeled
            unlabeled_count += len(batch) - len(labeled_ids)
                
        except Exception as e:
            print(f"\nError labeling batch starting at feature {batch[0]['index']}: {e}")
            unlabeled_count += len(batch)
    
    if not dry_run:
        print(f"\n  Labeled: {len(labeled_features)} features")
        print(f"  Unlabeled (not monosemantic): {unlabeled_count} features")
    
    return labeled_features


def save_labeled_features(
    labeled_features: List[LabeledFeature],
    output_path: Path,
):
    """Save labeled features to JSON."""
    data = [
        {
            "index": f.index,
            "label": f.label,
            "confidence": f.confidence,
            "max_act_tokens": f.max_act_tokens,
            "vocab_proj_tokens": f.vocab_proj_tokens,
            "reasoning": f.reasoning,
        }
        for f in labeled_features
    ]
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSaved {len(labeled_features)} labeled features to {output_path}")


def print_labeled_features(labeled_features: List[LabeledFeature]):
    """Pretty print labeled features."""
    print("\n" + "="*60)
    print("  LABELED MONOSEMANTIC FEATURES")
    print("="*60)
    
    # Group by confidence
    high = [f for f in labeled_features if f.confidence == "high"]
    medium = [f for f in labeled_features if f.confidence == "medium"]
    low = [f for f in labeled_features if f.confidence == "low"]
    
    for conf_level, features in [("HIGH CONFIDENCE", high), ("MEDIUM CONFIDENCE", medium), ("LOW CONFIDENCE", low)]:
        if not features:
            continue
        print(f"\n  {conf_level} ({len(features)} features)")
        print("  " + "-"*40)
        for f in features[:10]:  # Show top 10 per category
            print(f"    Feature {f.index:4d}: {f.label}")
            print(f"                  → {f.reasoning[:60]}...")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Label SAE features using OpenAI")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="OpenAI model to use (default: gpt-4o-mini)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview features without API calls")
    parser.add_argument("--max-features", type=int, default=1000,
                        help="Maximum features to label")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Features per API call (default: {BATCH_SIZE})")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path")
    args = parser.parse_args()
    
    # Load features
    if not FEATURES_PATH.exists():
        print(f"Error: {FEATURES_PATH} not found. Run main.py first.")
        return
    
    with open(FEATURES_PATH) as f:
        features = json.load(f)
    
    print(f"Loaded {len(features)} features from {FEATURES_PATH}")
    
    # Check API key
    if not args.dry_run:
        if not os.environ.get("OPENAI_API_KEY"):
            print("\nError: OPENAI_API_KEY not found.")
            print("Add your key to the .env file:")
            print("  OPENAI_API_KEY=sk-your-key-here")
            return
    
    # Label features
    labeled_features = label_features(
        features,
        model=args.model,
        dry_run=args.dry_run,
        max_features=args.max_features,
        batch_size=args.batch_size,
    )
    
    if args.dry_run:
        return
    
    # Print results
    print_labeled_features(labeled_features)
    
    # Save results
    output_path = Path(args.output) if args.output else OUTPUT_DIR / "labeled_features.json"
    save_labeled_features(labeled_features, output_path)


if __name__ == "__main__":
    main()
