#!/usr/bin/env python3
"""
Features are only labeled if:
1. MaxAct and VocabProj tokens are semantically coherent
2. The pattern is interpretable/meaningful

Usage:
    python label_features.py              # Label features (quality-filtered)
    python label_features.py --dry-run    # Preview without API calls
    python label_features.py --no-filter  # Skip quality filter, label all
    python label_features.py --min-freq 0.001 --max-freq 0.10  # Custom thresholds
"""

import json
import math
import os
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from config import MEDICAL_OUTPUT_DIR, MEDICAL_FEATURES_PATH


@dataclass
class LabeledFeature:
    """A feature with its auto-generated label."""
    index: int
    label: str
    confidence: str  # "high", "medium", "low"
    max_act_tokens: List[str]
    vocab_proj_tokens: List[str]
    reasoning: str
    quality_score: float = 0.0


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
                        "Your task is to label features extracted from a Sparse Autoencoder (SAE) trained on Llama 3.2 1B processing medical text. "
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
# Quality Filtering & Scoring
# =============================================================================

def compute_quality_score(feature: Dict[str, Any]) -> float:
    """
    Score a feature by its likelihood of being monosemantic (higher = better).

    Components:
    - freq_score:     Peaks at ~2% activation frequency. Penalizes always-on
                      (polysemantic) and near-dead features equally.
    - max_act_score:  More MaxAct examples → more evidence of what the feature detects.
    - vocab_score:    More VocabProj tokens → clearer output-centric signal.
    - act_score:      Higher mean activation → stronger, more distinctive feature.
    - diversity_score: MaxAct tokens should be diverse, not all the same token
                       (all-same → likely positional/syntactic catch-all).

    freq_score is double-weighted as the strongest signal for monosemanticity.
    """
    freq = feature.get("frequency", 0)

    # Frequency score: log-scale bell curve peaking at 2% (optimal monosemantic range)
    # log10(0.02) ≈ -1.7; penalize if more than 1.5 log-units away
    freq_score = max(0.0, 1.0 - abs(math.log10(max(freq, 1e-6)) - math.log10(0.02)) / 1.5)

    max_act_list = feature.get("max_activating_tokens", [])
    vocab_list = feature.get("vocab_projection", [])

    # Coverage scores: reward more data, capped at 10 examples
    max_act_score = min(len(max_act_list), 10) / 10
    vocab_score = min(len(vocab_list), 10) / 10

    # Activation strength: stronger = more distinctive
    mean_act = feature.get("mean_activation", 0)
    act_score = min(mean_act / 10.0, 1.0)

    # Token diversity: penalize if MaxAct is dominated by a single repeated token
    # (a feature that just fires on ' the' everywhere is likely positional)
    diversity_score = 1.0
    if max_act_list:
        tokens = []
        for item in max_act_list:
            if isinstance(item, dict):
                tokens.append(item.get("token", "").strip().lower())
            else:
                tokens.append(str(item).strip().lower())
        if tokens:
            most_common_ratio = tokens.count(max(set(tokens), key=tokens.count)) / len(tokens)
            # If >60% of top tokens are the same token, penalize heavily
            diversity_score = max(0.0, 1.0 - max(0.0, most_common_ratio - 0.4) / 0.6)

    return (freq_score * 2 + max_act_score + vocab_score + act_score + diversity_score) / 6


def filter_high_quality_features(
    features: List[Dict[str, Any]],
    min_freq: float = 0.001,   # 0.1%  — exclude near-dead features
    max_freq: float = 0.15,    # 15%   — exclude always-on polysemantic features
    min_max_act: int = 2,      # need ≥2 MaxAct examples to assess coherence
    min_vocab_proj: int = 3,   # need ≥3 VocabProj tokens to assess output signal
) -> List[Dict[str, Any]]:
    """
    Filter features to those most likely to be monosemantic and interpretable.

    Exclusion reasons logged for transparency:
    - too_rare:    frequency < min_freq (likely dead or near-dead)
    - too_common:  frequency > max_freq (likely polysemantic catch-all)
    - few_max_act: not enough MaxAct examples to assess
    - few_vocab:   not enough VocabProj tokens to assess
    """
    filtered = []
    stats = {"too_rare": 0, "too_common": 0, "few_max_act": 0, "few_vocab": 0}

    for f in features:
        freq = f.get("frequency", 0)

        if freq < min_freq:
            stats["too_rare"] += 1
            continue
        if freq > max_freq:
            stats["too_common"] += 1
            continue
        if len(f.get("max_activating_tokens", [])) < min_max_act:
            stats["few_max_act"] += 1
            continue
        if len(f.get("vocab_projection", [])) < min_vocab_proj:
            stats["few_vocab"] += 1
            continue

        filtered.append(f)

    print(f"  Quality filter: {len(features)} total → {len(filtered)} candidates")
    print(f"    Excluded: {stats['too_rare']} too rare (<{min_freq*100:.1f}%), "
          f"{stats['too_common']} too common (>{max_freq*100:.0f}%), "
          f"{stats['few_max_act']} few MaxAct, {stats['few_vocab']} few VocabProj")

    return filtered


# =============================================================================
# Feature Labeling Logic (Batched for efficiency)
# =============================================================================

BATCH_SIZE = 50  # Features per API call (can go up to ~100)


def extract_feature_tokens(feature: Dict[str, Any]) -> tuple:
    """Extract MaxAct and VocabProj tokens (with logits) from a feature."""
    max_act_tokens = []
    if "max_activating_tokens" in feature:
        for item in feature["max_activating_tokens"][:10]:
            if isinstance(item, dict):
                max_act_tokens.append((item.get("token", ""), item.get("activation", 0.0), item.get("context", "")))
            else:
                max_act_tokens.append((str(item), 0.0, ""))

    vocab_proj_tokens = feature.get("vocab_projection", [])[:10]
    vocab_proj_logits = feature.get("vocab_projection_logits", [])[:10]

    # Zip tokens with their logit values; fall back to None if logits not available
    vocab_proj = list(zip(vocab_proj_tokens, vocab_proj_logits)) if vocab_proj_logits else \
                 [(t, None) for t in vocab_proj_tokens]

    return max_act_tokens, vocab_proj


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

        vocab_parts = []
        for tok, logit in vocab_proj_tokens:
            if logit is not None:
                vocab_parts.append(f"'{tok}' ({logit:.2f})")
            else:
                vocab_parts.append(repr(tok))

        features_text += f"""
---
FEATURE {feature['index']} (freq: {feature['frequency']*100:.1f}%)
MaxAct (triggers): {', '.join(max_act_parts) if max_act_parts else 'No data'}
VocabProj (promotes, logit boost): {', '.join(vocab_parts) if vocab_parts else 'No data'}
---
"""

    prompt = f"""Analyze these neural network features from a Sparse Autoencoder trained on Llama 3.2 1B processing medical text.

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
    max_features: int = 3500,
    batch_size: int = BATCH_SIZE,
    min_freq: float = 0.001,
    max_freq: float = 0.15,
    no_filter: bool = False,
) -> List[LabeledFeature]:
    """Label features using OpenAI with batched API calls."""

    print(f"\nTotal features available: {len(features)}")

    if no_filter:
        candidates = features
        print("  Quality filter: disabled (--no-filter)")
    else:
        candidates = filter_high_quality_features(features, min_freq=min_freq, max_freq=max_freq)

    # Sort best candidates first so if we hit max_features, we label the most
    # interpretable ones rather than an arbitrary slice
    candidates.sort(key=compute_quality_score, reverse=True)
    features_to_label = candidates[:max_features]

    num_batches = (len(features_to_label) + batch_size - 1) // batch_size
    print(f"  Labeling top {len(features_to_label)} candidates with OpenAI ({model})...")
    print(f"  Batch size: {batch_size} | Total API calls: {num_batches}")

    labeled_features = []
    unlabeled_count = 0

    # Pre-compute quality scores for all features (stored in output JSON)
    quality_scores = {f["index"]: compute_quality_score(f) for f in features_to_label}

    # Create feature lookup for token extraction
    feature_lookup = {f["index"]: f for f in features_to_label}

    for i in tqdm(range(0, len(features_to_label), batch_size), desc="Batches"):
        batch = features_to_label[i:i + batch_size]

        if dry_run:
            for feature in batch:
                max_act, vocab_proj = extract_feature_tokens(feature)
                score = quality_scores[feature["index"]]
                print(f"\n--- Feature {feature['index']} (quality={score:.3f}, freq={feature['frequency']*100:.1f}%) ---")
                print(f"MaxAct:    {max_act[:3]}")
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
                        quality_score=quality_scores[feature_id],
                    ))
                    labeled_ids.add(feature_id)
                else:
                    unlabeled_count += 1
                    labeled_ids.add(feature_id)

            # Count features not in response as unlabeled
            unlabeled_count += len(batch) - len(labeled_ids)

        except Exception as e:
            print(f"\nError labeling batch starting at feature {batch[0]['index']}: {e}")
            unlabeled_count += len(batch)

    if not dry_run:
        label_rate = len(labeled_features) / len(features_to_label) * 100 if features_to_label else 0
        print(f"\n  Labeled: {len(labeled_features)} features ({label_rate:.1f}% label rate)")
        print(f"  Unlabeled (not monosemantic): {unlabeled_count} features")

    return labeled_features


def save_labeled_features(
    labeled_features: List[LabeledFeature],
    output_path: Path,
):
    """Save labeled features to JSON."""
    # Sort by quality score descending so the app shows best features first
    sorted_features = sorted(labeled_features, key=lambda f: f.quality_score, reverse=True)

    data = []
    for f in sorted_features:
        # vocab_proj_tokens is stored as [(token, logit), ...] — split for JSON
        # Keep vocab_proj_tokens as strings for app.py backwards compatibility
        vp_tokens = [t[0] if isinstance(t, (list, tuple)) else t for t in f.vocab_proj_tokens]
        vp_logits = [round(t[1], 4) if isinstance(t, (list, tuple)) and t[1] is not None else None
                     for t in f.vocab_proj_tokens]
        data.append({
            "index": f.index,
            "label": f.label,
            "confidence": f.confidence,
            "max_act_tokens": f.max_act_tokens,
            "vocab_proj_tokens": vp_tokens,
            "vocab_proj_logits": vp_logits,
            "reasoning": f.reasoning,
            "quality_score": round(f.quality_score, 4),
        })

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

    for conf_level, feats in [("HIGH CONFIDENCE", high), ("MEDIUM CONFIDENCE", medium), ("LOW CONFIDENCE", low)]:
        if not feats:
            continue
        # Show best quality first within each confidence tier
        feats_sorted = sorted(feats, key=lambda f: f.quality_score, reverse=True)
        print(f"\n  {conf_level} ({len(feats_sorted)} features)")
        print("  " + "-"*40)
        for f in feats_sorted[:10]:
            print(f"    Feature {f.index:4d} [q={f.quality_score:.2f}]: {f.label}")
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
    parser.add_argument("--max-features", type=int, default=3500,
                        help="Maximum features to label (default: 3500)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Features per API call (default: {BATCH_SIZE})")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path")
    parser.add_argument("--min-freq", type=float, default=0.001,
                        help="Min activation frequency to include (default: 0.001 = 0.1%%)")
    parser.add_argument("--max-freq", type=float, default=0.15,
                        help="Max activation frequency to include (default: 0.15 = 15%%)")
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable quality filter, label all features")
    args = parser.parse_args()

    # Load features from medical_outputs
    if not MEDICAL_FEATURES_PATH.exists():
        print(f"Error: {MEDICAL_FEATURES_PATH} not found. Run main.py first.")
        return

    with open(MEDICAL_FEATURES_PATH) as f:
        features = json.load(f)

    print(f"Loaded {len(features)} features from {MEDICAL_FEATURES_PATH}")

    # Check API key
    if not args.dry_run:
        if not os.environ.get("OPENAI_API_KEY"):
            print("\nError: OPENAI_API_KEY not found.")
            return

    # Label features
    labeled_features = label_features(
        features,
        model=args.model,
        dry_run=args.dry_run,
        max_features=args.max_features,
        batch_size=args.batch_size,
        min_freq=args.min_freq,
        max_freq=args.max_freq,
        no_filter=args.no_filter,
    )

    if args.dry_run:
        return

    # Print results
    print_labeled_features(labeled_features)

    # Save results
    output_path = Path(args.output) if args.output else MEDICAL_OUTPUT_DIR / "labeled_features.json"
    save_labeled_features(labeled_features, output_path)


if __name__ == "__main__":
    main()
