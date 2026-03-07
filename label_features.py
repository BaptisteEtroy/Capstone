#!/usr/bin/env python3
"""
Features are labeled using two complementary methods:

1. Heuristic labeling (free, no API): detects syntactic, morphological, and
   positional features using token string patterns, position statistics, and
   optionally spaCy POS tagging.

2. OpenAI semantic labeling: for features that pass a quality filter and were
   not already claimed by heuristics, batch API calls to gpt-4o-mini assign
   semantic labels ("proverbs and sayings", "US state names", etc.).

OpenAI labels take priority if both methods produce a label for the same feature.

Usage:
    python label_features.py              # Heuristic + OpenAI (default)
    python label_features.py --dry-run    # Preview without API calls
    python label_features.py --heuristic-only   # Skip OpenAI entirely (free)
    python label_features.py --no-heuristic     # OpenAI only (old behavior)
    python label_features.py --use-spacy        # Enable spaCy POS labeling
    python label_features.py --min-freq 0.001 --max-freq 0.10  # Custom thresholds
"""

import json
import math
import os
import re
import string
import argparse
from collections import Counter
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

from config import OUTPUT_DIR, FEATURES_PATH

MAX_TOKENS_PER_SEQ = 128  # must match max_tokens in collect_activations


# =============================================================================
# Data Class
# =============================================================================

@dataclass
class LabeledFeature:
    """A feature with its auto-generated label."""
    index: int
    label: str
    confidence: str        # "high", "medium", "low"
    max_act_tokens: List
    vocab_proj_tokens: List
    reasoning: str
    quality_score: float = 0.0
    label_source: str = "openai"   # "openai" | "heuristic_token" | "heuristic_positional" | "heuristic_pos"

    @property
    def category(self) -> str:
        """High-level feature category: semantic / syntactic / positional."""
        if self.label_source == "heuristic_positional":
            return "positional"
        if self.label_source in ("heuristic_token", "heuristic_pos"):
            return "syntactic"
        return "semantic"


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

    freq_score = max(0.0, 1.0 - abs(math.log10(max(freq, 1e-6)) - math.log10(0.02)) / 1.5)

    max_act_list = feature.get("max_activating_tokens", [])
    vocab_list = feature.get("vocab_projection", [])

    max_act_score = min(len(max_act_list), 10) / 10
    vocab_score = min(len(vocab_list), 10) / 10

    mean_act = feature.get("mean_activation", 0)
    act_score = min(mean_act / 10.0, 1.0)

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
            diversity_score = max(0.0, 1.0 - max(0.0, most_common_ratio - 0.4) / 0.6)

    return (freq_score * 2 + max_act_score + vocab_score + act_score + diversity_score) / 6


def filter_high_quality_features(
    features: List[Dict[str, Any]],
    min_freq: float = 0.001,
    max_freq: float = 0.15,
    min_max_act: int = 2,
    min_vocab_proj: int = 3,
) -> List[Dict[str, Any]]:
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
# Heuristic Labeling (no API required)
# =============================================================================

def _extract_raw_tokens(max_act_tokens: list) -> List[str]:
    """Extract raw token strings (with spaces preserved) from MaxAct list."""
    raw = []
    for item in max_act_tokens:
        if isinstance(item, dict):
            raw.append(item.get("token", ""))
        elif isinstance(item, (list, tuple)):
            raw.append(str(item[0]))
        else:
            raw.append(str(item))
    return raw


def _token_pattern_label(max_act_tokens: list) -> Optional[Tuple[str, str]]:
    """
    Classify by token string patterns alone. Returns (label, confidence) or None.

    Only catches obvious syntactic patterns that don't have semantic meaning:
      - Special tokens (<|endoftext|>)
      - Punctuation-only tokens
      - Numeric / digit tokens
    
    Removed: acronyms, subword continuations, word-initial — these often catch
    semantically meaningful features and mislabel them with syntactic descriptions.
    """
    if not max_act_tokens:
        return None

    raw = _extract_raw_tokens(max_act_tokens)
    clean = [t.strip() for t in raw]
    n = len(clean)
    if n == 0:
        return None

    # Special / boundary tokens
    n_special = sum(1 for t in raw if "<|" in t)
    if n_special / n >= 0.7:
        return "End-of-text / document boundary tokens", "high"

    # Punctuation only
    n_punct = sum(1 for t in clean if t and all(c in string.punctuation for c in t))
    if n_punct / n >= 0.7:
        common = Counter(clean).most_common(1)[0][0]
        conf = "high" if n_punct / n >= 0.85 else "medium"
        return f"Punctuation: '{common}'", conf

    # Digits / numeric
    n_digit = sum(1 for t in clean if re.match(r"^[\d,.\-\+%]+$", t) and t)
    if n_digit / n >= 0.65:
        conf = "high" if n_digit / n >= 0.8 else "medium"
        return "Numeric / digit tokens", conf

    return None


def _positional_label(
    position_mean: float,
    position_std: float,
    max_tokens: int = MAX_TOKENS_PER_SEQ,
) -> Optional[Tuple[str, str]]:
    """
    Classify by positional firing pattern. Returns (label, confidence) or None.

    Logic:
    - Only fires if position_std is low enough to indicate a consistent position bias.
    - Near position 0 → start feature; near end → end feature.
    - Very tight std anywhere → fixed-position feature.
    """
    # No position data (old features.json without position fields)
    if position_mean == 0.0 and position_std == 0.0:
        return None

    # Wide spread → fires all over, not a positional feature
    if position_std > 38:
        return None

    normalized = position_mean / max_tokens

    if position_std < 4 and position_mean <= 2:
        return "Document/sequence start feature (position 0–2)", "high"

    if normalized < 0.12 and position_std < 18:
        conf = "high" if position_std < 10 else "medium"
        return "Context-start / early-position feature", conf

    if normalized > 0.88 and position_std < 18:
        conf = "high" if position_std < 10 else "medium"
        return "Context-end / late-position feature", conf

    if position_std < 8:
        return f"Fixed-position feature (position ≈ {int(position_mean)})", "medium"

    return None


def _pos_tag_label(contexts: List[str], nlp) -> Optional[Tuple[str, str]]:
    """
    Use a pre-loaded spaCy model to determine syntactic role of MaxAct tokens.
    Returns (label, confidence) or None.

    The MaxAct context string marks the trigger token as [TOKEN] — we parse the
    surrounding sentence and find the POS/dep of that token.
    """
    pos_tags = []

    for ctx in contexts:
        if not ctx:
            continue
        match = re.search(r"\[([^\]]+)\]", ctx)
        if not match:
            continue
        token_text = match.group(1).strip()
        plain_ctx = re.sub(r"\[([^\]]+)\]", r"\1", ctx)

        try:
            doc = nlp(plain_ctx)
            for token in doc:
                if token.text.strip() == token_text or token.text == token_text:
                    pos_tags.append(token.pos_)
                    break
        except Exception:
            continue

    if len(pos_tags) < 3:
        return None

    counter = Counter(pos_tags)
    most_common_pos, count = counter.most_common(1)[0]
    ratio = count / len(pos_tags)

    if ratio < 0.55:
        return None

    pos_names = {
        "NOUN":  "Common noun tokens",
        "VERB":  "Verb tokens",
        "ADJ":   "Adjective tokens",
        "ADV":   "Adverb tokens",
        "PROPN": "Proper noun tokens",
        "DET":   "Determiner tokens (the/a/an)",
        "ADP":   "Preposition tokens",
        "CCONJ": "Coordinating conjunction tokens",
        "SCONJ": "Subordinating conjunction tokens",
        "PUNCT": "Punctuation tokens",
        "NUM":   "Numeric tokens",
        "PRON":  "Pronoun tokens",
        "PART":  "Particle tokens (to/not)",
        "AUX":   "Auxiliary verb tokens",
        "INTJ":  "Interjection tokens",
    }
    label = pos_names.get(most_common_pos)
    if label is None:
        return None

    conf = "high" if ratio >= 0.75 else "medium"
    return label, conf


def heuristic_label_feature(
    feature: Dict[str, Any],
    nlp=None,  # kept for API compatibility but no longer used
) -> Optional[LabeledFeature]:
    """
    Try to label a feature using rule-based heuristics (no API).

    Only catches obvious non-semantic patterns:
      1. Token pattern (punctuation, digits, special tokens)
      2. Positional (fires only at start/end of context)

    Removed: POS tagging, word-initial, subword, acronym heuristics — these
    were catching semantically meaningful features and mislabeling them.

    Returns LabeledFeature if any heuristic fires, else None.
    """
    # Skip dead features — they never activated, any signal is noise
    if feature.get("frequency", 0) < 1e-5:
        return None

    max_act_tokens = feature.get("max_activating_tokens", [])
    quality = compute_quality_score(feature)

    # Build (token, activation, context) tuples for storing in LabeledFeature
    def _act_tuples():
        result = []
        for item in max_act_tokens[:10]:
            if isinstance(item, dict):
                result.append((
                    item.get("token", ""),
                    item.get("activation", 0.0),
                    item.get("context", ""),
                ))
        return result

    def _vocab_tuples():
        vp_tokens = feature.get("vocab_projection", [])[:10]
        vp_logits = feature.get("vocab_projection_logits", [None] * 10)[:10]
        return list(zip(vp_tokens, vp_logits))

    # 1. Token pattern heuristics (only punctuation, numeric, special tokens)
    result = _token_pattern_label(max_act_tokens)
    if result:
        label, conf = result
        return LabeledFeature(
            index=feature["index"],
            label=label,
            confidence=conf,
            max_act_tokens=_act_tuples(),
            vocab_proj_tokens=_vocab_tuples(),
            reasoning="Heuristic: token string pattern analysis",
            quality_score=quality,
            label_source="heuristic_token",
        )

    # 2. Positional heuristics (start/end of context features)
    pos_mean = feature.get("position_mean", 0.0)
    pos_std  = feature.get("position_std",  0.0)
    result = _positional_label(pos_mean, pos_std)
    if result:
        label, conf = result
        return LabeledFeature(
            index=feature["index"],
            label=label,
            confidence=conf,
            max_act_tokens=_act_tuples(),
            vocab_proj_tokens=_vocab_tuples(),
            reasoning=f"Heuristic: positional analysis (mean={pos_mean:.1f}, std={pos_std:.1f})",
            quality_score=quality,
            label_source="heuristic_positional",
        )

    return None


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
# Feature Labeling Logic (Batched OpenAI)
# =============================================================================

OPENAI_BATCH_SIZE = 50


def extract_feature_tokens(feature: Dict[str, Any]) -> tuple:
    max_act_tokens = []
    if "max_activating_tokens" in feature:
        for item in feature["max_activating_tokens"][:10]:
            if isinstance(item, dict):
                max_act_tokens.append((item.get("token", ""), item.get("activation", 0.0), item.get("context", "")))
            else:
                max_act_tokens.append((str(item), 0.0, ""))

    vocab_proj_tokens = feature.get("vocab_projection", [])[:10]
    vocab_proj_logits = feature.get("vocab_projection_logits", [])[:10]
    vocab_proj = list(zip(vocab_proj_tokens, vocab_proj_logits)) if vocab_proj_logits else \
                 [(t, None) for t in vocab_proj_tokens]

    return max_act_tokens, vocab_proj


def build_batch_prompt(features_batch: List[Dict[str, Any]]) -> str:
    features_text = ""
    for feature in features_batch:
        max_act_tokens, vocab_proj_tokens = extract_feature_tokens(feature)

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
    results = []
    feature_ids = {f["index"] for f in features_batch}

    for line in response.split("\n"):
        line = line.strip()
        if not line.startswith("FEATURE"):
            continue
        try:
            parts = line.split(":", 1)
            if len(parts) < 2:
                continue
            feature_id = int(parts[0].replace("FEATURE", "").strip())
            if feature_id not in feature_ids:
                continue
            rest = parts[1].strip()
            segments = rest.split("|")
            label      = segments[0].strip() if len(segments) > 0 else "UNLABELED"
            confidence = segments[1].strip().lower() if len(segments) > 1 else "low"
            reasoning  = segments[2].strip() if len(segments) > 2 else ""
            results.append((feature_id, label, confidence, reasoning))
        except (ValueError, IndexError):
            continue

    return results


# =============================================================================
# Combined Labeling Pipeline
# =============================================================================

def label_features(
    features: List[Dict],
    model: str = "gpt-4o-mini",
    dry_run: bool = False,
    max_features: int = 3500,
    batch_size: int = OPENAI_BATCH_SIZE,
    min_freq: float = 0.001,
    max_freq: float = 0.15,
    no_filter: bool = False,
    heuristic_only: bool = False,
    no_heuristic: bool = False,
    use_spacy: bool = False,
) -> List[LabeledFeature]:
    """
    Label features using heuristics first, then OpenAI for the rest.

    Heuristics cover: punctuation, digits, subwords, positional, and (optionally)
    spaCy POS tags. They run on ALL 8192 features with no frequency filter.

    OpenAI runs on quality-filtered semantic candidates not already claimed.
    OpenAI labels win if both methods produce a result for the same feature.
    """
    print(f"\nTotal features available: {len(features)}")

    # -------------------------------------------------------------------------
    # Step 1: Heuristic labeling — all features, no filter
    # -------------------------------------------------------------------------
    heuristic_labeled: Dict[int, LabeledFeature] = {}

    if not no_heuristic:
        nlp = None
        if use_spacy:
            try:
                import spacy
                print("  Loading spaCy en_core_web_sm...")
                nlp = spacy.load("en_core_web_sm")
                print("  spaCy loaded.")
            except (ImportError, OSError) as e:
                print(f"  Warning: spaCy unavailable ({e}). Skipping POS labeling.")

        print("\nStep 1: Heuristic labeling (all features, no API)...")
        for f in tqdm(features, desc="  Heuristic"):
            result = heuristic_label_feature(f, nlp=nlp)
            if result:
                heuristic_labeled[f["index"]] = result

        # Break down by source
        by_source: Dict[str, int] = Counter(lf.label_source for lf in heuristic_labeled.values())
        print(f"  Heuristic labels: {len(heuristic_labeled)} features")
        for src, cnt in sorted(by_source.items()):
            print(f"    {src}: {cnt}")

    if heuristic_only or dry_run:
        if dry_run:
            print("\n[DRY RUN] Would also run OpenAI on quality-filtered features.")
            _preview_features(features, min_freq, max_freq, no_filter)
        return list(heuristic_labeled.values())

    # -------------------------------------------------------------------------
    # Step 2: OpenAI semantic labeling — quality-filtered, skip heuristic hits
    # -------------------------------------------------------------------------
    if not os.environ.get("OPENAI_API_KEY"):
        print("\nNo OPENAI_API_KEY found — skipping semantic labeling.")
        return list(heuristic_labeled.values())

    print("\nStep 2: OpenAI semantic labeling (quality-filtered features)...")

    if no_filter:
        candidates = features
        print("  Quality filter: disabled (--no-filter)")
    else:
        candidates = filter_high_quality_features(features, min_freq=min_freq, max_freq=max_freq)

    # Send ALL quality-filtered features to OpenAI — including heuristic hits.
    # OpenAI wins if it produces a real label; heuristic label is the fallback
    # when OpenAI returns UNLABELED (e.g. for caps/subword features that are
    # actually semantic, like stock tickers being mistaken for generic acronyms).
    semantic_candidates = sorted(candidates, key=compute_quality_score, reverse=True)
    features_to_label = semantic_candidates[:max_features]

    num_batches = (len(features_to_label) + batch_size - 1) // batch_size
    print(f"  Labeling top {len(features_to_label)} semantic candidates ({model})...")
    print(f"  Batch size: {batch_size} | Total API calls: {num_batches}")

    openai_labeled: List[LabeledFeature] = []
    unlabeled_count = 0
    quality_scores = {f["index"]: compute_quality_score(f) for f in features_to_label}
    feature_lookup = {f["index"]: f for f in features_to_label}

    for i in tqdm(range(0, len(features_to_label), batch_size), desc="  OpenAI batches"):
        batch = features_to_label[i:i + batch_size]
        try:
            prompt = build_batch_prompt(batch)
            response = call_openai(prompt, model)
            results = parse_batch_response(response, batch)

            labeled_ids = set()
            for feature_id, label, confidence, reasoning in results:
                if label.upper() != "UNLABELED":
                    max_act, vocab_proj = extract_feature_tokens(feature_lookup[feature_id])
                    openai_labeled.append(LabeledFeature(
                        index=feature_id,
                        label=label,
                        confidence=confidence,
                        max_act_tokens=max_act,
                        vocab_proj_tokens=vocab_proj,
                        reasoning=reasoning,
                        quality_score=quality_scores[feature_id],
                        label_source="openai",
                    ))
                    labeled_ids.add(feature_id)
                else:
                    unlabeled_count += 1
                    labeled_ids.add(feature_id)

            unlabeled_count += len(batch) - len(labeled_ids)

        except Exception as e:
            print(f"\nError labeling batch starting at feature {batch[0]['index']}: {e}")
            unlabeled_count += len(batch)

    label_rate = len(openai_labeled) / len(features_to_label) * 100 if features_to_label else 0
    print(f"\n  OpenAI labeled: {len(openai_labeled)} features ({label_rate:.1f}% label rate)")
    print(f"  Unlabeled (not monosemantic): {unlabeled_count} features")

    # -------------------------------------------------------------------------
    # Merge: OpenAI takes priority over heuristic for the same feature index
    # -------------------------------------------------------------------------
    openai_indices = {lf.index for lf in openai_labeled}
    final = [lf for lf in heuristic_labeled.values() if lf.index not in openai_indices]
    final.extend(openai_labeled)

    print(f"\n  Total labeled features: {len(final)}")
    print(f"    Heuristic: {sum(1 for lf in final if lf.label_source != 'openai')}")
    print(f"    OpenAI:    {sum(1 for lf in final if lf.label_source == 'openai')}")

    return final


def _preview_features(features, min_freq, max_freq, no_filter):
    """Print a preview of semantic candidates (used in dry-run mode)."""
    if no_filter:
        candidates = features
    else:
        candidates = filter_high_quality_features(features, min_freq=min_freq, max_freq=max_freq)
    candidates.sort(key=compute_quality_score, reverse=True)
    print(f"\n  Top 10 semantic candidates (would be sent to OpenAI):")
    for f in candidates[:10]:
        max_act, _ = extract_feature_tokens(f)
        score = compute_quality_score(f)
        print(f"  Feature {f['index']:4d} (q={score:.3f}, freq={f['frequency']*100:.1f}%): "
              f"{[t for t, _, _ in max_act[:3]]}")


# =============================================================================
# Save & Print
# =============================================================================

def save_labeled_features(labeled_features: List[LabeledFeature], output_path: Path):
    """Save labeled features to JSON, sorted by quality score descending."""
    sorted_features = sorted(labeled_features, key=lambda f: f.quality_score, reverse=True)

    data = []
    for f in sorted_features:
        vp_tokens = [t[0] if isinstance(t, (list, tuple)) else t for t in f.vocab_proj_tokens]
        vp_logits = [round(t[1], 4) if isinstance(t, (list, tuple)) and t[1] is not None else None
                     for t in f.vocab_proj_tokens]
        data.append({
            "index":            f.index,
            "label":            f.label,
            "confidence":       f.confidence,
            "label_source":     f.label_source,
            "category":         f.category,
            "max_act_tokens":   f.max_act_tokens,
            "vocab_proj_tokens": vp_tokens,
            "vocab_proj_logits": vp_logits,
            "reasoning":        f.reasoning,
            "quality_score":    round(f.quality_score, 4),
        })

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved {len(labeled_features)} labeled features to {output_path}")


def print_labeled_features(labeled_features: List[LabeledFeature]):
    print("\n" + "=" * 60)
    print("  LABELED FEATURES")
    print("=" * 60)

    # Source breakdown
    openai_feats    = [f for f in labeled_features if f.label_source == "openai"]
    heuristic_feats = [f for f in labeled_features if f.label_source != "openai"]
    print(f"\n  Sources: {len(openai_feats)} OpenAI | {len(heuristic_feats)} heuristic")

    # Heuristic breakdown by type
    if heuristic_feats:
        by_src = Counter(f.label_source for f in heuristic_feats)
        for src, cnt in sorted(by_src.items()):
            print(f"    {src}: {cnt}")

    # Show top features by confidence
    high   = [f for f in labeled_features if f.confidence == "high"]
    medium = [f for f in labeled_features if f.confidence == "medium"]
    low    = [f for f in labeled_features if f.confidence == "low"]

    for conf_level, feats in [("HIGH CONFIDENCE", high), ("MEDIUM CONFIDENCE", medium), ("LOW CONFIDENCE", low)]:
        if not feats:
            continue
        feats_sorted = sorted(feats, key=lambda f: f.quality_score, reverse=True)
        print(f"\n  {conf_level} ({len(feats_sorted)} features)")
        print("  " + "-" * 40)
        for f in feats_sorted[:10]:
            src_tag = f"[{f.label_source}]"
            print(f"    Feature {f.index:4d} [q={f.quality_score:.2f}] {src_tag}: {f.label}")
            print(f"                  → {f.reasoning[:70]}...")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Label SAE features")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without API calls (runs heuristics, previews OpenAI candidates)")
    parser.add_argument("--heuristic-only", action="store_true",
                        help="Run heuristic labeling only — no OpenAI API calls")
    parser.add_argument("--no-heuristic", action="store_true",
                        help="Skip heuristic labeling, use OpenAI only (old behavior)")
    parser.add_argument("--use-spacy", action="store_true",
                        help="Enable spaCy POS tagging (requires: pip install spacy && python -m spacy download en_core_web_sm)")
    parser.add_argument("--max-features", type=int, default=3500,
                        help="Max semantic features to send to OpenAI (default: 3500)")
    parser.add_argument("--batch-size", type=int, default=OPENAI_BATCH_SIZE,
                        help=f"Features per OpenAI API call (default: {OPENAI_BATCH_SIZE})")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--min-freq", type=float, default=0.001)
    parser.add_argument("--max-freq", type=float, default=0.15)
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable quality filter for OpenAI candidates")
    args = parser.parse_args()

    if not FEATURES_PATH.exists():
        print(f"Error: {FEATURES_PATH} not found. Run main.py first.")
        return

    with open(FEATURES_PATH) as f:
        features = json.load(f)
    print(f"Loaded {len(features)} features from {FEATURES_PATH}")

    labeled_features = label_features(
        features,
        model=args.model,
        dry_run=args.dry_run,
        max_features=args.max_features,
        batch_size=args.batch_size,
        min_freq=args.min_freq,
        max_freq=args.max_freq,
        no_filter=args.no_filter,
        heuristic_only=args.heuristic_only,
        no_heuristic=args.no_heuristic,
        use_spacy=args.use_spacy,
    )

    if args.dry_run:
        return

    print_labeled_features(labeled_features)

    output_path = Path(args.output) if args.output else OUTPUT_DIR / "labeled_features.json"
    save_labeled_features(labeled_features, output_path)


if __name__ == "__main__":
    main()
