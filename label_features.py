#!/usr/bin/env python3
"""
Features are labeled using two complementary methods:

1. Heuristic labeling (free, no API): detects syntactic, morphological, and
   positional features using token string patterns and position statistics.

2. gpt-4o-mini semantic labeling: for features that pass a quality filter and were
   not already claimed by heuristics, batch API calls to gpt-4o-mini assign
   semantic labels ("proverbs and sayings", "US state names", etc.).

gpt-4o-mini labels take priority if both methods produce a label for the same feature.

Usage:
    python label_features.py              # Heuristic + LLM (default)
    python label_features.py --dry-run    # Preview without API calls
    python label_features.py --heuristic-only   # Skip LLM entirely (free)
    python label_features.py --no-heuristic     # LLM only
    python label_features.py --min-freq 0.0005 --max-freq 0.30  # Custom thresholds
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

from config import MEDICAL_OUTPUT_DIR, TARGET_LAYER

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
    label_source: str = "gpt4omini"   # "gpt4omini" | "heuristic_token" | "heuristic_positional"
    maxact_entropy: float = 0.0    # Shannon entropy of MaxAct token distribution (low = monosemantic)
    source_breakdown: Dict[str, float] = field(default_factory=dict)  # fraction from each dataset
    category: str = "General/Other"  # broad semantic category for thesis figures


# =============================================================================
# Feature Categorisation
# =============================================================================

# Rules are checked in order; first match wins.  Each rule is
# (list_of_lowercase_keywords_to_search_in_label, category_name).
_CATEGORY_RULES: List[Tuple[List[str], str]] = [
    # Structural/Linguistic — expanded to catch heuristic positional labels
    (["structural", "positional", "punctuation", "special token", "numeric", "digit",
      "formatting", "whitespace", "early position", "late position", "sequence position",
      "early-position", "late-position", "fixed-position", "context-start", "context-end",
      "document/sequence start", "document boundary", "end-of-text",
      "conjunction", "pronoun", "preposition", "connective", "connector",
      "citation", "reference number", "journal", "publication", "bibliography",
      "abbreviation", "acronym", "parenthetical", "list marker"],
     "Structural/Linguistic"),
    # Research Methodology — expanded
    (["research", "study design", "trial", "randomized", "cohort", "meta-analysis",
      "systematic review", "statistical", "methodology", "experimental design",
      "clinical study", "sample size", "control group", "bias",
      "correlat", "regression", "confidence interval", "p-value", "odds ratio",
      "relative risk", "cross-sectional", "longitudinal", "prospective",
      "retrospective", "case-control", "blinding", "placebo", "survey",
      "questionnaire", "measurement", "instrument validation"],
     "Research Methodology"),
    # Pharmacology — expanded
    (["pharmacol", "drug", "medication", "therapeutic", "treatment", "dose", "dosage",
      "antibiotic", "inhibitor", "agonist", "antagonist", "pharmaceutical",
      "prescription", "side effect", "adverse", "toxicity",
      "chemotherap", "analgesic", "sedative", "anesthetic", "steroid",
      "anti-inflammatory", "antiviral", "antifungal", "antimicrobial",
      "contraindic", "drug interaction", "pharmacokinetic", "bioavailab"],
     "Pharmacology"),
    # NEW: Microbiology/Infectious Disease
    (["bacteri", "viral", "virus", "fung", "parasit", "infect",
      "pathogen", "microb", "sepsis", "steriliz", "contagi",
      "endotoxin", "exotoxin", "biofilm", "antimicrobial resistance",
      "gram-positive", "gram-negative", "flora", "prion"],
     "Microbiology/Infectious"),
    # NEW: Oncology
    (["cancer", "tumor", "tumour", "oncol", "malign", "metasta",
      "carcinoma", "lymphoma", "leukemia", "neoplasm", "sarcoma",
      "staging", "grading", "biopsy", "remission", "recurrence"],
     "Oncology"),
    # NEW: Mental Health/Psych
    (["psych", "depress", "anxiety", "cognitive", "emotion",
      "mental health", "schizoph", "autism", "bipolar", "adhd",
      "behavioral", "behaviour", "neuropsych", "dementia", "therapy session",
      "counseling", "counselling", "psychiatric", "mood"],
     "Mental Health/Psych"),
    # Biochemical/Molecular — expanded
    (["gene", "protein", "enzyme", "pathway", "metabolism", "biochem",
      "molecular", "cellular", "receptor", "hormone", "signal transduction",
      "amino acid", "nucleotide", "mrna", "expression",
      "homeosta", "acid-base", "coagulat", "electrolyte", "osmotic",
      "endocrin", "collagen", "lipid", "cholesterol", "glycol",
      "oxidat", "redox", "phosphoryl", "dna", "rna", "chromosom",
      "mitochond", "ribosom", "atp", "cytoplasm", "membrane transport",
      "ion channel", "catalytic", "substrate"],
     "Biochemical/Molecular"),
    # Epidemiological — expanded
    (["epidemiol", "prevalence", "incidence", "mortality", "public health",
      "outbreak", "surveillance", "population", "risk factor", "exposure",
      "screening", "socioeconomic", "disparit", "morbidity", "demographic",
      "age-adjusted", "endemic", "pandemic", "epidemic", "vaccination rate",
      "health outcome", "social determinant"],
     "Epidemiological"),
    # Anatomical — expanded
    (["anatomical", "anatomy", "organ", "tissue", "muscle", "bone", "nerve",
      "artery", "vein", "gland", "vessel", "tract", "cavity", "region",
      "ligament", "tendon", "cartilage", "fascia", "peritoneum", "pleura",
      "mediastin", "meninges", "spinal cord", "brain region", "cortex",
      "cerebr", "thorac", "abdomin", "pelvi", "cranial"],
     "Anatomical"),
    # Clinical/Diagnostic — expanded
    (["clinical", "diagnosis", "diagnostic", "symptom", "sign", "presentation",
      "complication", "syndrome", "disease", "disorder", "condition", "finding",
      "examination", "assessment", "patient", "surgical", "procedure", "patholog",
      "imaging", "mri", "ct scan", "ultrasound", "radiolog", "x-ray",
      "echocardiog", "endoscop", "prognosis", "differential diagnosis",
      "pain", "nausea", "fatigue", "fever", "chronic", "acute",
      "congenital", "prenatal", "neonat", "pediatr", "maternal",
      "wound", "fracture", "injury", "trauma", "rehab",
      "anemia", "diabetes", "hypertens", "asthma", "allerg",
      "dermatolog", "ophthalm", "dental", "auditory", "cardiac",
      "respiratory", "gastrointestin", "renal", "hepat", "urolog",
      "orthoped", "neurolog", "hematolog", "immunolog",
      "transplant", "prosthesis", "intensive care", "emergency",
      "nutrition", "diet", "obesity", "vitamin", "supplement",
      "glucose", "calori"],
     "Clinical/Diagnostic"),
]

_CATEGORY_FALLBACK = "General/Other"


def categorize_label(label: str) -> str:
    """
    Map a free-form feature label to a broad semantic category.

    Checks _CATEGORY_RULES in order (first match wins) against the lowercased label.
    Returns _CATEGORY_FALLBACK if no rule matches.
    """
    lower = label.lower()
    for keywords, category in _CATEGORY_RULES:
        if any(kw in lower for kw in keywords):
            return category
    return _CATEGORY_FALLBACK


# =============================================================================
# OpenAI API
# =============================================================================

_SYSTEM_PROMPT = (
    "You are an expert in mechanistic interpretability of neural networks. "
    "Your task is to label features extracted from a Sparse Autoencoder (SAE) "
    "trained on Llama 3.2 1B Instruct processing medical/clinical text. "
    "MaxAct tokens are real input tokens (with activation strengths) that triggered "
    "the feature — they show what the feature DETECTS. "
    "VocabProj tokens are output tokens the feature PROMOTES — they show what the "
    "feature CAUSES the model to predict. "
    "Give specific, concrete labels — avoid vague terms like 'language patterns', "
    "'text features', 'linguistic', or 'general concepts'."
)


def call_llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call OpenAI API for feature labeling."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()
    except ImportError:
        raise ImportError("Please install openai: pip install openai>=1.0.0")
    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {e}")


# =============================================================================
# Quality Filtering & Scoring
# =============================================================================

def compute_quality_score(feature: Dict[str, Any]) -> float:
    """
    Score a feature by its likelihood of being monosemantic (higher = better).

    Components:
    - freq_score:      Peaks at ~2% activation frequency. Penalises always-on
                       (polysemantic) and near-dead features equally.
    - max_act_score:   More MaxAct examples → more evidence of what the feature detects.
    - vocab_score:     More VocabProj tokens → clearer output-centric signal.
    - act_score:       Higher mean activation → stronger, more distinctive feature.
    - diversity_score: MaxAct tokens should be diverse (all-same → catch-all feature).
    - entropy_score:   Low Shannon entropy → tokens cluster on a narrow concept
                       (monosemantic). Normalized by log2(10) ≈ 3.32 bits (max for
                       10 unique tokens with uniform distribution).

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

    entropy = feature.get("maxact_entropy", None)
    if entropy is not None:
        _MAX_ENTROPY = math.log2(10)  # uniform over 10 unique tokens ≈ 3.32 bits
        entropy_score = max(0.0, 1.0 - entropy / _MAX_ENTROPY)
    else:
        entropy_score = diversity_score  # fallback: mirror diversity score

    return (freq_score * 2 + max_act_score + vocab_score + act_score + diversity_score + entropy_score) / 7


def filter_high_quality_features(
    features: List[Dict[str, Any]],
    min_freq: float = 0.0001,
    max_freq: float = 0.30,
    min_max_act: int = 2,
    min_vocab_proj: int = 0,
) -> List[Dict[str, Any]]:
    """
    Thresholds lowered to capture more valid features:
    - min_freq 0.0005 → 0.0001: the previous threshold was above the mean activation
      frequency of alive features, cutting out ~half the labelable set. Rare medical
      terminology (drug names, anatomy, procedures) fires infrequently but is highly
      interpretable.
    - min_vocab_proj 3 → 0: VocabProj always stores 10 tokens; this gate was a no-op
      for most features but silently dropped features with sparse decoder projections.
      Removed — quality score already penalises uninformative projections.
    """
    filtered = []
    stats = {"too_rare": 0, "too_common": 0, "few_max_act": 0}

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
        filtered.append(f)

    print(f"  Quality filter: {len(features)} total → {len(filtered)} candidates")
    print(f"    Excluded: {stats['too_rare']} too rare (<{min_freq*100:.3f}%), "
          f"{stats['too_common']} too common (>{max_freq*100:.0f}%), "
          f"{stats['few_max_act']} few MaxAct examples")

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
    Only catches obvious syntactic patterns: special tokens, punctuation, digits.
    """
    if not max_act_tokens:
        return None

    raw = _extract_raw_tokens(max_act_tokens)
    clean = [t.strip() for t in raw]
    n = len(clean)
    if n == 0:
        return None

    n_special = sum(1 for t in raw if "<|" in t)
    if n_special / n >= 0.7:
        return "End-of-text / document boundary tokens", "high"

    n_punct = sum(1 for t in clean if t and all(c in string.punctuation for c in t))
    if n_punct / n >= 0.7:
        common = Counter(clean).most_common(1)[0][0]
        conf = "high" if n_punct / n >= 0.85 else "medium"
        return f"Punctuation: '{common}'", conf

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
    """Classify by positional firing pattern. Returns (label, confidence) or None."""
    if position_mean == 0.0 and position_std == 0.0:
        return None
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


def heuristic_label_feature(feature: Dict[str, Any]) -> Optional[LabeledFeature]:
    """
    Try to label a feature using rule-based heuristics (no API).
    Returns LabeledFeature if any heuristic fires, else None.
    """
    if feature.get("frequency", 0) < 1e-6:
        return None

    max_act_tokens = feature.get("max_activating_tokens", [])
    quality = compute_quality_score(feature)

    def _act_tuples():
        result = []
        for item in max_act_tokens[:10]:
            if isinstance(item, dict):
                result.append((item.get("token", ""), item.get("activation", 0.0), item.get("context", "")))
        return result

    def _vocab_tuples():
        vp_tokens = feature.get("vocab_projection", [])[:10]
        vp_logits = feature.get("vocab_projection_logits", [None] * 10)[:10]
        return list(zip(vp_tokens, vp_logits))

    entropy = feature.get("maxact_entropy", 0.0)
    src_breakdown = feature.get("source_breakdown", {})

    result = _token_pattern_label(max_act_tokens)
    if result:
        label, conf = result
        return LabeledFeature(
            index=feature["index"], label=label, confidence=conf,
            max_act_tokens=_act_tuples(), vocab_proj_tokens=_vocab_tuples(),
            reasoning="Heuristic: token string pattern analysis",
            quality_score=quality, label_source="heuristic_token",
            maxact_entropy=entropy, source_breakdown=src_breakdown,
            category=categorize_label(label),
        )

    pos_mean = feature.get("position_mean", 0.0)
    pos_std  = feature.get("position_std",  0.0)
    result = _positional_label(pos_mean, pos_std)
    if result:
        label, conf = result
        return LabeledFeature(
            index=feature["index"], label=label, confidence=conf,
            max_act_tokens=_act_tuples(), vocab_proj_tokens=_vocab_tuples(),
            reasoning=f"Heuristic: positional (mean={pos_mean:.1f}, std={pos_std:.1f})",
            quality_score=quality, label_source="heuristic_positional",
            maxact_entropy=entropy, source_breakdown=src_breakdown,
            category=categorize_label(label),
        )

    return None


# =============================================================================
# Feature Labeling Logic (Batched LLM Calls)
# =============================================================================

LLM_BATCH_SIZE = 30   # smaller batches → better attention per feature


def extract_feature_tokens(feature: Dict[str, Any]) -> tuple:
    """
    Extract MaxAct and VocabProj token lists.

    MaxAct sampling strategy (EleutherAI arXiv:2410.13928):
    Show examples from DIFFERENT activation quantiles, not just the top tokens.
    This helps the labeling model understand what the feature truly detects
    vs what appears only at extreme activation levels.

    With 10 stored examples (sorted descending by activation), we pick:
      - indices 0-3  (top tier)
      - indices 5-6  (mid tier)
      - indices 8-9  (lower tier, still positive activations)
    """
    all_items = feature.get("max_activating_tokens", [])
    n = len(all_items)

    # Quantile sampling across the stored examples
    indices = []
    if n > 0:
        indices += list(range(min(4, n)))                             # top 4
    if n > 5:
        indices += list(range(5, min(7, n)))                          # mid 2
    if n > 8:
        indices += list(range(8, min(10, n)))                         # lower 2

    # Deduplicate while preserving order
    seen = set()
    sampled = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            sampled.append(all_items[i])

    max_act_tokens = []
    for item in sampled:
        if isinstance(item, dict):
            max_act_tokens.append((
                item.get("token", ""),
                item.get("activation", 0.0),
                item.get("context", ""),
            ))
        else:
            max_act_tokens.append((str(item), 0.0, ""))

    vocab_proj_tokens = feature.get("vocab_projection", [])[:10]
    vocab_proj_logits = feature.get("vocab_projection_logits", [])[:10]
    vocab_proj = list(zip(vocab_proj_tokens, vocab_proj_logits)) if vocab_proj_logits else \
                 [(t, None) for t in vocab_proj_tokens]

    return max_act_tokens, vocab_proj


def build_batch_prompt(features_batch: List[Dict[str, Any]]) -> str:
    """
    Build a labeling prompt that shows activation strengths prominently and
    samples from quantiles (not just top tokens) for better generalisation.

    Key changes from previous version:
    - Explicitly tells gpt-4o-mini the model is Llama 3.2 1B Instruct on medical chat data
    - Shows activation magnitude alongside context (not just token string)
    - Samples from top/mid/low tiers to reveal the feature's true range
    - No chain-of-thought in reasoning field (CoT doesn't improve quality per arXiv:2410.13928)
    """
    features_text = ""
    for feature in features_batch:
        max_act_tokens, vocab_proj_tokens = extract_feature_tokens(feature)

        max_act_parts = []
        for tok, act, ctx in max_act_tokens:
            # Truncate context to a short readable snippet around the trigger
            ctx_snippet = ""
            if ctx:
                # Find [TOKEN] marker and show ±30 chars
                m = re.search(r"\[([^\]]+)\]", ctx)
                if m:
                    start = max(0, m.start() - 30)
                    end = min(len(ctx), m.end() + 30)
                    ctx_snippet = f" | …{ctx[start:end]}…"
            max_act_parts.append(f"'{tok}' [act={act:.2f}]{ctx_snippet}")

        vocab_parts = []
        for tok, logit in vocab_proj_tokens:
            if logit is not None:
                vocab_parts.append(f"'{tok}' (+{logit:.2f})")
            else:
                vocab_parts.append(repr(tok))

        src_bd = feature.get("source_breakdown", {})
        src_str = ", ".join(f"{k}={v*100:.0f}%" for k, v in sorted(src_bd.items())) if src_bd else "unknown"
        features_text += f"""
---
FEATURE {feature['index']} (freq={feature['frequency']*100:.2f}%, mean_act={feature.get('mean_activation', 0):.3f}, sources: {src_str})
Triggers (MaxAct, sampled across activation quantiles):
  {chr(10).join(f'  {p}' for p in max_act_parts) if max_act_parts else '  No data'}
Promotes (VocabProj, logit boost):
  {', '.join(vocab_parts) if vocab_parts else 'No data'}
---"""

    prompt = f"""Analyze these SAE features from a Sparse Autoencoder trained on Llama 3.2 1B Instruct
processing medical/clinical Q&A conversations (medmcqa, pubmed_qa, clinical summaries).

Each feature:
- "Triggers" = tokens sampled from HIGH, MID, and LOW activation tiers that fire this feature
  (activation values shown in brackets — higher = stronger response)
- "Promotes" = tokens this feature boosts in the output vocabulary (logit boost shown)

Rules:
1. Label ONLY if the trigger tokens form a coherent semantic pattern across ALL tiers.
2. Use "UNLABELED" if tokens seem random, mixed across unrelated domains, or incoherent.
3. Be SPECIFIC: "drug dosage instructions", "patient age qualifiers", "surgical procedure names".
4. AVOID vague labels: "language patterns", "text features", "medical terms", "general concepts".
5. High confidence = pattern is clear across all activation tiers.
6. Medium confidence = mostly coherent but some noise, or only top tier is consistent.
7. Low confidence = weak signal, lean toward UNLABELED.

{features_text}

Respond with EXACTLY one line per feature:
FEATURE <id>: <LABEL or UNLABELED> | <high/medium/low> | <brief reasoning (no chain-of-thought)>

Examples:
FEATURE 123: drug dosage instructions | high | Triggers 'mg', '500', 'twice'; promotes 'daily', 'dose'
FEATURE 456: UNLABELED | low | Triggers span unrelated domains with no coherent pattern
FEATURE 789: patient symptom onset descriptions | medium | Triggers 'sudden', 'weeks', 'onset'; promotes 'symptoms'
FEATURE 101: surgical anatomy terms | high | Triggers 'artery', 'vein', 'incision'; promotes anatomical terms"""

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
    batch_size: int = LLM_BATCH_SIZE,
    min_freq: float = 0.0001,
    max_freq: float = 0.30,
    no_filter: bool = False,
    heuristic_only: bool = False,
    no_heuristic: bool = False,
) -> List[LabeledFeature]:
    """
    Label features using heuristics first, then gpt-4o-mini for the rest.

    Heuristics cover: punctuation, digits, special tokens, positional.
    They run on ALL features with no frequency filter and are always included
    in the output JSON (previously they were being lost due to a save bug — fixed).

    gpt-4o-mini runs on quality-filtered semantic candidates not already claimed.
    gpt-4o-mini labels win if both methods produce a result for the same feature.
    """
    print(f"\nTotal features available: {len(features)}")

    # ── Step 1: Heuristic labeling — all features, no filter ─────────────────
    heuristic_labeled: Dict[int, LabeledFeature] = {}

    if not no_heuristic:
        print("\nStep 1: Heuristic labeling (all features, no API)...")
        for f in tqdm(features, desc="  Heuristic"):
            result = heuristic_label_feature(f)
            if result:
                heuristic_labeled[f["index"]] = result

        by_source: Dict[str, int] = Counter(lf.label_source for lf in heuristic_labeled.values())
        print(f"  Heuristic labels: {len(heuristic_labeled)} features")
        for src, cnt in sorted(by_source.items()):
            print(f"    {src}: {cnt}")

    if heuristic_only or dry_run:
        if dry_run:
            print("\n[DRY RUN] Would also run gpt-4o-mini on quality-filtered features.")
            _preview_features(features, min_freq, max_freq, no_filter)
        return list(heuristic_labeled.values())

    # ── Step 2: LLM semantic labeling — quality-filtered, skip heuristic hits
    if not os.environ.get("OPENAI_API_KEY"):
        print("\nNo OPENAI_API_KEY found — skipping semantic labeling.")
        return list(heuristic_labeled.values())

    print(f"\nStep 2: OpenAI semantic labeling ({model})...")

    if no_filter:
        candidates = features
        print("  Quality filter: disabled (--no-filter)")
    else:
        candidates = filter_high_quality_features(features, min_freq=min_freq, max_freq=max_freq)

    semantic_candidates = sorted(candidates, key=compute_quality_score, reverse=True)
    features_to_label = semantic_candidates[:max_features]

    num_batches = (len(features_to_label) + batch_size - 1) // batch_size
    print(f"  Labeling top {len(features_to_label)} candidates | Batch size: {batch_size} | API calls: {num_batches}")

    llm_labeled: List[LabeledFeature] = []
    unlabeled_count = 0
    quality_scores = {f["index"]: compute_quality_score(f) for f in features_to_label}
    feature_lookup = {f["index"]: f for f in features_to_label}

    for i in tqdm(range(0, len(features_to_label), batch_size), desc="  gpt-4o-mini batches"):
        batch = features_to_label[i:i + batch_size]
        try:
            prompt = build_batch_prompt(batch)
            response = call_llm(prompt, model)
            results = parse_batch_response(response, batch)

            labeled_ids = set()
            for feature_id, label, confidence, reasoning in results:
                if feature_id in labeled_ids:
                    continue
                labeled_ids.add(feature_id)
                if "UNLABELED" not in label.upper():
                    max_act, vocab_proj = extract_feature_tokens(feature_lookup[feature_id])
                    llm_labeled.append(LabeledFeature(
                        index=feature_id,
                        label=label,
                        confidence=confidence,
                        max_act_tokens=max_act,
                        vocab_proj_tokens=vocab_proj,
                        reasoning=reasoning,
                        quality_score=quality_scores[feature_id],
                        label_source="gpt4omini",
                        maxact_entropy=feature_lookup[feature_id].get("maxact_entropy", 0.0),
                        source_breakdown=feature_lookup[feature_id].get("source_breakdown", {}),
                        category=categorize_label(label),
                    ))
                else:
                    unlabeled_count += 1

            unlabeled_count += len(batch) - len(labeled_ids)

        except Exception as e:
            print(f"\nError labeling batch at feature {batch[0]['index']}: {e}")
            unlabeled_count += len(batch)

    label_rate = len(llm_labeled) / len(features_to_label) * 100 if features_to_label else 0
    print(f"\n  gpt-4o-mini labeled: {len(llm_labeled)} features ({label_rate:.1f}% label rate)")
    print(f"  Unlabeled (not monosemantic): {unlabeled_count}")

    # ── Merge: gpt-4o-mini takes priority over heuristic for same feature index ────
    llm_indices = {lf.index for lf in llm_labeled}
    final = [lf for lf in heuristic_labeled.values() if lf.index not in llm_indices]
    final.extend(llm_labeled)

    print(f"\n  Total labeled features: {len(final)}")
    print(f"    Heuristic:   {sum(1 for lf in final if lf.label_source != 'gpt4omini')}")
    print(f"    gpt-4o-mini: {sum(1 for lf in final if lf.label_source == 'gpt4omini')}")

    return final


def _preview_features(features, min_freq, max_freq, no_filter):
    if no_filter:
        candidates = features
    else:
        candidates = filter_high_quality_features(features, min_freq=min_freq, max_freq=max_freq)
    candidates.sort(key=compute_quality_score, reverse=True)
    print(f"\n  Top 10 semantic candidates (would be sent to gpt-4o-mini):")
    for f in candidates[:10]:
        max_act, _ = extract_feature_tokens(f)
        score = compute_quality_score(f)
        print(f"  Feature {f['index']:4d} (q={score:.3f}, freq={f['frequency']*100:.2f}%): "
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
            "index":             f.index,
            "label":             f.label,
            "category":          f.category,
            "confidence":        f.confidence,
            "label_source":      f.label_source,
            "max_act_tokens":    f.max_act_tokens,
            "vocab_proj_tokens": vp_tokens,
            "vocab_proj_logits": vp_logits,
            "reasoning":         f.reasoning,
            "quality_score":     round(f.quality_score, 4),
            "maxact_entropy":    round(f.maxact_entropy, 4),
            "source_breakdown":  f.source_breakdown,
        })

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved {len(labeled_features)} labeled features to {output_path}")


def print_labeled_features(labeled_features: List[LabeledFeature]):
    print("\n" + "=" * 60)
    print("  LABELED FEATURES")
    print("=" * 60)

    llm_feats       = [f for f in labeled_features if f.label_source == "gpt4omini"]
    heuristic_feats = [f for f in labeled_features if f.label_source != "gpt4omini"]
    print(f"\n  Sources: {len(llm_feats)} gpt-4o-mini | {len(heuristic_feats)} heuristic")

    if heuristic_feats:
        by_src = Counter(f.label_source for f in heuristic_feats)
        for src, cnt in sorted(by_src.items()):
            print(f"    {src}: {cnt}")

    high   = [f for f in labeled_features if f.confidence == "high"]
    medium = [f for f in labeled_features if f.confidence == "medium"]
    low    = [f for f in labeled_features if f.confidence == "low"]

    for conf_level, feats in [("HIGH", high), ("MEDIUM", medium), ("LOW", low)]:
        if not feats:
            continue
        feats_sorted = sorted(feats, key=lambda f: f.quality_score, reverse=True)
        print(f"\n  {conf_level} CONFIDENCE ({len(feats_sorted)} features)")
        print("  " + "-" * 40)
        for f in feats_sorted[:10]:
            src_tag = f"[{f.label_source}]"
            print(f"    Feature {f.index:4d} [q={f.quality_score:.2f}] {src_tag}: {f.label}")
            if f.reasoning:
                print(f"                  → {f.reasoning[:70]}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Label SAE features using OpenAI + heuristics")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="OpenAI model ID (default: gpt-4o-mini)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without API calls (runs heuristics, previews gpt-4o-mini candidates)")
    parser.add_argument("--heuristic-only", action="store_true",
                        help="Heuristic labeling only — no gpt-4o-mini API calls")
    parser.add_argument("--no-heuristic", action="store_true",
                        help="Skip heuristic labeling, use gpt-4o-mini only")
    parser.add_argument("--max-features", type=int, default=3500,
                        help="Max semantic features to send to gpt-4o-mini (default: 3500)")
    parser.add_argument("--batch-size", type=int, default=LLM_BATCH_SIZE,
                        help=f"Features per gpt-4o-mini API call (default: {LLM_BATCH_SIZE})")
    parser.add_argument("--layer", type=int, default=TARGET_LAYER,
                        help=(
                            "Which layer's features to label (default: 8). "
                            "Resolves to medical_outputs/layer_N/ for multi-layer runs, "
                            "or medical_outputs/ for the legacy single-layer layout."
                        ))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--min-freq", type=float, default=0.0005,
                        help="Min activation frequency (default: 0.05%%)")
    parser.add_argument("--max-freq", type=float, default=0.30,
                        help="Max activation frequency (default: 30%%)")
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable quality filter for gpt-4o-mini candidates")
    args = parser.parse_args()

    # Resolve per-layer directory (mirrors the logic in main.py and server.py)
    layer_subdir = MEDICAL_OUTPUT_DIR / f"layer_{args.layer}"
    output_dir = layer_subdir if layer_subdir.exists() else MEDICAL_OUTPUT_DIR
    features_path = output_dir / "features.json"

    if not features_path.exists():
        print(f"Error: {features_path} not found. Run main.py first.")
        return

    with open(features_path) as f:
        features = json.load(f)

    print(f"Loaded {len(features)} features from {features_path} (layer {args.layer})")

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
    )

    if args.dry_run:
        return

    print_labeled_features(labeled_features)

    output_path = Path(args.output) if args.output else output_dir / "labeled_features.json"
    save_labeled_features(labeled_features, output_path)


if __name__ == "__main__":
    main()
