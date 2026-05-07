#!/usr/bin/env python3
"""Comprehensive analysis of SAE training outputs for code and medical domains."""

import json
import os
import sys
from collections import Counter, defaultdict

# Try to import torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: torch not available, skipping source_ids.pt analysis")

BASE = "/Users/baptisteetroy/Desktop/Capstone"
DOMAINS = ["medical_outputs", "code_outputs"]
LAYERS = [4, 8, 12]


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def sep(title="", width=80, char="="):
    if title:
        pad = (width - len(title) - 2) // 2
        print(char * pad + f" {title} " + char * (width - pad - len(title) - 2))
    else:
        print(char * width)


def subsep(title=""):
    print(f"\n--- {title} ---")


# ─────────────────────────────────────────────────────────────────────────────
# Collect all results for cross-domain comparison at the end
# ─────────────────────────────────────────────────────────────────────────────
cross_domain = defaultdict(dict)  # cross_domain[layer][domain] = {...}


def analyze_domain_layer(domain, layer):
    domain_dir = os.path.join(BASE, domain, f"layer_{layer}")
    sep(f"{domain.upper()} — Layer {layer}")

    # ── 1. summary.json ──────────────────────────────────────────────────────
    subsep("1. summary.json")
    summary = load_json(os.path.join(domain_dir, "summary.json"))
    if summary is None:
        print("  summary.json NOT FOUND")
    else:
        fields = ["n_tokens", "n_chunks", "dead_features", "avg_l0_sparsity",
                  "explained_variance", "final_loss"]
        for f in fields:
            print(f"  {f}: {summary.get(f, 'N/A')}")

    # ── 2. training_history.json ──────────────────────────────────────────────
    subsep("2. training_history.json")
    hist = load_json(os.path.join(domain_dir, "training_history.json"))
    if hist is None:
        print("  training_history.json NOT FOUND")
    else:
        for key in ["epoch_losses", "dead_features_per_epoch", "entropy_per_epoch"]:
            val = hist.get(key)
            if val is None:
                print(f"  {key}: N/A")
            elif isinstance(val, list):
                formatted = [f"{v:.6f}" if isinstance(v, float) else str(v) for v in val]
                print(f"  {key} ({len(val)} epochs): [{', '.join(formatted)}]")
            else:
                print(f"  {key}: {val}")

    # ── 3. features.json ─────────────────────────────────────────────────────
    subsep("3. features.json")
    features_path = os.path.join(domain_dir, "features.json")
    features = load_json(features_path)
    if features is None:
        print("  features.json NOT FOUND")
        return

    total = len(features)
    print(f"  Total features: {total}")

    # Alive/dead by frequency
    freqs = [f.get("frequency", 0) for f in features]
    alive = sum(1 for fq in freqs if fq > 0)
    dead_count = total - alive
    print(f"  Alive (freq > 0): {alive}")
    print(f"  Dead (freq == 0): {dead_count}")

    # Frequency buckets
    always_on = sum(1 for fq in freqs if fq > 0.5)
    high      = sum(1 for fq in freqs if 0.1  < fq <= 0.5)
    mid       = sum(1 for fq in freqs if 0.01 < fq <= 0.1)
    low       = sum(1 for fq in freqs if 0.001 < fq <= 0.01)
    rare      = sum(1 for fq in freqs if 0 < fq <= 0.001)
    print(f"  always-on (>0.5):      {always_on}")
    print(f"  high (0.1-0.5):        {high}")
    print(f"  mid (0.01-0.1):        {mid}")
    print(f"  low (0.001-0.01):      {low}")
    print(f"  rare (<0.001):         {rare}")

    # Max activation stats
    max_acts = [f.get("max", 0) for f in features if f.get("max") is not None]
    if max_acts:
        mean_max = sum(max_acts) / len(max_acts)
        max_max  = max(max_acts)
        print(f"  mean of 'max' field: {mean_max:.4f}")
        print(f"  max  of 'max' field: {max_max:.4f}")

    # MaxAct entropy
    entropies = [f.get("maxact_entropy", None) for f in features
                 if f.get("max_activating_tokens") and f.get("maxact_entropy") is not None]
    if entropies:
        mean_ent = sum(entropies) / len(entropies)
        print(f"  mean maxact_entropy (features with examples): {mean_ent:.4f}  (n={len(entropies)})")

    # KL stats
    kl_vals = [f.get("token_change_kl", None) for f in features
               if f.get("token_change_n_contexts", 0) > 0 and f.get("token_change_kl") is not None]
    if kl_vals:
        mean_kl = sum(kl_vals) / len(kl_vals)
        max_kl  = max(kl_vals)
        print(f"  KL mean (token_change_n_contexts>0): {mean_kl:.6f}  (n={len(kl_vals)})")
        print(f"  KL max:                              {max_kl:.6f}")

    # Top 20 most common tokens across all max_activating_tokens
    token_counter = Counter()
    for feat in features:
        for ex in feat.get("max_activating_tokens", []):
            tok = ex.get("token", "")
            if tok:
                token_counter[tok] += 1
    print(f"  Top 20 most common tokens across all maxact examples:")
    for tok, cnt in token_counter.most_common(20):
        print(f"    {repr(tok)}: {cnt}")

    # Source breakdown from source_ids.pt + source_list.json
    subsep("3b. Source breakdown (source_ids.pt)")
    source_ids_path  = os.path.join(domain_dir, "source_ids.pt")
    source_list_path = os.path.join(domain_dir, "source_list.json")
    if HAS_TORCH and os.path.exists(source_ids_path) and os.path.exists(source_list_path):
        try:
            source_ids  = torch.load(source_ids_path, map_location="cpu")
            source_list = load_json(source_list_path)
            # source_ids is a 1D tensor of integer indices into source_list
            id_counter = Counter(source_ids.tolist())
            # group by prefix (split on ":")
            prefix_counter = Counter()
            for idx, cnt in id_counter.items():
                if idx < len(source_list):
                    src = source_list[idx]
                    prefix = src.split(":")[0] if ":" in src else src
                    prefix_counter[prefix] += cnt
                else:
                    prefix_counter[f"unknown_idx_{idx}"] += cnt
            total_tokens = sum(prefix_counter.values())
            print(f"  Total tokens tracked: {total_tokens}")
            for src, cnt in sorted(prefix_counter.items(), key=lambda x: -x[1]):
                pct = 100.0 * cnt / total_tokens if total_tokens else 0
                print(f"    {src}: {cnt}  ({pct:.2f}%)")
        except Exception as e:
            print(f"  ERROR loading source data: {e}")
    else:
        print("  torch not available or files missing — skipping")

    # Top 20 most coherent features
    subsep("3c. Top 20 most coherent features")
    print("  Score = max_activation / (maxact_entropy + 0.1), freq>0.001, >=3 maxact examples")
    scored = []
    for feat in features:
        freq    = feat.get("frequency", 0)
        max_act = feat.get("max", 0) or 0
        entropy = feat.get("maxact_entropy") or 0
        examples = feat.get("max_activating_tokens", [])
        if freq > 0.001 and len(examples) >= 3:
            score = max_act / (entropy + 0.1)
            scored.append((score, feat))
    scored.sort(key=lambda x: -x[0])
    for rank, (score, feat) in enumerate(scored[:20], 1):
        idx      = feat.get("index", "?")
        freq     = feat.get("frequency", 0)
        max_a    = feat.get("max", 0)
        ent      = feat.get("maxact_entropy", None)
        kl       = feat.get("token_change_kl", None)
        tokens   = [ex.get("token", "") for ex in feat.get("max_activating_tokens", [])[:5]]
        vocab    = feat.get("vocab_projection", [])[:4]
        promoted = feat.get("token_change_promoted", [])[:3]
        ent_str  = f"{ent:.4f}" if ent is not None else "N/A"
        kl_str   = f"{kl:.6f}"  if kl  is not None else "N/A"
        print(f"  #{rank:2d} idx={idx:4d}  score={score:8.2f}  freq={freq:.5f}  "
              f"max={max_a:.3f}  entropy={ent_str}  kl={kl_str}")
        print(f"       tokens={tokens}  vocab={vocab}  promoted={promoted}")

    # Store for cross-domain comparison
    cross_domain[layer][domain] = {
        "dead_pct":          100.0 * dead_count / total if total else 0,
        "explained_variance": summary.get("explained_variance") if summary else None,
        "mean_entropy":       (sum(entropies) / len(entropies)) if entropies else None,
        "kl_mean":            (sum(kl_vals) / len(kl_vals)) if kl_vals else None,
        "kl_max":             max(kl_vals) if kl_vals else None,
    }

    # ── 4. labeled_features.json (medical only) ───────────────────────────────
    labeled_path = os.path.join(domain_dir, "labeled_features.json")
    if os.path.exists(labeled_path):
        subsep("4. labeled_features.json")
        labeled = load_json(labeled_path)
        if labeled is None:
            print("  Could not load")
        else:
            conf_counter   = Counter()
            source_counter = Counter()
            for lf in labeled:
                conf   = lf.get("confidence", "unknown")
                lsrc   = lf.get("label_source", "unknown")
                conf_counter[conf]   += 1
                source_counter[lsrc] += 1

            print(f"  Total labeled: {len(labeled)}")
            print("  By confidence:")
            for k, v in sorted(conf_counter.items()):
                print(f"    {k}: {v}")
            print("  By label_source:")
            for k, v in sorted(source_counter.items()):
                print(f"    {k}: {v}")

            # Top 10 by quality_score
            with_score = [(lf.get("quality_score", 0) or 0, lf)
                          for lf in labeled if lf.get("quality_score") is not None]
            with_score.sort(key=lambda x: -x[0])
            print("  Top 10 features by quality_score:")
            for rank, (qs, lf) in enumerate(with_score[:10], 1):
                idx   = lf.get("index", "?")
                label = lf.get("label", "")
                conf  = lf.get("confidence", "")
                lsrc  = lf.get("label_source", "")
                print(f"    #{rank:2d} idx={idx:4d}  score={qs:.4f}  conf={conf}  "
                      f"src={lsrc}  label='{label}'")


def analyze_cross_layer(domain):
    sep(f"CROSS-LAYER ANALYSIS — {domain.upper()}")
    cross_path = os.path.join(BASE, domain, "cross_layer_analysis.json")
    data = load_json(cross_path)
    if data is None:
        print("  cross_layer_analysis.json NOT FOUND")
        return

    print(f"  Top-level keys: {list(data.keys())}")
    print(f"  Layers: {data.get('layers')}")

    # pairs are nested under data["pairs"]
    pairs_container = data.get("pairs", data)  # fallback to top-level if no "pairs" key

    pair_keys = ["4_8", "4_12", "8_12"]
    for pair in pair_keys:
        subsep(f"Pair {pair}")
        pdata = pairs_container.get(pair)
        if pdata is None:
            print(f"  Pair {pair} not found")
            continue
        print(f"  layer_a={pdata.get('layer_a')}  layer_b={pdata.get('layer_b')}  "
              f"n_features_a={pdata.get('n_features_a')}  n_features_b={pdata.get('n_features_b')}  "
              f"similarity_threshold={pdata.get('similarity_threshold')}")
        for direction in ["a_to_b", "b_to_a"]:
            ddata = pdata.get(direction, {})
            mms   = ddata.get("mean_max_similarity", "N/A")
            med   = ddata.get("median_max_similarity", "N/A")
            std   = ddata.get("std_max_similarity", "N/A")
            sfc   = ddata.get("shared_features_count", "N/A")
            sfp   = ddata.get("shared_features_pct", "N/A")
            mms_s = f"{mms:.6f}" if isinstance(mms, float) else str(mms)
            med_s = f"{med:.6f}" if isinstance(med, float) else str(med)
            std_s = f"{std:.6f}" if isinstance(std, float) else str(std)
            sfp_s = f"{sfp:.4f}"  if isinstance(sfp, float) else str(sfp)
            print(f"  {direction}:  mean_max_sim={mms_s}  median={med_s}  std={std_s}  "
                  f"shared_count={sfc}  shared_pct={sfp_s}%")

        # Top 3 most-targeted layer_b_feature (many-to-one convergence)
        top_pairs = pdata.get("top_shared_pairs", [])
        if top_pairs:
            target_counter = Counter()
            for sp in top_pairs:
                # Key is "layer_b_feature" based on observed structure
                b_feat = sp.get("layer_b_feature", sp.get("b_feature", sp.get("feature_b", None)))
                if b_feat is not None:
                    target_counter[b_feat] += 1
            print(f"  Top 3 most-targeted b-features (many-to-one convergence):")
            for tgt, cnt in target_counter.most_common(3):
                print(f"    layer_b_feature={tgt}: targeted by {cnt} layer_a features")
            print(f"  Sample top_shared_pairs (first 5 of {len(top_pairs)}):")
            for sp in top_pairs[:5]:
                print(f"    {sp}")
        else:
            print("  top_shared_pairs: empty or not present")

        # Convergence patterns
        conv = pdata.get("convergence_patterns", {})
        if conv:
            print(f"  convergence_patterns: {conv}")


def cross_domain_compare():
    sep("CROSS-DOMAIN COMPARISON (code vs medical, same layer)")
    header = f"  {'Layer':<8} {'Domain':<20} {'Dead%':>8} {'ExplVar':>10} {'MeanEnt':>10} {'KL_mean':>10} {'KL_max':>10}"
    print(header)
    print("  " + "-" * 76)
    for layer in LAYERS:
        for domain in DOMAINS:
            d = cross_domain[layer].get(domain, {})
            dp  = d.get("dead_pct")
            ev  = d.get("explained_variance")
            me  = d.get("mean_entropy")
            km  = d.get("kl_mean")
            kmx = d.get("kl_max")
            dp_s  = f"{dp:.2f}%"  if dp  is not None else "N/A"
            ev_s  = f"{ev:.4f}"   if ev  is not None else "N/A"
            me_s  = f"{me:.4f}"   if me  is not None else "N/A"
            km_s  = f"{km:.6f}"   if km  is not None else "N/A"
            kmx_s = f"{kmx:.6f}"  if kmx is not None else "N/A"
            print(f"  {layer:<8} {domain:<20} {dp_s:>8} {ev_s:>10} {me_s:>10} {km_s:>10} {kmx_s:>10}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
sep("SAE TRAINING OUTPUT COMPREHENSIVE ANALYSIS", char="═")
print(f"Base directory: {BASE}")
print(f"Domains: {DOMAINS}")
print(f"Layers: {LAYERS}")
print(f"torch available: {HAS_TORCH}")

for domain in DOMAINS:
    for layer in LAYERS:
        analyze_domain_layer(domain, layer)
        print()

for domain in DOMAINS:
    analyze_cross_layer(domain)
    print()

cross_domain_compare()

sep("ANALYSIS COMPLETE", char="═")
