#!/usr/bin/env python3
"""
Medical Dataset Formatting
==========================
All data loading, cleaning, and streaming for the SAE training pipeline.

Instruction-formatted data is critical for instruct-tuned models: activations
at inference time look like chat templates, not raw prose. Training on plain
corpus text produces coarse, poorly-steerable features.
(FAST paper, arXiv:2506.07691; "SAEs are Highly Dataset Dependent", LessWrong 2024)

Sources (~1/3 each, ~150k total):
  1. Medical MCQ  — medmcqa (primary), bigbio/med_qa (fallback), MedQA-USMLE (2nd fallback)
  2. PubMed QA    — biomedical Q&A with long-form answers
  3. PubMed Abs   — abstracts wrapped as summarisation instructions
"""

from datasets import load_dataset
from typing import Iterator, Tuple


# =============================================================================
# Llama 3.2 Instruct Chat Template
# =============================================================================

_LLAMA_USER = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
_LLAMA_ASST = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
_LLAMA_END  = "<|eot_id|>"


def _chat(question: str, answer: str) -> str:
    """Wrap Q/A pair in Llama 3.2 Instruct chat template."""
    return f"{_LLAMA_USER}{question.strip()}{_LLAMA_ASST}{answer.strip()}{_LLAMA_END}"


# =============================================================================
# Source 1: Medical MCQ  (medmcqa → bigbio/med_qa → MedQA-USMLE fallback chain)
# =============================================================================

def _stream_medmcqa(max_count: int, max_tokens: int) -> Iterator[Tuple[str, str]]:
    """
    Load MedMCQA (Indian medical MCQ).
    Falls back to bigbio/med_qa (USMLE-style) then MedQA-USMLE if unavailable.
    """
    count = 0
    source_name = "medmcqa"

    # ── Primary: medmcqa ────────────────────────────────────────────────────
    try:
        ds = load_dataset("medmcqa", split="train", streaming=True)
        _opt_keys = ["opa", "opb", "opc", "opd"]
        for item in ds:
            q   = item.get("question", "").strip()
            exp = (item.get("exp") or "").strip()
            options = [item.get(k, "") for k in _opt_keys]
            cop = item.get("cop", 0)
            correct = options[cop] if 0 <= cop < len(options) else options[0]
            if not q:
                continue
            opts_str     = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(options) if o)
            question_text = f"{q}\n{opts_str}"
            answer_text   = exp if len(exp) > 20 else f"The correct answer is {chr(65+cop)}: {correct}."
            yield _chat(question_text, answer_text), f"{source_name}:{count}"
            count += 1
            if count >= max_count:
                return
        print(f"  MCQ source (medmcqa): {count} samples loaded")
        return
    except Exception as e:
        print(f"  medmcqa unavailable ({e}), trying bigbio/med_qa...")

    # ── Fallback 1: bigbio/med_qa (USMLE 4-option) ──────────────────────────
    source_name = "med_qa"
    try:
        ds = load_dataset(
            "bigbio/med_qa", "med_qa_en_4options_bigbio_qa",
            split="train", streaming=True, trust_remote_code=True,
        )
        for item in ds:
            q       = (item.get("question") or "").strip()
            choices = item.get("choices", [])
            answer  = item.get("answer", "")
            if isinstance(answer, list):
                answer = answer[0] if answer else ""
            if not q or not choices:
                continue
            opts_str      = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
            question_text = f"{q}\n{opts_str}"
            answer_text   = f"The correct answer is: {answer.strip()}" if answer else choices[0]
            yield _chat(question_text, answer_text), f"{source_name}:{count}"
            count += 1
            if count >= max_count:
                return
        print(f"  MCQ source (bigbio/med_qa): {count} samples loaded")
        return
    except Exception as e:
        print(f"  bigbio/med_qa unavailable ({e}), trying MedQA-USMLE...")

    # ── Fallback 2: GBaker/MedQA-USMLE-4-options ────────────────────────────
    source_name = "medqa_usmle"
    try:
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train", streaming=True)
        for item in ds:
            q      = (item.get("question") or "").strip()
            answer = (item.get("answer") or "").strip()
            opts   = item.get("options", {})
            if not q or not answer:
                continue
            opts_str      = "\n".join(f"{k}. {v}" for k, v in opts.items()) if opts else ""
            question_text = f"{q}\n{opts_str}" if opts_str else q
            yield _chat(question_text, answer), f"{source_name}:{count}"
            count += 1
            if count >= max_count:
                return
        print(f"  MCQ source (MedQA-USMLE): {count} samples loaded")
        return
    except Exception as e:
        print(f"  MedQA-USMLE unavailable ({e})")

    if count == 0:
        print("  WARNING: All medical MCQ sources failed — this third of training data is missing!")
    else:
        print(f"  MCQ: {count} samples loaded from fallback sources")


# =============================================================================
# Source 2: PubMed QA
# =============================================================================

def _stream_pubmed_qa(count_offset: int, max_count: int, max_tokens: int) -> Iterator[Tuple[str, str]]:
    """PubMed QA: biomedical questions with long-form answers."""
    count = count_offset
    try:
        ds = load_dataset(
            "pubmed_qa", "pqa_artificial", split="train",
            streaming=True, trust_remote_code=True,
        )
        for item in ds:
            q      = (item.get("question") or "").strip()
            answer = (item.get("long_answer") or "").strip()
            if not q or not answer:
                ctx_list = (
                    item.get("context", {}).get("contexts", [])
                    if isinstance(item.get("context"), dict)
                    else []
                )
                answer = " ".join(ctx_list)[: max_tokens * 3]
            if q and answer:
                yield _chat(q, answer[: max_tokens * 3]), f"pubmed_qa:{count}"
                count += 1
                if count >= max_count:
                    return
    except Exception as e:
        print(f"  PubMedQA unavailable: {e}")


# =============================================================================
# Source 3: PubMed Abstracts as Summarisation Instructions
# =============================================================================

def _stream_pubmed_abs(count_offset: int, num_samples: int, max_tokens: int) -> Iterator[Tuple[str, str]]:
    """PubMed abstracts wrapped as summarisation/explanation tasks."""
    count = count_offset
    try:
        ds = load_dataset(
            "ccdv/pubmed-summarization", "document", split="train", streaming=True,
        )
        for item in ds:
            abstract = (item.get("abstract") or "").strip()
            if not abstract:
                continue
            # First half = context presented to the model; second half = answer
            mid           = len(abstract) // 2
            question_text = (
                "Summarise and explain the clinical significance of this medical finding:\n\n"
                + abstract[:mid]
            )
            answer_text = abstract[mid: mid + max_tokens * 3]
            if answer_text.strip():
                yield _chat(question_text, answer_text), f"pubmed_abs:{count}"
                count += 1
                if count >= num_samples:
                    return
    except Exception as e:
        print(f"  PubMed abstracts unavailable: {e}")


# =============================================================================
# Public API — used by main.py
# =============================================================================

def stream_medical_texts(num_samples: int, max_tokens: int) -> Iterator[Tuple[str, str]]:
    """
    Stream (text, source_id) tuples formatted as Llama 3.2 Instruct chat conversations.

    Yields approximately num_samples / 3 samples from each of three medical sources.
    Prints a warning if any source fails to load.

    Args:
        num_samples: Total number of samples to stream.
        max_tokens:  Token budget per sample (used to cap answer length).

    Yields:
        (text, source_id) where text is a formatted chat string and source_id
        identifies the origin (e.g. "medmcqa:1234", "pubmed_qa:5678").
    """
    third = num_samples // 3

    # Source 1: Medical MCQ
    src1_count = 0
    for text, source_id in _stream_medmcqa(third, max_tokens):
        yield text, source_id
        src1_count += 1
    print(f"  [dataset] Source 1 (MCQ): {src1_count} / {third} samples")

    # Source 2: PubMed QA
    src2_count = 0
    src2_start = src1_count
    for text, source_id in _stream_pubmed_qa(src2_start, src2_start + third, max_tokens):
        yield text, source_id
        src2_count += 1
    print(f"  [dataset] Source 2 (PubMed QA): {src2_count} / {third} samples")

    # Source 3: PubMed abstracts — fill remainder up to num_samples
    src3_count = 0
    src3_start = src1_count + src2_count
    for text, source_id in _stream_pubmed_abs(src3_start, num_samples, max_tokens):
        yield text, source_id
        src3_count += 1
    print(f"  [dataset] Source 3 (PubMed Abs): {src3_count} samples")

    total = src1_count + src2_count + src3_count
    print(f"  [dataset] Total collected: {total} / {num_samples} samples")
    if total < num_samples * 0.8:
        print(
            f"  WARNING: Only {total}/{num_samples} samples collected "
            f"({100*total/num_samples:.0f}%). Some sources may be unavailable."
        )
