import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import math
    import os
    import random
    import sys
    import time
    from contextlib import nullcontext
    from dataclasses import asdict, dataclass
    from pathlib import Path
    from typing import Any

    import marimo as mo

    try:
        import pandas as pd
    except ModuleNotFoundError:
        pd = None

    try:
        import numpy as np
    except ModuleNotFoundError:
        np = None

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        plt = None

    try:
        import torch
        import torch.nn.functional as F
        from torch import nn
        from torch.nn.utils.rnn import pad_sequence
        from torch.utils.checkpoint import checkpoint as torch_checkpoint
        from torch.utils.data import DataLoader, Dataset
    except ModuleNotFoundError:
        torch = None
        F = None
        nn = None
        pad_sequence = None
        torch_checkpoint = None
        DataLoader = None
        Dataset = object

    try:
        from tokenizers import Tokenizer
    except ModuleNotFoundError:
        Tokenizer = None

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    return (
        DataLoader,
        Dataset,
        F,
        Path,
        Tokenizer,
        asdict,
        dataclass,
        json,
        math,
        mo,
        nn,
        np,
        nullcontext,
        pad_sequence,
        pd,
        plt,
        time,
        torch,
        torch_checkpoint,
    )


@app.cell
def _(mo):
    mo.md("""
    <style>
    .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 8px;
        background: linear-gradient(135deg, #102033 0%, #17324d 52%, #23415f 100%);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0 0 0.35rem 0;
        font-size: 2rem;
        letter-spacing: 0;
    }
    .hero p {
        margin: 0;
        max-width: 980px;
        color: #d8e6f2;
        line-height: 1.45;
    }
    .cards {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 0.75rem;
        margin: 0.75rem 0 1rem 0;
    }
    .card {
        border: 1px solid #d9e2ec;
        border-radius: 8px;
        padding: 0.85rem;
        background: #ffffff;
    }
    .card b {
        display: block;
        color: #14283d;
        margin-bottom: 0.25rem;
    }
    .card span {
        color: #4f6478;
        font-size: 0.92rem;
    }
    .ok { color: #18794e; font-weight: 600; }
    .warn { color: #b26b00; font-weight: 600; }
    .bad { color: #b42318; font-weight: 600; }
    .note {
        border-left: 4px solid #3a6ea5;
        padding: 0.7rem 0.9rem;
        background: #f4f8fb;
        border-radius: 0 8px 8px 0;
    }
    </style>

    <div class="hero">
      <h1>Gurmukhi Decoder-Only SLM Training</h1>
      <p>
      A research notebook for pretraining a compact EN &lt;-&gt; Punjabi Gurmukhi
      decoder-only translation model, then preparing it for teacher refinement
      and low-bit deployment experiments.
      </p>
    </div>

    <div class="cards">
      <div class="card"><b>Stage 1</b><span>Train a modern causal decoder on the cleaned parallel corpus.</span></div>
      <div class="card"><b>Stage 2</b><span>Refine with teacher-generated or teacher-scored translations.</span></div>
      <div class="card"><b>Stage 3</b><span>Run a quantization ladder: 8-bit, 4-bit, 1.58-bit, and true 1-bit.</span></div>
    </div>

    <div class="note">
    Research snapshot date: <b>2026-06-23</b>. This notebook intentionally replaces the older planning notes for the decoder-only branch.
    </div>
    """)
    return


@app.cell
def _(Path):
    PROJECT_ROOT = Path.cwd()
    DATA_PATH = PROJECT_ROOT / "datasets" / "cleaned.tsv"
    TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "hf_bpe24k_tokenizer.json"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "gur_slm_decoder"
    METRICS_DIR = CHECKPOINT_DIR / "metrics"
    return CHECKPOINT_DIR, DATA_PATH, METRICS_DIR, TOKENIZER_PATH


@app.cell
def _(DATA_PATH, TOKENIZER_PATH, json):
    def _format_size(path):
        if not path.exists():
            return "missing"
        size = path.stat().st_size
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024 or unit == "GB":
                return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
            size /= 1024

    def read_tokenizer_metadata(path):
        if not path.exists():
            return {
                "vocab_size": None,
                "special_tokens": {},
                "model_type": None,
                "normalizer": None,
                "pre_tokenizer": None,
            }
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        vocab = data.get("model", {}).get("vocab", {})
        special_tokens = [
            "<pad>",
            "<unk>",
            "<s>",
            "</s>",
            "<2en>",
            "<2pa>",
            "<legal>",
            "<general>",
            "<literal>",
            "<natural>",
        ]
        return {
            "vocab_size": len(vocab),
            "special_tokens": {token: vocab.get(token) for token in special_tokens},
            "model_type": data.get("model", {}).get("type"),
            "normalizer": data.get("normalizer", {}).get("type"),
            "pre_tokenizer": data.get("pre_tokenizer", {}).get("type"),
        }

    artifact_status = {
        "cleaned_tsv": {
            "path": DATA_PATH,
            "exists": DATA_PATH.exists(),
            "size": _format_size(DATA_PATH),
        },
        "tokenizer_json": {
            "path": TOKENIZER_PATH,
            "exists": TOKENIZER_PATH.exists(),
            "size": _format_size(TOKENIZER_PATH),
        },
    }
    tokenizer_meta = read_tokenizer_metadata(TOKENIZER_PATH)
    return artifact_status, tokenizer_meta


@app.cell
def _(artifact_status, mo, tokenizer_meta):
    _artifact_lines = []
    for _name, _info in artifact_status.items():
        _css = "ok" if _info["exists"] else "bad"
        _artifact_lines.append(
            f"<li><span class='{_css}'>{_name}</span>: `{_info['path']}` ({_info['size']})</li>"
        )

    _specials = tokenizer_meta["special_tokens"]
    _missing_specials = [token for token, token_id in _specials.items() if token_id is None]
    _special_status = (
        "<span class='ok'>all required special tokens found</span>"
        if not _missing_specials
        else f"<span class='bad'>missing {', '.join(_missing_specials)}</span>"
    )

    mo.md(
        f"""
        ## Artifact Check

        <ul>
        {''.join(_artifact_lines)}
        </ul>

        Tokenizer: **{tokenizer_meta['vocab_size']:,}** tokens, model **{tokenizer_meta['model_type']}**,
        normalizer **{tokenizer_meta['normalizer']}**, pre-tokenizer **{tokenizer_meta['pre_tokenizer']}**.

        Special-token check: {_special_status}.
        """
    )
    return


@app.cell
def _(DataLoader, F, Tokenizer, mo, nn, np, pad_sequence, pd, plt, torch):
    dependency_rows = [
        ("marimo", "loaded", True),
        ("pandas", "loaded" if pd is not None else "missing", pd is not None),
        ("numpy", "loaded" if np is not None else "missing", np is not None),
        ("matplotlib", "loaded" if plt is not None else "missing", plt is not None),
        ("torch", "loaded" if torch is not None else "missing", torch is not None),
        ("tokenizers", "loaded" if Tokenizer is not None else "missing", Tokenizer is not None),
        ("torch DataLoader", "loaded" if DataLoader is not None else "missing", DataLoader is not None),
        ("torch pad_sequence", "loaded" if pad_sequence is not None else "missing", pad_sequence is not None),
        ("torch nn/F", "loaded" if nn is not None and F is not None else "missing", nn is not None and F is not None),
    ]

    _lines = [
        f"- **{name}**: <span class='{'ok' if ok else 'bad'}'>{status}</span>"
        for name, status, ok in dependency_rows
    ]

    mo.md("## Runtime Dependencies\n\n" + "\n".join(_lines))
    return


@app.cell
def _(mo, pd):
    research_references = [
        {
            "phase": "Architecture",
            "reference": "LLaMA: Open and Efficient Foundation Language Models",
            "year": 2023,
            "url": "https://arxiv.org/abs/2302.13971",
            "use": "Decoder-only baseline pattern: pre-norm transformer, RoPE, RMSNorm, SwiGLU, tied embedding style.",
        },
        {
            "phase": "Architecture",
            "reference": "RoFormer: Enhanced Transformer with Rotary Position Embedding",
            "year": 2021,
            "url": "https://arxiv.org/abs/2104.09864",
            "use": "Replace absolute sinusoidal positions with RoPE for relative-position behavior in causal attention.",
        },
        {
            "phase": "Architecture",
            "reference": "Root Mean Square Layer Normalization",
            "year": 2019,
            "url": "https://arxiv.org/abs/1910.07467",
            "use": "Keep RMSNorm: cheaper than LayerNorm and common in modern decoder-only LMs.",
        },
        {
            "phase": "Architecture",
            "reference": "GLU Variants Improve Transformer",
            "year": 2020,
            "url": "https://arxiv.org/abs/2002.05202",
            "use": "Keep SwiGLU feed-forward blocks rather than ReLU/GELU MLPs.",
        },
        {
            "phase": "Architecture",
            "reference": "FlashAttention-2",
            "year": 2023,
            "url": "https://arxiv.org/abs/2307.08691",
            "use": "Use PyTorch scaled-dot-product attention so CUDA runtimes can select efficient attention kernels.",
        },
        {
            "phase": "Translation",
            "reference": "Samanantar",
            "year": 2021,
            "url": "https://arxiv.org/abs/2104.05596",
            "use": "Indic parallel corpus work: motivates careful corpus filtering and domain tracking.",
        },
        {
            "phase": "Translation",
            "reference": "No Language Left Behind",
            "year": 2022,
            "url": "https://arxiv.org/abs/2207.04672",
            "use": "Low-resource MT lessons: mining, quality filtering, language coverage, and safety evaluation.",
        },
        {
            "phase": "Translation",
            "reference": "IndicTrans2",
            "year": 2023,
            "url": "https://arxiv.org/abs/2305.16307",
            "use": "High-quality Indic MT reference point for benchmarks, corpus construction, and language tags.",
        },
        {
            "phase": "Translation",
            "reference": "Tower",
            "year": 2024,
            "url": "https://arxiv.org/abs/2402.17733",
            "use": "Modern LLM translation recipe: continued pretraining plus translation-workflow instruction tuning.",
        },
        {
            "phase": "Translation",
            "reference": "Babel",
            "year": 2025,
            "url": "https://arxiv.org/abs/2503.00865",
            "use": "Recent multilingual LLM direction for broader language coverage and layer-extension scaling.",
        },
        {
            "phase": "Tokenization",
            "reference": "Neural Machine Translation of Rare Words with Subword Units",
            "year": 2015,
            "url": "https://arxiv.org/abs/1508.07909",
            "use": "BPE/subword tokenization remains appropriate for rare words, names, and morphology.",
        },
        {
            "phase": "Tokenization",
            "reference": "SentencePiece",
            "year": 2018,
            "url": "https://arxiv.org/abs/1808.06226",
            "use": "Language-independent raw-text subword tokenization; useful alternative if retraining tokenizer.",
        },
        {
            "phase": "Tokenization",
            "reference": "Subword Regularization",
            "year": 2018,
            "url": "https://arxiv.org/abs/1804.10959",
            "use": "Future robustness option for low-resource/out-of-domain translation if using unigram tokenization.",
        },
        {
            "phase": "Teacher refinement",
            "reference": "Sequence-Level Knowledge Distillation",
            "year": 2016,
            "url": "https://arxiv.org/abs/1606.07947",
            "use": "Teacher translations can simplify NMT targets and improve small student decoding.",
        },
        {
            "phase": "Teacher refinement",
            "reference": "On-Policy Distillation of Language Models",
            "year": 2023,
            "url": "https://arxiv.org/abs/2306.13649",
            "use": "GKD trains students on self-generated outputs to reduce autoregressive train-test mismatch.",
        },
        {
            "phase": "Teacher refinement",
            "reference": "MiniLLM",
            "year": 2023,
            "url": "https://arxiv.org/abs/2306.08543",
            "use": "Reverse-KL style distillation can be better for generative student models than standard forward KL.",
        },
        {
            "phase": "Teacher refinement",
            "reference": "Minitron",
            "year": 2024,
            "url": "https://arxiv.org/abs/2408.11796",
            "use": "Pruning plus distillation is a practical path once a larger teacher or baseline is available.",
        },
        {
            "phase": "Teacher refinement",
            "reference": "On-Policy Context Distillation",
            "year": 2026,
            "url": "https://arxiv.org/abs/2602.12275",
            "use": "Recent direction: distill context-conditioned teacher behavior into a smaller student.",
        },
        {
            "phase": "Quantization",
            "reference": "SmoothQuant",
            "year": 2022,
            "url": "https://arxiv.org/abs/2211.10438",
            "use": "Practical W8A8 post-training quantization path for 8-bit inference.",
        },
        {
            "phase": "Quantization",
            "reference": "GPTQ",
            "year": 2022,
            "url": "https://arxiv.org/abs/2210.17323",
            "use": "One-shot 3/4-bit weight quantization baseline for decoder-only models.",
        },
        {
            "phase": "Quantization",
            "reference": "AWQ",
            "year": 2023,
            "url": "https://arxiv.org/abs/2306.00978",
            "use": "Activation-aware 4-bit weight-only quantization for deployment.",
        },
        {
            "phase": "Quantization",
            "reference": "QLoRA",
            "year": 2023,
            "url": "https://arxiv.org/abs/2305.14314",
            "use": "4-bit finetuning recipe if the model is converted to a Hugging Face style stack.",
        },
        {
            "phase": "Quantization",
            "reference": "BitNet",
            "year": 2023,
            "url": "https://arxiv.org/abs/2310.11453",
            "use": "Native low-bit training via BitLinear rather than naive post-training binarization.",
        },
        {
            "phase": "Quantization",
            "reference": "BitNet b1.58",
            "year": 2024,
            "url": "https://arxiv.org/abs/2402.17764",
            "use": "Ternary {-1, 0, 1} weights are the serious 1.58-bit target.",
        },
        {
            "phase": "Quantization",
            "reference": "BitNet b1.58 2B4T Technical Report",
            "year": 2025,
            "url": "https://arxiv.org/abs/2504.12285",
            "use": "Open 2B native 1.58-bit evidence: train low-bit natively when possible.",
        },
        {
            "phase": "Quantization",
            "reference": "Scaling Laws for Precision",
            "year": 2024,
            "url": "https://arxiv.org/abs/2411.04330",
            "use": "Low precision changes effective parameter count; compare low-bit models against compute and data budgets.",
        },
        {
            "phase": "Quantization",
            "reference": "Low-Bit Quantization Favors Undertrained LLMs",
            "year": 2024,
            "url": "https://arxiv.org/abs/2411.17691",
            "use": "Quantization degradation depends on training level; do not trust one checkpoint or one calibration set.",
        },
    ]

    if pd is not None:
        research_reference_frame = pd.DataFrame(research_references)
        _display = research_reference_frame
    else:
        research_reference_frame = None
        _display = "\n".join(
            f"- [{row['reference']}]({row['url']}) ({row['year']}): {row['use']}"
            for row in research_references
        )

    mo.md("## Research Reference Map")
    _display
    return (research_references,)


@app.cell
def _(mo):
    mo.md("""
    ## Decoder Critique and Changes

    **What is already good in `decoder.py`:**

    - Uses a decoder-only causal format, which is the right direction if later deployment should look like a small GPT/LLaMA artifact.
    - Uses RMSNorm, SwiGLU feed-forward blocks, tied embeddings, AMP, AdamW, gradient clipping, checkpointing, and bidirectional EN <-> PA prompt construction.
    - Keeps `loader_workers = 0`, which is conservative for marimo and Windows/spawn-style notebook runtimes.

    **What needed tightening for this research branch:**

    - `nn.MultiheadAttention` plus sinusoidal positions is older than the current decoder-only recipe. This notebook uses direct scaled-dot-product causal attention with RoPE.
    - The previous dataset can truncate away too much target text when prompts are long. This notebook reserves target budget before building labels.
    - Label smoothing was hardcoded in the training loop. It is now configurable.
    - The notebook now records metric history and plots loss, perplexity, token accuracy, corpus balance, length distributions, and token-length distributions.
    - `tokenization.py` references `pd` without importing pandas in its import cell. The saved tokenizer JSON is usable, but retraining the tokenizer should fix that notebook cell before rerun.

    **Research choice:** start with the full-precision modern decoder, establish clean metrics, then branch into teacher refinement and quantization. 1.58-bit and true 1-bit should be treated as separate low-bit training experiments, not as simple final-file conversions.
    """)
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Data and Tokenizer Audit

        Use a sample first. Full-corpus audit on a 1 GB TSV can take a while in a notebook.
        """
    )

    audit_sample_rows = mo.ui.number(
        value=100_000,
        start=1_024,
        stop=2_000_000,
        step=10_000,
        label="Audit sample rows",
    )
    audit_random_sample = mo.ui.checkbox(value=False, label="Randomize after loading sample")
    run_audit_button = mo.ui.run_button(label="Run data audit")

    mo.hstack([audit_sample_rows, audit_random_sample, run_audit_button])
    return audit_random_sample, audit_sample_rows, run_audit_button


@app.cell
def _(DATA_PATH, audit_random_sample, audit_sample_rows, pd, run_audit_button):
    if run_audit_button.value and pd is None:
        audit_frame = None
        audit_status = "pandas is not installed in this runtime."
    elif run_audit_button.value and not DATA_PATH.exists():
        audit_frame = None
        audit_status = f"Missing data file: {DATA_PATH}"
    elif run_audit_button.value:
        _usecols = [
            "id",
            "source",
            "domain",
            "en",
            "pa",
            "en_chars",
            "pa_chars",
            "en_words",
            "pa_words",
            "hard_drop",
            "review_flag",
        ]
        audit_frame = pd.read_csv(
            DATA_PATH,
            sep="\t",
            usecols=[col for col in _usecols],
            dtype={
                "id": "string",
                "source": "category",
                "domain": "category",
                "en": "string",
                "pa": "string",
                "en_chars": "int32",
                "pa_chars": "int32",
                "en_words": "int32",
                "pa_words": "int32",
                "hard_drop": "boolean",
                "review_flag": "boolean",
            },
            nrows=int(audit_sample_rows.value),
        )
        audit_frame["en"] = audit_frame["en"].fillna("")
        audit_frame["pa"] = audit_frame["pa"].fillna("")
        if audit_random_sample.value:
            audit_frame = audit_frame.sample(frac=1.0, random_state=42).reset_index(drop=True)
        audit_status = f"Loaded {len(audit_frame):,} rows from {DATA_PATH.name}."
    else:
        audit_frame = None
        audit_status = "Audit has not been run yet."

    audit_status
    return audit_frame, audit_status


@app.cell
def _(audit_frame, audit_status, mo, pd):
    if audit_frame is None or pd is None:
        corpus_summary = None
        domain_source_counts = None
        quality_summary = None
        audit_summary_view = mo.md(f"**Audit status:** {audit_status}")
    else:
        _length_view = audit_frame.assign(
            char_ratio=(audit_frame["pa_chars"] / audit_frame["en_chars"].clip(lower=1)).round(2),
            word_ratio=(audit_frame["pa_words"] / audit_frame["en_words"].clip(lower=1)).round(2),
        )
        corpus_summary = _length_view[
            ["en_chars", "pa_chars", "en_words", "pa_words", "char_ratio", "word_ratio"]
        ].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).round(2)

        domain_source_counts = (
            audit_frame.groupby(["source", "domain"], observed=True)
            .size()
            .reset_index(name="rows")
            .sort_values("rows", ascending=False)
        )

        def _has_gurmukhi(value):
            return any(0x0A00 <= ord(char) <= 0x0A7F for char in str(value))

        def _has_latin(value):
            return any(("A" <= char <= "Z") or ("a" <= char <= "z") for char in str(value))

        _en_has_gurmukhi = audit_frame["en"].map(_has_gurmukhi)
        _pa_has_gurmukhi = audit_frame["pa"].map(_has_gurmukhi)
        _pa_has_latin = audit_frame["pa"].map(_has_latin)

        quality_summary = pd.DataFrame(
            {
                "check": [
                    "rows",
                    "empty English",
                    "empty Punjabi",
                    "English contains Gurmukhi",
                    "Punjabi has no Gurmukhi",
                    "Punjabi contains Latin",
                    "review_flag rows",
                    "hard_drop rows",
                ],
                "rows": [
                    len(audit_frame),
                    int((audit_frame["en"].str.strip() == "").sum()),
                    int((audit_frame["pa"].str.strip() == "").sum()),
                    int(_en_has_gurmukhi.sum()),
                    int((~_pa_has_gurmukhi).sum()),
                    int(_pa_has_latin.sum()),
                    int(audit_frame["review_flag"].fillna(False).sum()) if "review_flag" in audit_frame else 0,
                    int(audit_frame["hard_drop"].fillna(False).sum()) if "hard_drop" in audit_frame else 0,
                ],
            }
        )

        audit_summary_view = mo.vstack(
            [
                mo.md(f"**Audit status:** {audit_status}"),
                quality_summary,
                domain_source_counts.head(20),
                corpus_summary,
            ]
        )

    audit_summary_view
    return


@app.cell
def _(audit_frame, mo, plt):
    if audit_frame is None:
        audit_visual = mo.md("Run the data audit to render corpus visualizations.")
    elif plt is None:
        audit_visual = mo.md("matplotlib is not installed, so plots cannot be rendered.")
    else:
        _plot_frame = audit_frame.assign(
            word_ratio=audit_frame["pa_words"] / audit_frame["en_words"].clip(lower=1),
            en_words_clipped=audit_frame["en_words"].clip(upper=180),
            pa_words_clipped=audit_frame["pa_words"].clip(upper=180),
        )
        _fig, _axes = plt.subplots(2, 2, figsize=(13, 8))

        _source_counts = audit_frame["source"].astype(str).value_counts().sort_values()
        _source_counts.plot(kind="barh", ax=_axes[0, 0], color="#3a6ea5")
        _axes[0, 0].set_title("Rows by source")
        _axes[0, 0].set_xlabel("rows")

        _domain_counts = audit_frame["domain"].astype(str).value_counts().sort_values()
        _domain_counts.plot(kind="barh", ax=_axes[0, 1], color="#2f9c95")
        _axes[0, 1].set_title("Rows by domain")
        _axes[0, 1].set_xlabel("rows")

        _axes[1, 0].hist(
            [_plot_frame["en_words_clipped"], _plot_frame["pa_words_clipped"]],
            bins=50,
            label=["English", "Punjabi"],
            color=["#3a6ea5", "#bf6f24"],
            alpha=0.72,
        )
        _axes[1, 0].set_title("Sentence length distribution, clipped at 180 words")
        _axes[1, 0].set_xlabel("words")
        _axes[1, 0].set_ylabel("rows")
        _axes[1, 0].legend()

        _axes[1, 1].hist(_plot_frame["word_ratio"].clip(upper=4), bins=60, color="#7b5ea7", alpha=0.82)
        _axes[1, 1].set_title("Punjabi / English word ratio, clipped at 4")
        _axes[1, 1].set_xlabel("word ratio")
        _axes[1, 1].set_ylabel("rows")

        _fig.tight_layout()
        audit_visual = _fig

    audit_visual
    return


@app.cell
def _(TOKENIZER_PATH, Tokenizer, audit_frame, mo, pd, tokenizer_meta):
    if Tokenizer is None:
        hf_tokenizer = None
        tokenizer_status = "tokenizers is not installed; only JSON metadata is available."
        tokenizer_length_frame = None
        tokenizer_stats_frame = None
    elif not TOKENIZER_PATH.exists():
        hf_tokenizer = None
        tokenizer_status = f"Missing tokenizer file: {TOKENIZER_PATH}"
        tokenizer_length_frame = None
        tokenizer_stats_frame = None
    else:
        hf_tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        tokenizer_status = (
            f"Loaded tokenizer with {hf_tokenizer.get_vocab_size():,} tokens. "
            f"Special IDs: {tokenizer_meta['special_tokens']}"
        )

        if audit_frame is not None and pd is not None:
            _sample = audit_frame.head(min(20_000, len(audit_frame))).copy()
            _sample["en_tokens"] = _sample["en"].map(lambda text: len(hf_tokenizer.encode(str(text)).ids))
            _sample["pa_tokens"] = _sample["pa"].map(lambda text: len(hf_tokenizer.encode(str(text)).ids))
            _sample["en_tokens_per_word"] = _sample["en_tokens"] / _sample["en_words"].clip(lower=1)
            _sample["pa_tokens_per_word"] = _sample["pa_tokens"] / _sample["pa_words"].clip(lower=1)
            tokenizer_length_frame = _sample[
                ["source", "domain", "en_words", "pa_words", "en_tokens", "pa_tokens", "en_tokens_per_word", "pa_tokens_per_word"]
            ]
            tokenizer_stats_frame = tokenizer_length_frame[
                ["en_tokens", "pa_tokens", "en_tokens_per_word", "pa_tokens_per_word"]
            ].describe(percentiles=[0.05, 0.5, 0.95, 0.99]).round(2)
        else:
            tokenizer_length_frame = None
            tokenizer_stats_frame = None

    mo.md(f"## Tokenizer Audit\n\n{tokenizer_status}")
    tokenizer_stats_frame if tokenizer_stats_frame is not None else tokenizer_meta
    return (tokenizer_length_frame,)


@app.cell
def _(mo, plt, tokenizer_length_frame):
    if tokenizer_length_frame is None:
        tokenizer_visual = mo.md("Run the data audit with `tokenizers` installed to render token-length plots.")
    elif plt is None:
        tokenizer_visual = mo.md("matplotlib is not installed, so tokenizer plots cannot be rendered.")
    else:
        _fig, _axes = plt.subplots(1, 2, figsize=(13, 4.5))
        _axes[0].hist(
            [
                tokenizer_length_frame["en_tokens"].clip(upper=320),
                tokenizer_length_frame["pa_tokens"].clip(upper=320),
            ],
            bins=50,
            color=["#3a6ea5", "#bf6f24"],
            label=["English", "Punjabi"],
            alpha=0.75,
        )
        _axes[0].set_title("Token lengths, clipped at 320")
        _axes[0].set_xlabel("tokens")
        _axes[0].set_ylabel("rows")
        _axes[0].legend()

        _axes[1].hist(
            [
                tokenizer_length_frame["en_tokens_per_word"].clip(upper=5),
                tokenizer_length_frame["pa_tokens_per_word"].clip(upper=5),
            ],
            bins=50,
            color=["#3a6ea5", "#bf6f24"],
            label=["English", "Punjabi"],
            alpha=0.75,
        )
        _axes[1].set_title("Tokens per word, clipped at 5")
        _axes[1].set_xlabel("tokens per word")
        _axes[1].legend()
        _fig.tight_layout()
        tokenizer_visual = _fig

    tokenizer_visual
    return


@app.cell
def _(dataclass):
    @dataclass
    class DecoderTrainConfig:
        vocab_size: int
        pad_id: int
        eos_id: int
        profile: str = "base"
        max_seq_len: int = 256
        min_target_tokens: int = 48
        d_model: int = 512
        nhead: int = 8
        num_layers: int = 8
        dim_feedforward: int = 2048
        dropout: float = 0.1
        rope_base: float = 10000.0
        gradient_checkpointing: bool = False
        batch_size: int = 32
        epochs: int = 1
        lr: float = 3e-4
        min_lr_ratio: float = 0.05
        warmup_steps: int = 2000
        weight_decay: float = 0.1
        grad_accum_steps: int = 1
        clip_grad_norm: float = 1.0
        label_smoothing: float = 0.05
        val_rows: int = 10_000
        val_fraction: float = 0.01
        seed: int = 42
        amp: str = "bf16"
        style_tag: str = "<natural>"

    MODEL_PROFILES = {
        "nano": {
            "d_model": 256,
            "nhead": 4,
            "num_layers": 4,
            "dim_feedforward": 1024,
        },
        "tiny": {
            "d_model": 384,
            "nhead": 6,
            "num_layers": 6,
            "dim_feedforward": 1536,
        },
        "base": {
            "d_model": 512,
            "nhead": 8,
            "num_layers": 8,
            "dim_feedforward": 2048,
        },
        "wide": {
            "d_model": 768,
            "nhead": 12,
            "num_layers": 10,
            "dim_feedforward": 3072,
        },
    }
    return DecoderTrainConfig, MODEL_PROFILES


@app.cell
def _(F, math, nn, torch, torch_checkpoint):
    RMSNorm = None
    RotaryEmbedding = None
    CausalSelfAttention = None
    SwiGLUFeedForward = None
    DecoderBlock = None
    ModernDecoderOnlyTransformer = None

    if torch is not None and nn is not None and F is not None:

        class RMSNorm(nn.Module):
            def __init__(self, dim: int, eps: float = 1e-6) -> None:
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(dim))

            def forward(self, x):
                dtype = x.dtype
                x = x.float()
                x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
                return (self.weight * x).to(dtype)


        def _rotate_half(x):
            x_even = x[..., 0::2]
            x_odd = x[..., 1::2]
            return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


        class RotaryEmbedding(nn.Module):
            def __init__(self, dim: int, base: float = 10000.0) -> None:
                super().__init__()
                if dim % 2 != 0:
                    raise ValueError("RoPE head dimension must be even")
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
                self.register_buffer("inv_freq", inv_freq, persistent=False)

            def forward(self, seq_len: int, device, dtype):
                positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
                freqs = torch.outer(positions, self.inv_freq.to(device))
                cos = freqs.cos().repeat_interleave(2, dim=-1).to(dtype=dtype)
                sin = freqs.sin().repeat_interleave(2, dim=-1).to(dtype=dtype)
                return cos[None, None, :, :], sin[None, None, :, :]


        def _apply_rope(x, cos, sin):
            return (x * cos) + (_rotate_half(x) * sin)


        class CausalSelfAttention(nn.Module):
            def __init__(self, d_model: int, nhead: int, dropout: float, rope_base: float) -> None:
                super().__init__()
                if d_model % nhead != 0:
                    raise ValueError("d_model must be divisible by nhead")
                self.nhead = nhead
                self.head_dim = d_model // nhead
                self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
                self.output = nn.Linear(d_model, d_model, bias=False)
                self.rope = RotaryEmbedding(self.head_dim, base=rope_base)
                self.attn_dropout = dropout
                self.resid_dropout = nn.Dropout(dropout)

            def forward(self, x, padding_mask):
                batch_size, seq_len, width = x.shape
                qkv = (
                    self.qkv(x)
                    .view(batch_size, seq_len, 3, self.nhead, self.head_dim)
                    .permute(2, 0, 3, 1, 4)
                )
                q, k, v = qkv[0], qkv[1], qkv[2]
                cos, sin = self.rope(seq_len, x.device, q.dtype)
                q = _apply_rope(q, cos, sin)
                k = _apply_rope(k, cos, sin)

                dropout_p = self.attn_dropout if self.training else 0.0
                if hasattr(F, "scaled_dot_product_attention"):
                    if padding_mask is None:
                        attn_out = F.scaled_dot_product_attention(
                            q,
                            k,
                            v,
                            dropout_p=dropout_p,
                            is_causal=True,
                        )
                    else:
                        causal = torch.ones(
                            seq_len,
                            seq_len,
                            device=x.device,
                            dtype=torch.bool,
                        ).tril()
                        key_keep = ~padding_mask[:, None, None, :]
                        attn_mask = key_keep & causal[None, None, :, :]
                        attn_out = F.scaled_dot_product_attention(
                            q,
                            k,
                            v,
                            attn_mask=attn_mask,
                            dropout_p=dropout_p,
                            is_causal=False,
                        )
                else:
                    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                    causal = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
                    scores = scores.masked_fill(~causal[None, None, :, :], torch.finfo(scores.dtype).min)
                    if padding_mask is not None:
                        scores = scores.masked_fill(padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
                    attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
                    attn = F.dropout(attn, p=dropout_p, training=self.training)
                    attn_out = torch.matmul(attn, v)

                attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, width)
                return self.resid_dropout(self.output(attn_out))


        class SwiGLUFeedForward(nn.Module):
            def __init__(self, d_model: int, dim_feedforward: int, dropout: float) -> None:
                super().__init__()
                self.w12 = nn.Linear(d_model, dim_feedforward * 2, bias=False)
                self.w3 = nn.Linear(dim_feedforward, d_model, bias=False)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                value, gate = self.w12(x).chunk(2, dim=-1)
                return self.w3(self.dropout(value * F.silu(gate)))


        class DecoderBlock(nn.Module):
            def __init__(
                self,
                d_model: int,
                nhead: int,
                dim_feedforward: int,
                dropout: float,
                rope_base: float,
            ) -> None:
                super().__init__()
                self.attn_norm = RMSNorm(d_model)
                self.ffn_norm = RMSNorm(d_model)
                self.attn = CausalSelfAttention(d_model, nhead, dropout, rope_base)
                self.ffn = SwiGLUFeedForward(d_model, dim_feedforward, dropout)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x, padding_mask):
                x = x + self.attn(self.attn_norm(x), padding_mask)
                x = x + self.dropout(self.ffn(self.ffn_norm(x)))
                return x


        class ModernDecoderOnlyTransformer(nn.Module):
            def __init__(
                self,
                vocab_size: int,
                pad_id: int,
                d_model: int,
                nhead: int,
                num_layers: int,
                dim_feedforward: int,
                dropout: float,
                max_seq_len: int,
                rope_base: float,
                gradient_checkpointing: bool = False,
            ) -> None:
                super().__init__()
                self.pad_id = pad_id
                self.max_seq_len = max_seq_len
                self.gradient_checkpointing = gradient_checkpointing
                self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
                self.layers = nn.ModuleList(
                    [
                        DecoderBlock(d_model, nhead, dim_feedforward, dropout, rope_base)
                        for _ in range(num_layers)
                    ]
                )
                self.norm = RMSNorm(d_model)
                self.output = nn.Linear(d_model, vocab_size, bias=False)
                self.output.weight = self.embedding.weight
                self.dropout = nn.Dropout(dropout)
                self.reset_parameters()

            def reset_parameters(self) -> None:
                nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    self.embedding.weight[self.pad_id].zero_()
                for module in self.modules():
                    if isinstance(module, nn.Linear) and module.weight is not self.embedding.weight:
                        nn.init.normal_(module.weight, mean=0.0, std=0.02)

            def forward(self, input_ids, padding_mask):
                if input_ids.size(1) > self.max_seq_len:
                    raise ValueError(f"Sequence length {input_ids.size(1)} exceeds max_seq_len={self.max_seq_len}")
                hidden = self.dropout(self.embedding(input_ids))
                for layer in self.layers:
                    if self.gradient_checkpointing and self.training and torch_checkpoint is not None:
                        hidden = torch_checkpoint(layer, hidden, padding_mask, use_reentrant=False)
                    else:
                        hidden = layer(hidden, padding_mask)
                return self.output(self.norm(hidden))
    return (ModernDecoderOnlyTransformer,)


@app.cell
def _(Dataset, pad_sequence, pd, torch):
    CausalTranslationDataset = None
    CausalCollator = None

    if torch is not None and pad_sequence is not None:

        class CausalTranslationDataset(Dataset):
            def __init__(
                self,
                frame,
                tokenizer,
                max_seq_len: int,
                min_target_tokens: int,
                style_tag: str,
            ) -> None:
                self.frame = frame.reset_index(drop=True)
                self.tokenizer = tokenizer
                self.max_seq_len = max_seq_len
                self.min_target_tokens = min_target_tokens
                self.style_tag = style_tag
                self.eos_id = tokenizer.token_to_id("</s>")

            def __len__(self) -> int:
                return len(self.frame) * 2

            def __getitem__(self, index: int):
                row = self.frame.iloc[index // 2]
                domain_tag = "<legal>" if str(row["domain"]) == "legal" else "<general>"

                if index % 2 == 0:
                    prompt = f"<2pa> {domain_tag} {self.style_tag} {row['en']}\n"
                    target = str(row["pa"])
                else:
                    prompt = f"<2en> {domain_tag} {self.style_tag} {row['pa']}\n"
                    target = str(row["en"])

                prompt_ids = self.tokenizer.encode(prompt).ids
                max_prompt_len = max(8, self.max_seq_len - self.min_target_tokens - 1)
                prompt_ids = prompt_ids[:max_prompt_len]
                target_budget = max(1, self.max_seq_len - len(prompt_ids) - 1)
                target_ids = self.tokenizer.encode(target).ids[:target_budget] + [self.eos_id]

                full_ids = prompt_ids + target_ids
                if len(full_ids) < 2:
                    full_ids = full_ids + [self.eos_id]

                input_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
                labels = torch.tensor(full_ids[1:], dtype=torch.long)
                prompt_label_cutoff = max(0, min(len(prompt_ids) - 1, labels.numel()))
                labels[:prompt_label_cutoff] = -100

                return {"input_ids": input_ids, "labels": labels}


        class CausalCollator:
            def __init__(self, pad_id: int) -> None:
                self.pad_id = pad_id

            def __call__(self, batch):
                input_ids = pad_sequence(
                    [item["input_ids"] for item in batch],
                    batch_first=True,
                    padding_value=self.pad_id,
                )
                labels = pad_sequence(
                    [item["labels"] for item in batch],
                    batch_first=True,
                    padding_value=-100,
                )
                return {
                    "input_ids": input_ids,
                    "labels": labels,
                    "padding_mask": input_ids.eq(self.pad_id),
                }


    def load_training_frame(path, rows: int | None):
        if pd is None:
            raise RuntimeError("pandas is required to load the training TSV")
        frame = pd.read_csv(
            path,
            sep="\t",
            usecols=["id", "source", "domain", "en", "pa", "en_words", "pa_words"],
            dtype={
                "id": "string",
                "source": "category",
                "domain": "category",
                "en": "string",
                "pa": "string",
                "en_words": "int32",
                "pa_words": "int32",
            },
            nrows=rows,
        )
        frame["en"] = frame["en"].fillna("")
        frame["pa"] = frame["pa"].fillna("")
        return frame


    def split_train_val_frame(frame, val_rows: int, val_fraction: float, seed: int):
        if len(frame) < 3:
            return frame, frame
        shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        requested_val = min(val_rows, max(1, int(len(shuffled) * val_fraction)))
        val_size = min(requested_val, len(shuffled) - 1)
        return (
            shuffled.iloc[val_size:].reset_index(drop=True),
            shuffled.iloc[:val_size].reset_index(drop=True),
        )

    return (
        CausalCollator,
        CausalTranslationDataset,
        load_training_frame,
        split_train_val_frame,
    )


@app.cell
def _(math, nullcontext, torch):
    def set_seed(seed: int) -> None:
        random_state = seed
        import random as _random

        _random.seed(random_state)
        if torch is not None:
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_state)


    def amp_dtype(device, amp: str):
        if torch is None or device.type != "cuda" or amp == "none":
            return None
        if amp == "bf16":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            print("bf16 is not supported on this CUDA device; falling back to fp16")
            return torch.float16
        if amp == "fp16":
            return torch.float16
        raise ValueError(f"Unsupported amp mode: {amp}")


    def autocast_context(device, dtype):
        if torch is not None and device.type == "cuda" and dtype is not None:
            return torch.autocast(device_type="cuda", dtype=dtype)
        return nullcontext()


    def make_grad_scaler(use_fp16: bool):
        try:
            return torch.amp.GradScaler("cuda", enabled=use_fp16)
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler(enabled=use_fp16)


    def make_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float):
        warmup_steps = max(1, warmup_steps)
        total_steps = max(warmup_steps + 1, total_steps)

        def lr_lambda(step: int) -> float:
            step = max(1, step)
            if step <= warmup_steps:
                return step / warmup_steps
            progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    def parameter_count(model) -> int:
        return sum(param.numel() for param in model.parameters())


    def perplexity_from_loss(loss: float) -> float:
        return float(math.exp(min(loss, 20.0)))


    def format_count(value: int) -> str:
        if value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.2f}B"
        if value >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        if value >= 1_000:
            return f"{value / 1_000:.2f}K"
        return str(value)

    return (
        amp_dtype,
        autocast_context,
        format_count,
        make_cosine_scheduler,
        make_grad_scaler,
        parameter_count,
        perplexity_from_loss,
        set_seed,
    )


@app.cell
def _(F, amp_dtype, autocast_context, torch):
    def train_one_epoch(
        model,
        loader,
        optimizer,
        scheduler,
        scaler,
        device,
        amp: str,
        grad_accum_steps: int,
        clip_grad_norm: float,
        label_smoothing: float,
        max_steps: int | None,
        epoch: int,
        start_global_step: int,
        log_every: int = 100,
    ):
        model.train()
        dtype = amp_dtype(device, amp)
        total_loss = 0.0
        micro_steps = 0
        optimizer_steps = 0
        global_step = start_global_step
        records = []
        optimizer.zero_grad(set_to_none=True)

        for batch_index, batch in enumerate(loader, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}

            with autocast_context(device, dtype):
                logits = model(batch["input_ids"], batch["padding_mask"])
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch["labels"].reshape(-1),
                    ignore_index=-100,
                    label_smoothing=label_smoothing,
                )

            total_loss += float(loss.detach().cpu())
            scaler.scale(loss / grad_accum_steps).backward()
            micro_steps += 1

            should_step = batch_index % grad_accum_steps == 0 or batch_index == len(loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                global_step += 1

                if optimizer_steps % log_every == 0:
                    avg_loss = total_loss / max(micro_steps, 1)
                    lr = scheduler.get_last_lr()[0]
                    records.append(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "train_loss": avg_loss,
                            "lr": lr,
                        }
                    )
                    print(
                        f"epoch {epoch} step {optimizer_steps:,}: "
                        f"train_loss={avg_loss:.4f} lr={lr:.6g}"
                    )

                if max_steps is not None and optimizer_steps >= max_steps:
                    break

        train_loss = total_loss / max(micro_steps, 1)
        records.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "lr": scheduler.get_last_lr()[0],
            }
        )
        return train_loss, optimizer_steps, global_step, records


    @torch.no_grad() if torch is not None else (lambda fn: fn)
    def evaluate_model(model, loader, device, amp: str):
        model.eval()
        dtype = amp_dtype(device, amp)
        total_loss = 0.0
        total_tokens = 0
        correct_tokens = 0
        batches = 0

        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with autocast_context(device, dtype):
                logits = model(batch["input_ids"], batch["padding_mask"])
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch["labels"].reshape(-1),
                    ignore_index=-100,
                )

            mask = batch["labels"].ne(-100)
            pred = logits.argmax(dim=-1)
            correct_tokens += int((pred.eq(batch["labels"]) & mask).sum().detach().cpu())
            total_tokens += int(mask.sum().detach().cpu())
            total_loss += float(loss.detach().cpu())
            batches += 1

        return total_loss / max(batches, 1), correct_tokens / max(total_tokens, 1)

    return evaluate_model, train_one_epoch


@app.cell
def _(torch):
    @torch.no_grad() if torch is not None else (lambda fn: fn)
    def generate_translation(
        model,
        tokenizer,
        text: str,
        target_lang: str,
        domain: str,
        style_tag: str,
        device,
        max_seq_len: int,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        top_k: int = 0,
    ) -> str:
        model.eval()
        eos_id = tokenizer.token_to_id("</s>")
        pad_id = tokenizer.token_to_id("<pad>")
        target_tag = "<2pa>" if target_lang == "pa" else "<2en>"
        domain_tag = "<legal>" if domain == "legal" else "<general>"
        prompt = f"{target_tag} {domain_tag} {style_tag} {text}\n"
        ids = tokenizer.encode(prompt).ids
        prompt_len = len(ids)

        for _ in range(max_new_tokens):
            input_ids = torch.tensor([ids[-max_seq_len:]], dtype=torch.long, device=device)
            padding_mask = input_ids.eq(pad_id)
            logits = model(input_ids, padding_mask)[:, -1, :]

            if temperature and temperature > 0:
                logits = logits / temperature
                if top_k and top_k > 0:
                    values, indices = torch.topk(logits, k=min(top_k, logits.size(-1)))
                    probs = torch.softmax(values, dim=-1)
                    next_id = int(indices.gather(-1, torch.multinomial(probs, 1)).item())
                else:
                    probs = torch.softmax(logits, dim=-1)
                    next_id = int(torch.multinomial(probs, 1).item())
            else:
                next_id = int(logits.argmax(dim=-1).item())

            ids.append(next_id)
            if next_id == eos_id:
                break

        generated_ids = ids[prompt_len:]
        if eos_id in generated_ids:
            generated_ids = generated_ids[: generated_ids.index(eos_id)]
        return tokenizer.decode(generated_ids)

    return (generate_translation,)


@app.cell
def _(MODEL_PROFILES, mo):
    mo.md("## Training Controls")

    decoder_profile = mo.ui.dropdown(
        options=list(MODEL_PROFILES.keys()),
        value="base",
        label="Decoder profile",
    )
    train_style_tag = mo.ui.dropdown(
        options=["<natural>", "<literal>"],
        value="<natural>",
        label="Style tag",
    )
    train_batch_size = mo.ui.number(value=192, start=1, stop=1024, step=1, label="Batch size")
    train_epochs = mo.ui.number(value=1, start=1, stop=100, step=1, label="Epochs")
    train_max_rows = mo.ui.number(value=0, start=0, stop=2_000_000, step=10_000, label="Max rows; 0 = full corpus")
    train_max_steps = mo.ui.number(value=0, start=0, stop=1_000_000, step=100, label="Max steps/epoch; 0 = full epoch")
    train_max_seq_len = mo.ui.number(value=256, start=64, stop=2048, step=32, label="Max sequence length")
    train_min_target_tokens = mo.ui.number(value=48, start=8, stop=512, step=8, label="Reserved target tokens")
    train_grad_accum_steps = mo.ui.number(value=1, start=1, stop=128, step=1, label="Grad accumulation")
    train_lr = mo.ui.number(value=3e-4, start=1e-5, stop=2e-3, step=1e-5, label="Learning rate")
    train_weight_decay = mo.ui.number(value=0.1, start=0.0, stop=0.5, step=0.01, label="Weight decay")
    train_label_smoothing = mo.ui.number(value=0.05, start=0.0, stop=0.2, step=0.01, label="Label smoothing")
    train_dropout = mo.ui.number(value=0.1, start=0.0, stop=0.5, step=0.01, label="Dropout")
    train_num_workers = mo.ui.number(value=0, start=0, stop=32, step=1, label="DataLoader workers")
    train_amp = mo.ui.dropdown(options=["bf16", "fp16", "none"], value="bf16", label="AMP")
    train_compile = mo.ui.checkbox(value=False, label="torch.compile")
    train_checkpointing = mo.ui.checkbox(value=False, label="Gradient checkpointing")

    mo.vstack(
        [
            mo.hstack([decoder_profile, train_style_tag, train_amp, train_compile, train_checkpointing]),
            mo.hstack([train_batch_size, train_epochs, train_grad_accum_steps, train_num_workers]),
            mo.hstack([train_max_rows, train_max_steps, train_max_seq_len, train_min_target_tokens]),
            mo.hstack([train_lr, train_weight_decay, train_label_smoothing, train_dropout]),
        ]
    )
    return (
        decoder_profile,
        train_amp,
        train_batch_size,
        train_checkpointing,
        train_compile,
        train_dropout,
        train_epochs,
        train_grad_accum_steps,
        train_label_smoothing,
        train_lr,
        train_max_rows,
        train_max_seq_len,
        train_max_steps,
        train_min_target_tokens,
        train_num_workers,
        train_style_tag,
        train_weight_decay,
    )


@app.cell
def _(
    DecoderTrainConfig,
    MODEL_PROFILES,
    TOKENIZER_PATH,
    Tokenizer,
    asdict,
    decoder_profile,
    mo,
    train_amp,
    train_batch_size,
    train_checkpointing,
    train_dropout,
    train_epochs,
    train_grad_accum_steps,
    train_label_smoothing,
    train_lr,
    train_max_seq_len,
    train_min_target_tokens,
    train_style_tag,
    train_weight_decay,
):
    if Tokenizer is not None and TOKENIZER_PATH.exists():
        _config_tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        _pad_id = _config_tokenizer.token_to_id("<pad>")
        _eos_id = _config_tokenizer.token_to_id("</s>")
        _vocab_size = _config_tokenizer.get_vocab_size()
    else:
        _pad_id = 0
        _eos_id = 3
        _vocab_size = 24_000

    selected_profile_config = MODEL_PROFILES[decoder_profile.value]
    selected_train_config = DecoderTrainConfig(
        vocab_size=_vocab_size,
        pad_id=_pad_id,
        eos_id=_eos_id,
        profile=decoder_profile.value,
        max_seq_len=int(train_max_seq_len.value),
        min_target_tokens=int(train_min_target_tokens.value),
        batch_size=int(train_batch_size.value),
        epochs=int(train_epochs.value),
        lr=float(train_lr.value),
        weight_decay=float(train_weight_decay.value),
        grad_accum_steps=int(train_grad_accum_steps.value),
        label_smoothing=float(train_label_smoothing.value),
        dropout=float(train_dropout.value),
        amp=train_amp.value,
        style_tag=train_style_tag.value,
        gradient_checkpointing=bool(train_checkpointing.value),
        **selected_profile_config,
    )

    mo.md(
        "Selected config:\n\n"
        + "```python\n"
        + "\n".join(f"{key} = {value!r}" for key, value in asdict(selected_train_config).items())
        + "\n```"
    )
    return selected_profile_config, selected_train_config


@app.cell
def _(mo):
    dry_run_train_button = mo.ui.run_button(label="Run decoder smoke train")
    full_train_button = mo.ui.run_button(label="Train selected decoder")
    resume_training = mo.ui.checkbox(value=False, label="Resume from checkpoint")
    resume_checkpoint_path = mo.ui.text(
        value="checkpoints/gur_slm_decoder/base_best.pt",
        label="Resume checkpoint path",
    )
    mo.vstack(
        [
            mo.hstack([dry_run_train_button, full_train_button]),
            mo.hstack([resume_training, resume_checkpoint_path]),
        ]
    )
    return dry_run_train_button, full_train_button, resume_checkpoint_path, resume_training


@app.cell
def _(
    CHECKPOINT_DIR,
    CausalCollator,
    CausalTranslationDataset,
    DATA_PATH,
    DataLoader,
    DecoderTrainConfig,
    METRICS_DIR,
    ModernDecoderOnlyTransformer,
    TOKENIZER_PATH,
    Tokenizer,
    amp_dtype,
    asdict,
    dry_run_train_button,
    evaluate_model,
    format_count,
    full_train_button,
    generate_translation,
    load_training_frame,
    make_cosine_scheduler,
    make_grad_scaler,
    math,
    mo,
    parameter_count,
    pd,
    perplexity_from_loss,
    resume_checkpoint_path,
    resume_training,
    selected_profile_config,
    selected_train_config,
    set_seed,
    split_train_val_frame,
    time,
    torch,
    train_compile,
    train_max_rows,
    train_max_steps,
    train_num_workers,
    train_one_epoch,
):
    trained_model = None
    trained_tokenizer = None
    trained_config = selected_train_config
    training_history_frame = None
    training_report = {"status": "not started"}

    _run_requested = bool(dry_run_train_button.value or full_train_button.value)

    if _run_requested:
        _missing = []
        if torch is None:
            _missing.append("torch")
        if pd is None:
            _missing.append("pandas")
        if Tokenizer is None:
            _missing.append("tokenizers")
        if DataLoader is None or CausalTranslationDataset is None or ModernDecoderOnlyTransformer is None:
            _missing.append("model/data classes")
        if _missing:
            training_report = {"status": f"missing dependencies: {', '.join(_missing)}"}
        elif not DATA_PATH.exists() or not TOKENIZER_PATH.exists():
            training_report = {"status": "missing data or tokenizer artifact"}
        else:
            set_seed(selected_train_config.seed)
            if torch.cuda.is_available() and hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            row_limit = 4096 if dry_run_train_button.value else (int(train_max_rows.value) if int(train_max_rows.value) > 0 else None)
            max_steps_value = 2 if dry_run_train_button.value else (int(train_max_steps.value) if int(train_max_steps.value) > 0 else None)

            tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
            pad_id = tokenizer.token_to_id("<pad>")
            eos_id = tokenizer.token_to_id("</s>")
            vocab_size = tokenizer.get_vocab_size()

            config = DecoderTrainConfig(
                vocab_size=vocab_size,
                pad_id=pad_id,
                eos_id=eos_id,
                profile=selected_train_config.profile,
                max_seq_len=selected_train_config.max_seq_len,
                min_target_tokens=selected_train_config.min_target_tokens,
                batch_size=4 if dry_run_train_button.value else selected_train_config.batch_size,
                epochs=1 if dry_run_train_button.value else selected_train_config.epochs,
                lr=selected_train_config.lr,
                weight_decay=selected_train_config.weight_decay,
                grad_accum_steps=selected_train_config.grad_accum_steps,
                label_smoothing=selected_train_config.label_smoothing,
                dropout=selected_train_config.dropout,
                amp=selected_train_config.amp,
                style_tag=selected_train_config.style_tag,
                gradient_checkpointing=selected_train_config.gradient_checkpointing,
                **selected_profile_config,
            )

            frame = load_training_frame(DATA_PATH, row_limit)
            train_frame, val_frame = split_train_val_frame(
                frame,
                val_rows=min(config.val_rows, max(1, len(frame) // 20)),
                val_fraction=config.val_fraction,
                seed=config.seed,
            )

            train_dataset = CausalTranslationDataset(
                train_frame,
                tokenizer,
                config.max_seq_len,
                config.min_target_tokens,
                config.style_tag,
            )
            val_dataset = CausalTranslationDataset(
                val_frame,
                tokenizer,
                config.max_seq_len,
                config.min_target_tokens,
                config.style_tag,
            )

            loader_workers = int(train_num_workers.value)
            collator = CausalCollator(pad_id)
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=loader_workers,
                collate_fn=collator,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=loader_workers > 0,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=loader_workers,
                collate_fn=collator,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=loader_workers > 0,
            )

            model = ModernDecoderOnlyTransformer(
                vocab_size=config.vocab_size,
                pad_id=config.pad_id,
                d_model=config.d_model,
                nhead=config.nhead,
                num_layers=config.num_layers,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                max_seq_len=config.max_seq_len,
                rope_base=config.rope_base,
                gradient_checkpointing=config.gradient_checkpointing,
            ).to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.lr,
                betas=(0.9, 0.95),
                weight_decay=config.weight_decay,
            )
            total_steps = max(
                1,
                config.epochs
                * math.ceil(len(train_loader) / max(1, config.grad_accum_steps)),
            )
            if max_steps_value is not None:
                total_steps = min(total_steps, max_steps_value * config.epochs)
            scheduler = make_cosine_scheduler(optimizer, config.warmup_steps, total_steps, config.min_lr_ratio)
            dtype = amp_dtype(device, config.amp)
            scaler = make_grad_scaler(use_fp16=(device.type == "cuda" and dtype == torch.float16))

            resume_epoch = 0
            best_val_loss = float("inf")
            global_step = 0
            if bool(resume_training.value) and not dry_run_train_button.value:
                checkpoint_path = (TOKENIZER_PATH.parent.parent / resume_checkpoint_path.value).expanduser()
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                scheduler.load_state_dict(checkpoint["scheduler"])
                resume_epoch = int(checkpoint.get("epoch", 0))
                best_val_loss = float(checkpoint.get("val_loss", best_val_loss))
                if max_steps_value is not None:
                    global_step = resume_epoch * max_steps_value
                print(
                    f"resumed from {checkpoint_path} "
                    f"at saved_epoch={resume_epoch} best_val_loss={best_val_loss:.4f}"
                )

            if bool(train_compile.value) and hasattr(torch, "compile"):
                model = torch.compile(model)

            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            METRICS_DIR.mkdir(parents=True, exist_ok=True)

            print(f"device: {device}")
            print(f"amp: {config.amp}")
            print(f"profile: {config.profile}")
            print(f"vocab_size: {vocab_size:,}")
            print(f"parameters: {parameter_count(model):,} ({format_count(parameter_count(model))})")
            print(f"train examples including both directions: {len(train_dataset):,}")
            print(f"val examples including both directions: {len(val_dataset):,}")
            print(f"train batches per epoch: {len(train_loader):,}")
            print(f"loader_workers: {loader_workers}")

            history_records = []
            started_at = time.time()
            start_epoch = resume_epoch + 1
            end_epoch = resume_epoch + config.epochs

            for epoch in range(start_epoch, end_epoch + 1):
                train_loss, steps, global_step, train_records = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                    scaler,
                    device,
                    config.amp,
                    config.grad_accum_steps,
                    config.clip_grad_norm,
                    config.label_smoothing,
                    max_steps=max_steps_value,
                    epoch=epoch,
                    start_global_step=global_step,
                    log_every=25 if dry_run_train_button.value else 100,
                )
                val_loss, val_acc = evaluate_model(model, val_loader, device, config.amp)
                epoch_record = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_ppl": perplexity_from_loss(val_loss),
                    "val_token_acc": val_acc,
                    "lr": scheduler.get_last_lr()[0],
                    "optimizer_steps": steps,
                }
                history_records.extend(train_records)
                history_records.append(epoch_record)

                model_for_save = model._orig_mod if hasattr(model, "_orig_mod") else model
                payload = {
                    "epoch": epoch,
                    "model": model_for_save.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": asdict(config),
                    "val_loss": val_loss,
                    "vocab_size": vocab_size,
                }
                epoch_path = CHECKPOINT_DIR / f"{config.profile}_epoch_{epoch}.pt"
                last_path = CHECKPOINT_DIR / f"{config.profile}_last.pt"
                torch.save(payload, epoch_path)
                torch.save(payload, last_path)

                best_note = ""
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = CHECKPOINT_DIR / f"{config.profile}_best.pt"
                    torch.save(payload, best_path)
                    best_note = f" best={best_path}"

                print(
                    f"epoch {epoch}: train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f} val_ppl={perplexity_from_loss(val_loss):.2f} "
                    f"val_token_acc={val_acc:.4f} steps={steps} checkpoint={epoch_path}{best_note}"
                )
                sample = generate_translation(
                    model_for_save,
                    tokenizer,
                    "The agreement shall remain in force for five years.",
                    target_lang="pa",
                    domain="legal",
                    style_tag=config.style_tag,
                    device=device,
                    max_seq_len=config.max_seq_len,
                )
                print(f"sample en->pa: {sample}")

            training_history_frame = pd.DataFrame(history_records)
            metrics_path = METRICS_DIR / f"{config.profile}_history.csv"
            training_history_frame.to_csv(metrics_path, index=False)

            trained_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            trained_tokenizer = tokenizer
            trained_config = config
            training_report = {
                "status": "complete",
                "seconds": round(time.time() - started_at, 2),
                "resumed_from_epoch": resume_epoch,
                "completed_through_epoch": end_epoch,
                "best_val_loss": best_val_loss,
                "best_val_ppl": perplexity_from_loss(best_val_loss),
                "metrics_path": str(metrics_path),
                "checkpoint_dir": str(CHECKPOINT_DIR),
            }

    mo.md(f"**Training status:** {training_report['status']}")
    training_report
    return (
        trained_config,
        trained_model,
        trained_tokenizer,
        training_history_frame,
    )


@app.cell
def _(mo, plt, training_history_frame):
    if training_history_frame is None:
        training_visual = mo.md("Train or smoke-train the model to render metric curves.")
    elif plt is None:
        training_visual = mo.md("matplotlib is not installed, so training curves cannot be rendered.")
    else:
        _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.5))
        _frame = training_history_frame.copy()
        _x = _frame["global_step"] if "global_step" in _frame else _frame.index

        if "train_loss" in _frame:
            _axes[0].plot(_x, _frame["train_loss"], color="#3a6ea5", label="train")
        if "val_loss" in _frame:
            _val_frame = _frame.dropna(subset=["val_loss"])
            _axes[0].plot(_val_frame["global_step"], _val_frame["val_loss"], marker="o", color="#bf6f24", label="val")
        _axes[0].set_title("Loss")
        _axes[0].set_xlabel("global step")
        _axes[0].legend()

        if "val_ppl" in _frame:
            _val_frame = _frame.dropna(subset=["val_ppl"])
            _axes[1].plot(_val_frame["global_step"], _val_frame["val_ppl"], marker="o", color="#7b5ea7")
        _axes[1].set_title("Validation perplexity")
        _axes[1].set_xlabel("global step")

        if "val_token_acc" in _frame:
            _val_frame = _frame.dropna(subset=["val_token_acc"])
            _axes[2].plot(_val_frame["global_step"], _val_frame["val_token_acc"], marker="o", color="#2f9c95")
        _axes[2].set_title("Validation token accuracy")
        _axes[2].set_xlabel("global step")

        _fig.tight_layout()
        training_visual = _fig

    training_visual
    return


@app.cell
def _(mo):
    mo.md("## Quick Generation Check")
    sample_text = mo.ui.text_area(
        value="The agreement shall remain in force for five years.",
        label="Source text",
        rows=3,
    )
    sample_target_lang = mo.ui.dropdown(options=["pa", "en"], value="pa", label="Target language")
    sample_domain = mo.ui.dropdown(options=["legal", "general"], value="legal", label="Domain")
    sample_style = mo.ui.dropdown(options=["<natural>", "<literal>"], value="<natural>", label="Style")
    sample_temperature = mo.ui.number(value=0.0, start=0.0, stop=1.5, step=0.1, label="Temperature; 0 = greedy")
    sample_button = mo.ui.run_button(label="Generate sample")

    mo.vstack(
        [
            sample_text,
            mo.hstack([sample_target_lang, sample_domain, sample_style, sample_temperature, sample_button]),
        ]
    )
    return (
        sample_button,
        sample_domain,
        sample_style,
        sample_target_lang,
        sample_temperature,
        sample_text,
    )


@app.cell
def _(
    generate_translation,
    mo,
    sample_button,
    sample_domain,
    sample_style,
    sample_target_lang,
    sample_temperature,
    sample_text,
    torch,
    trained_config,
    trained_model,
    trained_tokenizer,
):
    if not sample_button.value:
        sample_output = mo.md("Train or load a model, then run a sample.")
    elif trained_model is None or trained_tokenizer is None or torch is None:
        sample_output = mo.md("No trained model is available in this notebook state.")
    else:
        _device = next(trained_model.parameters()).device
        _translation = generate_translation(
            trained_model,
            trained_tokenizer,
            sample_text.value,
            target_lang=sample_target_lang.value,
            domain=sample_domain.value,
            style_tag=sample_style.value,
            device=_device,
            max_seq_len=trained_config.max_seq_len,
            temperature=float(sample_temperature.value),
        )
        sample_output = mo.md(f"**Output:**\n\n{_translation}")

    sample_output
    return


@app.cell
def _(mo):
    mo.md("""
    ## New Research Plan

    ```mermaid
    flowchart LR
      A[Cleaned EN-PA corpus] --> B[Modern decoder pretraining]
      B --> C[Evaluation set: legal and general]
      C --> D[Teacher refinement]
      D --> E[Distilled decoder checkpoint]
      E --> F[8-bit PTQ]
      E --> G[4-bit PTQ or QLoRA]
      E --> H[1.58-bit BitLinear branch]
      E --> I[True 1-bit binary branch]
    ```

    **Phase 1 - supervised decoder pretraining**

    - Train the RoPE + RMSNorm + SwiGLU decoder in this notebook.
    - Track validation loss, perplexity, token accuracy, and generated legal/general samples.
    - Add external MT metrics next: chrF and BLEU with `sacrebleu`, plus a small manually reviewed challenge set.

    **Phase 2 - teacher refinement**

    - First teacher pass: sequence-level distillation. Generate cleaner EN->PA and PA->EN targets from a stronger teacher and train on mixed gold + teacher outputs.
    - Second teacher pass: on-policy distillation. Let this student generate candidates, then score or correct with the teacher to reduce exposure bias.
    - Keep domains explicit. Legal translation should have separate validation and review because fluency can hide terminology errors.

    **Phase 3 - quantization ladder**

    - 8-bit: start with post-training W8A8 or weight-only quantization and compare against FP/BF16 checkpoint loss.
    - 4-bit: use GPTQ/AWQ style calibration for inference; use QLoRA only for finetuning branches.
    - 1.58-bit: train or finetune a BitLinear variant with straight-through estimation and distill from the full-precision decoder.
    - True 1-bit: treat as a stress experiment. Compare quality, speed, and memory honestly against the 1.58-bit ternary branch.

    **Decision rule:** do not quantize the first model blindly. Quantize after a reproducible full-precision baseline and teacher-refined checkpoint exist.
    """)
    return


@app.cell
def _(mo, research_references):
    _quant_refs = [
        row
        for row in research_references
        if row["phase"] in {"Teacher refinement", "Quantization"}
    ]
    _lines = "\n".join(
        f"- [{row['reference']}]({row['url']}) ({row['year']}): {row['use']}"
        for row in _quant_refs
    )
    mo.md("## Distillation and Quantization References\n\n" + _lines)
    return


@app.cell
def _(CHECKPOINT_DIR, mo):
    mo.md("## Checkpoint Inventory")
    checkpoint_files = sorted(CHECKPOINT_DIR.glob("**/*.pt")) if CHECKPOINT_DIR.exists() else []
    checkpoint_files[-20:]
    return


if __name__ == "__main__":
    app.run()
