import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import importlib.util as importlib_util
    import json
    import math
    import os
    import random
    import shutil
    import sys
    import time
    from contextlib import nullcontext
    from dataclasses import asdict, dataclass
    from pathlib import Path

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
        from torch.utils.data import DataLoader, Dataset
    except ModuleNotFoundError:
        torch = None
        F = None
        nn = None
        pad_sequence = None
        DataLoader = None
        Dataset = object

    try:
        from tokenizers import Tokenizer
    except ModuleNotFoundError:
        Tokenizer = None

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ModuleNotFoundError:
        hf_hub_download = None
        snapshot_download = None

    try:
        import sacrebleu
    except ModuleNotFoundError:
        sacrebleu = None

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError:
        AutoModelForCausalLM = None
        AutoTokenizer = None

    try:
        _original_sys_path = list(sys.path)
        _repo_root = str(Path.cwd().resolve())
        _shadowed_datasets = sys.modules.pop("datasets", None)
        sys.path = [
            _path for _path in sys.path
            if _path and str(Path(_path).resolve()) != _repo_root
        ]
        from datasets import load_dataset
    except Exception:
        load_dataset = None
    finally:
        sys.path = _original_sys_path
        if load_dataset is None and _shadowed_datasets is not None:
            sys.modules["datasets"] = _shadowed_datasets

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataLoader,
        Dataset,
        F,
        Path,
        Tokenizer,
        dataclass,
        hf_hub_download,
        importlib_util,
        load_dataset,
        mo,
        nn,
        np,
        pad_sequence,
        pd,
        plt,
        sacrebleu,
        time,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    <style>
    .hero {
        padding: 1.15rem 1.35rem;
        border-radius: 8px;
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 { margin: 0 0 0.3rem 0; font-size: 1.95rem; letter-spacing: 0; }
    .hero p { margin: 0; color: #dceaf5; max-width: 1050px; line-height: 1.45; }
    .cards {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 0.75rem;
        margin: 0.75rem 0 1.1rem 0;
    }
    .card {
        border: 1px solid #d9e2ec;
        border-radius: 8px;
        padding: 0.85rem;
        background: #fff;
    }
    .card b { display: block; color: #13283d; margin-bottom: 0.25rem; }
    .card span { color: #536879; font-size: 0.92rem; }
    .ok { color: #18794e; font-weight: 600; }
    .warn { color: #a45f00; font-weight: 600; }
    .bad { color: #b42318; font-weight: 600; }
    </style>

    <div class="hero">
      <h1>Reverse-KL On-Policy Distillation for Gurmukhi Translation</h1>
      <p>
      A paper-implementation notebook inspired by MiniLLM: use a stronger translation
      teacher to refine your decoder-only EN <-> Punjabi Gurmukhi SLM, then compare
      baseline, teacher, and distilled student with BLEU, chrF, challenge-set slices,
      and qualitative inspection.
      </p>
    </div>

    <div class="cards">
      <div class="card"><b>Paper Core</b><span>Reverse KL is mode-seeking and avoids overestimating low-probability teacher regions.</span></div>
      <div class="card"><b>Translation Adaptation</b><span>Use teacher-generated labels plus sequence-level reverse-KL rewards when teacher/student tokenizers differ.</span></div>
      <div class="card"><b>Notebook Extension</b><span>Quality-gated, domain-aware distillation with manual challenge-set diagnostics.</span></div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(Path):
    PROJECT_ROOT = Path.cwd()
    RUN_DIR = PROJECT_ROOT / "distillation_runs"
    HF_BASE_REPO_ID = "Ajaple/gur-slm-decoder-base"
    DATA_PATH = PROJECT_ROOT / "datasets" / "cleaned.tsv"
    TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "hf_bpe24k_tokenizer.json"
    BASE_LOCAL_DIR = RUN_DIR / "base_checkpoint"
    CACHE_DIR = RUN_DIR / "teacher_cache"
    DISTILLED_DIR = RUN_DIR / "distilled_checkpoints"
    METRICS_DIR = RUN_DIR / "metrics"
    for _path in [RUN_DIR, BASE_LOCAL_DIR, CACHE_DIR, DISTILLED_DIR, METRICS_DIR]:
        _path.mkdir(parents=True, exist_ok=True)
    return (
        BASE_LOCAL_DIR,
        CACHE_DIR,
        DATA_PATH,
        DISTILLED_DIR,
        HF_BASE_REPO_ID,
        PROJECT_ROOT,
        TOKENIZER_PATH,
    )


@app.cell(hide_code=True)
def _(mo, pd):
    distillation_references = [
        {
            "topic": "Core paper",
            "reference": "MiniLLM: On-Policy Distillation of Large Language Models",
            "url": "https://arxiv.org/abs/2306.08543",
            "use": "Reverse KL and on-policy student sampling for generative KD.",
        },
        {
            "topic": "Translation teacher",
            "reference": "Sarvam-Translate",
            "url": "https://huggingface.co/sarvamai/sarvam-translate",
            "use": "Primary sequence-level teacher for English to Punjabi Gurmukhi distillation; disclose GPL-3.0 before publishing derivatives.",
        },
        {
            "topic": "Benchmark",
            "reference": "FLORES+",
            "url": "https://huggingface.co/datasets/openlanguagedata/flores_plus",
            "use": "Public evaluation set for eng_Latn -> pan_Guru reporting; use dev for debugging and devtest for final tables.",
        },
        {
            "topic": "Metrics",
            "reference": "sacreBLEU",
            "url": "https://github.com/mjpost/sacrebleu",
            "use": "BLEU and chrF with reproducible signatures for MT comparisons.",
        },
        {
            "topic": "Contrast",
            "reference": "Sequence-Level Knowledge Distillation",
            "url": "https://arxiv.org/abs/1606.07947",
            "use": "Hard teacher translations are a strong baseline before the MiniLLM-style on-policy extension.",
        },
    ]
    if pd is not None:
        _reference_view = pd.DataFrame(distillation_references)
    else:
        _reference_view = "\n".join(
            f"- [{_row['reference']}]({_row['url']}): {_row['use']}"
            for _row in distillation_references
        )
    mo.md("## Research Map")
    _reference_view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## How This Mirrors MiniLLM

    MiniLLM argues that standard forward KL makes a student cover too many
    low-probability regions of a generative teacher. Reverse KL is more
    mode-seeking: the student is rewarded for concentrating on outputs the
    teacher also likes. The paper also emphasizes on-policy student-generated
    outputs to reduce exposure bias.

    **Exact token-level reverse KL requires compatible vocabularies.** Your
    student checkpoint is a custom decoder-only BPE model, while
    Sarvam-Translate has its own tokenizer and chat format. This notebook
    therefore implements two practical levels:

    - **Sequence KD:** generate Sarvam translations, filter them, and fine-tune
      the 58M from-scratch student on accepted source-target pairs.
    - **MiniLLM-style sequence reverse KL:** sample translations from the
      current student, score those full translations under Sarvam, and optimize
      a sequence-level reverse-KL policy-gradient surrogate.

    **Custom extension for the challenge:** quality-gated distillation. The
    notebook filters teacher targets with script checks, length-ratio checks,
    and optional chrF/BLEU diagnostics before using them for training.
    """)
    return


@app.cell(hide_code=True)
def distillation_architecture(mo, pd):
    distillation_architecture_frame = pd.DataFrame(
        [
            {
                "MiniLLM component": "Reverse KL objective",
                "Notebook implementation": "Sequence-level surrogate: student samples are scored by Sarvam with KL-style reward `log p_student - log p_teacher`.",
                "Status": "Adapted for tokenizer mismatch",
            },
            {
                "MiniLLM component": "On-policy student sampling",
                "Notebook implementation": "The RKL demo samples from the current 58M decoder before scoring with Sarvam.",
                "Status": "Implemented as small-batch extension",
            },
            {
                "MiniLLM component": "Teacher/student same token space",
                "Notebook implementation": "Sarvam and the from-scratch decoder do not share a tokenizer, so the main scalable run uses sequence KD on accepted teacher translations.",
                "Status": "Explicitly not token-level RKL",
            },
            {
                "MiniLLM component": "Stable distillation baseline",
                "Notebook implementation": "SeqKD and mixed gold+teacher CE fine-tune the student after teacher quality gates.",
                "Status": "Primary training path",
            },
            {
                "MiniLLM component": "Evaluation against the starting student",
                "Notebook implementation": "Baseline, Sarvam teacher, and post-distillation outputs are compared with BLEU, chrF, script errors, English leakage, and challenge examples.",
                "Status": "Reader-facing result path",
            },
        ]
    ) if pd is not None else None

    if distillation_architecture_frame is None:
        distillation_architecture_view = mo.md("pandas is required to show the distillation architecture table.")
    else:
        distillation_architecture_view = mo.vstack(
            [
                mo.md(
                    """
                    ## Distillation Architecture

                    This notebook mirrors MiniLLM at the **generative objective level**:
                    use on-policy student samples and a reverse-KL-style teacher score.
                    Because Sarvam and the 58M decoder use different tokenizers, the
                    scalable training path is intentionally sequence-level KD first,
                    then a small MiniLLM-style RKL extension.
                    """
                ),
                distillation_architecture_frame,
            ]
        )
    distillation_architecture_view
    return


@app.cell(hide_code=True)
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    DataLoader,
    Tokenizer,
    hf_hub_download,
    importlib_util,
    load_dataset,
    mo,
    pd,
    plt,
    sacrebleu,
    torch,
):
    dependency_rows = [
        ("torch", torch is not None),
        ("pandas", pd is not None),
        ("matplotlib", plt is not None),
        ("tokenizers", Tokenizer is not None),
        ("huggingface_hub", hf_hub_download is not None),
        ("datasets", load_dataset is not None),
        ("sacrebleu", sacrebleu is not None),
        ("transformers causal LM", AutoModelForCausalLM is not None and AutoTokenizer is not None),
        ("accelerate", importlib_util.find_spec("accelerate") is not None),
        ("sentencepiece", importlib_util.find_spec("sentencepiece") is not None),
        ("safetensors", importlib_util.find_spec("safetensors") is not None),
        ("hf_transfer", importlib_util.find_spec("hf_transfer") is not None),
        ("torch DataLoader", DataLoader is not None),
    ]
    mo.md(
        "## Dependency Check\n\n"
        + "\n".join(
            f"- **{_name}**: <span class='{'ok' if _ok else 'bad'}'>{'loaded' if _ok else 'missing'}</span>"
            for _name, _ok in dependency_rows
        )
        + "\n\nInstall missing runtime packages in cloud if needed:\n\n"
        + "```bash\npip install -U marimo torch transformers accelerate tokenizers datasets sacrebleu pandas matplotlib sentencepiece protobuf safetensors huggingface_hub hf_transfer\n```"
    )
    return


@app.cell(hide_code=True)
def cloud_gpu_checklist(mo):
    mo.md("""
    ## Cloud GPU Run Checklist

    Use a CUDA runtime with at least **16 GB VRAM** for Sarvam-Translate; 24 GB+
    is more comfortable when teacher generation and student training share the
    session. Log in to Hugging Face before loading Sarvam or gated FLORES+.

    ```bash
    pip install -U marimo torch transformers accelerate tokenizers datasets sacrebleu pandas matplotlib sentencepiece protobuf safetensors huggingface_hub hf_transfer
    huggingface-cli login
    marimo edit distillation.py --host 0.0.0.0 --port 2718
    ```

    Recommended cloud environment variables:

    ```bash
    export TOKENIZERS_PARALLELISM=true
    export HF_HOME=/workspace/hf_cache
    export HF_HUB_ENABLE_HF_TRANSFER=1
    ```

    Run order: dependency check -> download/load `base_best.pt` -> load FLORES+ dev
    sample -> load Sarvam teacher -> generate teacher cache -> inspect qualification
    -> run SeqKD or mixed KD.
    """)
    return


@app.cell(hide_code=True)
def _(BASE_LOCAL_DIR, HF_BASE_REPO_ID, TOKENIZER_PATH, mo):
    mo.md("## Base Checkpoint")
    base_repo_id_input = mo.ui.text(value=HF_BASE_REPO_ID, label="Baseline Hugging Face repo")
    download_base_button = mo.ui.run_button(label="Download baseline checkpoint")
    local_checkpoint_path_input = mo.ui.text(
        value=str(BASE_LOCAL_DIR / "base_best.pt"),
        label="Local base checkpoint path",
    )
    local_tokenizer_path_input = mo.ui.text(
        value=str(TOKENIZER_PATH),
        label="Local tokenizer path",
    )
    mo.vstack(
        [
            mo.hstack([base_repo_id_input, download_base_button]),
            local_checkpoint_path_input,
            local_tokenizer_path_input,
        ]
    )
    return (
        base_repo_id_input,
        download_base_button,
        local_checkpoint_path_input,
        local_tokenizer_path_input,
    )


@app.cell(hide_code=True)
def _(
    BASE_LOCAL_DIR,
    PROJECT_ROOT,
    base_repo_id_input,
    download_base_button,
    hf_hub_download,
    local_checkpoint_path_input,
    local_tokenizer_path_input,
    mo,
):
    base_checkpoint_path = BASE_LOCAL_DIR / "base_best.pt"
    base_config_path = BASE_LOCAL_DIR / "decoder_config.json"
    base_tokenizer_path = BASE_LOCAL_DIR / "hf_bpe24k_tokenizer.json"
    base_download_report = {"status": "not downloaded this run"}

    if download_base_button.value:
        if hf_hub_download is None:
            base_download_report = {"status": "missing huggingface_hub"}
        else:
            _repo_id = base_repo_id_input.value
            _ckpt = hf_hub_download(_repo_id, "base_best.pt", local_dir=str(BASE_LOCAL_DIR))
            _tok = hf_hub_download(_repo_id, "hf_bpe24k_tokenizer.json", local_dir=str(BASE_LOCAL_DIR))
            try:
                _cfg = hf_hub_download(_repo_id, "decoder_config.json", local_dir=str(BASE_LOCAL_DIR))
            except Exception as _exc:
                _cfg = None
                print(f"decoder_config.json not found or not downloadable: {_exc}")
            base_checkpoint_path = BASE_LOCAL_DIR / "base_best.pt"
            base_tokenizer_path = BASE_LOCAL_DIR / "hf_bpe24k_tokenizer.json"
            base_download_report = {
                "status": "downloaded",
                "checkpoint": str(_ckpt),
                "tokenizer": str(_tok),
                "config": str(_cfg) if _cfg else None,
            }
    else:
        _manual_ckpt = local_checkpoint_path_input.value.strip()
        _manual_tok = local_tokenizer_path_input.value.strip()
        if _manual_ckpt:
            base_checkpoint_path = (PROJECT_ROOT / _manual_ckpt).expanduser()
        if _manual_tok:
            base_tokenizer_path = (PROJECT_ROOT / _manual_tok).expanduser()

    mo.md(
        f"""
        **Base checkpoint:** `{base_checkpoint_path}`

        **Base tokenizer:** `{base_tokenizer_path}`

        Status: `{base_download_report['status']}`
        """
    )
    return base_checkpoint_path, base_tokenizer_path


@app.cell(hide_code=True)
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
        seed: int = 42
        amp: str = "bf16"
        style_tag: str = "<natural>"

    return


@app.cell(hide_code=True)
def _(F, nn, torch):
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
                _dtype = x.dtype
                x = x.float()
                x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
                return (self.weight * x).to(_dtype)


        def _rotate_half(x):
            _even = x[..., 0::2]
            _odd = x[..., 1::2]
            return torch.stack((-_odd, _even), dim=-1).flatten(-2)


        class RotaryEmbedding(nn.Module):
            def __init__(self, dim: int, base: float = 10000.0) -> None:
                super().__init__()
                _inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
                self.register_buffer("inv_freq", _inv_freq, persistent=False)

            def forward(self, seq_len: int, device, dtype):
                _positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
                _freqs = torch.outer(_positions, self.inv_freq.to(device))
                _cos = _freqs.cos().repeat_interleave(2, dim=-1).to(dtype=dtype)
                _sin = _freqs.sin().repeat_interleave(2, dim=-1).to(dtype=dtype)
                return _cos[None, None, :, :], _sin[None, None, :, :]


        def _apply_rope(x, cos, sin):
            return (x * cos) + (_rotate_half(x) * sin)


        class CausalSelfAttention(nn.Module):
            def __init__(self, d_model: int, nhead: int, dropout: float, rope_base: float) -> None:
                super().__init__()
                self.nhead = nhead
                self.head_dim = d_model // nhead
                self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
                self.output = nn.Linear(d_model, d_model, bias=False)
                self.rope = RotaryEmbedding(self.head_dim, base=rope_base)
                self.attn_dropout = dropout
                self.resid_dropout = nn.Dropout(dropout)

            def forward(self, x, padding_mask):
                _batch, _seq_len, _width = x.shape
                _qkv = (
                    self.qkv(x)
                    .view(_batch, _seq_len, 3, self.nhead, self.head_dim)
                    .permute(2, 0, 3, 1, 4)
                )
                _q, _k, _v = _qkv[0], _qkv[1], _qkv[2]
                _cos, _sin = self.rope(_seq_len, x.device, _q.dtype)
                _q = _apply_rope(_q, _cos, _sin)
                _k = _apply_rope(_k, _cos, _sin)
                _dropout_p = self.attn_dropout if self.training else 0.0
                if padding_mask is None:
                    _attn_out = F.scaled_dot_product_attention(
                        _q, _k, _v, dropout_p=_dropout_p, is_causal=True
                    )
                else:
                    _causal = torch.ones(_seq_len, _seq_len, device=x.device, dtype=torch.bool).tril()
                    _key_keep = ~padding_mask[:, None, None, :]
                    _attn_mask = _key_keep & _causal[None, None, :, :]
                    _attn_out = F.scaled_dot_product_attention(
                        _q, _k, _v, attn_mask=_attn_mask, dropout_p=_dropout_p, is_causal=False
                    )
                _attn_out = _attn_out.transpose(1, 2).contiguous().view(_batch, _seq_len, _width)
                return self.resid_dropout(self.output(_attn_out))


        class SwiGLUFeedForward(nn.Module):
            def __init__(self, d_model: int, dim_feedforward: int, dropout: float) -> None:
                super().__init__()
                self.w12 = nn.Linear(d_model, dim_feedforward * 2, bias=False)
                self.w3 = nn.Linear(dim_feedforward, d_model, bias=False)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                _value, _gate = self.w12(x).chunk(2, dim=-1)
                return self.w3(self.dropout(_value * F.silu(_gate)))


        class DecoderBlock(nn.Module):
            def __init__(self, d_model, nhead, dim_feedforward, dropout, rope_base) -> None:
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
                vocab_size,
                pad_id,
                d_model,
                nhead,
                num_layers,
                dim_feedforward,
                dropout,
                max_seq_len,
                rope_base,
                gradient_checkpointing=False,
            ) -> None:
                super().__init__()
                self.pad_id = pad_id
                self.max_seq_len = max_seq_len
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
                self.gradient_checkpointing = gradient_checkpointing
                self.reset_parameters()

            def reset_parameters(self) -> None:
                nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    self.embedding.weight[self.pad_id].zero_()
                for _module in self.modules():
                    if isinstance(_module, nn.Linear) and _module.weight is not self.embedding.weight:
                        nn.init.normal_(_module.weight, mean=0.0, std=0.02)

            def forward(self, input_ids, padding_mask):
                _hidden = self.dropout(self.embedding(input_ids))
                for _layer in self.layers:
                    _hidden = _layer(_hidden, padding_mask)
                return self.output(self.norm(_hidden))
    return (ModernDecoderOnlyTransformer,)


@app.cell(hide_code=True)
def _(
    ModernDecoderOnlyTransformer,
    Tokenizer,
    base_checkpoint_path,
    base_tokenizer_path,
    torch,
):
    def load_student_checkpoint(checkpoint_path, tokenizer_path, device=None, dropout=0.0):
        if torch is None or Tokenizer is None or ModernDecoderOnlyTransformer is None:
            raise RuntimeError("torch, tokenizers, and the student model class are required")
        _device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _checkpoint = torch.load(checkpoint_path, map_location=_device)
        _config = dict(_checkpoint["config"])
        _tokenizer = Tokenizer.from_file(str(tokenizer_path))
        _model = ModernDecoderOnlyTransformer(
            vocab_size=int(_checkpoint.get("vocab_size", _config["vocab_size"])),
            pad_id=int(_config["pad_id"]),
            d_model=int(_config["d_model"]),
            nhead=int(_config["nhead"]),
            num_layers=int(_config["num_layers"]),
            dim_feedforward=int(_config["dim_feedforward"]),
            dropout=dropout,
            max_seq_len=int(_config["max_seq_len"]),
            rope_base=float(_config.get("rope_base", 10000.0)),
            gradient_checkpointing=False,
        ).to(_device)
        _model.load_state_dict(_checkpoint["model"])
        _model.eval()
        return _model, _tokenizer, _config, _checkpoint


    def student_prompt(text, target_lang, domain="general", style_tag="<natural>"):
        _target_tag = "<2pa>" if target_lang == "pa" else "<2en>"
        _domain_tag = "<legal>" if domain == "legal" else "<general>"
        return f"{_target_tag} {_domain_tag} {style_tag} {text}\n"


    @torch.no_grad() if torch is not None else (lambda _fn: _fn)
    def generate_student_translation(
        model,
        tokenizer,
        text,
        target_lang,
        domain,
        style_tag,
        device,
        max_seq_len,
        max_new_tokens=96,
        temperature=0.0,
        top_k=0,
    ):
        _eos_id = tokenizer.token_to_id("</s>")
        _pad_id = tokenizer.token_to_id("<pad>")
        _prompt = student_prompt(text, target_lang, domain, style_tag)
        _ids = tokenizer.encode(_prompt).ids
        _prompt_len = len(_ids)
        for _ in range(max_new_tokens):
            _input_ids = torch.tensor([_ids[-max_seq_len:]], dtype=torch.long, device=device)
            _padding_mask = _input_ids.eq(_pad_id)
            _logits = model(_input_ids, _padding_mask)[:, -1, :]
            if temperature and temperature > 0:
                _logits = _logits / temperature
                if top_k and top_k > 0:
                    _values, _indices = torch.topk(_logits, k=min(top_k, _logits.size(-1)))
                    _probs = torch.softmax(_values, dim=-1)
                    _next_id = int(_indices.gather(-1, torch.multinomial(_probs, 1)).item())
                else:
                    _probs = torch.softmax(_logits, dim=-1)
                    _next_id = int(torch.multinomial(_probs, 1).item())
            else:
                _next_id = int(_logits.argmax(dim=-1).item())
            _ids.append(_next_id)
            if _next_id == _eos_id:
                break
        _generated = _ids[_prompt_len:]
        if _eos_id in _generated:
            _generated = _generated[: _generated.index(_eos_id)]
        return tokenizer.decode(_generated)

    base_model_available = base_checkpoint_path.exists() and base_tokenizer_path.exists()
    return (
        base_model_available,
        generate_student_translation,
        load_student_checkpoint,
        student_prompt,
    )


@app.cell(hide_code=True)
def _(base_checkpoint_path, base_model_available, base_tokenizer_path, mo):
    _status = "found" if base_model_available else "missing"
    mo.md(
        f"""
        ## Student Baseline Loader

        Baseline checkpoint status: **{_status}**

        - checkpoint: `{base_checkpoint_path}`
        - tokenizer: `{base_tokenizer_path}`
        """
    )
    return


@app.cell(hide_code=True)
def _(pd):
    challenge_rows = [
        {
            "id": "legal_duration_en_pa",
            "direction": "en-pa",
            "domain": "legal",
            "source": "The agreement shall remain in force for five years.",
            "reference": "ਸਮਝੌਤਾ ਪੰਜ ਸਾਲਾਂ ਲਈ ਲਾਗੂ ਰਹੇਗਾ.",
            "phenomenon": "legal duration; ਲਈ vs ਤੋਂ",
        },
        {
            "id": "instruction_wrapper_en_pa",
            "direction": "en-pa",
            "domain": "legal",
            "source": "Translate only this: The agreement shall remain in force for five years.",
            "reference": "ਸਮਝੌਤਾ ਪੰਜ ਸਾਲਾਂ ਲਈ ਲਾਗੂ ਰਹੇਗਾ.",
            "phenomenon": "instruction wrapper should be ignored",
        },
        {
            "id": "number_date_en_pa",
            "direction": "en-pa",
            "domain": "general",
            "source": "The meeting will start at 9:30 AM on 15 August 2026.",
            "reference": "ਮੀਟਿੰਗ 15 ਅਗਸਤ 2026 ਨੂੰ ਸਵੇਰੇ 9:30 ਵਜੇ ਸ਼ੁਰੂ ਹੋਵੇਗੀ.",
            "phenomenon": "numbers and date",
        },
        {
            "id": "negation_en_pa",
            "direction": "en-pa",
            "domain": "general",
            "source": "The application was not approved because the documents were incomplete.",
            "reference": "ਦਸਤਾਵੇਜ਼ ਅਧੂਰੇ ਹੋਣ ਕਰਕੇ ਅਰਜ਼ੀ ਮਨਜ਼ੂਰ ਨਹੀਂ ਕੀਤੀ ਗਈ.",
            "phenomenon": "negation and cause",
        },
        {
            "id": "pa_en_legal",
            "direction": "pa-en",
            "domain": "legal",
            "source": "ਸਮਝੌਤਾ ਪੰਜ ਸਾਲਾਂ ਲਈ ਲਾਗੂ ਰਹੇਗਾ.",
            "reference": "The agreement shall remain in force for five years.",
            "phenomenon": "Punjabi to English legal duration",
        },
        {
            "id": "pa_en_general",
            "direction": "pa-en",
            "domain": "general",
            "source": "ਕਿਸਾਨਾਂ ਨੂੰ ਨਵੀਂ ਤਕਨੀਕ ਬਾਰੇ ਸਿਖਲਾਈ ਦਿੱਤੀ ਗਈ.",
            "reference": "The farmers were given training about the new technology.",
            "phenomenon": "general domain",
        },
    ]
    challenge_frame = pd.DataFrame(challenge_rows) if pd is not None else challenge_rows
    challenge_frame
    return (challenge_frame,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Evaluation and Cache Controls")
    eval_sample_rows_input = mo.ui.number(value=256, start=8, stop=20_000, step=8, label="Corpus sample rows for teacher cache")
    eval_seed_input = mo.ui.number(value=42, start=0, stop=100_000, step=1, label="Sample seed")
    include_challenge_set = mo.ui.checkbox(value=True, label="Include manual challenge set")
    mo.hstack([eval_sample_rows_input, eval_seed_input, include_challenge_set])
    return eval_sample_rows_input, eval_seed_input, include_challenge_set


@app.cell(hide_code=True)
def flores_gate_md(mo):
    mo.md("""
    ## FLORES+ Benchmark Gate

    FLORES+ is the public reporting benchmark for `eng_Latn -> pan_Guru`.
    The dataset is gated on Hugging Face and is intended for evaluation, not
    training. Use `dev` for notebook debugging and `devtest` for final paper
    tables after accepting the dataset terms.

    If `datasets.load_dataset` fails with a multiprocessing/RLock error, this
    notebook falls back to direct gated JSONL downloads through
    `huggingface_hub`: `{split}/eng_Latn.jsonl` and `{split}/pan_Guru.jsonl`.
    """)
    return


@app.cell(hide_code=True)
def flores_controls(mo):
    flores_split_input = mo.ui.dropdown(
        options=["dev", "devtest"],
        value="dev",
        label="FLORES+ split",
    )
    flores_sample_rows_input = mo.ui.number(
        value=10,
        start=1,
        stop=1012,
        step=1,
        label="FLORES+ rows",
    )
    include_flores_in_cache = mo.ui.checkbox(
        value=True,
        label="Add loaded FLORES+ rows to teacher cache",
    )
    load_flores_button = mo.ui.run_button(label="Load FLORES+")
    mo.hstack([flores_split_input, flores_sample_rows_input, include_flores_in_cache, load_flores_button])
    return (
        flores_sample_rows_input,
        flores_split_input,
        include_flores_in_cache,
        load_flores_button,
    )


@app.cell(hide_code=True)
def flores_loader(
    flores_sample_rows_input,
    flores_split_input,
    hf_hub_download,
    load_dataset,
    load_flores_button,
    mo,
    pd,
):
    flores_pa_benchmark = None
    flores_report = {"status": "not loaded"}

    def _normalize_flores_frame(frame, lang_code):
        _frame = frame.copy()
        if "text" not in _frame.columns:
            _candidate_cols = [
                _col for _col in ["sentence", "raw_text", "content"] if _col in _frame.columns
            ]
            if not _candidate_cols:
                raise ValueError(f"No text column found for {lang_code}; columns={list(_frame.columns)}")
            _frame = _frame.rename(columns={_candidate_cols[0]: "text"})
        if "id" not in _frame.columns:
            _frame["id"] = range(len(_frame))
        if "domain" not in _frame.columns:
            _frame["domain"] = "flores+"
        if "topic" not in _frame.columns:
            _frame["topic"] = "unknown"
        return _frame[["id", "text", "domain", "topic"]]


    def _load_flores_lang(split, lang_code):
        _datasets_error = None
        if load_dataset is not None:
            try:
                _dataset = load_dataset("openlanguagedata/flores_plus", lang_code, split=split)
                return _normalize_flores_frame(_dataset.to_pandas(), lang_code), "datasets.load_dataset"
            except Exception as _exc:
                _datasets_error = _exc
        if hf_hub_download is None:
            raise RuntimeError(f"datasets loader failed and huggingface_hub is unavailable: {_datasets_error}")
        _path = hf_hub_download(
            repo_id="openlanguagedata/flores_plus",
            filename=f"{split}/{lang_code}.jsonl",
            repo_type="dataset",
        )
        _note = "jsonl fallback"
        if _datasets_error is not None:
            _note += f" after datasets error: {_datasets_error}"
        return _normalize_flores_frame(pd.read_json(_path, lines=True), lang_code), _note


    if load_flores_button.value:
        if pd is None:
            flores_report = {"status": "missing pandas"}
        else:
            try:
                _split = flores_split_input.value
                _eng_df, _eng_loader = _load_flores_lang(_split, "eng_Latn")
                _pan_df, _pan_loader = _load_flores_lang(_split, "pan_Guru")
                _eng_df = _eng_df.rename(columns={"text": "source"})
                _pan_df = _pan_df[["id", "text"]].rename(columns={"text": "reference"})
                _merged = _eng_df.merge(_pan_df, on="id", how="inner")
                _merged = _merged.head(int(flores_sample_rows_input.value)).copy()
                _merged["id"] = "flores:" + _merged["id"].astype(str)
                _merged["direction"] = "en-pa"
                _merged["phenomenon"] = "flores+ benchmark"
                flores_pa_benchmark = _merged[
                    ["id", "direction", "domain", "source", "reference", "phenomenon", "topic"]
                ]
                flores_report = {
                    "status": "loaded",
                    "split": _split,
                    "rows": len(flores_pa_benchmark),
                    "eng_loader": _eng_loader,
                    "pan_loader": _pan_loader,
                }
            except Exception as _exc:
                flores_report = {"status": f"FLORES+ load failed: {_exc}"}

    mo.md(f"**FLORES+ status:** `{flores_report['status']}`")
    flores_pa_benchmark.head(8) if flores_pa_benchmark is not None else flores_report
    return (flores_pa_benchmark,)


@app.cell(hide_code=True)
def _(
    DATA_PATH,
    challenge_frame,
    eval_sample_rows_input,
    eval_seed_input,
    flores_pa_benchmark,
    include_challenge_set,
    include_flores_in_cache,
    pd,
):
    def load_eval_seed_frame(path, sample_rows, seed, include_challenge=True):
        if pd is None:
            raise RuntimeError("pandas is required")
        _frames = []
        if path.exists():
            _raw = pd.read_csv(
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
            )
            _raw["en"] = _raw["en"].fillna("")
            _raw["pa"] = _raw["pa"].fillna("")
            _sample = _raw.sample(n=min(sample_rows, len(_raw)), random_state=seed).reset_index(drop=True)
            _en_pa = pd.DataFrame(
                {
                    "id": _sample["id"].astype(str) + ":en-pa",
                    "direction": "en-pa",
                    "domain": _sample["domain"].astype(str),
                    "source": _sample["en"].astype(str),
                    "reference": _sample["pa"].astype(str),
                    "phenomenon": "corpus sample",
                }
            )
            _pa_en = pd.DataFrame(
                {
                    "id": _sample["id"].astype(str) + ":pa-en",
                    "direction": "pa-en",
                    "domain": _sample["domain"].astype(str),
                    "source": _sample["pa"].astype(str),
                    "reference": _sample["en"].astype(str),
                    "phenomenon": "corpus sample",
                }
            )
            _frames.extend([_en_pa, _pa_en])
        if include_challenge and isinstance(challenge_frame, pd.DataFrame):
            _frames.append(challenge_frame.copy())
        if bool(include_flores_in_cache.value) and isinstance(flores_pa_benchmark, pd.DataFrame):
            _frames.append(flores_pa_benchmark.copy())
        if not _frames:
            return pd.DataFrame(columns=["id", "direction", "domain", "source", "reference", "phenomenon"])
        return pd.concat(_frames, ignore_index=True)


    eval_seed_frame = load_eval_seed_frame(
        DATA_PATH,
        int(eval_sample_rows_input.value),
        int(eval_seed_input.value),
        bool(include_challenge_set.value),
    )
    eval_seed_frame.head(12)
    return (eval_seed_frame,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Teacher Model Choice

    This notebook uses one runnable teacher: `sarvamai/sarvam-translate`.
    Sarvam-Translate supports Punjabi and produced strong Gurmukhi outputs in the
    10-row FLORES+ cloud sanity check, so the distillation path is intentionally
    simple:

    - generate candidate teacher translations with Sarvam,
    - reject outputs with script, instruction-leakage, and length checks,
    - fine-tune the from-scratch 58M decoder on accepted pairs,
    - run the small MiniLLM-style on-policy section only after the cache passes.

    **License disclosure:** the Sarvam-Translate model card lists GPL-3.0. This is
    fine for a research notebook, but be cautious before publishing distilled
    weights or commercial artifacts derived from teacher outputs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    teacher_backend = mo.ui.dropdown(
        options=[
            "cache-only",
            "sarvamai/sarvam-translate",
        ],
        value="sarvamai/sarvam-translate",
        label="Teacher backend",
    )
    teacher_num_beams = mo.ui.number(value=1, start=1, stop=4, step=1, label="Teacher beams")
    teacher_max_new_tokens = mo.ui.number(value=160, start=16, stop=512, step=16, label="Teacher max new tokens")
    load_teacher_button = mo.ui.run_button(label="Load teacher")
    mo.hstack([teacher_backend, teacher_num_beams, teacher_max_new_tokens, load_teacher_button])
    return (
        load_teacher_button,
        teacher_backend,
        teacher_max_new_tokens,
        teacher_num_beams,
    )


@app.cell(hide_code=True)
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    load_teacher_button,
    mo,
    teacher_backend,
    torch,
):
    teacher_bundle = None
    teacher_load_report = {"status": "not loaded"}

    if load_teacher_button.value:
        if teacher_backend.value == "cache-only":
            teacher_load_report = {"status": "cache-only; no teacher loaded"}
        elif AutoModelForCausalLM is None or AutoTokenizer is None or torch is None:
            teacher_load_report = {"status": "missing transformers or torch"}
        else:
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if _device.type == "cuda":
                _dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                _dtype = torch.float32
            _teacher_id = teacher_backend.value
            _model_kwargs = {"torch_dtype": _dtype, "low_cpu_mem_usage": True}
            if _device.type == "cuda":
                _model_kwargs["device_map"] = "auto"
            try:
                _tokenizer = AutoTokenizer.from_pretrained(_teacher_id)
                if _tokenizer.pad_token_id is None and _tokenizer.eos_token_id is not None:
                    _tokenizer.pad_token = _tokenizer.eos_token
                _model = AutoModelForCausalLM.from_pretrained(_teacher_id, **_model_kwargs)
                if _device.type != "cuda":
                    _model = _model.to(_device)
                _model.eval()
                _model_device = next(_model.parameters()).device
                teacher_bundle = {
                    "kind": "sarvam_translate",
                    "id": _teacher_id,
                    "tokenizer": _tokenizer,
                    "model": _model,
                    "device": _model_device,
                    "license": "GPL-3.0",
                }
                teacher_load_report = {
                    "status": "loaded",
                    "teacher_id": _teacher_id,
                    "device": str(_model_device),
                    "dtype": str(_dtype).replace("torch.", ""),
                    "license": "GPL-3.0",
                }
            except Exception as _exc:
                teacher_load_report = {"status": f"teacher load failed: {_exc}"}

    mo.md(f"**Teacher status:** `{teacher_load_report['status']}`")
    return (teacher_bundle,)


@app.cell(hide_code=True)
def _(torch):
    def sarvam_target_language(direction):
        if direction == "en-pa":
            return "Punjabi in Gurmukhi script"
        if direction == "pa-en":
            return "English"
        raise ValueError(f"Unknown direction: {direction}")


    def sarvam_messages(text, direction):
        _target = sarvam_target_language(direction)
        return [
            {
                "role": "system",
                "content": f"Translate the text below to {_target}. Return only the translation.",
            },
            {"role": "user", "content": str(text)},
        ]


    def sarvam_prompt(tokenizer, text, direction):
        return tokenizer.apply_chat_template(
            sarvam_messages(text, direction),
            tokenize=False,
            add_generation_prompt=True,
        )


    def clean_teacher_text(text):
        return str(text).strip().strip('"').strip("'").strip()


    @torch.no_grad() if torch is not None else (lambda _fn: _fn)
    def translate_with_sarvam(bundle, texts, directions, max_new_tokens=160, num_beams=1):
        _tokenizer = bundle["tokenizer"]
        _model = bundle["model"]
        _device = bundle["device"]
        _pad_id = _tokenizer.pad_token_id or _tokenizer.eos_token_id
        _outputs = []
        for _text, _direction in zip(texts, directions, strict=False):
            _prompt = sarvam_prompt(_tokenizer, _text, _direction)
            _inputs = _tokenizer([_prompt], return_tensors="pt", truncation=True, max_length=4096).to(_device)
            _generate_kwargs = {
                "max_new_tokens": int(max_new_tokens),
                "do_sample": False,
                "pad_token_id": _pad_id,
            }
            if int(num_beams) > 1:
                _generate_kwargs["num_beams"] = int(num_beams)
            _generated = _model.generate(**_inputs, **_generate_kwargs)
            _output_ids = _generated[0, _inputs["input_ids"].shape[-1]:]
            _outputs.append(clean_teacher_text(_tokenizer.decode(_output_ids, skip_special_tokens=True)))
        return _outputs


    @torch.no_grad() if torch is not None else (lambda _fn: _fn)
    def score_with_sarvam(bundle, sources, candidates, directions, max_length=2048):
        _tokenizer = bundle["tokenizer"]
        _model = bundle["model"]
        _device = bundle["device"]
        _scores = []
        for _source, _candidate, _direction in zip(sources, candidates, directions, strict=False):
            _candidate = clean_teacher_text(_candidate)
            if not _candidate:
                _scores.append(-1e9)
                continue
            _prompt = sarvam_prompt(_tokenizer, _source, _direction)
            _prompt_ids = _tokenizer(_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
            _candidate_text = _candidate + (_tokenizer.eos_token or "")
            _candidate_ids = _tokenizer(_candidate_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
            _budget = int(max_length) - _prompt_ids.shape[-1]
            if _budget < 2:
                _scores.append(-1e9)
                continue
            _candidate_ids = _candidate_ids[:, :_budget]
            _full_ids = torch.cat([_prompt_ids, _candidate_ids], dim=-1).to(_device)
            if _full_ids.shape[-1] < 2:
                _scores.append(-1e9)
                continue
            _output = _model(input_ids=_full_ids[:, :-1])
            _logits = _output.logits
            _shift_labels = _full_ids[:, 1:]
            _mask = torch.zeros_like(_shift_labels, dtype=torch.bool)
            _mask[:, max(0, _prompt_ids.shape[-1] - 1):] = True
            if _tokenizer.pad_token_id is not None:
                _mask &= _shift_labels.ne(_tokenizer.pad_token_id)
            _log_probs = torch.log_softmax(_logits.float(), dim=-1)
            _token_lp = _log_probs.gather(-1, _shift_labels.unsqueeze(-1)).squeeze(-1)
            _scores.append(float((_token_lp * _mask).sum().detach().cpu() / _mask.sum().clamp_min(1).detach().cpu()))
        return _scores


    def translate_with_teacher(bundle, texts, directions, max_new_tokens=128, num_beams=4):
        if bundle is None:
            return [""] * len(texts)
        if bundle["kind"] == "sarvam_translate":
            return translate_with_sarvam(bundle, texts, directions, max_new_tokens, num_beams)
        raise ValueError(f"Unsupported teacher kind: {bundle['kind']}")

    return score_with_sarvam, translate_with_teacher


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Build Teacher Cache")
    cache_name_input = mo.ui.text(value="sarvam_teacher_cache.csv", label="Cache filename")
    generate_cache_button = mo.ui.run_button(label="Generate baseline + teacher cache")
    mo.hstack([cache_name_input, generate_cache_button])
    return cache_name_input, generate_cache_button


@app.cell(hide_code=True)
def _(
    CACHE_DIR,
    base_checkpoint_path,
    base_model_available,
    base_tokenizer_path,
    cache_name_input,
    eval_seed_frame,
    generate_cache_button,
    generate_student_translation,
    load_student_checkpoint,
    mo,
    pd,
    teacher_backend,
    teacher_bundle,
    teacher_max_new_tokens,
    teacher_num_beams,
    torch,
    translate_with_teacher,
):
    teacher_cache_path = CACHE_DIR / cache_name_input.value
    teacher_cache_frame = None
    teacher_cache_report = {"status": "not generated this run", "path": str(teacher_cache_path)}

    if generate_cache_button.value:
        if pd is None or torch is None:
            teacher_cache_report = {"status": "missing pandas or torch"}
        else:
            _rows = eval_seed_frame.copy().reset_index(drop=True)
            _baseline_outputs = []
            if base_model_available:
                _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                _student, _student_tok, _cfg, _ckpt = load_student_checkpoint(
                    base_checkpoint_path,
                    base_tokenizer_path,
                    device=_device,
                    dropout=0.0,
                )
                for _row in _rows.itertuples(index=False):
                    _target_lang = "pa" if _row.direction == "en-pa" else "en"
                    _baseline_outputs.append(
                        generate_student_translation(
                            _student,
                            _student_tok,
                            _row.source,
                            target_lang=_target_lang,
                            domain=_row.domain,
                            style_tag="<natural>",
                            device=_device,
                            max_seq_len=int(_cfg["max_seq_len"]),
                            max_new_tokens=96,
                            temperature=0.0,
                        )
                    )
            else:
                _baseline_outputs = [""] * len(_rows)
            _rows["baseline_output"] = _baseline_outputs

            if teacher_bundle is not None:
                _rows["teacher_output"] = translate_with_teacher(
                    teacher_bundle,
                    _rows["source"].tolist(),
                    _rows["direction"].tolist(),
                    max_new_tokens=int(teacher_max_new_tokens.value),
                    num_beams=int(teacher_num_beams.value),
                )
                _rows["teacher_id"] = teacher_backend.value
            elif "teacher_output" not in _rows.columns:
                _rows["teacher_output"] = ""
                _rows["teacher_id"] = teacher_backend.value

            _rows.to_csv(teacher_cache_path, index=False)
            teacher_cache_frame = _rows
            teacher_cache_report = {"status": "generated", "rows": len(_rows), "path": str(teacher_cache_path)}
    elif teacher_cache_path.exists() and pd is not None:
        teacher_cache_frame = pd.read_csv(teacher_cache_path)
        teacher_cache_report = {"status": "loaded existing", "rows": len(teacher_cache_frame), "path": str(teacher_cache_path)}

    mo.md(f"**Teacher cache:** `{teacher_cache_report['status']}` at `{teacher_cache_path}`")
    teacher_cache_frame.head(8) if teacher_cache_frame is not None else teacher_cache_report
    return (teacher_cache_frame,)


@app.cell(hide_code=True)
def _(sacrebleu):
    def has_gurmukhi(text):
        return any(0x0A00 <= ord(_char) <= 0x0A7F for _char in str(text))


    def has_latin(text):
        return any(("A" <= _char <= "Z") or ("a" <= _char <= "z") for _char in str(text))


    def instruction_leakage(text):
        _lower = str(text).lower()
        return int(any(_phrase in _lower for _phrase in ["translate", "only this", "ਅਨੁਵਾਦ ਕਰੋ", "ਸਿਰਫ"]))


    def compute_mt_metrics(frame, hypothesis_col):
        _view = frame.dropna(subset=["reference", hypothesis_col]).copy()
        _view[hypothesis_col] = _view[hypothesis_col].astype(str)
        _view["reference"] = _view["reference"].astype(str)
        if len(_view) == 0:
            return {
                "system": hypothesis_col,
                "rows": 0,
                "bleu": None,
                "chrf": None,
                "script_error_rate": None,
                "english_leak_rate": None,
                "instruction_leak_rate": None,
                "mean_len_ratio": None,
            }
        _hyps = _view[hypothesis_col].tolist()
        _refs = [_view["reference"].tolist()]
        if sacrebleu is not None:
            _bleu = float(sacrebleu.corpus_bleu(_hyps, _refs).score)
            _chrf = float(sacrebleu.corpus_chrf(_hyps, _refs).score)
        else:
            _bleu = None
            _chrf = None
        _script_errors = []
        _english_leaks = []
        for _row in _view.itertuples(index=False):
            _hyp = getattr(_row, hypothesis_col)
            if _row.direction == "en-pa":
                _script_errors.append(not has_gurmukhi(_hyp))
                _english_leaks.append(has_latin(_hyp))
            else:
                _script_errors.append(not has_latin(_hyp))
                _english_leaks.append(False)
        _len_ratio = [
            len(str(_hyp).split()) / max(1, len(str(_ref).split()))
            for _hyp, _ref in zip(_hyps, _view["reference"].tolist(), strict=False)
        ]
        return {
            "system": hypothesis_col,
            "rows": len(_view),
            "bleu": _bleu,
            "chrf": _chrf,
            "script_error_rate": sum(_script_errors) / len(_script_errors),
            "english_leak_rate": sum(_english_leaks) / len(_english_leaks),
            "instruction_leak_rate": sum(instruction_leakage(_hyp) for _hyp in _hyps) / len(_hyps),
            "mean_len_ratio": sum(_len_ratio) / len(_len_ratio),
        }

    return compute_mt_metrics, has_gurmukhi, has_latin, instruction_leakage


@app.cell(hide_code=True)
def sarvam_teacher_qualification(
    has_gurmukhi,
    has_latin,
    instruction_leakage,
    mo,
    pd,
    teacher_cache_frame,
):
    teacher_quality_frame = None
    qualified_teacher_frame = None

    if teacher_cache_frame is None or pd is None:
        teacher_quality_view = mo.md("Generate or load a Sarvam teacher cache to qualify distillation rows.")
    elif "teacher_output" not in teacher_cache_frame.columns:
        teacher_quality_view = mo.md("Teacher cache has no `teacher_output` column.")
    else:
        _frame = teacher_cache_frame.copy()
        _frame["teacher_output"] = _frame["teacher_output"].fillna("").astype(str)
        _frame["reference"] = _frame["reference"].fillna("").astype(str)
        _frame["teacher_len_ratio"] = [
            len(_hyp) / max(1, len(_ref))
            for _hyp, _ref in zip(_frame["teacher_output"], _frame["reference"], strict=False)
        ]
        _frame["teacher_script_ok"] = [
            has_gurmukhi(_hyp) and not has_latin(_hyp)
            if _direction == "en-pa"
            else has_latin(_hyp)
            for _hyp, _direction in zip(_frame["teacher_output"], _frame["direction"], strict=False)
        ]
        _frame["teacher_no_instruction_leak"] = [
            not bool(instruction_leakage(_hyp)) for _hyp in _frame["teacher_output"]
        ]
        _frame["teacher_length_ok"] = _frame["teacher_len_ratio"].between(0.5, 1.8)
        _frame["teacher_qualified"] = (
            _frame["teacher_output"].str.strip().ne("")
            & _frame["teacher_script_ok"]
            & _frame["teacher_no_instruction_leak"]
            & _frame["teacher_length_ok"]
        )
        teacher_quality_frame = _frame
        qualified_teacher_frame = _frame[_frame["teacher_qualified"]].copy()
        _summary = pd.DataFrame(
            [
                {"check": "rows", "value": len(_frame)},
                {"check": "qualified rows", "value": len(qualified_teacher_frame)},
                {"check": "qualification rate", "value": round(float(_frame["teacher_qualified"].mean()), 3)},
                {"check": "script pass rate", "value": round(float(_frame["teacher_script_ok"].mean()), 3)},
                {"check": "length pass rate", "value": round(float(_frame["teacher_length_ok"].mean()), 3)},
            ]
        )
        teacher_quality_view = mo.vstack([
            mo.md("## Sarvam Teacher Qualification"),
            _summary,
            qualified_teacher_frame.head(8),
        ])
    teacher_quality_view
    return (qualified_teacher_frame,)


@app.cell(hide_code=True)
def cloud_flores10_sanity(mo, pd):
    cloud_flores10_score_frame = pd.DataFrame(
        [
            {
                "model": "base_best",
                "bleu": 0.2653043503761413,
                "chrf": 3.921556105471328,
                "script_error_rate": 0.5,
                "english_leak_rate": 0.5,
            },
            {
                "model": "sarvam_teacher",
                "bleu": 25.72790987330312,
                "chrf": 56.217154219237045,
                "script_error_rate": 0.0,
                "english_leak_rate": 0.0,
            },
        ]
    ) if pd is not None else None

    if cloud_flores10_score_frame is None:
        cloud_flores10_score_view = mo.md("pandas is required to show the FLORES+ cloud sanity score.")
    else:
        _delta_bleu = cloud_flores10_score_frame.loc[1, "bleu"] - cloud_flores10_score_frame.loc[0, "bleu"]
        _delta_chrf = cloud_flores10_score_frame.loc[1, "chrf"] - cloud_flores10_score_frame.loc[0, "chrf"]
        cloud_flores10_score_view = mo.vstack(
            [
                mo.md(
                    f"""
                    ## FLORES+ 10-Sentence Cloud Sanity Check

                    These scores came from the cloud notebook using `base_best.pt` and
                    `sarvamai/sarvam-translate` on a 10-row FLORES+ `eng_Latn -> pan_Guru`
                    sample. Because this local kernel is missing `sacrebleu`, treat this
                    as the recorded cloud run rather than a local recomputation.

                    Sarvam is ahead by **{_delta_bleu:.2f} BLEU** and **{_delta_chrf:.2f} chrF**.
                    The base model has script/English leakage on **5 of 10** examples;
                    Sarvam has none in this sample.
                    """
                ),
                cloud_flores10_score_frame,
            ]
        )
    cloud_flores10_score_view
    return (cloud_flores10_score_frame,)


@app.cell(hide_code=True)
def _(compute_mt_metrics, mo, pd, teacher_cache_frame):
    if teacher_cache_frame is None or pd is None:
        metric_frame = None
        metric_view = mo.md("Generate or load a teacher cache to compute BLEU/chrF.")
    else:
        _systems = ["baseline_output"]
        if "teacher_output" in teacher_cache_frame.columns and teacher_cache_frame["teacher_output"].astype(str).str.len().sum() > 0:
            _systems.append("teacher_output")
        metric_frame = pd.DataFrame([compute_mt_metrics(teacher_cache_frame, _system) for _system in _systems])
        metric_view = metric_frame
    metric_view
    return (metric_frame,)


@app.cell(hide_code=True)
def _(metric_frame, mo, plt):
    if metric_frame is None:
        metric_plot = mo.md("No metrics to plot yet.")
    elif plt is None:
        metric_plot = mo.md("matplotlib is not installed.")
    else:
        _plot_cols = [_col for _col in ["bleu", "chrf", "script_error_rate", "instruction_leak_rate"] if _col in metric_frame]
        _fig, _axes = plt.subplots(1, len(_plot_cols), figsize=(4.4 * len(_plot_cols), 4))
        if len(_plot_cols) == 1:
            _axes = [_axes]
        for _axis, _col in zip(_axes, _plot_cols, strict=False):
            _axis.bar(metric_frame["system"], metric_frame[_col], color=["#3a6ea5", "#bf6f24", "#2f9c95"][: len(metric_frame)])
            _axis.set_title(_col)
            _axis.tick_params(axis="x", rotation=25)
        _fig.tight_layout()
        metric_plot = _fig
    metric_plot
    return


@app.cell(hide_code=True)
def _(mo, pd, teacher_cache_frame):
    if teacher_cache_frame is None or pd is None:
        challenge_comparison = mo.md("Generate/load the teacher cache to view challenge examples.")
    else:
        _mask = teacher_cache_frame["phenomenon"].astype(str).ne("corpus sample")
        _cols = ["id", "direction", "domain", "source", "reference", "baseline_output", "teacher_output", "phenomenon"]
        challenge_comparison = teacher_cache_frame.loc[_mask, [_col for _col in _cols if _col in teacher_cache_frame.columns]]
    challenge_comparison
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Reverse-KL Intuition Lab

    Forward KL is often **mean-seeking**: it pays heavily for missing any
    teacher-supported region. Reverse KL is often **mode-seeking**: it is
    comfortable choosing one strong mode and avoiding low-probability regions.

    For translation, that means reverse KL can help the student prefer a
    crisp high-probability phrasing instead of averaging several plausible
    phrasings into a less natural sentence. The risk is mode collapse, so
    the notebook keeps gold/teacher CE and challenge-set metrics in the loop.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, pd, plt):
    if np is None or pd is None:
        toy_kl_view = mo.md("numpy and pandas are needed for the toy KL lab.")
    else:
        _teacher = np.array([0.44, 0.05, 0.02, 0.05, 0.44])
        _forward_style = np.array([0.25, 0.18, 0.14, 0.18, 0.25])
        _reverse_style = np.array([0.88, 0.06, 0.02, 0.02, 0.02])
        _labels = ["mode A", "near A", "bridge", "near B", "mode B"]

        def _kl(p, q):
            return float(np.sum(p * (np.log(p + 1e-9) - np.log(q + 1e-9))))

        toy_kl_frame = pd.DataFrame(
            {
                "region": _labels,
                "teacher": _teacher,
                "forward_kl_style_student": _forward_style,
                "reverse_kl_style_student": _reverse_style,
            }
        )
        toy_kl_summary = pd.DataFrame(
            [
                {"student": "forward style", "KL_teacher_to_student": _kl(_teacher, _forward_style), "KL_student_to_teacher": _kl(_forward_style, _teacher)},
                {"student": "reverse style", "KL_teacher_to_student": _kl(_teacher, _reverse_style), "KL_student_to_teacher": _kl(_reverse_style, _teacher)},
            ]
        )
        if plt is not None:
            _fig, _axis = plt.subplots(figsize=(9, 4))
            _x = np.arange(len(_labels))
            _axis.bar(_x - 0.25, _teacher, width=0.25, label="teacher", color="#13283d")
            _axis.bar(_x, _forward_style, width=0.25, label="forward KL style", color="#3a6ea5")
            _axis.bar(_x + 0.25, _reverse_style, width=0.25, label="reverse KL style", color="#bf6f24")
            _axis.set_xticks(_x)
            _axis.set_xticklabels(_labels)
            _axis.set_ylabel("probability")
            _axis.legend()
            _fig.tight_layout()
            toy_kl_view = mo.vstack([_fig, toy_kl_summary])
        else:
            toy_kl_view = mo.vstack([toy_kl_frame, toy_kl_summary])
    toy_kl_view
    return


@app.cell(hide_code=True)
def _(Dataset, pad_sequence, torch):
    DistillationDataset = None
    DistillationCollator = None

    if torch is not None and pad_sequence is not None:

        class DistillationDataset(Dataset):
            def __init__(
                self,
                frame,
                tokenizer,
                max_seq_len,
                target_mix="teacher",
                style_tag="<natural>",
                min_target_tokens=48,
            ) -> None:
                self.frame = frame.reset_index(drop=True)
                self.tokenizer = tokenizer
                self.max_seq_len = max_seq_len
                self.target_mix = target_mix
                self.style_tag = style_tag
                self.min_target_tokens = min_target_tokens
                self.eos_id = tokenizer.token_to_id("</s>")

            def __len__(self):
                return len(self.frame)

            def __getitem__(self, index):
                _row = self.frame.iloc[index]
                _target_lang = "pa" if _row["direction"] == "en-pa" else "en"
                _target_tag = "<2pa>" if _target_lang == "pa" else "<2en>"
                _domain_tag = "<legal>" if str(_row["domain"]) == "legal" else "<general>"
                _prompt = f"{_target_tag} {_domain_tag} {self.style_tag} {_row['source']}\n"
                if self.target_mix == "gold" or not str(_row.get("teacher_output", "")).strip():
                    _target = str(_row["reference"])
                elif self.target_mix == "teacher":
                    _target = str(_row["teacher_output"])
                else:
                    _target = str(_row["teacher_output"]) if index % 2 == 0 else str(_row["reference"])
                _prompt_ids = self.tokenizer.encode(_prompt).ids
                _max_prompt_len = max(8, self.max_seq_len - self.min_target_tokens - 1)
                _prompt_ids = _prompt_ids[:_max_prompt_len]
                _target_budget = max(1, self.max_seq_len - len(_prompt_ids) - 1)
                _target_ids = self.tokenizer.encode(_target).ids[:_target_budget] + [self.eos_id]
                _full = _prompt_ids + _target_ids
                _input_ids = torch.tensor(_full[:-1], dtype=torch.long)
                _labels = torch.tensor(_full[1:], dtype=torch.long)
                _cutoff = max(0, min(len(_prompt_ids) - 1, _labels.numel()))
                _labels[:_cutoff] = -100
                return {"input_ids": _input_ids, "labels": _labels}


        class DistillationCollator:
            def __init__(self, pad_id):
                self.pad_id = pad_id

            def __call__(self, batch):
                _input_ids = pad_sequence(
                    [_item["input_ids"] for _item in batch],
                    batch_first=True,
                    padding_value=self.pad_id,
                )
                _labels = pad_sequence(
                    [_item["labels"] for _item in batch],
                    batch_first=True,
                    padding_value=-100,
                )
                return {
                    "input_ids": _input_ids,
                    "labels": _labels,
                    "padding_mask": _input_ids.eq(self.pad_id),
                }
    return DistillationCollator, DistillationDataset


@app.cell(hide_code=True)
def _(F, torch):
    def ce_translation_loss(model, batch):
        _logits = model(batch["input_ids"], batch["padding_mask"])
        return F.cross_entropy(
            _logits.reshape(-1, _logits.size(-1)),
            batch["labels"].reshape(-1),
            ignore_index=-100,
        )


    def student_sequence_logprob(model, tokenizer, prompt, output_text, device, max_seq_len):
        _pad_id = tokenizer.token_to_id("<pad>")
        _prompt_ids = tokenizer.encode(prompt).ids
        _output_ids = tokenizer.encode(output_text).ids + [tokenizer.token_to_id("</s>")]
        _full = (_prompt_ids + _output_ids)[-max_seq_len:]
        _input_ids = torch.tensor([_full[:-1]], dtype=torch.long, device=device)
        _labels = torch.tensor([_full[1:]], dtype=torch.long, device=device)
        _padding = _input_ids.eq(_pad_id)
        _logits = model(_input_ids, _padding)
        _log_probs = torch.log_softmax(_logits.float(), dim=-1)
        _token_lp = _log_probs.gather(-1, _labels.unsqueeze(-1)).squeeze(-1)
        _mask = torch.ones_like(_labels, dtype=torch.bool)
        _prompt_cutoff = max(0, min(len(_prompt_ids) - 1, _labels.numel()))
        _mask[:, :_prompt_cutoff] = False
        _score = (_token_lp * _mask).sum() / _mask.sum().clamp_min(1)
        return _score


    def rkl_policy_loss(student_logps, teacher_scores, normalize=True):
        _student_scores = torch.stack(student_logps)
        _teacher_scores = torch.tensor(teacher_scores, dtype=_student_scores.dtype, device=_student_scores.device)
        _advantage = (_student_scores.detach() - _teacher_scores).detach()
        if normalize and _advantage.numel() > 1:
            _advantage = (_advantage - _advantage.mean()) / _advantage.std().clamp_min(1e-5)
        return (_advantage * _student_scores).mean()

    return ce_translation_loss, rkl_policy_loss, student_sequence_logprob


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Distillation Training Controls")
    distill_mode = mo.ui.dropdown(
        options=["seqkd_ce", "mixed_gold_teacher_ce"],
        value="mixed_gold_teacher_ce",
        label="Distillation mode",
    )
    distill_batch_size = mo.ui.number(value=16, start=1, stop=256, step=1, label="Batch size")
    distill_steps = mo.ui.number(value=500, start=1, stop=50_000, step=100, label="Max optimizer steps")
    distill_lr = mo.ui.number(value=1e-5, start=1e-6, stop=1e-4, step=1e-6, label="Learning rate")
    run_distill_button = mo.ui.run_button(label="Run distillation")
    mo.hstack([distill_mode, distill_batch_size, distill_steps, distill_lr, run_distill_button])
    return (
        distill_batch_size,
        distill_lr,
        distill_mode,
        distill_steps,
        run_distill_button,
    )


@app.cell(hide_code=True)
def _(
    DISTILLED_DIR,
    DataLoader,
    DistillationCollator,
    DistillationDataset,
    Tokenizer,
    base_checkpoint_path,
    base_model_available,
    base_tokenizer_path,
    ce_translation_loss,
    distill_batch_size,
    distill_lr,
    distill_mode,
    distill_steps,
    load_student_checkpoint,
    mo,
    pd,
    qualified_teacher_frame,
    run_distill_button,
    time,
    torch,
):
    distillation_history_frame = None
    distillation_report = {"status": "not started"}
    distilled_checkpoint_path = DISTILLED_DIR / "student_distilled_last.pt"

    if run_distill_button.value:
        if not base_model_available:
            distillation_report = {"status": "missing base checkpoint"}
        elif qualified_teacher_frame is None or pd is None:
            distillation_report = {"status": "missing qualified teacher cache"}
        elif torch is None or Tokenizer is None or DistillationDataset is None:
            distillation_report = {"status": "missing training dependencies"}
        else:
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _model, _tokenizer, _cfg, _checkpoint = load_student_checkpoint(
                base_checkpoint_path,
                base_tokenizer_path,
                device=_device,
                dropout=0.1,
            )
            _model.train()
            _train_frame = qualified_teacher_frame.copy()
            if "teacher_output" in _train_frame.columns:
                _train_frame = _train_frame[_train_frame["teacher_output"].astype(str).str.strip().ne("")]
            if len(_train_frame) == 0:
                distillation_report = {"status": "no qualified teacher rows"}
            else:
                _target_mix = "teacher" if distill_mode.value == "seqkd_ce" else "mixed"
                _dataset = DistillationDataset(
                    _train_frame,
                    _tokenizer,
                    max_seq_len=int(_cfg["max_seq_len"]),
                    target_mix=_target_mix,
                    style_tag="<natural>",
                )
                _loader = DataLoader(
                    _dataset,
                    batch_size=int(distill_batch_size.value),
                    shuffle=True,
                    collate_fn=DistillationCollator(_tokenizer.token_to_id("<pad>")),
                )
                _optimizer = torch.optim.AdamW(_model.parameters(), lr=float(distill_lr.value), weight_decay=0.05)
                _records = []
                _step = 0
                _started = time.time()
                while _step < int(distill_steps.value):
                    for _batch in _loader:
                        _batch = {k: v.to(_device) for k, v in _batch.items()}
                        _loss = ce_translation_loss(_model, _batch)
                        _optimizer.zero_grad(set_to_none=True)
                        _loss.backward()
                        torch.nn.utils.clip_grad_norm_(_model.parameters(), 1.0)
                        _optimizer.step()
                        _step += 1
                        if _step % 25 == 0 or _step == 1:
                            _records.append({"step": _step, "loss": float(_loss.detach().cpu()), "mode": distill_mode.value})
                            print(f"distill step {_step}: loss={float(_loss.detach().cpu()):.4f}")
                        if _step >= int(distill_steps.value):
                            break
                _payload = {
                    "model": _model.state_dict(),
                    "config": _cfg,
                    "base_checkpoint": str(base_checkpoint_path),
                    "mode": distill_mode.value,
                    "steps": _step,
                    "teacher_rows": len(_train_frame),
                    "teacher_id": str(_train_frame["teacher_id"].iloc[0]) if "teacher_id" in _train_frame.columns else "sarvamai/sarvam-translate",
                }
                DISTILLED_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(_payload, distilled_checkpoint_path)
                distillation_history_frame = pd.DataFrame(_records)
                distillation_report = {
                    "status": "complete",
                    "steps": _step,
                    "teacher_rows": len(_train_frame),
                    "seconds": round(time.time() - _started, 2),
                    "checkpoint": str(distilled_checkpoint_path),
                    "mode": distill_mode.value,
                }

    mo.md(f"**Distillation status:** `{distillation_report['status']}`")
    distillation_report
    return (distillation_history_frame,)


@app.cell(hide_code=True)
def _(distillation_history_frame, mo, plt):
    if distillation_history_frame is None:
        distill_plot = mo.md("Run distillation to plot training loss.")
    elif plt is None:
        distill_plot = mo.md("matplotlib is not installed.")
    else:
        _fig, _axis = plt.subplots(figsize=(8, 4))
        _axis.plot(distillation_history_frame["step"], distillation_history_frame["loss"], color="#3a6ea5")
        _axis.set_title("Distillation training loss")
        _axis.set_xlabel("step")
        _axis.set_ylabel("loss")
        _fig.tight_layout()
        distill_plot = _fig
    distill_plot
    return


@app.cell(hide_code=True)
def post_distillation_eval_intro(mo):
    mo.md("""
    ## Post-Distillation Evaluation

    A reader should not have to infer whether distillation helped. After training,
    run this section to generate outputs from `student_distilled_last.pt` on the
    same cache used for baseline and teacher comparison. The result table reports
    absolute scores and deltas against `base_best`.
    """)
    return


@app.cell(hide_code=True)
def post_distillation_eval_controls(mo):
    evaluate_distilled_button = mo.ui.run_button(label="Evaluate distilled checkpoint")
    distilled_eval_rows = mo.ui.number(value=100, start=1, stop=20_000, step=10, label="Evaluation rows")
    mo.hstack([evaluate_distilled_button, distilled_eval_rows])
    return distilled_eval_rows, evaluate_distilled_button


@app.cell(hide_code=True)
def post_distillation_eval(
    DISTILLED_DIR,
    base_model_available,
    base_tokenizer_path,
    compute_mt_metrics,
    distilled_eval_rows,
    evaluate_distilled_button,
    generate_student_translation,
    load_student_checkpoint,
    mo,
    pd,
    teacher_cache_frame,
    torch,
):
    distilled_eval_frame = None
    distilled_metric_frame = None
    distilled_eval_report = {"status": "not run"}
    _distilled_checkpoint_path = DISTILLED_DIR / "student_distilled_last.pt"

    if evaluate_distilled_button.value:
        if teacher_cache_frame is None or pd is None:
            distilled_eval_report = {"status": "missing teacher cache"}
        elif not _distilled_checkpoint_path.exists():
            distilled_eval_report = {"status": f"missing checkpoint: {_distilled_checkpoint_path}"}
        elif not base_model_available:
            distilled_eval_report = {"status": "missing base tokenizer/checkpoint context"}
        elif torch is None:
            distilled_eval_report = {"status": "missing torch"}
        else:
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _model, _tokenizer, _cfg, _checkpoint = load_student_checkpoint(
                _distilled_checkpoint_path,
                base_tokenizer_path,
                device=_device,
                dropout=0.0,
            )
            _rows = teacher_cache_frame.head(int(distilled_eval_rows.value)).copy().reset_index(drop=True)
            _distilled_outputs = []
            for _row in _rows.itertuples(index=False):
                _target_lang = "pa" if _row.direction == "en-pa" else "en"
                _distilled_outputs.append(
                    generate_student_translation(
                        _model,
                        _tokenizer,
                        _row.source,
                        target_lang=_target_lang,
                        domain=_row.domain,
                        style_tag="<natural>",
                        device=_device,
                        max_seq_len=int(_cfg["max_seq_len"]),
                        max_new_tokens=96,
                        temperature=0.0,
                    )
                )
            _rows["distilled_output"] = _distilled_outputs
            _systems = ["baseline_output"]
            if "teacher_output" in _rows.columns and _rows["teacher_output"].astype(str).str.strip().ne("").any():
                _systems.append("teacher_output")
            _systems.append("distilled_output")
            distilled_metric_frame = pd.DataFrame([compute_mt_metrics(_rows, _system) for _system in _systems])
            _baseline = distilled_metric_frame[distilled_metric_frame["system"].eq("baseline_output")]
            if len(_baseline) == 1:
                _base_bleu = _baseline["bleu"].iloc[0]
                _base_chrf = _baseline["chrf"].iloc[0]
                distilled_metric_frame["delta_bleu_vs_base"] = distilled_metric_frame["bleu"].apply(
                    lambda _score: None if _score is None or _base_bleu is None else _score - _base_bleu
                )
                distilled_metric_frame["delta_chrf_vs_base"] = distilled_metric_frame["chrf"].apply(
                    lambda _score: None if _score is None or _base_chrf is None else _score - _base_chrf
                )
            distilled_eval_frame = _rows
            _metrics_dir = DISTILLED_DIR.parent / "metrics"
            _metrics_dir.mkdir(parents=True, exist_ok=True)
            _rows.to_csv(_metrics_dir / "distilled_eval_outputs.csv", index=False)
            distilled_metric_frame.to_csv(_metrics_dir / "distilled_eval_metrics.csv", index=False)
            distilled_eval_report = {
                "status": "complete",
                "rows": len(_rows),
                "checkpoint": str(_distilled_checkpoint_path),
                "outputs": str(_metrics_dir / "distilled_eval_outputs.csv"),
                "metrics": str(_metrics_dir / "distilled_eval_metrics.csv"),
            }

    mo.md(f"**Distilled evaluation:** `{distilled_eval_report['status']}`")
    distilled_metric_frame if distilled_metric_frame is not None else distilled_eval_report
    return (distilled_metric_frame,)


@app.cell(hide_code=True)
def improvement_dashboard(
    cloud_flores10_score_frame,
    distilled_metric_frame,
    mo,
    pd,
):
    improvement_dashboard_view = None

    if pd is None:
        improvement_dashboard_view = mo.md("pandas is required for the improvement dashboard.")
    else:
        _parts = [mo.md("## Improvement Dashboard")]
        _parts.append(
            mo.md(
                """
                The fixed 10-row FLORES+ cloud sanity check establishes the starting gap:
                `base_best` is weak on this sample, while Sarvam is a qualified teacher.
                The distilled checkpoint row appears after running post-distillation
                evaluation, using the same cache as the baseline and teacher outputs.
                """
            )
        )
        if cloud_flores10_score_frame is not None:
            _cloud = cloud_flores10_score_frame.copy()
            _base = _cloud[_cloud["model"].eq("base_best")]
            if len(_base) == 1:
                _cloud["delta_bleu_vs_base"] = _cloud["bleu"] - float(_base["bleu"].iloc[0])
                _cloud["delta_chrf_vs_base"] = _cloud["chrf"] - float(_base["chrf"].iloc[0])
            _parts.extend([mo.md("### Recorded FLORES+ 10-row cloud result"), _cloud])
        if distilled_metric_frame is not None:
            _parts.extend([mo.md("### Current post-distillation evaluation"), distilled_metric_frame])
        else:
            _parts.append(mo.md("Run **Evaluate distilled checkpoint** after training to add the distilled-student row."))
        improvement_dashboard_view = mo.vstack(_parts)

    improvement_dashboard_view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Reverse-KL On-Policy Extension Cell

    The CE distillation loop above is the stable first experiment. The MiniLLM-style
    extension below is intentionally small-batch and slow: it samples student
    outputs, scores them with Sarvam, and uses the sequence-level reverse-KL policy
    surrogate.

    Use it on the manual challenge set first. If it improves instruction leakage
    and `ਲਈ` vs `ਤੋਂ` examples, scale it to a larger cache.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    run_rkl_demo_button = mo.ui.run_button(label="Run small RKL on-policy demo")
    rkl_demo_rows = mo.ui.number(value=4, start=1, stop=32, step=1, label="Demo rows")
    mo.hstack([run_rkl_demo_button, rkl_demo_rows])
    return rkl_demo_rows, run_rkl_demo_button


@app.cell(hide_code=True)
def _(
    base_checkpoint_path,
    base_model_available,
    base_tokenizer_path,
    eval_seed_frame,
    generate_student_translation,
    load_student_checkpoint,
    pd,
    rkl_demo_rows,
    rkl_policy_loss,
    run_rkl_demo_button,
    score_with_sarvam,
    student_prompt,
    student_sequence_logprob,
    teacher_bundle,
    torch,
):
    rkl_demo_report = {"status": "not run"}
    if run_rkl_demo_button.value:
        if not base_model_available or teacher_bundle is None or torch is None or pd is None:
            rkl_demo_report = {"status": "needs base checkpoint, Sarvam teacher, torch, and pandas"}
        else:
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _model, _tokenizer, _cfg, _checkpoint = load_student_checkpoint(base_checkpoint_path, base_tokenizer_path, device=_device, dropout=0.0)
            _model.train()
            _demo = eval_seed_frame.head(int(rkl_demo_rows.value)).copy()
            _outputs = []
            _student_logps = []
            for _row in _demo.itertuples(index=False):
                _target_lang = "pa" if _row.direction == "en-pa" else "en"
                _candidate = generate_student_translation(
                    _model,
                    _tokenizer,
                    _row.source,
                    target_lang=_target_lang,
                    domain=_row.domain,
                    style_tag="<natural>",
                    device=_device,
                    max_seq_len=int(_cfg["max_seq_len"]),
                    max_new_tokens=96,
                    temperature=0.8,
                    top_k=40,
                )
                _outputs.append(_candidate)
                _prompt = student_prompt(_row.source, _target_lang, _row.domain, "<natural>")
                _student_logps.append(student_sequence_logprob(_model, _tokenizer, _prompt, _candidate, _device, int(_cfg["max_seq_len"])))
            _teacher_scores = score_with_sarvam(
                teacher_bundle,
                _demo["source"].tolist(),
                _outputs,
                _demo["direction"].tolist(),
            )
            _loss = rkl_policy_loss(_student_logps, _teacher_scores)
            rkl_demo_report = {
                "status": "computed",
                "rkl_policy_loss": float(_loss.detach().cpu()),
                "mean_teacher_score": sum(_teacher_scores) / max(1, len(_teacher_scores)),
            }
            _demo["student_sample"] = _outputs
            _demo["teacher_len_norm_logp"] = _teacher_scores
            rkl_demo_report["samples"] = _demo
    rkl_demo_report["samples"] if isinstance(rkl_demo_report.get("samples"), pd.DataFrame) else rkl_demo_report
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Suggested Experiment Grid for the Paper Challenge

    1. **Baseline:** `Ajaple/gur-slm-decoder-base` on FLORES+ plus the manual challenge set.
    2. **Teacher:** Sarvam-Translate outputs on the same examples.
    3. **Qualification:** keep only script-clean, length-sane, leakage-free Sarvam rows.
    4. **SeqKD / Mixed KD:** fine-tune on teacher outputs only or 50% teacher + 50% gold reference.
    5. **MiniLLM-style extension:** add small-batch sequence reverse-KL on student samples scored by Sarvam.

    Report:

    - BLEU and chrF overall.
    - BLEU and chrF by direction and domain.
    - Script error and English leakage rate.
    - Challenge-set table with baseline, teacher, and distilled outputs.
    - A short error analysis: legal duration, numbers, negation, and wrapper prompts.
    """)
    return


if __name__ == "__main__":
    app.run()
