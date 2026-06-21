import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    import random
    import subprocess
    import sys
    from contextlib import nullcontext
    from dataclasses import asdict, dataclass
    from pathlib import Path
    from typing import Any

    import marimo as mo
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from tokenizers import Tokenizer
    from torch import nn
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader, Dataset

    return (
        Any,
        DataLoader,
        Dataset,
        F,
        Path,
        Tokenizer,
        asdict,
        dataclass,
        math,
        mo,
        nn,
        nullcontext,
        pad_sequence,
        pd,
        random,
        subprocess,
        sys,
        torch,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # Decoder-Only Gurmukhi Translation SLM

        This notebook trains a GPT/Llama-style causal decoder model for
        English<->Punjabi Gurmukhi translation.

        This is a separate research branch from `Gur_slm.py`.

        - `Gur_slm.py`: encoder-decoder translation model, better fit for MT.
        - `decoder.py`: decoder-only translation model, easier to package for
          GGUF/llama.cpp-style deployment later.
        """
    )
    return


@app.cell
def _(Path):
    PROJECT_ROOT = Path.cwd()
    DATA_PATH = PROJECT_ROOT / "datasets" / "cleaned.tsv"
    TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "hf_bpe24k_tokenizer.json"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "decoder_only"
    return CHECKPOINT_DIR, DATA_PATH, PROJECT_ROOT, TOKENIZER_PATH


@app.cell
def _(DATA_PATH, TOKENIZER_PATH, mo):
    path_status = {
        "datasets/cleaned.tsv": DATA_PATH.exists(),
        "tokenizer/hf_bpe24k_tokenizer.json": TOKENIZER_PATH.exists(),
    }
    mo.md(
        "\n".join(
            [
                "## Required Files",
                "",
                *[
                    f"- `{name}`: {'found' if exists else 'missing'}"
                    for name, exists in path_status.items()
                ],
            ]
        )
    )
    return


@app.cell
def _(dataclass):
    @dataclass
    class DecoderConfig:
        vocab_size: int
        pad_id: int
        max_seq_len: int = 256
        d_model: int = 512
        nhead: int = 8
        num_layers: int = 8
        dim_feedforward: int = 2048
        dropout: float = 0.1
        batch_size: int = 32
        epochs: int = 1
        lr: float = 3e-4
        min_lr_ratio: float = 0.05
        warmup_steps: int = 2000
        weight_decay: float = 0.01
        grad_accum_steps: int = 1
        clip_grad_norm: float = 1.0
        label_smoothing: float = 0.05
        seed: int = 42
        amp: str = "bf16"

    PROFILES = {
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
        "large": {
            "d_model": 768,
            "nhead": 12,
            "num_layers": 12,
            "dim_feedforward": 3072,
        },
    }

    return DecoderConfig, PROFILES


@app.cell
def _(F, math, nn, torch):
    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            dtype = x.dtype
            x = x.float()
            x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return (self.weight * x).to(dtype)


    class SinusoidalPositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 4096) -> None:
            super().__init__()
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.pe[:, : x.size(1)]


    class SwiGLUFeedForward(nn.Module):
        def __init__(self, d_model: int, dim_feedforward: int, dropout: float) -> None:
            super().__init__()
            self.w12 = nn.Linear(d_model, dim_feedforward * 2, bias=False)
            self.w3 = nn.Linear(dim_feedforward, d_model, bias=False)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            value, gate = self.w12(x).chunk(2, dim=-1)
            return self.w3(self.dropout(value * F.silu(gate)))


    class DecoderBlock(nn.Module):
        def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float) -> None:
            super().__init__()
            self.self_norm = RMSNorm(d_model)
            self.ffn_norm = RMSNorm(d_model)
            self.self_attn = nn.MultiheadAttention(
                d_model,
                nhead,
                dropout=dropout,
                batch_first=True,
            )
            self.ffn = SwiGLUFeedForward(d_model, dim_feedforward, dropout)
            self.dropout = nn.Dropout(dropout)

        def forward(
            self,
            x: torch.Tensor,
            causal_mask: torch.Tensor,
            padding_mask: torch.Tensor,
        ) -> torch.Tensor:
            attn_input = self.self_norm(x)
            attn_out, _ = self.self_attn(
                attn_input,
                attn_input,
                attn_input,
                attn_mask=causal_mask,
                key_padding_mask=padding_mask,
                need_weights=False,
            )
            x = x + self.dropout(attn_out)
            x = x + self.dropout(self.ffn(self.ffn_norm(x)))
            return x


    class DecoderOnlyTransformer(nn.Module):
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
        ) -> None:
            super().__init__()
            self.pad_id = pad_id
            self.d_model = d_model
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
            self.positional = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len + 16)
            self.layers = nn.ModuleList(
                [
                    DecoderBlock(d_model, nhead, dim_feedforward, dropout)
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
                    nn.init.xavier_uniform_(module.weight)

        def forward(self, input_ids: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
            seq_len = input_ids.size(1)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
                diagonal=1,
            )
            hidden = self.embedding(input_ids) * math.sqrt(self.d_model)
            hidden = self.dropout(self.positional(hidden))
            for layer in self.layers:
                hidden = layer(hidden, causal_mask, padding_mask)
            return self.output(self.norm(hidden))

    return DecoderOnlyTransformer


@app.cell
def _(Dataset, pad_sequence, pd, torch):
    class CausalTranslationDataset(Dataset):
        def __init__(self, frame, tokenizer, max_seq_len: int) -> None:
            self.frame = frame.reset_index(drop=True)
            self.tokenizer = tokenizer
            self.max_seq_len = max_seq_len
            self.eos_id = tokenizer.token_to_id("</s>")

        def __len__(self) -> int:
            return len(self.frame) * 2

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            row = self.frame.iloc[index // 2]
            domain_tag = "<legal>" if str(row["domain"]) == "legal" else "<general>"

            if index % 2 == 0:
                prompt = f"<2pa> {domain_tag} {row['en']}\n"
                target = str(row["pa"])
            else:
                prompt = f"<2en> {domain_tag} {row['pa']}\n"
                target = str(row["en"])

            prompt_ids = self.tokenizer.encode(prompt).ids
            target_ids = self.tokenizer.encode(target).ids + [self.eos_id]
            full_ids = (prompt_ids + target_ids)[: self.max_seq_len]

            input_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
            labels = torch.tensor(full_ids[1:], dtype=torch.long)

            prompt_label_cutoff = max(0, min(len(prompt_ids) - 1, labels.numel()))
            labels[:prompt_label_cutoff] = -100

            return {"input_ids": input_ids, "labels": labels}


    def causal_collate(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
        input_ids = pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=pad_id,
        )
        labels = pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=-100,
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "padding_mask": input_ids.eq(pad_id),
        }


    def load_frame(path, rows: int | None):
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


    def split_train_val(frame, val_rows: int, val_fraction: float, seed: int):
        if len(frame) < 3:
            return frame, frame
        shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        requested_val = min(val_rows, max(1, int(len(shuffled) * val_fraction)))
        val_size = min(requested_val, len(shuffled) - 1)
        return (
            shuffled.iloc[val_size:].reset_index(drop=True),
            shuffled.iloc[:val_size].reset_index(drop=True),
        )

    return CausalTranslationDataset, causal_collate, load_frame, split_train_val


@app.cell
def _(mo):
    mo.md("## Training Controls")
    return


@app.cell
def _(mo):
    profile = mo.ui.dropdown(
        options=["tiny", "base", "large"],
        value="base",
        label="Decoder profile",
    )
    batch_size = mo.ui.number(value=32, start=1, stop=256, step=1, label="Batch size")
    epochs = mo.ui.number(value=1, start=1, stop=100, step=1, label="Epochs")
    max_rows = mo.ui.number(value=0, start=0, stop=2_000_000, step=1024, label="Max rows; 0 = full corpus")
    max_steps = mo.ui.number(value=0, start=0, stop=1_000_000, step=100, label="Max steps/epoch; 0 = full epoch")
    max_seq_len = mo.ui.number(value=256, start=64, stop=1024, step=16, label="Max sequence length")
    grad_accum_steps = mo.ui.number(value=1, start=1, stop=64, step=1, label="Grad accumulation")
    num_workers = mo.ui.number(value=0, start=0, stop=32, step=1, label="DataLoader workers")
    amp = mo.ui.dropdown(options=["bf16", "fp16", "none"], value="bf16", label="AMP")

    mo.vstack(
        [
            mo.hstack([profile, amp]),
            mo.hstack([batch_size, epochs, grad_accum_steps, num_workers]),
            mo.hstack([max_rows, max_steps, max_seq_len]),
        ]
    )
    return amp, batch_size, epochs, grad_accum_steps, max_rows, max_seq_len, max_steps, num_workers, profile


@app.cell
def _(mo):
    mo.md("## Training Helpers")
    return


@app.cell
def _(Any, nullcontext, torch):
    def amp_dtype(device: torch.device, amp: str):
        if device.type != "cuda" or amp == "none":
            return None
        if amp == "bf16":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            print("bf16 is not supported on this CUDA device; falling back to fp16")
            return torch.float16
        if amp == "fp16":
            return torch.float16
        raise ValueError(f"Unsupported amp mode: {amp}")


    def autocast_context(device: torch.device, dtype):
        if device.type == "cuda" and dtype is not None:
            return torch.autocast(device_type="cuda", dtype=dtype)
        return nullcontext()


    def make_grad_scaler(use_fp16: bool):
        try:
            return torch.amp.GradScaler("cuda", enabled=use_fp16)
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler(enabled=use_fp16)


    def make_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, min_lr_ratio: float):
        warmup_steps = max(1, warmup_steps)

        def lr_lambda(step: int) -> float:
            step = max(1, step)
            warmup = min(1.0, step / warmup_steps)
            decay = (warmup_steps / step) ** 0.5
            return max(min_lr_ratio, min(warmup, decay))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    def parameter_count(model: torch.nn.Module) -> int:
        return sum(param.numel() for param in model.parameters())

    return amp_dtype, autocast_context, make_grad_scaler, make_scheduler, parameter_count


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
        max_steps: int | None,
        log_every: int = 100,
    ):
        model.train()
        dtype = amp_dtype(device, amp)
        total_loss = 0.0
        micro_steps = 0
        optimizer_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_index, batch in enumerate(loader, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}

            with autocast_context(device, dtype):
                logits = model(batch["input_ids"], batch["padding_mask"])
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch["labels"].reshape(-1),
                    ignore_index=-100,
                    label_smoothing=0.05,
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

                if optimizer_steps % log_every == 0:
                    print(
                        f"step {optimizer_steps:,}: "
                        f"train_loss={total_loss / max(micro_steps, 1):.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.6g}"
                    )

                if max_steps is not None and optimizer_steps >= max_steps:
                    break

        return total_loss / max(micro_steps, 1), optimizer_steps


    @torch.no_grad()
    def evaluate(model, loader, device, amp: str):
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

    return evaluate, train_one_epoch


@app.cell
def _(torch):
    @torch.no_grad()
    def generate_translation(
        model,
        tokenizer,
        text: str,
        target_lang: str,
        domain: str,
        device,
        max_seq_len: int,
        max_new_tokens: int = 96,
    ) -> str:
        model.eval()
        eos_id = tokenizer.token_to_id("</s>")
        pad_id = tokenizer.token_to_id("<pad>")
        target_tag = "<2pa>" if target_lang == "pa" else "<2en>"
        domain_tag = "<legal>" if domain == "legal" else "<general>"
        prompt = f"{target_tag} {domain_tag} {text}\n"
        ids = tokenizer.encode(prompt).ids
        prompt_len = len(ids)

        for _ in range(max_new_tokens):
            input_ids = torch.tensor([ids[-max_seq_len:]], dtype=torch.long, device=device)
            padding_mask = input_ids.eq(pad_id)
            logits = model(input_ids, padding_mask)
            next_id = int(logits[:, -1].argmax(dim=-1).item())
            ids.append(next_id)
            if next_id == eos_id:
                break

        generated_ids = ids[prompt_len:]
        if eos_id in generated_ids:
            generated_ids = generated_ids[: generated_ids.index(eos_id)]
        return tokenizer.decode(generated_ids)

    return (generate_translation,)


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Train decoder-only model")
    train_button
    return (train_button,)


@app.cell
def _(
    CHECKPOINT_DIR,
    CausalTranslationDataset,
    DATA_PATH,
    DataLoader,
    DecoderConfig,
    DecoderOnlyTransformer,
    PROFILES,
    TOKENIZER_PATH,
    Tokenizer,
    amp,
    amp_dtype,
    asdict,
    batch_size,
    causal_collate,
    epochs,
    evaluate,
    generate_translation,
    grad_accum_steps,
    load_frame,
    make_grad_scaler,
    make_scheduler,
    max_rows,
    max_seq_len,
    max_steps,
    num_workers,
    parameter_count,
    profile,
    random,
    split_train_val,
    torch,
    train_button,
    train_one_epoch,
):
    if train_button.value:
        tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        pad_id = tokenizer.token_to_id("<pad>")
        vocab_size = tokenizer.get_vocab_size()
        profile_config = PROFILES[profile.value]
        config = DecoderConfig(
            vocab_size=vocab_size,
            pad_id=pad_id,
            max_seq_len=int(max_seq_len.value),
            batch_size=int(batch_size.value),
            epochs=int(epochs.value),
            grad_accum_steps=int(grad_accum_steps.value),
            amp=amp.value,
            **profile_config,
        )

        random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        row_limit = int(max_rows.value) if int(max_rows.value) > 0 else None
        frame = load_frame(DATA_PATH, row_limit)
        train_frame, val_frame = split_train_val(frame, val_rows=10_000, val_fraction=0.01, seed=config.seed)

        train_dataset = CausalTranslationDataset(train_frame, tokenizer, config.max_seq_len)
        val_dataset = CausalTranslationDataset(val_frame, tokenizer, config.max_seq_len)
        # Keep notebook DataLoaders single-process. Multiprocessing workers in
        # marimo/cloud notebooks need picklable top-level callables, and cell
        # closures/lambdas are not reliable under spawn-based multiprocessing.
        loader_workers = 0

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=loader_workers,
            collate_fn=lambda batch: causal_collate(batch, pad_id),
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=loader_workers,
            collate_fn=lambda batch: causal_collate(batch, pad_id),
            pin_memory=torch.cuda.is_available(),
        )

        model = DecoderOnlyTransformer(
            vocab_size=config.vocab_size,
            pad_id=config.pad_id,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.98),
            weight_decay=config.weight_decay,
        )
        scheduler = make_scheduler(optimizer, config.warmup_steps, config.min_lr_ratio)
        dtype = amp_dtype(device, config.amp)
        scaler = make_grad_scaler(use_fp16=(device.type == "cuda" and dtype == torch.float16))

        print(f"device: {device}")
        print(f"profile: {profile.value}")
        print(f"vocab_size: {vocab_size:,}")
        print(f"parameters: {parameter_count(model):,}")
        print(f"train examples including both directions: {len(train_dataset):,}")
        print(f"val examples including both directions: {len(val_dataset):,}")

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        best_val_loss = float("inf")
        max_steps_value = int(max_steps.value) if int(max_steps.value) > 0 else None

        for epoch in range(1, config.epochs + 1):
            train_loss, steps = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                device,
                config.amp,
                config.grad_accum_steps,
                config.clip_grad_norm,
                max_steps=max_steps_value,
            )
            val_loss, val_acc = evaluate(model, val_loader, device, config.amp)

            payload = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": asdict(config),
                "val_loss": val_loss,
                "vocab_size": vocab_size,
            }
            epoch_path = CHECKPOINT_DIR / f"decoder_{profile.value}_epoch_{epoch}.pt"
            last_path = CHECKPOINT_DIR / f"decoder_{profile.value}_last.pt"
            torch.save(payload, epoch_path)
            torch.save(payload, last_path)

            best_note = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = CHECKPOINT_DIR / f"decoder_{profile.value}_best.pt"
                torch.save(payload, best_path)
                best_note = f" best={best_path}"

            print(
                f"epoch {epoch}: train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_token_acc={val_acc:.4f} "
                f"steps={steps} checkpoint={epoch_path}{best_note}"
            )
            sample = generate_translation(
                model,
                tokenizer,
                "The agreement shall remain in force for five years.",
                target_lang="pa",
                domain="legal",
                device=device,
                max_seq_len=config.max_seq_len,
            )
            print(f"sample en->pa: {sample}")

        train_status = 0
    else:
        train_status = None

    train_status
    return


if __name__ == "__main__":
    app.run()
