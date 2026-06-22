"""Train a compact English <-> Punjabi translation model.

The script is intended for two workflows:

Local smoke tests on a smaller GPU:

    python Gur_slm.py --dry-run --device cuda --profile small --batch-size 4
    python Gur_slm.py --train --device cuda --profile small --max-rows 2048 --max-steps 100

Full training on a larger online GPU:

    python Gur_slm.py --train --device cuda --profile base --batch-size 64 --amp bf16 --epochs 5
"""

from __future__ import annotations

import argparse
import math
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


MODEL_PROFILES: dict[str, dict[str, int]] = {
    "tiny": {
        "d_model": 384,
        "nhead": 6,
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
        "dim_feedforward": 1536,
    },
    "small": {
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
        "dim_feedforward": 2048,
    },
    "base": {
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
    },
    "large": {
        "d_model": 768,
        "nhead": 12,
        "num_encoder_layers": 8,
        "num_decoder_layers": 8,
        "dim_feedforward": 3072,
    },
}


@dataclass
class TrainConfig:
    data_path: Path = Path("datasets/cleaned.tsv")
    tokenizer_path: Path = Path("tokenizer/hf_bpe24k_tokenizer.json")
    output_dir: Path = Path("checkpoints")
    profile: str = "base"
    dry_run_rows: int = 2048
    max_rows: int | None = None
    val_rows: int = 10_000
    val_fraction: float = 0.01
    max_src_len: int = 192
    max_tgt_len: int = 192
    batch_size: int = 16
    epochs: int = 1
    lr: float = 3e-4
    min_lr_ratio: float = 0.05
    warmup_steps: int = 4000
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    grad_accum_steps: int = 1
    clip_grad_norm: float = 1.0
    num_workers: int = 0
    dropout: float = 0.1
    seed: int = 42
    amp: str = "bf16"
    compile_model: bool = False

    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_profile(config: TrainConfig) -> TrainConfig:
    profile = MODEL_PROFILES[config.profile]
    config.d_model = profile["d_model"]
    config.nhead = profile["nhead"]
    config.num_encoder_layers = profile["num_encoder_layers"]
    config.num_decoder_layers = profile["num_decoder_layers"]
    config.dim_feedforward = profile["dim_feedforward"]
    return config


def config_for_checkpoint(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    data["data_path"] = str(config.data_path)
    data["tokenizer_path"] = str(config.tokenizer_path)
    data["output_dir"] = str(config.output_dir)
    return data


class TranslationDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: Tokenizer,
        max_src_len: int,
        max_tgt_len: int,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.bos_id = tokenizer.token_to_id("<s>")
        self.eos_id = tokenizer.token_to_id("</s>")

    def __len__(self) -> int:
        return len(self.frame) * 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index // 2]
        domain_tag = "<legal>" if str(row["domain"]) == "legal" else "<general>"

        if index % 2 == 0:
            src_text = f"<2pa> {domain_tag} {row['en']}"
            tgt_text = str(row["pa"])
        else:
            src_text = f"<2en> {domain_tag} {row['pa']}"
            tgt_text = str(row["en"])

        src_ids = self.tokenizer.encode(src_text).ids[: self.max_src_len - 1]
        tgt_ids = self.tokenizer.encode(tgt_text).ids[: self.max_tgt_len - 2]

        src = torch.tensor(src_ids + [self.eos_id], dtype=torch.long)
        tgt_in = torch.tensor([self.bos_id] + tgt_ids, dtype=torch.long)
        tgt_out = torch.tensor(tgt_ids + [self.eos_id], dtype=torch.long)
        return {"src": src, "tgt_in": tgt_in, "tgt_out": tgt_out}


def collate_batch(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    src = pad_sequence([item["src"] for item in batch], batch_first=True, padding_value=pad_id)
    tgt_in = pad_sequence([item["tgt_in"] for item in batch], batch_first=True, padding_value=pad_id)
    tgt_out = pad_sequence([item["tgt_out"] for item in batch], batch_first=True, padding_value=pad_id)
    return {
        "src": src,
        "tgt_in": tgt_in,
        "tgt_out": tgt_out,
        "src_padding_mask": src.eq(pad_id),
        "tgt_padding_mask": tgt_in.eq(pad_id),
    }


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


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.w12 = nn.Linear(d_model, dim_feedforward * 2, bias=False)
        self.w3 = nn.Linear(dim_feedforward, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.dropout(value * F.silu(gate)))


class EncoderLayer(nn.Module):
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

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        attn_input = self.self_norm(x)
        attn_out, _ = self.self_attn(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = RMSNorm(d_model)
        self.cross_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
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
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_padding_mask: torch.Tensor,
        memory_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        self_input = self.self_norm(x)
        self_out, _ = self.self_attn(
            self_input,
            self_input,
            self_input,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(self_out)

        cross_input = self.cross_norm(x)
        cross_out, _ = self.cross_attn(
            cross_input,
            memory,
            memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(cross_out)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class ModernSeq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.positional = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len + 16)
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_decoder_layers)
            ]
        )
        self.encoder_norm = RMSNorm(d_model)
        self.decoder_norm = RMSNorm(d_model)
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

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.positional(self.embedding(tokens) * math.sqrt(self.d_model)))

    def encode(self, src: torch.Tensor, src_padding_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(src)
        for layer in self.encoder_layers:
            hidden = layer(hidden, src_padding_mask)
        return self.encoder_norm(hidden)

    def decode_hidden(
        self,
        tgt_in: torch.Tensor,
        memory: torch.Tensor,
        tgt_padding_mask: torch.Tensor,
        memory_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt_mask = torch.triu(
            torch.ones(tgt_in.size(1), tgt_in.size(1), device=tgt_in.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.embed(tgt_in)
        for layer in self.decoder_layers:
            hidden = layer(hidden, memory, tgt_mask, tgt_padding_mask, memory_padding_mask)
        return self.decoder_norm(hidden)

    def decode_logits(
        self,
        tgt_in: torch.Tensor,
        memory: torch.Tensor,
        tgt_padding_mask: torch.Tensor,
        memory_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.output(self.decode_hidden(tgt_in, memory, tgt_padding_mask, memory_padding_mask))

    def forward(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
        src_padding_mask: torch.Tensor,
        tgt_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_padding_mask)
        return self.decode_logits(tgt_in, memory, tgt_padding_mask, src_padding_mask)


def load_frame(path: Path, rows: int | None) -> pd.DataFrame:
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


def split_train_val(frame: pd.DataFrame, config: TrainConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(frame) < 3:
        return frame, frame
    shuffled = frame.sample(frac=1.0, random_state=config.seed).reset_index(drop=True)
    requested_val = min(config.val_rows, max(1, int(len(shuffled) * config.val_fraction)))
    val_size = min(requested_val, len(shuffled) - 1)
    val_frame = shuffled.iloc[:val_size].reset_index(drop=True)
    train_frame = shuffled.iloc[val_size:].reset_index(drop=True)
    return train_frame, val_frame


def build_loaders(
    config: TrainConfig,
    tokenizer: Tokenizer,
    dry_run: bool,
) -> tuple[DataLoader, DataLoader]:
    rows = config.dry_run_rows if dry_run else config.max_rows
    frame = load_frame(config.data_path, rows)
    train_frame, val_frame = split_train_val(frame, config)
    pad_id = tokenizer.token_to_id("<pad>")

    train_dataset = TranslationDataset(train_frame, tokenizer, config.max_src_len, config.max_tgt_len)
    val_dataset = TranslationDataset(val_frame, tokenizer, config.max_src_len, config.max_tgt_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=lambda batch: collate_batch(batch, pad_id),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=lambda batch: collate_batch(batch, pad_id),
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, warmup_steps)

    def lr_lambda(step: int) -> float:
        step = max(1, step)
        warmup = min(1.0, step / warmup_steps)
        decay = math.sqrt(warmup_steps / step)
        return max(min_lr_ratio, min(warmup, decay))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def amp_dtype(device: torch.device, amp: str) -> torch.dtype | None:
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


def autocast_context(device: torch.device, dtype: torch.dtype | None):
    if device.type == "cuda" and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def make_grad_scaler(use_fp16: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=use_fp16)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_fp16)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: Any,
    criterion: nn.Module,
    device: torch.device,
    config: TrainConfig,
    max_steps: int | None,
    log_every: int,
) -> tuple[float, int]:
    model.train()
    dtype = amp_dtype(device, config.amp)
    total_loss = 0.0
    optimizer_steps = 0
    micro_steps = 0
    optimizer.zero_grad(set_to_none=True)

    for batch_index, batch in enumerate(loader, start=1):
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}

        with autocast_context(device, dtype):
            logits = model(
                batch["src"],
                batch["tgt_in"],
                batch["src_padding_mask"],
                batch["tgt_padding_mask"],
            )
            loss = criterion(logits.reshape(-1, logits.size(-1)), batch["tgt_out"].reshape(-1))

        total_loss += float(loss.detach().cpu())
        scaled_loss = loss / config.grad_accum_steps
        scaler.scale(scaled_loss).backward()
        micro_steps += 1

        should_step = batch_index % config.grad_accum_steps == 0 or batch_index == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

            if optimizer_steps % log_every == 0:
                lr = scheduler.get_last_lr()[0]
                avg_loss = total_loss / max(micro_steps, 1)
                print(f"step {optimizer_steps:,}: train_loss={avg_loss:.4f} lr={lr:.6g}")

            if max_steps is not None and optimizer_steps >= max_steps:
                break

    return total_loss / max(micro_steps, 1), optimizer_steps


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: TrainConfig,
    max_batches: int | None = None,
) -> tuple[float, float]:
    model.eval()
    dtype = amp_dtype(device, config.amp)
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    batches = 0

    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with autocast_context(device, dtype):
            logits = model(
                batch["src"],
                batch["tgt_in"],
                batch["src_padding_mask"],
                batch["tgt_padding_mask"],
            )
            loss = criterion(logits.reshape(-1, logits.size(-1)), batch["tgt_out"].reshape(-1))

        target = batch["tgt_out"]
        mask = target.ne(model.pad_id)
        prediction = logits.argmax(dim=-1)
        correct_tokens += int((prediction.eq(target) & mask).sum().detach().cpu())
        total_tokens += int(mask.sum().detach().cpu())
        total_loss += float(loss.detach().cpu())
        batches += 1

        if max_batches is not None and batches >= max_batches:
            break

    return total_loss / max(batches, 1), correct_tokens / max(total_tokens, 1)


@torch.no_grad()
def greedy_translate(
    model: ModernSeq2SeqTransformer,
    tokenizer: Tokenizer,
    text: str,
    target_lang: str,
    domain: str,
    device: torch.device,
    max_new_tokens: int = 96,
) -> str:
    model.eval()
    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")
    pad_id = tokenizer.token_to_id("<pad>")
    target_tag = "<2pa>" if target_lang == "pa" else "<2en>"
    domain_tag = "<legal>" if domain == "legal" else "<general>"
    src_ids = tokenizer.encode(f"{target_tag} {domain_tag} {text}").ids + [eos_id]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_padding_mask = src.eq(pad_id)
    memory = model.encode(src, src_padding_mask)
    generated = torch.tensor([[bos_id]], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        tgt_padding_mask = generated.eq(pad_id)
        logits = model.decode_logits(generated, memory, tgt_padding_mask, src_padding_mask)
        next_id = int(logits[:, -1].argmax(dim=-1).item())
        generated = torch.cat(
            [generated, torch.tensor([[next_id]], dtype=torch.long, device=device)],
            dim=1,
        )
        if next_id == eos_id:
            break

    ids = generated[0, 1:].tolist()
    if eos_id in ids:
        ids = ids[: ids.index(eos_id)]
    return tokenizer.decode(ids)


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: TrainConfig,
    epoch: int,
    val_loss: float,
    vocab_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config_for_checkpoint(config),
            "val_loss": val_loss,
            "vocab_size": vocab_size,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run a small forward/backward validation.")
    parser.add_argument("--train", action="store_true", help="Run training.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--profile", choices=sorted(MODEL_PROFILES), default="base")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dry-run-rows", type=int, default=2048)
    parser.add_argument("--max-rows", type=int, default=None, help="Limit loaded corpus rows for train smoke tests.")
    parser.add_argument("--max-steps", type=int, default=None, help="Max optimizer steps per epoch.")
    parser.add_argument("--max-src-len", type=int, default=192)
    parser.add_argument("--max-tgt-len", type=int, default=192)
    parser.add_argument("--val-rows", type=int, default=10_000)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=4000)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", choices=["none", "fp16", "bf16"], default="bf16")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile when available.")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dry_run and not args.train:
        args.dry_run = True

    config = apply_profile(
        TrainConfig(
            output_dir=args.output_dir,
            profile=args.profile,
            batch_size=args.batch_size,
            epochs=args.epochs,
            dry_run_rows=args.dry_run_rows,
            max_rows=args.max_rows,
            val_rows=args.val_rows,
            val_fraction=args.val_fraction,
            max_src_len=args.max_src_len,
            max_tgt_len=args.max_tgt_len,
            lr=args.lr,
            warmup_steps=args.warmup_steps,
            grad_accum_steps=args.grad_accum_steps,
            num_workers=args.num_workers,
            amp=args.amp,
            compile_model=args.compile,
        )
    )
    set_seed(config.seed)

    tokenizer = Tokenizer.from_file(str(config.tokenizer_path))
    pad_id = tokenizer.token_to_id("<pad>")
    vocab_size = tokenizer.get_vocab_size()
    device = torch.device(args.device)

    train_loader, val_loader = build_loaders(config, tokenizer, dry_run=args.dry_run)
    model = ModernSeq2SeqTransformer(
        vocab_size=vocab_size,
        pad_id=pad_id,
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_seq_len=max(config.max_src_len, config.max_tgt_len),
    ).to(device)

    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.98),
        weight_decay=config.weight_decay,
    )
    scheduler = make_scheduler(optimizer, config.warmup_steps, config.min_lr_ratio)
    dtype = amp_dtype(device, config.amp)
    scaler = make_grad_scaler(use_fp16=(device.type == "cuda" and dtype == torch.float16))

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint.get("val_loss", best_val_loss))
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    print(f"device: {device}")
    print(f"amp: {config.amp}")
    print(f"profile: {config.profile}")
    print(f"vocab_size: {vocab_size:,}")
    print(f"parameters: {parameter_count(model):,}")
    print(f"train examples including both directions: {len(train_loader.dataset):,}")
    print(f"val examples including both directions: {len(val_loader.dataset):,}")
    print(f"train batches per epoch: {len(train_loader):,}")
    print(f"grad_accum_steps: {config.grad_accum_steps}")

    if args.dry_run:
        batch = next(iter(train_loader))
        print("dry batch shapes:", {key: tuple(value.shape) for key, value in batch.items()})
        loss, steps = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            criterion,
            device,
            config,
            max_steps=2,
            log_every=1,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, config, max_batches=2)
        print(f"dry run train_loss={loss:.4f} optimizer_steps={steps}")
        print(f"dry run val_loss={val_loss:.4f} val_token_acc={val_acc:.4f}")
        print("dry run passed")
        return

    config.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, config.epochs + 1):
        train_loss, steps = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            criterion,
            device,
            config,
            max_steps=args.max_steps,
            log_every=args.log_every,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, config)

        epoch_path = config.output_dir / f"{config.profile}_epoch_{epoch}.pt"
        last_path = config.output_dir / f"{config.profile}_last.pt"
        save_checkpoint(epoch_path, model, optimizer, scheduler, config, epoch, val_loss, vocab_size)
        save_checkpoint(last_path, model, optimizer, scheduler, config, epoch, val_loss, vocab_size)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = config.output_dir / f"{config.profile}_best.pt"
            save_checkpoint(best_path, model, optimizer, scheduler, config, epoch, val_loss, vocab_size)
            best_note = f" best={best_path}"
        else:
            best_note = ""

        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_token_acc={val_acc:.4f} steps={steps} checkpoint={epoch_path}{best_note}"
        )

        sample_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        sample = greedy_translate(
            sample_model,
            tokenizer,
            "The agreement shall remain in force for five years.",
            target_lang="pa",
            domain="legal",
            device=device,
        )
        print(f"sample en->pa: {sample}")


if __name__ == "__main__":
    main()
