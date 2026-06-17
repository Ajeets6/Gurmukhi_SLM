#!/usr/bin/env python
"""Combine raw English-Punjabi parallel corpora into one exploration TSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import unicodedata
from collections import Counter
from pathlib import Path


FIELDNAMES = [
    "id",
    "source",
    "domain",
    "src_lang",
    "tgt_lang",
    "en",
    "pa",
    "en_chars",
    "pa_chars",
    "en_words",
    "pa_words",
    "pair_hash",
]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = html.unescape(text)
    return " ".join(text.strip().split())


def pair_hash(en: str, pa: str) -> str:
    normalized = f"{en}\t{pa}".casefold()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def row_for(source: str, domain: str, idx: int, en: str, pa: str) -> dict[str, str | int]:
    en = normalize_text(en)
    pa = normalize_text(pa)
    return {
        "id": f"{source}:{idx}",
        "source": source,
        "domain": domain,
        "src_lang": "eng_Latn",
        "tgt_lang": "pan_Guru",
        "en": en,
        "pa": pa,
        "en_chars": len(en),
        "pa_chars": len(pa),
        "en_words": len(en.split()),
        "pa_words": len(pa.split()),
        "pair_hash": pair_hash(en, pa),
    }


def read_pan_guru_tsv(path: Path) -> tuple[list[dict[str, str | int]], Counter]:
    rows: list[dict[str, str | int]] = []
    stats: Counter = Counter()

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"src", "tgt", "src_lang", "tgt_lang"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        for idx, record in enumerate(reader, start=1):
            if record.get("src_lang") != "eng_Latn" or record.get("tgt_lang") != "pan_Guru":
                stats["skipped_wrong_language"] += 1
                continue
            en = record.get("src", "")
            pa = record.get("tgt", "")
            if not en.strip() or not pa.strip():
                stats["skipped_empty"] += 1
                continue
            rows.append(row_for("pan_Guru", "general", idx, en, pa))
            stats["kept"] += 1

    return rows, stats


def read_parallel_files(
    en_path: Path,
    pa_path: Path,
    source: str,
    domain: str,
) -> tuple[list[dict[str, str | int]], Counter]:
    rows: list[dict[str, str | int]] = []
    stats: Counter = Counter()

    with en_path.open("r", encoding="utf-8", errors="replace") as en_file:
        with pa_path.open("r", encoding="utf-8", errors="replace") as pa_file:
            for idx, (en, pa) in enumerate(zip(en_file, pa_file), start=1):
                if not en.strip() or not pa.strip():
                    stats["skipped_empty"] += 1
                    continue
                rows.append(row_for(source, domain, idx, en, pa))
                stats["kept"] += 1

            extra_en = sum(1 for _ in en_file)
            extra_pa = sum(1 for _ in pa_file)
            if extra_en:
                stats["extra_en_lines"] = extra_en
            if extra_pa:
                stats["extra_pa_lines"] = extra_pa

    return rows, stats


def write_tsv(rows: list[dict[str, str | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, str | int]], summary_path: Path, source_stats: dict[str, Counter]) -> None:
    source_counts = Counter(row["source"] for row in rows)
    domain_counts = Counter(row["domain"] for row in rows)
    hashes = Counter(row["pair_hash"] for row in rows)
    duplicate_pairs = sum(count - 1 for count in hashes.values() if count > 1)

    en_words = [int(row["en_words"]) for row in rows]
    pa_words = [int(row["pa_words"]) for row in rows]

    def avg(values: list[int]) -> float:
        return sum(values) / len(values) if values else 0.0

    lines = [
        "# Combined Corpus Summary",
        "",
        f"total_rows: {len(rows)}",
        f"duplicate_exact_pairs: {duplicate_pairs}",
        f"avg_en_words: {avg(en_words):.2f}",
        f"avg_pa_words: {avg(pa_words):.2f}",
        "",
        "## Rows By Source",
        "",
    ]
    lines.extend(f"- {source}: {count}" for source, count in sorted(source_counts.items()))
    lines.extend(["", "## Rows By Domain", ""])
    lines.extend(f"- {domain}: {count}" for domain, count in sorted(domain_counts.items()))
    lines.extend(["", "## Loader Stats", ""])
    for source, stats in sorted(source_stats.items()):
        lines.append(f"### {source}")
        for key, value in sorted(stats.items()):
            lines.append(f"- {key}: {value}")
        lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--output", type=Path, default=Path("datasets/combined_raw.tsv"))
    parser.add_argument("--summary", type=Path, default=Path("datasets/combined_raw_summary.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir

    loaders = [
        ("pan_Guru", lambda: read_pan_guru_tsv(data_dir / "pan_Guru.tsv")),
        (
            "trainclean",
            lambda: read_parallel_files(
                data_dir / "trainclean.eng",
                data_dir / "trainclean.pun",
                "trainclean",
                "general",
            ),
        ),
        (
            "judicial",
            lambda: read_parallel_files(
                data_dir / "judicial_train.en",
                data_dir / "judicial_train.pa",
                "judicial",
                "legal",
            ),
        ),
    ]

    all_rows: list[dict[str, str | int]] = []
    source_stats: dict[str, Counter] = {}
    for source, load in loaders:
        rows, stats = load()
        all_rows.extend(rows)
        source_stats[source] = stats

    write_tsv(all_rows, args.output)
    write_summary(all_rows, args.summary, source_stats)

    print(f"Wrote {len(all_rows):,} rows to {args.output}")
    print(f"Wrote summary to {args.summary}")


if __name__ == "__main__":
    main()
