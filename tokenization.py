import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import os
    from pathlib import Path

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = "12"  # set to your physical/logical CPU thread count

    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.normalizers import NFC, Sequence
    from tokenizers.pre_tokenizers import Metaspace
    from tokenizers.decoders import Metaspace as MetaspaceDecoder
    from tokenizers.trainers import BpeTrainer


    return (
        BPE,
        BpeTrainer,
        Metaspace,
        MetaspaceDecoder,
        NFC,
        Path,
        Sequence,
        Tokenizer,
    )


@app.cell
def _(Path):

    DATA_PATH = Path("datasets/cleaned.tsv")
    TOKENIZER_DIR = Path("tokenizer")
    TOKENIZER_TEXT_PATH = TOKENIZER_DIR / "spm_train_text.txt"
    TOKENIZER_PREFIX = TOKENIZER_DIR / "spm_bpe24k"
    TOKENIZER_MODEL_PATH = TOKENIZER_PREFIX.with_suffix(".model")
    TOKENIZER_VOCAB_PATH = TOKENIZER_PREFIX.with_suffix(".vocab")

    VOCAB_SIZE = 24_000

    SPECIAL_TOKENS = [
        "<2en>",
        "<2pa>",
        "<legal>",
        "<general>",
        "<literal>",
        "<natural>",
    ]
    return DATA_PATH, TOKENIZER_TEXT_PATH


@app.cell
def _(DATA_PATH, pd):

    cleaned = pd.read_csv(
        DATA_PATH,
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

    cleaned["en"] = cleaned["en"].fillna("")
    cleaned["pa"] = cleaned["pa"].fillna("")

    cleaned.shape
    return (cleaned,)


@app.cell
def _(cleaned):

    cleaned.groupby(["source", "domain"], observed=True).size().reset_index(name="rows")
    return


@app.cell
def _(Path, pd):

    def write_sentencepiece_training_text(
        df: pd.DataFrame,
        output_path: Path,
        max_rows: int | None = None,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        view = df if max_rows is None else df.sample(max_rows, random_state=42)

        with output_path.open("w", encoding="utf-8", newline="\n") as f:
            for en, pa in zip(view["en"], view["pa"], strict=False):
                en = str(en).strip()
                pa = str(pa).strip()
                if en:
                    f.write(en + "\n")
                if pa:
                    f.write(pa + "\n")

        return output_path


    return (write_sentencepiece_training_text,)


@app.cell
def _(TOKENIZER_TEXT_PATH, cleaned, write_sentencepiece_training_text):

    # Full corpus tokenizer training text. This writes about 3.18M lines.
    # If iteration is slow, temporarily pass max_rows=300_000, then retrain on full data later.
    spm_training_text = write_sentencepiece_training_text(
        cleaned,
        TOKENIZER_TEXT_PATH,
        max_rows=None,
    )

    spm_training_text
    return


@app.cell
def _(
    BPE,
    BpeTrainer,
    Metaspace,
    MetaspaceDecoder,
    NFC,
    Path,
    Sequence,
    Tokenizer,
):
    TOKENIZER_DIR = Path("tokenizer")
    TOKENIZER_TEXT_PATH = TOKENIZER_DIR / "spm_train_text.txt"
    HF_TOKENIZER_PATH = TOKENIZER_DIR / "hf_bpe24k_tokenizer.json"

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = Sequence([NFC()])

    # SentencePiece-like whitespace handling.
    tokenizer.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="always")
    tokenizer.decoder = MetaspaceDecoder(replacement="▁", prepend_scheme="always")

    trainer = BpeTrainer(
        vocab_size=24_000,
        min_frequency=2,
        show_progress=True,
        special_tokens=[
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
        ],
    )

    tokenizer.train(
        files=[str(TOKENIZER_TEXT_PATH)],
        trainer=trainer,
    )

    tokenizer.save(str(HF_TOKENIZER_PATH))
    return HF_TOKENIZER_PATH, TOKENIZER_TEXT_PATH


@app.cell
def _(HF_TOKENIZER_PATH, Tokenizer):
    hf_tokenizer = Tokenizer.from_file(str(HF_TOKENIZER_PATH))

    sample = "<2pa> <legal> The agreement shall remain in force for five years."
    encoded = hf_tokenizer.encode(sample)

    encoded.tokens, encoded.ids, hf_tokenizer.decode(encoded.ids)
    return


if __name__ == "__main__":
    app.run()
