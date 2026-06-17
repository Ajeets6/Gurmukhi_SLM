import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():

    from pathlib import Path

    import pandas as pd
    import sentencepiece as spm


    return Path, pd, spm


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

    return (
        DATA_PATH,
        SPECIAL_TOKENS,
        TOKENIZER_MODEL_PATH,
        TOKENIZER_PREFIX,
        TOKENIZER_TEXT_PATH,
        TOKENIZER_VOCAB_PATH,
        VOCAB_SIZE,
    )


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
    SPECIAL_TOKENS,
    TOKENIZER_MODEL_PATH,
    TOKENIZER_PREFIX,
    TOKENIZER_TEXT_PATH,
    TOKENIZER_VOCAB_PATH,
    VOCAB_SIZE,
    spm,
):

    spm.SentencePieceTrainer.train(
        input=str(TOKENIZER_TEXT_PATH),
        model_prefix=str(TOKENIZER_PREFIX),
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0,
        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=SPECIAL_TOKENS,
        num_threads=8,
    )

    TOKENIZER_MODEL_PATH, TOKENIZER_VOCAB_PATH

    return


@app.cell
def _(TOKENIZER_MODEL_PATH, pd, spm):

    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL_PATH))

    sample_en = "The agreement shall remain in force for five years."
    sample_pa = "ਇਹ ਸਮਝੌਤਾ ਪੰਜ ਸਾਲਾਂ ਲਈ ਲਾਗੂ ਰਹੇਗਾ।"

    pd.DataFrame(
        {
            "text": [sample_en, sample_pa, "<2pa> <legal> " + sample_en],
            "pieces": [
                sp.encode(sample_en, out_type=str),
                sp.encode(sample_pa, out_type=str),
                sp.encode("<2pa> <legal> " + sample_en, out_type=str),
            ],
            "ids": [
                sp.encode(sample_en, out_type=int),
                sp.encode(sample_pa, out_type=int),
                sp.encode("<2pa> <legal> " + sample_en, out_type=int),
            ],
        }
    )

    return


if __name__ == "__main__":
    app.run()
