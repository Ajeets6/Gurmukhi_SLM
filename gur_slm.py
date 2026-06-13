import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    return mo, pd


@app.cell
def _(pd):
    corpus = pd.read_csv(
        r"datasets\combined_raw.tsv",
        sep="\t",
        dtype={
            "id": "string",
            "source": "category",
            "domain": "category",
            "src_lang": "category",
            "tgt_lang": "category",
            "en": "string",
            "pa": "string",
            "pair_hash": "string",
        },
    )
    corpus["en"] = corpus["en"].fillna("")
    corpus["pa"] = corpus["pa"].fillna("")
    return (corpus,)


@app.cell
def _(corpus, mo):
    mo.md(f"""
    ## Corpus Overview

    Rows: **{len(corpus):,}**

    Columns: `{", ".join(corpus.columns)}`
    """)
    return


@app.cell
def _(corpus):
    corpus.head(10)
    return


@app.cell
def _(corpus):
    corpus.groupby(["source", "domain"], observed=True).size().reset_index(name="rows")
    return


@app.cell
def _(corpus):
    corpus[["en_chars", "pa_chars", "en_words", "pa_words"]].describe(
        percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]
    )
    return


@app.cell
def _(corpus, pd):
    length_view = corpus.assign(
        char_ratio=(corpus["pa_chars"] / corpus["en_chars"].clip(lower=1)).round(2),
        word_ratio=(corpus["pa_words"] / corpus["en_words"].clip(lower=1)).round(2),
    )

    length_flags = pd.DataFrame(
        {
            "check": [
                "empty English",
                "empty Punjabi",
                "very short English (<3 words)",
                "very short Punjabi (<3 words)",
                "long English (>200 words)",
                "long Punjabi (>220 words)",
                "high word ratio (>3x)",
                "low word ratio (<0.33x)",
            ],
            "rows": [
                (length_view["en"].str.strip() == "").sum(),
                (length_view["pa"].str.strip() == "").sum(),
                (length_view["en_words"] < 3).sum(),
                (length_view["pa_words"] < 3).sum(),
                (length_view["en_words"] > 200).sum(),
                (length_view["pa_words"] > 220).sum(),
                (length_view["word_ratio"] > 3).sum(),
                (length_view["word_ratio"] < 0.33).sum(),
            ],
        }
    )
    length_flags
    return (length_view,)


@app.cell
def _(corpus, pd):
    duplicate_summary = pd.DataFrame(
        {
            "check": [
                "duplicate exact EN/PA pair hash",
                "duplicate English source",
                "duplicate Punjabi target",
            ],
            "rows": [
                corpus.duplicated("pair_hash").sum(),
                corpus.duplicated("en").sum(),
                corpus.duplicated("pa").sum(),
            ],
        }
    )
    duplicate_summary
    return


@app.cell
def _(corpus):
    duplicate_examples = corpus[corpus.duplicated("pair_hash", keep=False)].sort_values(
        "pair_hash"
    )
    duplicate_examples.head(30)
    return


@app.cell
def _(corpus, pd):
    script_view = corpus.assign(
        en_has_gurmukhi=corpus["en"].str.contains(r"[\u0A00-\u0A7F]", regex=True),
        pa_has_gurmukhi=corpus["pa"].str.contains(r"[\u0A00-\u0A7F]", regex=True),
        pa_has_latin=corpus["pa"].str.contains(r"[A-Za-z]", regex=True),
        en_has_latin=corpus["en"].str.contains(r"[A-Za-z]", regex=True),
    )

    script_flags = pd.DataFrame(
        {
            "check": [
                "English contains Gurmukhi",
                "Punjabi has no Gurmukhi",
                "Punjabi contains Latin letters",
                "English has no Latin letters",
            ],
            "rows": [
                script_view["en_has_gurmukhi"].sum(),
                (~script_view["pa_has_gurmukhi"]).sum(),
                script_view["pa_has_latin"].sum(),
                (~script_view["en_has_latin"]).sum(),
            ],
        }
    )
    script_flags
    return (script_view,)


@app.cell
def _(script_view):
    script_view[
        script_view["en_has_gurmukhi"]
        | ~script_view["pa_has_gurmukhi"]
        | script_view["pa_has_latin"]
        | ~script_view["en_has_latin"]
    ][["id", "source", "domain", "en", "pa"]].head(50)
    return


@app.cell
def _(length_view):
    length_view[
        (length_view["en_words"] > 200)
        | (length_view["pa_words"] > 220)
        | (length_view["word_ratio"] > 3)
        | (length_view["word_ratio"] < 0.33)
    ][["id", "source", "domain", "en_words", "pa_words", "word_ratio", "en", "pa"]].head(50)
    return


@app.cell
def _(corpus):
    corpus.sample(20, random_state=7)[["id", "source", "domain", "en", "pa"]]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
