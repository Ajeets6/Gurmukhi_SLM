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
    def _has_gurmukhi(value):
        return any(0x0A00 <= ord(char) <= 0x0A7F for char in str(value))


    def _has_latin(value):
        return any(("A" <= char <= "Z") or ("a" <= char <= "z") for char in str(value))


    script_view = corpus.assign(
        en_has_gurmukhi=corpus["en"].map(_has_gurmukhi),
        pa_has_gurmukhi=corpus["pa"].map(_has_gurmukhi),
        pa_has_latin=corpus["pa"].map(_has_latin),
        en_has_latin=corpus["en"].map(_has_latin),
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
def _(corpus, pd):

    # Remove only exact duplicate translation pairs first.
    # Repeated English with different Punjabi is useful translation variation, so keep it.
    clean = corpus.drop_duplicates("pair_hash", keep="first").copy()

    _url_pattern = r"https?://|www\."
    _url_only_pattern = r"\s*(?:https?://\S+|www\.\S+)\s*"

    clean = clean.assign(
        char_ratio=(clean["pa_chars"] / clean["en_chars"].clip(lower=1)).round(2),
        word_ratio=(clean["pa_words"] / clean["en_words"].clip(lower=1)).round(2),
        en_too_short=clean["en_words"] < 3,
        pa_too_short=clean["pa_words"] < 3,
        en_too_long=clean["en_words"] > 200,
        pa_too_long=clean["pa_words"] > 220,
        ratio_too_high=(clean["pa_words"] / clean["en_words"].clip(lower=1)) > 3,
        ratio_too_low=(clean["pa_words"] / clean["en_words"].clip(lower=1)) < 0.33,
        en_is_punctuation_only=clean["en"].str.fullmatch(r"[\W_]+", na=False),
        pa_is_punctuation_only=clean["pa"].str.fullmatch(r"[\W_]+", na=False),
        en_has_url=clean["en"].str.contains(_url_pattern, case=False, regex=True, na=False),
        pa_has_url=clean["pa"].str.contains(_url_pattern, case=False, regex=True, na=False),
        en_is_url_only=clean["en"].str.fullmatch(_url_only_pattern, case=False, na=False),
        pa_is_url_only=clean["pa"].str.fullmatch(_url_only_pattern, case=False, na=False),
    )

    clean = clean.assign(
        hard_drop=(
            clean["en_is_punctuation_only"]
            | clean["pa_is_punctuation_only"]
            | clean["en_is_url_only"]
            | clean["pa_is_url_only"]
            | clean["en_too_long"]
            | clean["pa_too_long"]
            | clean["ratio_too_high"]
            | clean["ratio_too_low"]
        ),
        review_flag=(
            clean["en_too_short"]
            | clean["pa_too_short"]
            | clean["en_too_long"]
            | clean["pa_too_long"]
            | clean["ratio_too_high"]
            | clean["ratio_too_low"]
            | clean["en_is_punctuation_only"]
            | clean["pa_is_punctuation_only"]
            | clean["en_has_url"]
            | clean["pa_has_url"]
            | clean["en_is_url_only"]
            | clean["pa_is_url_only"]
        ),
    )

    clean_candidate = clean.loc[~clean["hard_drop"]].copy()

    cleaning_summary = pd.DataFrame(
        {
            "check": [
                "raw rows",
                "after exact pair dedupe",
                "removed exact pair duplicates",
                "punctuation-only English",
                "punctuation-only Punjabi",
                "English has URL",
                "Punjabi has URL",
                "URL-only English",
                "URL-only Punjabi",
                "very short English (<3 words)",
                "very short Punjabi (<3 words)",
                "long English (>200 words)",
                "long Punjabi (>220 words)",
                "high word ratio (>3x)",
                "low word ratio (<0.33x)",
                "hard drop rows",
                "clean candidate rows",
                "review flag rows",
            ],
            "rows": [
                len(corpus),
                len(clean),
                len(corpus) - len(clean),
                clean["en_is_punctuation_only"].sum(),
                clean["pa_is_punctuation_only"].sum(),
                clean["en_has_url"].sum(),
                clean["pa_has_url"].sum(),
                clean["en_is_url_only"].sum(),
                clean["pa_is_url_only"].sum(),
                clean["en_too_short"].sum(),
                clean["pa_too_short"].sum(),
                clean["en_too_long"].sum(),
                clean["pa_too_long"].sum(),
                clean["ratio_too_high"].sum(),
                clean["ratio_too_low"].sum(),
                clean["hard_drop"].sum(),
                len(clean_candidate),
                clean["review_flag"].sum(),
            ],
        }
    )

    cleaning_summary

    return clean, clean_candidate


@app.cell
def _(corpus):
    en_variants = (
        corpus.groupby("en")
        .agg(
            rows=("id", "count"),
            unique_pa=("pa", "nunique"),
            sources=("source", lambda x: ", ".join(sorted(set(map(str, x))))),
            avg_en_words=("en_words", "mean"),
        )
        .query("rows > 1 and unique_pa > 1")
        .sort_values(["unique_pa", "rows"], ascending=False)
    )

    en_variants.head(50)
    return (en_variants,)


@app.cell
def _(corpus, en_variants):
    example_en = en_variants.index[0]

    corpus.loc[
        corpus["en"] == example_en,
        ["id", "source", "domain", "en", "pa"]
    ].head(30)
    return


@app.cell
def _(corpus):
    variant_groups = (
        corpus.groupby("en")
        .agg(
            rows=("id", "count"),
            unique_pa=("pa", "nunique"),
            domain=("domain", lambda x: ", ".join(sorted(set(map(str, x))))),
            source=("source", lambda x: ", ".join(sorted(set(map(str, x))))),
        )
        .query("rows > 1 and unique_pa > 1")
        .sort_values(["unique_pa", "rows"], ascending=False)
    )

    variant_groups.head(50)
    return


@app.cell
def _():
    return


@app.cell
def _(clean):

    hard_drop_examples = clean.loc[
        clean["hard_drop"],
        [
            "id",
            "source",
            "domain",
            "en_words",
            "pa_words",
            "word_ratio",
            "en_is_punctuation_only",
            "pa_is_punctuation_only",
            "en",
            "pa",
        ],
    ]

    hard_drop_examples.head(100)

    return


@app.cell
def _(clean):

    _short_mask = (clean["en_too_short"] | clean["pa_too_short"]) & ~clean["hard_drop"]
    short_review_examples = clean.loc[
        _short_mask,
        ["id", "source", "domain", "en_words", "pa_words", "en", "pa"],
    ]

    short_review_examples.sample(min(100, len(short_review_examples)), random_state=11)

    return


@app.cell
def _(clean):

    url_review_examples = clean.loc[
        (clean["en_has_url"] | clean["pa_has_url"]) & ~clean["hard_drop"],
        [
            "id",
            "source",
            "domain",
            "en_words",
            "pa_words",
            "en_is_url_only",
            "pa_is_url_only",
            "en",
            "pa",
        ],
    ]

    url_review_examples.head(100)

    return


@app.cell
def _(clean_candidate, pd):

    web_phrase_pattern = (
        r"\b(?:log on|visit|available at|downloaded from|website|web site|"
        r"online booking|book online|for reservation|reservation visit)\b"
    )

    clean_train_v1 = clean_candidate.assign(
        en_has_web_phrase=clean_candidate["en"].str.contains(
            web_phrase_pattern, case=False, regex=True, na=False
        ),
        pa_has_web_phrase=clean_candidate["pa"].str.contains(
            web_phrase_pattern, case=False, regex=True, na=False
        ),
    )

    clean_train_v1 = clean_train_v1.loc[
        ~(
            clean_train_v1["en_has_url"]
            | clean_train_v1["pa_has_url"]
            | clean_train_v1["en_has_web_phrase"]
            | clean_train_v1["pa_has_web_phrase"]
        )
    ].copy()

    web_filter_summary = pd.DataFrame(
        {
            "check": [
                "clean candidate rows",
                "rows with URL in English or Punjabi",
                "rows with web/navigation phrase",
                "clean_train_v1 rows",
                "rows removed from clean_candidate for train_v1",
            ],
            "rows": [
                len(clean_candidate),
                ((clean_candidate["en_has_url"] | clean_candidate["pa_has_url"])).sum(),
                (
                    clean_candidate["en"].str.contains(web_phrase_pattern, case=False, regex=True, na=False)
                    | clean_candidate["pa"].str.contains(web_phrase_pattern, case=False, regex=True, na=False)
                ).sum(),
                len(clean_train_v1),
                len(clean_candidate) - len(clean_train_v1),
            ],
        }
    )

    web_filter_summary

    return (web_phrase_pattern,)


@app.cell
def _(clean_candidate, web_phrase_pattern):

    web_removed_examples = clean_candidate.loc[
        clean_candidate["en_has_url"]
        | clean_candidate["pa_has_url"]
        | clean_candidate["en"].str.contains(web_phrase_pattern, case=False, regex=True, na=False)
        | clean_candidate["pa"].str.contains(web_phrase_pattern, case=False, regex=True, na=False),
        ["id", "source", "domain", "en_words", "pa_words", "en", "pa"],
    ]

    web_removed_examples.head(100)

    return


if __name__ == "__main__":
    app.run()
