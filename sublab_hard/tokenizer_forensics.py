"""
Sublab Harder - why Kazakh costs more,
and what a homoglyph does to a word.
"""

import json
import sys
import unicodedata
from pathlib import Path

import tiktoken


# === Dataset configuration ===
DATA = Path(__file__).resolve().parent.parent / "data"
PARALLEL = DATA / "parallel.json"
KAZAKH_ERRORS = DATA / "kazakh_errors.json"


# === Tokenizers and languages used in the experiment ===
ENCODINGS = ["cl100k_base", "o200k_base"]
LANGS = ["kk", "ru", "en"]


# === Load parallel Kazakh, Russian, and English sentences ===
def load_triplets() -> list[dict]:
    """Six meanings, each written in Kazakh, Russian and English."""
    return json.loads(
        PARALLEL.read_text(encoding="utf-8")
    )["triplets"]


# === Load corrupted Kazakh sentences from Sublab Medium ===
def load_sentences() -> list[dict]:
    """The corrupted Kazakh sentences from Sublab Medium."""
    return json.loads(
        KAZAKH_ERRORS.read_text(encoding="utf-8")
    )["sentences"]


# === Convert text into token IDs using the selected tokenizer ===
def encode(
    text: str,
    encoding_name: str = "o200k_base"
) -> list[int]:

    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.encode(text)


# === Convert token IDs back into individual token pieces ===
def pieces(
    ids: list[int],
    encoding_name: str = "o200k_base"
) -> list[str]:

    encoding = tiktoken.get_encoding(encoding_name)

    return [
        encoding.decode([token_id])
        for token_id in ids
    ]


# === Calculate how many tokens are used per character ===
def tokens_per_char(
    text: str,
    ids: list[int]
) -> float:

    if not text:
        return 0.0

    return len(ids) / len(text)


# === Find where two token streams first become different ===
def first_divergence(
    a: list[int],
    b: list[int]
) -> int | None:

    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index

    if len(a) != len(b):
        return min(len(a), len(b))

    return None


# === Detect alphabetic characters that are not Cyrillic ===
def foreign_chars(
    text: str
) -> list[tuple[int, str, str]]:

    result = []

    for index, character in enumerate(text):

        if not character.isalpha():
            continue

        name = unicodedata.name(
            character,
            "UNKNOWN"
        )

        if "CYRILLIC" not in name:
            result.append(
                (index, character, name)
            )

    return result


# === Compare token usage between Kazakh, Russian, and English ===
def language_table(
    encoding_name: str
) -> dict[str, dict]:

    totals = {
        lang: {
            "tokens": 0,
            "chars": 0,
            "tok_per_char": 0.0
        }
        for lang in LANGS
    }

    for triplet in load_triplets():

        for lang in LANGS:

            text = triplet[lang]

            totals[lang]["tokens"] += len(
                encode(text, encoding_name)
            )

            totals[lang]["chars"] += len(text)

    for lang in LANGS:

        chars = totals[lang]["chars"]

        totals[lang]["tok_per_char"] = (
            totals[lang]["tokens"] / chars
            if chars
            else 0.0
        )

    return totals


# === Estimate the input cost for 1,000 sentences ===
def cost_per_thousand(
    tok_per_char: float,
    chars: int,
    rate_in: float = 5.00
) -> float:

    tokens_per_sentence = (
        tok_per_char * chars
    )

    tokens_for_thousand = (
        tokens_per_sentence * 1_000
    )

    return (
        tokens_for_thousand
        * rate_in
        / 1_000_000
    )


# === Compare correct and corrupted text affected by homoglyphs ===
def homoglyph_report(
    corrupted: str,
    correct: str,
    encoding_name: str = "o200k_base"
) -> dict:

    correct_ids = encode(
        correct,
        encoding_name
    )

    corrupted_ids = encode(
        corrupted,
        encoding_name
    )

    return {
        "foreign":
            foreign_chars(corrupted),

        "tokens_correct":
            len(correct_ids),

        "tokens_corrupted":
            len(corrupted_ids),

        "delta":
            len(corrupted_ids) - len(correct_ids),

        "diverge_at":
            first_divergence(
                correct_ids,
                corrupted_ids
            ),

        "pieces_correct":
            pieces(
                correct_ids,
                encoding_name
            ),

        "pieces_corrupted":
            pieces(
                corrupted_ids,
                encoding_name
            ),
    }


# === Display tokenization changes caused by Latin homoglyphs ===
def show_homoglyphs(
    encoding_name: str = "o200k_base"
) -> None:

    rows = [
        r
        for r in load_sentences()
        if "latin_homoglyph" in r["errors"]
    ]

    if not rows:
        print(
            "  no latin_homoglyph rows in the dataset"
        )
        return

    for row in rows:

        rep = homoglyph_report(
            row["corrupted"],
            row["correct"],
            encoding_name
        )

        print(
            "\n  [%s]  %+d tokens (%d -> %d), "
            "diverging at index %s"
            % (
                row["id"],
                rep["delta"],
                rep["tokens_correct"],
                rep["tokens_corrupted"],
                rep["diverge_at"]
            )
        )

        for idx, ch, name in rep["foreign"]:
            print(
                "    char %d is %r - %s"
                % (idx, ch, name)
            )

        d = rep["diverge_at"] or 0

        print(
            "    correct  : %s"
            % rep["pieces_correct"][
                max(0, d - 1):d + 5
            ]
        )

        print(
            "    corrupted: %s"
            % rep["pieces_corrupted"][
                max(0, d - 1):d + 5
            ]
        )


# === Run all tokenizer experiments and print the results ===
if __name__ == "__main__":

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(
            encoding="utf-8"
        )

    # Part A: Compare three languages using two tokenizers
    print(
        "=== A. the same six meanings, "
        "three languages, two tokenizers ==="
    )

    for enc_name in ENCODINGS:

        table = language_table(enc_name)

        print("\n  %s" % enc_name)

        print(
            "    %-4s %8s %8s %12s"
            % (
                "lang",
                "tokens",
                "chars",
                "tok/char"
            )
        )

        for lang in LANGS:

            row = table[lang]

            print(
                "    %-4s %8d %8d %12.3f"
                % (
                    lang,
                    row["tokens"],
                    row["chars"],
                    row["tok_per_char"]
                )
            )

        base = table["en"]["tok_per_char"]

        for lang in LANGS:

            print(
                "    %s costs %.2fx English"
                % (
                    lang,
                    table[lang]["tok_per_char"]
                    / base
                )
            )


    # Part B: Show the effect of Latin homoglyphs
    print(
        "\n=== B. what a Latin homoglyph "
        "does to the token stream ==="
    )

    show_homoglyphs("o200k_base")


    # Part C: Compare old and new tokenizer efficiency
    print(
        "\n=== C. did the newer tokenizer "
        "narrow the gap? ==="
    )

    old, new = (
        language_table(e)
        for e in ENCODINGS
    )

    for lang in LANGS:

        print(
            "  %s: %.3f -> %.3f tok/char"
            % (
                lang,
                old[lang]["tok_per_char"],
                new[lang]["tok_per_char"]
            )
        )

    print(
        "\n  Now answer question 2 in "
        "SUBMISSION.md, with these numbers in hand."
    )