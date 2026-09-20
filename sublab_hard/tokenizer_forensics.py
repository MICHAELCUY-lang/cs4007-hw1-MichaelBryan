"""Sublab Harder - why Kazakh costs more, and what a homoglyph does to a word.

Sublab Medium gave you six models and a table of what each one repaired. This
sublab explains part of that table, and it does it without calling any model at
all. A tokenizer is a fixed, inspectable piece of software: you can open it,
run text through it, and see exactly what the model was handed.

Two tokenizers, both real:

  cl100k_base   the GPT-4 / GPT-3.5-turbo vocabulary
  o200k_base    the newer, larger one used from GPT-4o onwards

Three measurements:

  A. the same meaning in Kazakh, Russian and English - how many tokens each
     costs, in both tokenizers;
  B. what a Latin homoglyph does to the token stream of a Kazakh word;
  C. whether the newer tokenizer narrowed the gap.

No API keys. No network at run time (tiktoken downloads its vocabulary files
once and caches them). Nothing here costs money, which means there is no excuse
for not running it several times.

Fill in every `TODO`. Do not change the function signatures.
"""

import json
import sys
import unicodedata
from pathlib import Path

import tiktoken

DATA = Path(__file__).resolve().parent.parent / "data"
PARALLEL = DATA / "parallel.json"
KAZAKH_ERRORS = DATA / "kazakh_errors.json"

# Both are real tiktoken encodings. Do not use tiktoken.encoding_for_model():
# the models in this course are newer than your installed tiktoken and it will
# not know their names.
ENCODINGS = ["cl100k_base", "o200k_base"]

LANGS = ["kk", "ru", "en"]


def load_triplets() -> list[dict]:
    """Six meanings, each written in Kazakh, Russian and English."""
    return json.loads(PARALLEL.read_text(encoding="utf-8"))["triplets"]


def load_sentences() -> list[dict]:
    """The corrupted Kazakh sentences from Sublab Medium."""
    return json.loads(KAZAKH_ERRORS.read_text(encoding="utf-8"))["sentences"]


# --------------------------------------------------------------------------
# The tokenizer itself
# --------------------------------------------------------------------------

def encode(text: str, encoding_name: str = "o200k_base") -> list[int]:
    """Token ids for `text` under the named encoding.

    Two lines: get the encoding, encode the text.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.encode(text)


def pieces(ids: list[int], encoding_name: str = "o200k_base") -> list[str]:
    """The text of each token, one string per id.

    This is the function that makes the whole sublab visible - it is the
    difference between "8 tokens" and seeing the word come apart. Decode each
    id on its own, not the list as a whole.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return [encoding.decode([token_id]) for token_id in ids]


# --------------------------------------------------------------------------
# Pure measurements. No tokenizer in here - these take ids you already have,
# so you can check them by hand against the examples in each docstring before
# you run anything against real text.
# --------------------------------------------------------------------------

def tokens_per_char(text: str, ids: list[int]) -> float:
    """How many tokens each character of `text` cost.

    The comparison across languages only works per character; a Kazakh sentence
    and its English translation are not the same length, so raw token counts
    would be measuring the translation, not the tokenizer.

    Return 0.0 for empty text rather than dividing by zero.

    >>> tokens_per_char("abcd", [1, 2])
    0.5
    >>> tokens_per_char("", [])
    0.0
    """
    if not text:
        return 0.0
    return len(ids) / len(text)


def first_divergence(a: list[int], b: list[int]) -> int | None:
    """Index of the first position where two token streams differ.

    Returns None if one is a prefix of the other and they are the same length,
    i.e. if the streams are identical. If they share a prefix and then differ -
    including when one simply runs out - return the length of the shared prefix.

    >>> first_divergence([1, 2, 3], [1, 2, 3])

    >>> first_divergence([1, 2, 3], [1, 9, 3])
    1
    >>> first_divergence([1, 2], [1, 2, 3])
    2
    """
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index

    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def foreign_chars(text: str) -> list[tuple[int, str, str]]:
    """Every character that is a letter but not a Cyrillic one.

    This is how you find a homoglyph without knowing in advance where it is.
    A Latin `c` sitting inside a Kazakh word looks identical to a Cyrillic `с`
    on screen; it is a different code point, and `unicodedata.name` says so.

    Returns:
        A list of (index, character, unicode name), in order. Skip anything
        that is not a letter - spaces, digits and punctuation are not the
        interesting case.

    >>> foreign_chars("аcа")[0][:2]
    (1, 'c')
    >>> foreign_chars("Астана")
    []
    """
    result = []
    for index, character in enumerate(text):
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "UNKNOWN")
        if "CYRILLIC" not in name:
            result.append((index, character, name))
    return result


# --------------------------------------------------------------------------
# A. The price of a language
# --------------------------------------------------------------------------

def language_table(encoding_name: str) -> dict[str, dict]:
    """Total tokens, total characters and tokens-per-character, per language.

    Sum over all six triplets, do not average the per-sentence ratios - a long
    sentence and a short one should not count equally.

    Returns:
        {"kk": {"tokens": int, "chars": int, "tok_per_char": float}, "ru": ..., "en": ...}
    """
    totals = {
        lang: {"tokens": 0, "chars": 0, "tok_per_char": 0.0}
        for lang in LANGS
    }

    for triplet in load_triplets():
        for lang in LANGS:
            text = triplet[lang]
            totals[lang]["tokens"] += len(encode(text, encoding_name))
            totals[lang]["chars"] += len(text)

    for lang in LANGS:
        chars = totals[lang]["chars"]
        totals[lang]["tok_per_char"] = (
            totals[lang]["tokens"] / chars if chars else 0.0
        )

    return totals


def cost_per_thousand(tok_per_char: float, chars: int,
                      rate_in: float = 5.00) -> float:
    """What 1,000 sentences of this length would cost as input tokens.

    `rate_in` is dollars per million tokens; the default is gpt-5.6-sol's input
    rate. This turns a ratio into the only unit anyone outside this classroom
    cares about.

    >>> round(cost_per_thousand(0.5, 100, 10.0), 6)
    0.5
    """
    tokens_per_sentence = tok_per_char * chars
    tokens_for_thousand = tokens_per_sentence * 1_000
    return tokens_for_thousand * rate_in / 1_000_000


# --------------------------------------------------------------------------
# B. What the homoglyph did
# --------------------------------------------------------------------------

def homoglyph_report(corrupted: str, correct: str,
                     encoding_name: str = "o200k_base") -> dict:
    """Side-by-side forensics on one corrupted sentence.

    Returns:
        {
          "foreign": [(index, char, unicode_name), ...],   # from `foreign_chars`
          "tokens_correct": int,
          "tokens_corrupted": int,
          "delta": int,                 # corrupted minus correct
          "diverge_at": int | None,     # from `first_divergence`
          "pieces_correct": [str, ...],
          "pieces_corrupted": [str, ...],
        }
    """
    correct_ids = encode(correct, encoding_name)
    corrupted_ids = encode(corrupted, encoding_name)

    return {
        "foreign": foreign_chars(corrupted),
        "tokens_correct": len(correct_ids),
        "tokens_corrupted": len(corrupted_ids),
        "delta": len(corrupted_ids) - len(correct_ids),
        "diverge_at": first_divergence(correct_ids, corrupted_ids),
        "pieces_correct": pieces(correct_ids, encoding_name),
        "pieces_corrupted": pieces(corrupted_ids, encoding_name),
    }


def show_homoglyphs(encoding_name: str = "o200k_base") -> None:
    """Given. Print the report for every latin_homoglyph row in the dataset."""
    rows = [r for r in load_sentences() if "latin_homoglyph" in r["errors"]]
    if not rows:
        print("  no latin_homoglyph rows in the dataset")
        return
    for row in rows:
        rep = homoglyph_report(row["corrupted"], row["correct"], encoding_name)
        print("\n  [%s]  %+d tokens (%d -> %d), diverging at index %s"
              % (row["id"], rep["delta"], rep["tokens_correct"],
                 rep["tokens_corrupted"], rep["diverge_at"]))
        for idx, ch, name in rep["foreign"]:
            print("    char %d is %r - %s" % (idx, ch, name))
        d = rep["diverge_at"] or 0
        print("    correct  : %s" % rep["pieces_correct"][max(0, d - 1):d + 5])
        print("    corrupted: %s" % rep["pieces_corrupted"][max(0, d - 1):d + 5])


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== A. the same six meanings, three languages, two tokenizers ===")
    for enc_name in ENCODINGS:
        table = language_table(enc_name)
        print("\n  %s" % enc_name)
        print("    %-4s %8s %8s %12s" % ("lang", "tokens", "chars", "tok/char"))
        for lang in LANGS:
            row = table[lang]
            print("    %-4s %8d %8d %12.3f"
                  % (lang, row["tokens"], row["chars"], row["tok_per_char"]))
        base = table["en"]["tok_per_char"]
        for lang in LANGS:
            print("    %s costs %.2fx English"
                  % (lang, table[lang]["tok_per_char"] / base))

    print("\n=== B. what a Latin homoglyph does to the token stream ===")
    show_homoglyphs("o200k_base")

    print("\n=== C. did the newer tokenizer narrow the gap? ===")
    old, new = (language_table(e) for e in ENCODINGS)
    for lang in LANGS:
        print("  %s: %.3f -> %.3f tok/char"
              % (lang, old[lang]["tok_per_char"], new[lang]["tok_per_char"]))
    print("\n  Now answer question 2 in SUBMISSION.md, with these numbers in hand.")
