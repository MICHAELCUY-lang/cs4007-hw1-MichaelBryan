"""
Sublab Medium - one Kazakh-correction task, six models.
"""

import json
import os
import sys
from pathlib import Path

# === Import functions from Sublab Easy ===
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sublab_easy.registration_bot import (
    RATES_PER_MTOK,
    ask_once,
    estimate_cost
)


# === Dataset and models configuration ===
DATA = Path(__file__).resolve().parent.parent / "data" / "kazakh_errors.json"

MODELS = [
    ("openrouter", "google/gemma-4-26b-a4b-it:free"),
    ("openrouter", "qwen/qwen3.8-27b"),
    ("openrouter", "deepseek/deepseek-v4-flash-0731"),
    ("openai", "gpt-5.6-luna"),
    ("openai", "gpt-5.6-terra"),
    ("openai", "gpt-5.6-sol"),
]


# === Load the eight corrupted Kazakh sentences ===
def load_sentences() -> list[dict]:
    return json.loads(DATA.read_text(encoding="utf-8"))["sentences"]


# === Build the correction prompt with a fixed JSON output format ===
def build_prompt(corrupted: str) -> str:
    sentence = json.dumps(corrupted, ensure_ascii=False)

    return f"""Correct the Kazakh sentence below. It may contain incorrect
Kazakh letters, joined words, doubled letters, missing hyphens, or characters
from the wrong alphabet such as Latin letters mixed into Cyrillic text.

Return exactly one JSON object and nothing else, using this shape:
{{"corrected": "...", "changes": ["...", "..."]}}

Put the fully corrected sentence in "corrected". In "changes", briefly list
each correction you made. Do not add Markdown fences or explanatory prose.

Sentence: {sentence}"""


# === Extract and validate JSON from the model response ===
def parse_response(text: str) -> dict:
    decoder = json.JSONDecoder()

    for start, character in enumerate(text):
        if character != "{":
            continue

        try:
            candidate, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue

        if not isinstance(candidate, dict):
            continue

        corrected = candidate.get("corrected")
        changes = candidate.get("changes")

        if not isinstance(corrected, str) or not isinstance(changes, list):
            continue

        if not all(isinstance(change, str) for change in changes):
            continue

        return {
            "corrected": corrected,
            "changes": changes
        }

    raise ValueError(
        "No valid correction JSON found in response: " + repr(text)
    )


# === Send one corrupted sentence to one AI model ===
def correct_with(model: str, corrupted: str, via: str) -> dict:
    response = ask_once(
        build_prompt(corrupted),
        model=model,
        via=via
    )

    correction = parse_response(response["text"])

    return {
        "corrected": correction["corrected"],
        "changes": correction["changes"],
        "input_tokens": response["input_tokens"],
        "output_tokens": response["output_tokens"],
        "model": response["model"],
    }


# === Compare the model correction with the published original ===
def score_correction(returned: str, expected: str) -> dict:
    positional_differences = sum(
        returned_char != expected_char
        for returned_char, expected_char in zip(returned, expected)
    )

    char_diff = (
        positional_differences
        + abs(len(returned) - len(expected))
    )

    return {
        "exact": returned == expected,
        "char_diff": char_diff
    }


# === Run all 6 models against all 8 sentences ===
def run_all() -> list[dict]:
    rows = []

    for via, model in MODELS:
        for s in load_sentences():

            try:
                r = correct_with(
                    model,
                    s["corrupted"],
                    via
                )

            except Exception as exc:
                rows.append({
                    "model": model,
                    "id": s["id"],
                    "errors": s["errors"],
                    "failed": repr(exc)
                })
                continue

            rate_in, rate_out = RATES_PER_MTOK[model]

            rows.append({
                "model": model,
                "id": s["id"],
                "errors": s["errors"],
                "corrected": r["corrected"],
                "changes": r["changes"],

                **score_correction(
                    r["corrected"],
                    s["correct"]
                ),

                "cost": estimate_cost(
                    r["input_tokens"],
                    r["output_tokens"],
                    rate_in,
                    rate_out
                ),

                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
            })

    return rows


# === Summarise exact matches, failures, tokens, and cost per model ===
def summarise(rows: list[dict]) -> None:
    print(
        f"{'model':38}"
        f"{'exact':>7}"
        f"{'failed':>8}"
        f"{'tokens':>9}"
        f"{'cost $':>10}"
    )

    print("-" * 72)

    for _, model in MODELS:
        mine = [
            r for r in rows
            if r["model"] == model
        ]

        exact = sum(
            1 for r in mine
            if r.get("exact")
        )

        failed = sum(
            1 for r in mine
            if r.get("failed")
        )

        toks = sum(
            r.get("input_tokens", 0)
            + r.get("output_tokens", 0)
            for r in mine
        )

        cost = sum(
            r.get("cost", 0.0)
            for r in mine
        )

        print(
            f"{model:38}"
            f"{exact:>7}"
            f"{failed:>8}"
            f"{toks:>9}"
            f"{cost:>10.5f}"
        )


# === Run experiment, print summary, and save results to JSON ===
if __name__ == "__main__":
    out = run_all()

    summarise(out)

    dest = (
        Path(__file__).resolve().parent.parent
        / "outputs"
    )

    dest.mkdir(exist_ok=True)

    (dest / "corrections.json").write_text(
        json.dumps(
            out,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        f"\nwrote outputs/corrections.json "
        f"({len(out)} rows)"
    )