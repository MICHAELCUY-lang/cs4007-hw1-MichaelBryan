"""Sublab Medium - one Kazakh-correction task, six models.

Six models, one prompt, eight sentences. What you are producing is evidence:
a table that says which models repaired which kind of damage, and what each one
charged you for the attempt.

Fill in every `TODO`. Keep the function signatures.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sublab_easy.registration_bot import (RATES_PER_MTOK,  # noqa: E402
                                          ask_once, estimate_cost)

DATA = Path(__file__).resolve().parent.parent / "data" / "kazakh_errors.json"

# Every model you must run. Keep the order - it is the order of your table.
MODELS = [
    ("openrouter", "google/gemma-4-26b-a4b-it:free"),
    ("openrouter", "qwen/qwen3.8-27b"),
    ("openrouter", "deepseek/deepseek-v4-flash-0731"),
    ("openai", "gpt-5.6-luna"),
    ("openai", "gpt-5.6-terra"),
    ("openai", "gpt-5.6-sol"),
]


def load_sentences() -> list[dict]:
    """The eight corrupted sentences and their published originals."""
    return json.loads(DATA.read_text(encoding="utf-8"))["sentences"]


def build_prompt(corrupted: str) -> str:
    """Ask for a corrected sentence AND a list of the changes made.

    Requirements:
      - state that the text is Kazakh and may contain wrong letters, joined
        words, or letters from the wrong alphabet;
      - demand exactly this JSON and nothing else:
            {"corrected": "...", "changes": ["...", "..."]}
      - do not include the correct answer in the prompt. You are testing the
        model, not your own typing.

    Asking for a fixed shape instead of prose is how you make six models
    comparable. Week 3 turns this into a topic.
    """
    sentence = json.dumps(corrupted, ensure_ascii=False)
    return f"""Correct the Kazakh sentence below. It may contain incorrect
Kazakh letters, joined words, doubled letters, missing hyphens, or characters
from the wrong alphabet such as Latin letters mixed into Cyrillic text.

Return exactly one JSON object and nothing else, using this shape:
{{"corrected": "...", "changes": ["...", "..."]}}

Put the fully corrected sentence in "corrected". In "changes", briefly list
each correction you made. Do not add Markdown fences or explanatory prose.

Sentence: {sentence}"""


def parse_response(text: str) -> dict:
    """Pull {"corrected": str, "changes": list} out of the model's reply.

    Models wrap JSON in prose, or in ```json fences, more often than you would
    like. Be forgiving: find the JSON, parse it, and raise ValueError with the
    offending text if you truly cannot.
    """
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

        return {"corrected": corrected, "changes": changes}

    raise ValueError("No valid correction JSON found in response: " + repr(text))


def correct_with(model: str, corrupted: str, via: str) -> dict:
    """Send one sentence to one model.

    Returns:
        {"corrected": str, "changes": list, "input_tokens": int,
         "output_tokens": int, "model": str}

    `via` is "openai" or "openrouter" and goes straight through to
    `ask_once` from sublab_easy - there is no conversation here, just one
    prompt and one reply, eight times per model.
    """
    response = ask_once(build_prompt(corrupted), model=model, via=via)
    correction = parse_response(response["text"])

    return {
        "corrected": correction["corrected"],
        "changes": correction["changes"],
        "input_tokens": response["input_tokens"],
        "output_tokens": response["output_tokens"],
        "model": response["model"],
    }


def score_correction(returned: str, expected: str) -> dict:
    """Compare a model's output against the published original.

    Returns {"exact": bool, "char_diff": int} where char_diff is the number of
    differing characters (a simple positional comparison is enough; count the
    length difference too).

    READ THIS: `exact` is a signal, not a grade. Good Kazakh that differs from
    the original still counts as a correction. Your written analysis is where
    you make that call.
    """
    positional_differences = sum(
        returned_char != expected_char
        for returned_char, expected_char in zip(returned, expected)
    )
    char_diff = positional_differences + abs(len(returned) - len(expected))
    return {"exact": returned == expected, "char_diff": char_diff}


def run_all() -> list[dict]:
    """Every model against every sentence. One row per (model, sentence)."""
    rows = []
    for via, model in MODELS:
        for s in load_sentences():
            try:
                r = correct_with(model, s["corrupted"], via)
            except Exception as exc:            # a model failing IS a result
                rows.append({"model": model, "id": s["id"],
                             "errors": s["errors"], "failed": repr(exc)})
                continue
            rate_in, rate_out = RATES_PER_MTOK[model]
            rows.append({
                "model": model,
                "id": s["id"],
                "errors": s["errors"],
                "corrected": r["corrected"],
                "changes": r["changes"],
                **score_correction(r["corrected"], s["correct"]),
                "cost": estimate_cost(r["input_tokens"], r["output_tokens"],
                                      rate_in, rate_out),
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
            })
    return rows


def summarise(rows: list[dict]) -> None:
    """Per-model totals, to paste into SUBMISSION.md."""
    print(f"{'model':38}{'exact':>7}{'failed':>8}{'tokens':>9}{'cost $':>10}")
    print("-" * 72)
    for _, model in MODELS:
        mine = [r for r in rows if r["model"] == model]
        exact = sum(1 for r in mine if r.get("exact"))
        failed = sum(1 for r in mine if r.get("failed"))
        toks = sum(r.get("input_tokens", 0) + r.get("output_tokens", 0) for r in mine)
        cost = sum(r.get("cost", 0.0) for r in mine)
        print(f"{model:38}{exact:>7}{failed:>8}{toks:>9}{cost:>10.5f}")


if __name__ == "__main__":
    out = run_all()
    summarise(out)
    dest = Path(__file__).resolve().parent.parent / "outputs"
    dest.mkdir(exist_ok=True)
    (dest / "corrections.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote outputs/corrections.json ({len(out)} rows)")
