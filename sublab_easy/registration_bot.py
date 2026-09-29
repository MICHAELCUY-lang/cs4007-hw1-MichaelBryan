"""Sublab Easy - a course-registration chatbot, and the bill it runs up."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# === Load environment variables and basic configuration ===
load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

CATALOGUE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "courses.json"
)

RATES_PER_MTOK = {
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-sol": (5.00, 30.00),
    "google/gemma-4-26b-a4b-it:free": (0.00, 0.00),
    "qwen/qwen3.8-27b": (0.45, 3.20),
    "deepseek/deepseek-v4-flash-0731": (0.14, 0.28),
}


# === Load the course catalogue from JSON ===
def load_catalogue() -> dict:
    """The course catalogue, the registration rules, and the student."""
    return json.loads(
        CATALOGUE.read_text(encoding="utf-8")
    )


# === Create an OpenAI client ===
def openai_client() -> OpenAI:
    """A client pointed at OpenAI itself."""

    key = os.environ.get("OPENAI_API_KEY")

    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env."
        )

    return OpenAI(api_key=key)


# === Create an OpenRouter client using the same OpenAI interface ===
def openrouter_client() -> OpenAI:
    """A client pointed at OpenRouter."""

    key = os.environ.get("OPENROUTER_API_KEY")

    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env."
        )

    return OpenAI(
        api_key=key,
        base_url=OPENROUTER_BASE_URL
    )


# === Select which provider to use ===
def client_for(via: str) -> OpenAI:
    """Given. `via` is "openai" or "openrouter"."""

    if via == "openai":
        return openai_client()

    if via == "openrouter":
        return openrouter_client()

    raise ValueError(
        'via must be "openai" or "openrouter", got '
        + repr(via)
    )


# === Build the system prompt with catalogue and registration rules ===
def build_system_prompt(catalogue: dict) -> str:
    """Write the system message that turns a language model into a registrar."""

    prompt_catalogue = json.loads(
        json.dumps(catalogue)
    )

    for course in prompt_catalogue["courses"]:
        course["seats_remaining"] = (
            course["seats_total"]
            - course["seats_taken"]
        )

    catalogue_text = json.dumps(
        prompt_catalogue,
        ensure_ascii=False,
        indent=2
    )

    return f"""
You are a university course-registration advisor. Base every answer only on
the catalogue and student record below.

Before approving a registration, check all of the following:
- the requested course exists in the catalogue;
- the student has completed every prerequisite;
- the student has not already completed the course;
- the course has at least one remaining seat;
- its meeting times do not overlap another requested or registered course;
- the resulting credit total obeys the stated minimum and maximum limits.

Never invent a course or any course details. If a course is not present in the
catalogue, explicitly refuse the request. Explain any rejection using the
specific catalogue rule that prevents registration.

The JSON below is the complete and authoritative catalogue. It includes every
course's code, title, credits, prerequisites, meeting times, seat totals,
seats taken and seats remaining, as well as the student's completed courses
and the credit limits.

CATALOGUE AND STUDENT RECORD:
{catalogue_text}
""".strip()


# === Send messages to the selected AI model and collect token usage ===
def chat(
    messages: list[dict],
    model: str = "gpt-5.6-luna",
    via: str = "openai"
) -> dict:

    request_options = {}

    if via == "openrouter" and model in {
        "qwen/qwen3.8-27b",
        "deepseek/deepseek-v4-flash-0731",
    }:
        request_options["extra_body"] = {
            "reasoning": {
                "enabled": False
            }
        }

    response = client_for(via).chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=1024,
        **request_options,
    )

    if not response.choices:
        raise RuntimeError(
            "The model returned no choices."
        )

    if response.usage is None:
        raise RuntimeError(
            "The provider returned no token usage."
        )

    text = response.choices[0].message.content

    if text is None:
        raise RuntimeError(
            "The model returned no text content."
        )

    return {
        "text": text,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "model": model,
    }


# === Send a single prompt without conversation history ===
def ask_once(
    prompt: str,
    model: str = "gpt-5.6-luna",
    via: str = "openai"
) -> dict:

    return chat(
        [
            {
                "role": "user",
                "content": prompt
            }
        ],
        model=model,
        via=via
    )


# === Start a new conversation with the system prompt ===
def new_conversation(
    catalogue: dict
) -> list[dict]:

    return [
        {
            "role": "system",
            "content": build_system_prompt(catalogue)
        }
    ]


# === Run one conversation turn and update the history ===
def run_turn(
    history: list[dict],
    user_text: str,
    model: str = "gpt-5.6-luna",
    via: str = "openai"
) -> tuple[list[dict], dict]:

    history = history + [
        {
            "role": "user",
            "content": user_text
        }
    ]

    reply = chat(
        history,
        model=model,
        via=via
    )

    history = history + [
        {
            "role": "assistant",
            "content": reply["text"]
        }
    ]

    return history, reply


# === Calculate the cost of one API request ===
def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    rate_in: float,
    rate_out: float
) -> float:

    return (
        input_tokens * rate_in
        + output_tokens * rate_out
    ) / 1_000_000


# === Calculate cost based on model and token usage ===
def cost_of(usage: dict) -> float:

    rate_in, rate_out = RATES_PER_MTOK[
        usage["model"]
    ]

    return estimate_cost(
        usage["input_tokens"],
        usage["output_tokens"],
        rate_in,
        rate_out
    )


# === Calculate the total cost of the whole conversation ===
def conversation_cost(
    usages: list[dict]
) -> float:

    return sum(
        (cost_of(usage) for usage in usages),
        0.0
    )


# === Five conversation turns used for the experiment ===
SCRIPT = [
    "I am a third-year student. Which courses am I still eligible to register for?",
    "Register me for CSS-4007 and CSS-4102.",
    "How many credits would that be in total, and am I within the limit?",
    "Add CSS-4090 Quantum Machine Learning to my schedule.",
    "Я студент третьего курса. На какие курсы я всё ещё могу зарегистрироваться?",
]


# === Run all five turns and display token usage and cost ===
def run_script(
    model: str,
    via: str
) -> list[dict]:

    history = new_conversation(
        load_catalogue()
    )

    usages = []

    print(
        "\n===== "
        + via
        + " / "
        + model
        + " ====="
    )

    for i, user_text in enumerate(
        SCRIPT,
        start=1
    ):

        history, usage = run_turn(
            history,
            user_text,
            model=model,
            via=via
        )

        usages.append(usage)

        print("\n--- turn %d ---" % i)
        print("you: " + user_text)
        print("bot: " + usage["text"])

        print(
            "     in=%6d  out=%5d  $%.6f"
            % (
                usage["input_tokens"],
                usage["output_tokens"],
                cost_of(usage)
            )
        )

    print(
        "\n%10s%8s%7s%12s"
        % (
            "",
            "in",
            "out",
            "cost"
        )
    )

    for i, u in enumerate(
        usages,
        start=1
    ):
        print(
            "turn %-5d%8d%7d%12.6f"
            % (
                i,
                u["input_tokens"],
                u["output_tokens"],
                cost_of(u)
            )
        )

    print(
        "%25s%s"
        % (
            "",
            "-" * 12
        )
    )

    print(
        "%25s%12.6f"
        % (
            "total",
            conversation_cost(usages)
        )
    )

    return usages


# === Main: run the same experiment with OpenAI and OpenRouter ===
if __name__ == "__main__":

    if any(
        t.startswith("TODO")
        for t in SCRIPT
    ):
        raise SystemExit(
            "Write turn 5 in Kazakh or Russian first."
        )

    run_script(
        "gpt-5.6-luna",
        "openai"
    )

    run_script(
        "google/gemma-4-26b-a4b-it:free",
        "openrouter"
    )