"""
create_data.py — Part 4: Generate training data (English prompt → Hebrew answer)

Calls the Gemini API to generate Hebrew answers for English instructions
from tatsu-lab/alpaca.  Requires a Gemini API key in the environment:

    export GEMINI_API_KEY="your-key-here"

Usage:
    python code/part4/create_data.py              # generate 1000 examples
    python code/part4/create_data.py --n 10       # quick smoke test
    python code/part4/create_data.py --resume     # continue from a partial run
    python code/part4/create_data.py --model gemini-3-flash-preview  # override model

Output:
    data/train_data.jsonl   (one JSON object per line: {"prompt": ..., "response": ...})
"""

import argparse
import json
import os
import time

from datasets import load_dataset
from google import genai
from google.genai import types
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(BASE))           # workspace root

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gemini-3-flash-preview"
SYSTEM_PROMPT = "You are a helpful assistant. Always answer in Hebrew."
RATE_LIMIT_SLEEP = 1.0   # seconds between API calls (free tier: 60 RPM)

# All 20 evaluation inputs — must NOT appear in training data.
EVAL_INPUTS = {
    # 10 provided by the assignment
    "Explain why the sky looks blue during the day.",
    "Give two advantages and two disadvantages of public transportation.",
    "Write a short email asking a professor for an extension on an assignment.",
    "Describe how to make a simple omelette.",
    "What is the difference between supervised and unsupervised learning?",
    "Summarize the story of Cinderella in three sentences.",
    "Suggest three ways to reduce smartphone distraction while studying.",
    "Explain what happens when water boils.",
    "Give a polite refusal to an invitation to a party.",
    'Turn the idea "practice makes progress" into advice for a student.',
    # 10 custom evaluation inputs
    "What is the capital of France?",
    "How many continents are there on Earth?",
    "Explain the concept of gravity in simple terms.",
    "Write a short poem about rain.",
    "What are three benefits of regular exercise?",
    "How do you make a cup of tea?",
    "What is the difference between a noun and a verb?",
    "Name two famous painters from the Renaissance.",
    "Why do people dream?",
    "Give advice to someone who is starting a new job.",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_eval_input(text: str) -> bool:
    return text.strip() in EVAL_INPUTS


def load_alpaca_prompts(n_needed: int, already_done: set) -> list:
    """Pull clean English instructions from tatsu-lab/alpaca.

    Only uses examples where `input` is empty (pure single-turn instructions).
    Excludes eval inputs and any prompt already written to the output file.
    Returns up to n_needed prompts.
    """
    print("Downloading tatsu-lab/alpaca …")
    ds = load_dataset("tatsu-lab/alpaca", split="train",
                      token=os.environ.get("HF_TOKEN"))
    prompts = []
    for row in ds:
        instruction = row["instruction"].strip()
        if not instruction:
            continue
        if row["input"].strip():       # skip multi-part examples for simplicity
            continue
        if is_eval_input(instruction):
            continue
        if instruction in already_done:
            continue
        prompts.append(instruction)
        if len(prompts) >= n_needed:
            break
    return prompts


def generate_hebrew_answer(client: genai.Client, model_name: str, prompt: str) -> str:
    response = client.models.generate_content(
        model=model_name,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=512,
            temperature=0.7,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
        contents=prompt,
    )
    return response.text


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int, default=1000,
                        help="Total number of examples to generate (default: 1000)")
    parser.add_argument("--output", default=os.path.join(ROOT, "data", "train_data.jsonl"),
                        help="Output JSONL file path")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from an existing partial output file")
    parser.add_argument("--model",  default=DEFAULT_MODEL,
                        help=f"Gemini model name (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")
    client = genai.Client(api_key=api_key)
    print(f"Gemini model: {args.model}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # ── Resume: read already-written prompts ──────────────────────────────────
    already_done: set = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                already_done.add(rec["prompt"])
        print(f"Resuming — {len(already_done)} examples already done.")

    n_needed = args.n - len(already_done)
    if n_needed <= 0:
        print(f"Already have {len(already_done)} examples — nothing to do.")
        return

    # ── Select prompts ────────────────────────────────────────────────────────
    prompts = load_alpaca_prompts(n_needed, already_done)
    print(f"  {len(prompts)} prompts selected (target: {n_needed})")

    if len(prompts) < n_needed:
        print(f"  Warning: only found {len(prompts)} suitable prompts (requested {n_needed}).")

    # ── Generate via Gemini API ───────────────────────────────────────────────
    mode = "a" if args.resume else "w"
    with open(args.output, mode, encoding="utf-8") as out_f:
        for prompt in tqdm(prompts, desc="Generating"):
            response = generate_hebrew_answer(client, args.model, prompt)
            out_f.write(json.dumps({"prompt": prompt, "response": response},
                                   ensure_ascii=False) + "\n")
            out_f.flush()        # safe to Ctrl-C and resume
            time.sleep(RATE_LIMIT_SLEEP)

    total = len(already_done) + len(prompts)
    print(f"\nDone.  Wrote {total} examples to: {args.output}")


if __name__ == "__main__":
    main()
