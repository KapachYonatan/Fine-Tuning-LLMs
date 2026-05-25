"""
create_data.py — Part 4: Generate training data (English prompt → Hebrew answer)

Strategy A: load Qwen/Qwen2.5-7B-Instruct with a system prompt that forces Hebrew
answers, then feed it English instructions from tatsu-lab/alpaca.

Usage:
    python code/part4/create_data.py              # generate 1000 examples
    python code/part4/create_data.py --n 10       # quick smoke test
    python code/part4/create_data.py --resume     # continue from a partial run

Output:
    data/train_data.jsonl   (one JSON object per line: {"prompt": ..., "response": ...})
"""

import argparse
import gc
import json
import os

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(BASE))           # workspace root

# ── Constants ─────────────────────────────────────────────────────────────────

GENERATOR_MODEL  = "Qwen/Qwen2.5-7B-Instruct"
SYSTEM_PROMPT    = "You are a helpful assistant. Always answer in Hebrew."
MAX_NEW_TOKENS   = 256

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
    ds = load_dataset("tatsu-lab/alpaca", split="train")
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


def generate_hebrew_answer(model, tokenizer, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs    = tokenizer(formatted, return_tensors="pt")
    device    = next(model.parameters()).device
    inputs    = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int, default=1000,
                        help="Total number of examples to generate (default: 1000)")
    parser.add_argument("--output", default=os.path.join(ROOT, "data", "train_data.jsonl"),
                        help="Output JSONL file path")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from an existing partial output file")
    args = parser.parse_args()

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

    # ── Load generator model ──────────────────────────────────────────────────
    print(f"\nLoading generator model: {GENERATOR_MODEL}")
    device    = "cuda" if torch.cuda.is_available() else "cpu"

    # Load in 8-bit on GPU: ~7 GB for 7B, good quality/memory balance on T4.
    # On CPU, quantization is not supported so we fall back to fp32.
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs = dict(quantization_config=bnb_config, device_map="auto")
    else:
        model_kwargs = dict(torch_dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(
        GENERATOR_MODEL, use_fast=True, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        GENERATOR_MODEL,
        trust_remote_code=True,
        **model_kwargs,
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    print(f"  Loaded on {device}.\n")

    # ── Generate ──────────────────────────────────────────────────────────────
    mode = "a" if args.resume else "w"
    with open(args.output, mode, encoding="utf-8") as out_f:
        for prompt in tqdm(prompts, desc="Generating"):
            response = generate_hebrew_answer(model, tokenizer, prompt)
            out_f.write(json.dumps({"prompt": prompt, "response": response},
                                   ensure_ascii=False) + "\n")
            out_f.flush()   # safe to Ctrl-C and resume

    total = len(already_done) + len(prompts)
    print(f"\nDone.  Wrote {total} examples to: {args.output}")

    # ── Free GPU memory ───────────────────────────────────────────────────────
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
