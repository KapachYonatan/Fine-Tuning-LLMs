"""
run_eval.py — Part 4: Evaluate base and fine-tuned Qwen/Qwen2.5-1.5B-Instruct

Runs both the original model and the LoRA-adapted model on the 20 evaluation
prompts and writes eval_outputs.jsonl.

Usage:
    python code/part4/run_eval.py                         # run both models (default)
    python code/part4/run_eval.py --model base            # base model only
    python code/part4/run_eval.py --model finetuned       # fine-tuned model only
    python code/part4/run_eval.py --num_cpu_threads 8     # limit CPU threads (default: 16)

Output:
    eval_outputs.jsonl  (one JSON object per prompt)
"""

import argparse
import gc
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(BASE))           # workspace root

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_MODEL      = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER = os.path.join(ROOT, "lora_adapter")
DEFAULT_OUTPUT  = os.path.join(ROOT, "eval_outputs.jsonl")
MAX_NEW_TOKENS  = 256

# ── Evaluation prompts ────────────────────────────────────────────────────────

EVAL_PROMPTS = [
    # ── 10 provided by the assignment ────────────────────────────────────────
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
    # ── 10 custom evaluation inputs ───────────────────────────────────────────
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
]


# ── Inference ─────────────────────────────────────────────────────────────────

def generate_response(model, tokenizer, prompt: str) -> str:
    """Greedy generation using the model's chat template."""
    messages  = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs    = tokenizer(formatted, return_tensors="pt")
    device    = next(model.parameters()).device
    inputs    = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    new_ids = output_ids[0, input_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def load_model(base_model: str, adapter_path: str | None, use_gpu: bool):
    """Load the base model, optionally wrapping it with a LoRA adapter."""
    dtype = torch.float16 if use_gpu else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, use_fast=True, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto" if use_gpu else None,
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    if not use_gpu:
        model = model.to("cpu")
    model.eval()
    return model, tokenizer


def run_model(label: str, base_model: str, adapter_path: str | None,
              use_gpu: bool) -> dict:
    """Run all eval prompts with one model configuration; return {prompt: output}."""
    print(f"\n── {label} {'─' * (60 - len(label))}")
    model, tokenizer = load_model(base_model, adapter_path, use_gpu)
    outputs = {}
    for prompt in EVAL_PROMPTS:
        print(f"  {prompt[:70]}…")
        outputs[prompt] = generate_response(model, tokenizer, prompt)
    del model, tokenizer
    gc.collect()
    if use_gpu:
        torch.cuda.empty_cache()
    return outputs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",           choices=["base", "finetuned", "both"],
                        default="both")
    parser.add_argument("--adapter",         default=DEFAULT_ADAPTER,
                        help="Path to the saved LoRA adapter directory")
    parser.add_argument("--output",          default=DEFAULT_OUTPUT)
    parser.add_argument("--num_cpu_threads", type=int, default=16,
                        help="Max CPU threads for PyTorch (default: 16)")
    args = parser.parse_args()

    torch.set_num_threads(args.num_cpu_threads)
    os.environ["OMP_NUM_THREADS"] = str(args.num_cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.num_cpu_threads)

    use_gpu = torch.cuda.is_available()
    print(f"Device: {'cuda' if use_gpu else 'cpu'}  |  CPU threads: {args.num_cpu_threads}")

    base_outputs:      dict = {}
    finetuned_outputs: dict = {}

    if args.model in ("base", "both"):
        base_outputs = run_model(
            "Base model", BASE_MODEL, adapter_path=None, use_gpu=use_gpu
        )

    if args.model in ("finetuned", "both"):
        finetuned_outputs = run_model(
            "Fine-tuned model", BASE_MODEL, adapter_path=args.adapter, use_gpu=use_gpu
        )

    # ── Write output ──────────────────────────────────────────────────────────
    records = [
        {
            "prompt":           prompt,
            "base_output":      base_outputs.get(prompt, ""),
            "finetuned_output": finetuned_outputs.get(prompt, ""),
            "notes":            "",
        }
        for prompt in EVAL_PROMPTS
    ]

    with open(args.output, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(records)} records to: {args.output}")


if __name__ == "__main__":
    main()
