"""
train.py — Part 4: Fine-tune Qwen/Qwen2.5-1.5B-Instruct with LoRA

Applies LoRA to the attention projections, trains on (English prompt →
Hebrew answer) pairs, and saves the adapter weights.

Usage:
    python code/part4/train.py                          # full training run
    python code/part4/train.py --overfit                # sanity check (2 examples, 20 steps)
    python code/part4/train.py --num_cpu_threads 8      # limit CPU threads (default: 16)
    python code/part4/train.py --data path/to/data.jsonl

Output:
    lora_adapter/   (adapter weights + tokenizer config; no base weights)
"""

import argparse
import json
import os
import warnings

import torch

# Suppress deprecation noise from bitsandbytes registering Enum types with
# torch's pytree system (triggered transitively via peft).
warnings.filterwarnings(
    "ignore",
    message=".*register_constant.*Enum subclass.*",
    category=UserWarning,
)
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


class DataCollatorForCompletionOnlyLM:
    """Masks labels so the loss is computed only on the assistant response.

    Tokens before (and including) the *last* occurrence of ``response_template``
    in each sequence are set to ``ignore_index`` (-100), so the model learns
    only to predict the assistant's reply.
    """

    def __init__(self, response_template, tokenizer, ignore_index: int = -100):
        self.response_template = (
            response_template
            if isinstance(response_template, list)
            else tokenizer.encode(response_template, add_special_tokens=False)
        )
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    def __call__(self, features):
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        tpl, tpl_len = self.response_template, len(self.response_template)

        for i in range(len(labels)):
            ids = labels[i].tolist()
            last_match = -1
            for j in range(len(ids) - tpl_len + 1):
                if ids[j : j + tpl_len] == tpl:
                    last_match = j
            if last_match == -1:
                labels[i] = torch.full_like(labels[i], self.ignore_index)
            else:
                labels[i, : last_match + tpl_len] = self.ignore_index

        # Mask padding tokens
        if self.tokenizer.pad_token_id is not None:
            labels[batch["input_ids"] == self.tokenizer.pad_token_id] = self.ignore_index

        batch["labels"] = labels
        return batch

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(BASE))           # workspace root

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_MODEL     = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_DATA   = os.path.join(ROOT, "data", "train_data.jsonl")
DEFAULT_OUTPUT = os.path.join(ROOT, "lora_adapter")

# Qwen2.5 chat format: assistant turn always starts with this header.
# We use token IDs (not a raw string) for reliable matching inside the collator.
ASSISTANT_HEADER = "<|im_start|>assistant\n"


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_example(example: dict, tokenizer) -> str:
    """Format a (prompt, response) pair with the model's chat template.

    The full turn is formatted so the collator can mask the user portion
    and compute loss only on the assistant (Hebrew) response.
    """
    messages = [
        {"role": "user",      "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",            default=DEFAULT_DATA,
                        help="Path to train_data.jsonl")
    parser.add_argument("--output",          default=DEFAULT_OUTPUT,
                        help="Directory to save the LoRA adapter")
    parser.add_argument("--overfit",         action="store_true",
                        help="Sanity check: use 2 examples, 20 steps")
    parser.add_argument("--num_cpu_threads", type=int, default=16,
                        help="Max CPU threads for PyTorch (default: 16)")
    args = parser.parse_args()

    # ── Thread limits (respected on CPU and alongside GPU BLAS) ───────────────
    torch.set_num_threads(args.num_cpu_threads)
    os.environ["OMP_NUM_THREADS"] = str(args.num_cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.num_cpu_threads)

    use_gpu = torch.cuda.is_available()
    device  = "cuda" if use_gpu else "cpu"
    print(f"Device: {device}  |  CPU threads: {args.num_cpu_threads}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"\nLoading data: {args.data}")
    records = load_jsonl(args.data)
    if args.overfit:
        records = records[:2]
        print(f"  --overfit: using {len(records)} examples for 20 steps")
    else:
        print(f"  {len(records)} examples loaded")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"Loading model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16 if use_gpu else torch.float32,
        device_map="auto"  if use_gpu else None,
        trust_remote_code=True,
    )
    if not use_gpu:
        model = model.to("cpu")

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────────
    texts   = [format_example(r, tokenizer) for r in records]
    dataset = Dataset.from_dict({"text": texts})

    # ── Label masking: compute loss only on the assistant's Hebrew response ────
    # Encode the assistant header to token IDs so the collator can find it
    # reliably (special tokens like <|im_start|> are single token IDs in Qwen).
    response_template_ids = tokenizer.encode(
        ASSISTANT_HEADER, add_special_tokens=False
    )
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=args.output,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,      # effective batch size = 8
        num_train_epochs=3 if not args.overfit else 1,
        max_steps=20     if args.overfit    else -1,
        learning_rate=2e-4,
        fp16=use_gpu,
        bf16=False,
        logging_steps=5,
        save_steps=50,
        save_total_limit=1,
        report_to="none",
        dataset_text_field="text",
        max_length=512,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    print("\nStarting training …")
    trainer.train()

    # ── Save adapter ──────────────────────────────────────────────────────────
    print(f"\nSaving adapter to: {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
