"""
run_part3.py
Run this ONCE from the terminal (not in Jupyter) to:
  A) Identify Hebrew-eligible tokens for Qwen and Mistral and write two JSON files.
  B) Run constrained and unconstrained inference for 10 English queries and write
     decoding_outputs.jsonl.

Run from anywhere — all paths are computed relative to this file.

Output files written to the workspace root:
  hebrew_allowed_tokens_qwen.json
  hebrew_allowed_tokens_mistral.json
  decoding_outputs.jsonl

NOTE: Loading 7B models on CPU requires ~14 GB RAM (float16).
      Inference is slow on CPU — expect several minutes per query.
      Close other applications before running.
"""

import gc
import json
import os
import unicodedata

import torch

# ─────────────────────────────────────────────────────────────────────────────
# CPU core limit
# ─────────────────────────────────────────────────────────────────────────────
CPU_CORES = 16
os.environ["OMP_NUM_THREADS"]     = str(CPU_CORES)
os.environ["MKL_NUM_THREADS"]     = str(CPU_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_CORES)
torch.set_num_threads(CPU_CORES)
try:
    os.sched_setaffinity(0, set(range(CPU_CORES)))
except AttributeError:
    pass  # not available on all platforms
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList,
)

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(BASE))  # workspace root (two levels up from code/part3/)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    "Qwen/Qwen2.5-7B-Instruct":           os.path.join(ROOT, "hebrew_allowed_tokens_qwen.json"),
    "mistralai/Mistral-7B-Instruct-v0.3": os.path.join(ROOT, "hebrew_allowed_tokens_mistral.json"),
}

# 10 English evaluation queries (the 10 provided Part 4 evaluation inputs)
QUERIES = [
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
]

MAX_NEW_TOKENS = 300

# ─────────────────────────────────────────────────────────────────────────────
# A — Hebrew Token Identification
# ─────────────────────────────────────────────────────────────────────────────

# Hebrew Unicode ranges:
#   U+0590–U+05FF  Hebrew block (letters, niqqud, cantillation, punctuation,
#                  geresh U+05F3, gershayim U+05F4)
#   U+FB1D–U+FB4F  Hebrew Presentation Forms
_HEBREW_RANGES = [(0x0590, 0x05FF)]


def _is_hebrew_char(c: str) -> bool:
    cp = ord(c)
    return any(lo <= cp <= hi for lo, hi in _HEBREW_RANGES)


def token_may_be_hebrew(s: str) -> bool:
    """
    Return True if token string ``s`` is compatible with Hebrew text.

    Rejection rule: a character that (a) is a Unicode letter (general category
    starts with 'L') AND (b) does NOT fall inside a Hebrew Unicode range causes
    the token to be rejected.

    Punctuation characters (categories Po, Ps, Pe, Pd, …) are never tested
    against the letter-rejection condition, so ASCII apostrophe (') and double-
    quote (") as well as all standard punctuation pass through freely.
    Digits and whitespace also pass freely.
    """
    for c in s:
        if unicodedata.category(c).startswith("L") and not _is_hebrew_char(c):
            return False
    return True


# Accumulate token data for Section B
_allowed_ids_per_model: dict[str, list[int]] = {}

for model_id, out_path in MODELS.items():
    print(f"\n{'='*60}", flush=True)
    print(f"[A] Hebrew token identification: {model_id}", flush=True)
    print(f"{'='*60}", flush=True)

    print("  Loading tokenizer ...", flush=True)
    tok = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=True, token=True
    )

    vocab = tok.get_vocab()   # {token_str: token_id}
    vocab_size = len(vocab)
    print(f"  Vocab size (full): {vocab_size}", flush=True)

    # Use tok.decode([id]) so that byte-level fallback tokens (e.g. <0xD7>)
    # are resolved to their actual Unicode character before the Hebrew check.
    print("  Scanning vocabulary ...", flush=True)
    allowed_ids: list[int] = []
    for token_str, token_id in vocab.items():
        decoded = tok.decode([token_id])
        if token_may_be_hebrew(decoded):
            allowed_ids.append(token_id)

    # EOS must always be included so generation can terminate.
    eos_id = tok.eos_token_id
    if eos_id is not None and eos_id not in set(allowed_ids):
        allowed_ids.append(eos_id)
        print(f"  Force-added EOS token id={eos_id}", flush=True)

    allowed_ids.sort()
    print(f"  Allowed tokens: {len(allowed_ids):,} / {vocab_size:,}", flush=True)

    # Split allowed tokens into two buckets for a representative display:
    #   hebrew_ids  — tokens that contain at least one Hebrew character
    #   other_ids   — tokens that passed the filter but contain no Hebrew
    #                 characters (digits, punctuation, whitespace, EOS, …)
    hebrew_ids = []
    other_ids  = []
    for tid in allowed_ids:
        decoded = tok.decode([tid])
        if any(_is_hebrew_char(c) for c in decoded):
            hebrew_ids.append(tid)
        else:
            other_ids.append(tid)

    print(f"  Breakdown: {len(hebrew_ids):,} contain Hebrew chars, "
          f"{len(other_ids):,} are digits/punctuation/special", flush=True)

    # Sample ~20 Hebrew-bearing tokens spread evenly across the sorted list
    # so the sample is representative of the whole vocab range, not just the
    # lowest IDs.
    sample_size = 20
    if hebrew_ids:
        step = max(1, len(hebrew_ids) // sample_size)
        sampled = [tok.decode([hebrew_ids[i]]) for i in range(0, len(hebrew_ids), step)]
        sampled = sampled[:sample_size]
        print(f"  Sample Hebrew tokens ({len(sampled)} evenly spaced):", flush=True)
        print(f"    {[repr(t) for t in sampled]}", flush=True)
    else:
        print("  No Hebrew-character tokens found — check tokenizer.", flush=True)

    payload = {"model_id": model_id, "allowed_token_ids": allowed_ids}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  Wrote {out_path}", flush=True)

    _allowed_ids_per_model[model_id] = allowed_ids

    del tok
    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# B — Constrained Decoding: LogitsProcessor + Inference
# ─────────────────────────────────────────────────────────────────────────────

class HebrewOnlyLogitsProcessor(LogitsProcessor):
    """
    Sets logits of all tokens outside the Hebrew-allowed set to -inf.

    The boolean mask is built lazily on the first call so it always matches the
    actual vocabulary size reported by the model's logits tensor, regardless of
    any mismatch between tokenizer.vocab_size and model.config.vocab_size.
    The mask is also moved to the correct device on first use.
    """

    def __init__(self, allowed_ids: list[int]) -> None:
        self._allowed_ids_tensor = torch.tensor(allowed_ids, dtype=torch.long)
        self._mask: torch.Tensor | None = None
        self._mask_vocab_size: int = -1

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        vocab_size = scores.shape[-1]
        if vocab_size != self._mask_vocab_size:
            mask = torch.zeros(vocab_size, dtype=torch.bool)
            valid = self._allowed_ids_tensor[self._allowed_ids_tensor < vocab_size]
            mask[valid] = True
            self._mask = mask.to(scores.device)
            self._mask_vocab_size = vocab_size
        scores[:, ~self._mask] = float("-inf")
        return scores


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    allowed_ids: list[int] | None = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """
    Generate a response for ``prompt`` using the model's chat template.
    If ``allowed_ids`` is provided, a HebrewOnlyLogitsProcessor is applied
    (constrained decoding); otherwise generation is unconstrained.
    Greedy decoding (do_sample=False) is used for determinism.
    """
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    # Move inputs to the same device as the model to avoid cross-device errors
    # when device_map="auto" places the model on GPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs: dict = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    if allowed_ids is not None:
        processor = HebrewOnlyLogitsProcessor(allowed_ids=allowed_ids)
        gen_kwargs["logits_processor"] = LogitsProcessorList([processor])

    with torch.no_grad():
        output_ids = model.generate(**gen_kwargs)

    new_ids = output_ids[0, input_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ── Resumable: load any results already written in a previous run ─────────────
jsonl_path = os.path.join(ROOT, "decoding_outputs.jsonl")
results: list[dict] = []
if os.path.exists(jsonl_path):
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    print(f"\n[B] Resuming — loaded {len(results)} existing record(s) from {jsonl_path}", flush=True)

done_models = {rec["model"] for rec in results}

# Open in append mode so every record is written to disk immediately after
# generation.  If the script crashes mid-run the file already contains all
# records produced so far, and the next run will skip completed models.
jsonl_file = open(jsonl_path, "a", encoding="utf-8")

for model_id, token_json_path in MODELS.items():
    print(f"\n{'='*60}", flush=True)
    print(f"[B] Running inference: {model_id}", flush=True)
    print(f"{'='*60}", flush=True)

    if model_id in done_models:
        print("  Already complete — skipping.", flush=True)
        continue

    # Load allowed_ids from the JSON written in Section A (works even if this
    # script was restarted and Section A did not run in the current session).
    print(f"  Loading allowed token IDs from {token_json_path} ...", flush=True)
    with open(token_json_path, encoding="utf-8") as f:
        allowed_ids = json.load(f)["allowed_token_ids"]
    print(f"  Hebrew-allowed token count: {len(allowed_ids):,}", flush=True)

    if torch.cuda.is_available():
        print(
            "  Loading model (int8 quantization, device_map=auto) ...\n"
            "  NOTE: inference may take several minutes per query.",
            flush=True,
        )
        load_kwargs = dict(
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto",
        )
    else:
        print(
            "  Loading model (float16, device_map=cpu) ...\n"
            "  NOTE: CPU inference on a 7B model is slow — expect several minutes per query.\n"
            "  TIP: Run on Colab with a GPU for int8 quantization and much faster inference.",
            flush=True,
        )
        load_kwargs = dict(
            torch_dtype=torch.float16,
            device_map="cpu",
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **load_kwargs,
        trust_remote_code=True,
        token=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=True, token=True
    )
    print("  Model loaded.", flush=True)

    for i, query in enumerate(QUERIES, 1):
        print(f"\n  Query {i}/{len(QUERIES)}: {query[:70]}", flush=True)

        print("    Running unconstrained ...", flush=True)
        unconstrained = generate_response(model, tokenizer, query, allowed_ids=None)
        print(f"    Done ({len(unconstrained.split())} words).", flush=True)

        print("    Running constrained (Hebrew-only tokens) ...", flush=True)
        constrained = generate_response(model, tokenizer, query, allowed_ids=allowed_ids)
        print(f"    Done ({len(constrained.split())} words).", flush=True)

        rec = {
            "prompt": query,
            "model": model_id,
            "unconstrained_output": unconstrained,
            "constrained_output": constrained,
        }
        results.append(rec)
        jsonl_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jsonl_file.flush()

    print(f"\n  Finished {model_id}. Unloading model ...", flush=True)
    del model, tokenizer
    gc.collect()
    gc.collect()  # second pass catches any cyclic references missed by the first

    if torch.cuda.is_available():
        torch.cuda.synchronize()    # wait for all GPU ops to finish
        torch.cuda.empty_cache()    # release cached-but-free GPU memory back to the driver
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved()  / 1e9
        print(f"  GPU memory after unload: {allocated:.2f} GB allocated, "
              f"{reserved:.2f} GB reserved", flush=True)

jsonl_file.close()
print(f"\nWrote {jsonl_path}  ({len(results)} records total)")
