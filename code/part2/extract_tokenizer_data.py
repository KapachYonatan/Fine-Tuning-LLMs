"""
extract_tokenizer_data.py
Run this ONCE from the terminal (not in Jupyter) to precompute all tokenizer
data and write tokenizer_data.json.  The notebook then loads only the JSON.
"""
import json, os, sys

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(BASE))  # workspace root (two levels up from code/part2/)

ENGLISH_CORPUS = open(os.path.join(ROOT, "docs", "prague_text_en.txt"), encoding="utf-8").read().strip()
HEBREW_CORPUS  = open(os.path.join(ROOT, "docs", "prague_text_he.txt"), encoding="utf-8").read().strip()
CHESS_TEXT     = open(os.path.join(ROOT, "docs", "english_text_part2.txt"), encoding="utf-8").read().strip()

SPANS = ["Gukesh Dommaraju", "The World Chess Championship"]

MODEL_IDS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
    "allenai/OLMo-2-1124-7B-Instruct",
    "ibm-granite/granite-3.3-8b-instruct",
    "deepseek-ai/DeepSeek-V3",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "microsoft/Phi-4-mini-instruct",
    "tiiuae/Falcon3-7B-Instruct",
    "dicta-il/dictalm2.0-instruct",
]

def detect_boundary_strategy(tok):
    ids = tok.encode("hello world test", add_special_tokens=False)
    toks = tok.convert_ids_to_tokens(ids)
    for t in toks[1:]:
        if isinstance(t, str):
            if t.startswith("\u0120"):   # Ġ — GPT-2/BPE
                return "space_prefix (\u0120)"
            if t.startswith("\u2581"):   # ▁ — SentencePiece
                return "space_prefix (\u2581)"
    for t in toks[1:]:
        if isinstance(t, str) and t.startswith(" "):
            return "space_in_token (tiktoken-style)"
    return "byte_level / other"

def avg_tokens_per_word(tok, text):
    n_tokens = len(tok.encode(text, add_special_tokens=False))
    n_words  = len(text.split())
    return round(n_tokens / n_words, 2)

from transformers import AutoTokenizer

data = {}
for mid in MODEL_IDS:
    print(f"Loading {mid} ...", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(
            mid, use_fast=True, trust_remote_code=True, token=True
        )
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)
        continue

    special = tok.all_special_tokens
    vocab   = tok.get_vocab()

    avg_en  = avg_tokens_per_word(tok, ENGLISH_CORPUS)
    avg_he  = avg_tokens_per_word(tok, HEBREW_CORPUS)

    span_toks = {}
    for span in SPANS:
        span_toks[span] = list(tok.tokenize(span))

    data[mid] = {
        "tokenizer_class":              type(tok).__name__,
        "vocab_size":                   tok.vocab_size,
        "vocab_size_full":              len(vocab),
        "boundary_strategy":            detect_boundary_strategy(tok),
        "has_byte_tokens":              any("<0x" in t for t in vocab),
        "special_token_count":          len(special),
        "special_token_examples":       special[:6],
        "avg_tokens_per_english_word":  avg_en,
        "avg_tokens_per_hebrew_word":   avg_he,
        "he_en_ratio":                  round(avg_he / avg_en, 2),
        "total_chess_tokens":           len(tok.encode(CHESS_TEXT, add_special_tokens=False)),
        "span_tokenizations":           span_toks,
    }
    print(f"  OK  class={type(tok).__name__}  vocab={tok.vocab_size}", flush=True)

out_path = os.path.join(BASE, "tokenizer_data.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\nWrote {out_path}  ({len(data)}/{len(MODEL_IDS)} models)")
