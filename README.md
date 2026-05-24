# Assignment 2: Architectural Choices, Tokenizers, Decoding, and Fine Tuning

## Structure

```
architecture.csv              Part 1 output — one row per model
tokenizers.csv                Part 2 output — one row per model
hebrew_allowed_tokens_*.json  Part 3 output — Hebrew-eligible token IDs per model
decoding_outputs.jsonl        Part 3 output — unconstrained + constrained responses
eval_outputs.jsonl            Part 4 output — base vs. fine-tuned responses

code/
  part1/
    part1_architectural_choices.ipynb   Analysis + writes architecture.csv
  part2/
    extract_tokenizer_data.py           Heavy script — run once from terminal
    tokenizer_data.json                 Precomputed tokenizer data (generated)
    part2_tokenizers.ipynb              Display notebook — loads tokenizer_data.json
  part3/
    run_part3.py                        Heavy script — run once from terminal
    part3_constrained_decoding.ipynb    Display notebook — loads JSON/JSONL outputs
  part4/
    (to be added)

docs/
  prague_text_en.txt     English corpus used for avg-tokens-per-word (Part 2)
  prague_text_he.txt     Hebrew corpus used for avg-tokens-per-word (Part 2)
  english_text_part2.txt Tokenization-differences example text (Part 2)
  model_architectures/   Per-model architecture notes (Part 1 reference)

report/
  report.tex             LaTeX source
```

## How to Run

### Part 1 — Architectural Choices
Open and run `code/part1/part1_architectural_choices.ipynb`.  
Writes `architecture.csv` to the workspace root.

### Part 2 — Tokenizers
```bash
python code/part2/extract_tokenizer_data.py
```
Then open and run `code/part2/part2_tokenizers.ipynb`.  
Writes `tokenizers.csv` to the workspace root.

### Part 3 — Constrained Decoding
```bash
python code/part3/run_part3.py
```
Writes `hebrew_allowed_tokens_qwen.json`, `hebrew_allowed_tokens_mistral.json`,
and `decoding_outputs.jsonl` to the workspace root.  
Then open and run `code/part3/part3_constrained_decoding.ipynb` to display results.

> **Note:** Loads 7B models in float16. Requires ~14 GB RAM and takes several
> minutes per query on CPU. Close other applications before running.

### Part 4 — Fine Tuning
*(to be added)*

## Dependencies

```
torch
transformers
accelerate
peft          # Part 4 (LoRA)
pandas
```

Install with:
```bash
pip install torch transformers accelerate peft pandas
```

A HuggingFace token (`token=True`) is required for gated models (Llama, Mistral).
Log in with `huggingface-cli login` before running any script.
