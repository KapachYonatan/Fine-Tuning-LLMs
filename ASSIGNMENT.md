# Assignment 2: Architectural Choices, Tokenizers, Decoding, and Fine Tuning

---

## Part 1 — Architectural Choices

### Models to Analyze (all 10)

| # | Model ID |
|---|----------|
| 1 | `meta-llama/Llama-3.1-8B-Instruct` |
| 2 | `mistralai/Mistral-7B-Instruct-v0.3` |
| 3 | `Qwen/Qwen2.5-7B-Instruct` |
| 4 | `allenai/OLMo-2-1124-7B-Instruct` |
| 5 | `ibm-granite/granite-3.3-8b-instruct` |
| 6 | `deepseek-ai/DeepSeek-V3` |
| 7 | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |
| 8 | `microsoft/Phi-4-mini-instruct` |
| 9 | `tiiuae/Falcon3-7B-Instruct` |
| 10 | `dicta-il/dictalm2.0-instruct` |

### 1.1 Extract Architectural Information (per model)

For each model, extract:
1. Number of layers, width of layers
2. Number of attention heads
3. Sizes of everything:
   - MLP layer dimensions (if applicable)
   - MoE details: number of experts and dimensions involved (if applicable)
   - Dimensions of all attention components (Q, K, V, output projections)
   - Sizes of embedding and unembedding layers
4. Position encoding: method used, max position supported (if any), hyper-parameters and their values
5. Activation functions used (same everywhere or mixed?)
6. Normalization: type (LayerNorm, RMSNorm, etc.), placement (pre-norm, post-norm, other), which components
7. Any other interesting property

> **Note:** For each property, record **where** you extracted it from and **how** (e.g., `config.json`, model card, code inspection).

### 1.2 Required Output File: `architecture.csv`

One row per model, columns:

```
model_id,hidden_size,num_layers,num_attention_heads,num_kv_heads,mlp_size,activation,norm_type,position_encoding,context_length,vocab_size,moe_details
```

Use `NA` for fields that do not apply or cannot be found.

### 1.3 Report Contents (Architectural Choices section)

1. Short explanation of how architectural information was extracted
2. Any uncertainty, missing fields, or disagreements between sources
3. Readable summary of the main architectural differences between models
4. Reflection and analysis (see below)

### 1.4 Reflection and Analysis

Inspect the extracted data and identify general trends:
- Are there choices consistent across all/most models?
- Are there choices where there is no consensus?
- Are some models deviating from the norm?
- Are there ratios or rules of thumb (e.g., "size X is often k times size Y", "size X is often a power of 10")?
- **If you were to build a new model, what settings would you choose?**

---

## Part 2 — Tokenizers

Consider the tokenizers of all 10 models listed above.

### 2.1 Architectural Choices (per tokenizer)

- How many tokens are in the tokenizer (vocab size)?
- What strategy is used to indicate that a token is not an incomplete word? How are tokens supposed to merge back to words?
- Are there tokens that do not correspond to words or parts of words? How many, what do they encode, and how are they represented?
- For each tokenizer, compute or estimate the **average number of tokens per word** for:
  - **English**
  - **Hebrew**
  - Explain exactly how you computed/estimated this and what assumptions the method relies on.
  - Explain how you counted "words."

> Answers that repeat across models: it is ok to respond with "the same as in model X."

### 2.2 Required Output File: `tokenizers.csv`

One row per model, columns:

```
model_id,tokenizer_type,vocab_size,special_tokens,word_boundary_strategy,byte_fallback_or_byte_level,avg_tokens_per_english_word,avg_tokens_per_hebrew_word
```

Use `NA` for fields that do not apply or cannot be found.

### 2.3 Tokenization Differences Example

- Find an **English text** that is tokenized **differently by at least 3 models**.
- Provide: the text, the tokenizations by those 3 models, and a short description of the likely cause.
- How many of the remaining 7 tokenizers agree with each of the 3 tokenizers on this text?

### 2.4 Report Contents (Tokenizers section)

1. Short explanation of how the tokenizers were inspected
2. How the average-tokens-per-word numbers were computed/estimated
3. How words were counted
4. The tokenization-differences example
5. Discussion of average-tokens-per-word numbers and what they mean

---

## Part 3 — Constrained Decoding

### 3.1 Get Basic Inference Working

- Load `Qwen/Qwen2.5-7B-Instruct` and `mistralai/Mistral-7B-Instruct-v0.3`
- Verify that you can query them with English prompts and obtain reasonable answers
- **Important:** These are instruction-tuned models — use the tokenizer's chat template or the model card's recommended prompting format (user/assistant turns); otherwise the model may behave worse than expected

### 3.2 Identify Hebrew-Related Tokens

- For each model, identify all tokenizer tokens that **may participate in Hebrew text**
  - Include: Hebrew characters, numbers, punctuation
  - Exclude: words/tokens from other languages
- Describe the strategy used for identifying these tokens (in the report)

**Submit two JSON files:**

`hebrew_allowed_tokens_qwen.json`
`hebrew_allowed_tokens_mistral.json`

Each with this structure:
```json
{
  "model_id": "...",
  "allowed_token_ids": [1, 2, 3]
}
```

### 3.3 Constrained Decoding Implementation

- Modify the inference procedure to perform simple constrained decoding: **only allow tokens from the Hebrew-allowed set to be generated**
- Run the English queries (from 3.1, or others) using the constrained decoder on both models

### 3.4 Required Output File: `decoding_outputs.jsonl`

Each line is a JSON object:
```json
{"prompt":"Explain why leaves are green.","model":"Qwen/Qwen2.5-7B-Instruct","unconstrained_output":"...","constrained_output":"..."}
```

### 3.5 Report Contents (Constrained Decoding section)

1. How constrained decoding was implemented (brief description)
2. 10 English language queries, and for each query and each model:
   - (a) Unconstrained decoding response
   - (b) Constrained decoding response
3. Brief discussion of results:
   - Is the Hebrew text meaningful?
   - Does it relate to the question?
   - Does it differ between models?
   - Are some languages favored over others?

---

## Part 4 — Fine Tuning

**Goal:** Fine-tune `Qwen/Qwen2.5-1.5B-Instruct` so that it answers in Hebrew on English queries.  
The answers must be **relevant to the query** (not just a fixed Hebrew sentence like "לא יודע").  
The model should at least appear to attempt to answer the prompt, even if incorrectly.

### 4.1 Data

- Find or create training data (English prompt → Hebrew answer pairs)
- Options for creating data:
  - Find existing relevant data
  - Create from scratch
  - Modify existing data (e.g., translate answers to Hebrew)
  - Use an LLM to help with transformations or data generation
- **Training data must NOT include any of the 20 evaluation inputs listed below**

### 4.2 Fine-Tuning

- Can use full fine-tuning or lightweight methods like **LoRA** (recommended)
  - Main LoRA library in HuggingFace ecosystem: `peft`
- Technical tips:
  1. LoRA is recommended over full fine-tuning
  2. First check that regular inference works before training
  3. Try to overfit a tiny dataset of 1–2 examples as a sanity check
  4. Use the model's **chat template consistently** for both training and inference
  5. Make sure training examples match the actual task: English prompt in, Hebrew answer out

### 4.3 Evaluation — "Did It Work?"

Run both the **original (base) model** and the **fine-tuned model** on:

**10 provided English evaluation inputs:**
1. Explain why the sky looks blue during the day.
2. Give two advantages and two disadvantages of public transportation.
3. Write a short email asking a professor for an extension on an assignment.
4. Describe how to make a simple omelette.
5. What is the difference between supervised and unsupervised learning?
6. Summarize the story of Cinderella in three sentences.
7. Suggest three ways to reduce smartphone distraction while studying.
8. Explain what happens when water boils.
9. Give a polite refusal to an invitation to a party.
10. Turn the idea "practice makes progress" into advice for a student.

**Plus 10 additional English inputs of your own choice** (also not in training data).

### 4.4 Required Output File: `eval_outputs.jsonl`

Each line is a JSON object:
```json
{"prompt":"Explain why the sky looks blue during the day.","base_output":"...","finetuned_output":"...","notes":"..."}
```

### 4.5 Report Contents (Fine Tuning section)

- How the training data was created
- How the model was fine-tuned (method, hyperparameters, etc.)
- What evaluation was run
- Whether fine-tuning worked, and why you think so
- Outputs on all 20 inputs with explanation of what changed after fine-tuning
- Optional: quantitative summary (e.g., % of answers in Hebrew, manual relevance score)
- **Both** anecdotal before/after examples and a constructed evaluation set with metrics are required

---

## What to Submit

### Required Files

| File | Description |
|------|-------------|
| `report.pdf` | Full report (PDF) covering all sections. Must include names and ID numbers prominently at the top. |
| `architecture.csv` | Structured architectural facts, one row per model |
| `tokenizers.csv` | Structured tokenizer facts, one row per model |
| `hebrew_allowed_tokens_qwen.json` | Hebrew-allowed token IDs for Qwen |
| `hebrew_allowed_tokens_mistral.json` | Hebrew-allowed token IDs for Mistral |
| `decoding_outputs.jsonl` | Constrained/unconstrained decoding results |
| `eval_outputs.jsonl` | Before/after fine-tuning generation results on 20 inputs |
| Code | All code used: token identification, constrained decoding, data creation/processing, fine-tuning, evaluation |
| Training data | If it fits in Moodle; if >1000 examples, submit the first 1000; if too large, submit a representative sample + generation script |

### Do NOT Submit

- Full model weights
- LoRA adapter is optional (submit if small enough)

### Report Quality Note

> "Large part of your grade will be based on the report quality and clarity: the report is the main window we have into your work."

The report must be easy to navigate, with clearly labeled sections matching the assignment structure.

---

## Key Constraints Summary

- Training data must **not** include any of the 20 evaluation inputs (10 provided + 10 self-chosen)
- For constrained decoding: tokens from other languages (non-Hebrew) must be excluded
- For tokenizer avg-tokens-per-word: must explain computation method and assumptions clearly
- The fine-tuned model must answer in Hebrew in a **query-relevant** way
