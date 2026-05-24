Here is the architectural information for the **SmolLM2-1.7B-Instruct** model, extracted directly from the provided sources.

**1. Number of layers, width of layers**
*   **Number of layers:** The model features **24 hidden layers** (Source 1).
*   **Width of layers:** The hidden dimension size (width) is **2048** (Source 1).

**2. Number of attention heads**
*   **Query Heads:** The model uses **32 attention heads** per layer (Source 1).
*   **Key-Value Heads:** The model also uses **32 key-value heads**, meaning it utilizes standard Multi-Head Attention rather than Grouped-Query Attention (Source 1).

**3. Sizes of everything**
*   **MLP layers:** The Feed-Forward Network (MLP) layers feature an intermediate dimension size of **8192** (Source 1).
*   **MoE layers:** The sources do **not mention** Mixture of Experts (MoE) layers for this model. The configuration indicates a standard dense Transformer structure (`LlamaForCausalLM`) (Source 1).
*   **Attention components:** The specific dimensions for the individual attention components (such as the exact head dimension size) are **missing** from the provided text (though mathematically it can be inferred as 64 by dividing the hidden size by the number of heads).
*   **Embeddings and Unembedding layers:** The vocabulary size is **49,152** (Source 1, Source 2). Additionally, the model utilizes **tied word embeddings** (`"tie_word_embeddings": true`), which means the weights for the input embedding matrix and the final output language model head matrix are shared (Source 1, Source 2).

**4. Position encoding**
*   **Method:** The model utilizes **Rotary Positional Embeddings (RoPE)**, which is indicated by the presence of `rope_theta` and `rope_scaling` in its configuration (Source 1). 
*   **Max position supported:** There is a direct **disagreement** between the sources regarding the maximum context window. Source 1 (the raw `config.json`) states that the `max_position_embeddings` is **8192** tokens. Conversely, Source 2 explicitly states that the context length for the 1.7B model is **2,048** tokens.
*   **Hyper-parameters:** The base `rope_theta` is set to **130,000**, and the `rope_scaling` hyper-parameter is set to **null**, meaning no additional context-length scaling factors are applied out-of-the-box (Source 1).

**5. Activations**
*   **Type of activation:** The model uses the **SiLU** (Swish Gated Linear Unit) activation function (Source 1).
*   **Consistency:** The configuration specifies `"hidden_act": "silu"` without mentioning any mixing of different activation types, indicating it is applied consistently across the feed-forward layers (Source 1).

**6. Normalizations**
*   **Type of normalization:** The model uses **RMSNorm** with an epsilon (`rms_norm_eps`) of **1e-05** (Source 1).
*   **Components and Placement:** The exact placement of the normalizations (e.g., whether the model relies on a pre-norm or post-norm configuration) is **missing** from the provided source texts.

**7. Any other property of interest**
*   **Parameter Count:** The model has exactly **1.7 Billion parameters** (Source 2, Source 3).
*   **Pre-training Data Scale:** The base model was pre-trained on a massive **11 Trillion tokens**. The training corpus included high-quality educational web content (FineWeb-Edu), DCLM, The Stack, Cosmopedia v2, and specialized math and coding datasets (Source 2, Source 3).
*   **Post-Training / Alignment:** To create the "Instruct" version, the model underwent Supervised Fine-Tuning (SFT) on a mix of public and custom datasets, followed by Direct Preference Optimization (DPO) utilizing UltraFeedback (Source 3).
*   **Compute Hardware:** Training was conducted in bfloat16 precision using 256 NVIDIA H100 GPUs and the Nanotron distributed training framework (Source 2, Source 3).

***

**Indexed List of Sources:**
*   Source 1: <a href="https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/blob/main/config.json">config.json · HuggingFaceTB/SmolLM2-1.7B-Instruct at main - Hugging Face</a>
*   Source 2: <a href="https://collabnix.com/hugging-face-small-language-model-a-complete-guide/">Hugging Face Small Language Model: A Complete Guide - Collabnix</a>
*   Source 3: <a href="https://mirai.us/models/SmolLM2-1.7B-Instruct">SmolLM2-1.7B-Instruct from HuggingFace – Run On-Device with Mirai.</a>