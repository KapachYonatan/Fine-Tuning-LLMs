Here is the architectural information for the **microsoft/Phi-4-mini-instruct** model, extracted directly from the provided sources.

**1. Number of layers, width of layers**
*   **Number of layers:** The model features **32 hidden layers** (Source 2).
*   **Width of layers:** The hidden dimension (width or `hidden_size`) is **3072** (Source 2).

**2. Number of attention heads**
*   **Query Heads:** The model uses **24 attention heads** per layer (Source 2).
*   **Key-Value Heads:** The model utilizes Grouped-Query Attention (GQA) and has **8 key-value heads** (Source 2, Source 3, Source 4).

**3. Sizes of everything**
*   **MLP layers:** The Feed-Forward Network (MLP) layers feature an intermediate dimension size of **8192** (Source 2).
*   **MoE layers:** The model does **not** have Mixture of Experts (MoE) layers; it is built on a standard dense decoder-only Transformer architecture (Source 1, Source 3, Source 4).
*   **Embeddings and Unembedding layers:** The vocabulary size is notably large at **200,064 tokens** (Source 2, Source 3, Source 4). Additionally, the model utilizes **shared embeddings** (`"tie_word_embeddings": true`), meaning the weights for the input embedding matrix and the final output unembedding matrix are tied (Source 1, Source 2, Source 3, Source 4).
*   **Attention components:** The explicit head dimension ($d_h$) is **missing** from the provided text, but it can mathematically be inferred as 128 by dividing the hidden size by the number of query heads (3072 / 24).

**4. Position encoding**
*   **Method:** The model utilizes a specific Rotary Positional Embeddings (RoPE) implementation called **LongRoPE** (`"type": "longrope"`) (Source 2). It also applies a `partial_rotary_factor` of **0.75**, meaning RoPE is only applied to 75% of the head dimension (Source 2). 
*   **Max position supported:** The model supports an extended context window of **128,000 (128K) tokens** (Source 1, Source 3, Source 4). The exact `max_position_embeddings` configured in the code is **131,072** (Source 2). 
*   **Hyper-parameters:** The model's LongRoPE implementation uses a base `rope_theta` of **10,000.0** and an `original_max_position_embeddings` of **4096** (Source 2). To extrapolate to 128k, it utilizes a massive `rope_scaling` dictionary that contains two fields: a `short_factor` array consisting entirely of 1.0s, and a `long_factor` array with 48 dynamically scaling multipliers (ranging from 1.0 to 47.77) (Source 2).

**5. Activations**
*   **Type of activation:** The model uses the **SiLU** activation function (Source 2).
*   **Consistency:** The configuration simply denotes `"hidden_act": "silu"` without mentioning any mixture of activation types, indicating it is applied consistently across all layers (Source 2).

**6. Normalizations**
*   **Type of normalization:** The model uses **RMSNorm** with an epsilon (`rms_norm_eps`) of **1e-05** (Source 2).
*   **Components and Placement:** The exact placement of the normalizations (e.g., whether it uses a pre-norm or post-norm configuration) is **missing** from the provided texts.

**7. Any other property of interest**
*   **Parameter Count:** The model features exactly **3.8 Billion parameters** (Source 1, Source 3, Source 4).
*   **Bias and Dropout:** The model relies on zero dropout (`attention_dropout`, `embd_pdrop`, and `resid_pdrop` are all **0.0**). Additionally, it operates without attention or MLP biases (`"attention_bias": false`, `"mlp_bias": false`) (Source 2).
*   **Training Data:** It was trained on **5 Trillion tokens** consisting of high-quality educational data, code, synthetic "textbook-like" data, and heavily filtered public documents with a knowledge cutoff of **June 2024** (Source 3, Source 4). 
*   **Alignment:** The model underwent Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Reinforcement Learning from Human Feedback (RLHF) to enforce strict safety boundaries and enhance instruction adherence (Source 1, Source 3, Source 4). 

***

**Indexed List of Sources:**
*   Source 1: <a href="https://catalog.models.ai.microsoft.com/">Phi-4-mini-instruct - AI Model Catalog | Microsoft Foundry Models</a>
*   Source 2: <a href="https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/main/config.json">config.json · microsoft/Phi-4-mini-instruct at main - Hugging Face</a>
*   Source 3: <a href="https://build.nvidia.com/microsoft/phi-4-mini-instruct">microsoft / phi-4-mini-instruct - NVIDIA API Documentation</a>
*   Source 4: <a href="https://huggingface.co/microsoft/Phi-4-mini-instruct">microsoft/Phi-4-mini-instruct - Hugging Face</a>