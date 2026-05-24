Here is the architectural information for the **ibm-granite/granite-3.3-8b-instruct** model, extracted directly from the provided sources. 

*(Note: Because the instruct version is directly fine-tuned from the base model without altering the underlying network geometry, the precise architectural parameters below are extracted from the documentation of its base model, `granite-3.3-8b-base`)*.

**1. Number of layers, width of layers**
*   **Number of layers:** The model features **40 layers** (Source 1).
*   **Width of layers:** The hidden dimension (embedding size) of the model is **4096** (Source 1).

**2. Number of attention heads**
*   **Query Heads:** The model uses **32 attention heads** per layer (Source 1).
*   **Key-Value Heads:** The model utilizes Grouped-Query Attention (GQA) with **8 Key-Value (KV) heads** (Source 1).

**3. Sizes of everything**
*   **MLP layers:** The Feed-Forward Network (MLP) layers feature a hidden size of **12,800** (Source 1).
*   **MoE layers:** The model is a **dense** decoder-only transformer, meaning it does **not** feature Mixture of Experts (MoE) layers (Source 1).
*   **Attention components:** The attention head dimension size is **128** (Source 1).
*   **Embeddings and Unembedding layers:** The model uses **shared input/output embeddings** (Source 1). However, the exact sizes of the vocabulary and the embedding matrices are **missing** from the provided text.

**4. Position encoding**
*   **Method:** The model utilizes **Rotary Positional Embeddings (RoPE)** (Source 1).
*   **Max position supported:** The model supports a massive **128,000-token (128K)** context window (Source 1, Source 2, Source 3).
*   **Hyper-parameters:** The specific base/theta values or potential scaling factors for the RoPE implementation are **missing** from the provided sources.

**5. Activations**
*   **Type of activation:** The model uses the **SwiGLU** activation function (Source 1).
*   **Consistency:** The source specifies "MLP activation: SwiGLU" without mentioning any mixture of activation types, indicating it is applied consistently across all layers (Source 1).

**6. Normalizations**
*   **Type of normalization:** The model utilizes **RMSNorm** (Source 1).
*   **Components and Placement:** The exact placement of the normalizations (i.e., whether it uses a pre-norm or post-norm configuration) and its epsilon value are **missing** from the provided texts.

**7. Any other property of interest**
*   **Parameter Count:** The model has exactly **8.1 Billion total parameters**, all of which are active during inference (Source 1).
*   **Training Data Scale:** The base model was pretrained on **12 Trillion tokens** (Source 1). 
*   **Structured Reasoning:** The instruct model is uniquely optimized to support structured reasoning out-of-the-box by utilizing `<think>` and `<response>` tags, allowing it to clearly separate its internal thought traces from the final user output (Source 2).
*   **Fill-in-the-Middle (FIM):** The model natively supports Fill-in-the-Middle code completions using specialized tokens to generate content conditioned on both a prefix and a suffix (Source 1).
*   **Multilingual Capabilities:** The model supports 12 languages (English, German, Spanish, French, Japanese, Portuguese, Arabic, Czech, Italian, Korean, Dutch, and Chinese) (Source 2).

***

**Indexed List of Sources:**
*   Source 1: <a href="https://huggingface.co/ibm-granite/granite-3.3-8b-base">ibm-granite/granite-3.3-8b-base - Hugging Face</a>
*   Source 2: <a href="https://build.nvidia.com/ibm/granite-3-3-8b-instruct">granite-3.3-8b-instruct Model by IBM | NVIDIA NIM</a>
*   Source 3: <a href="https://navigator.ufl.edu/">Granite 3.3 8B | NaviGator AI - University of Florida</a>