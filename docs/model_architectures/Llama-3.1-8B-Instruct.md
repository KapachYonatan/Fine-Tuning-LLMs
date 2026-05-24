Here is the architectural information for the **meta-llama/Llama-3.1-8B-Instruct** model, extracted directly from the provided sources. 

**1. Number of layers, width of layers**
*   **Number of layers:** The model features **32 layers** (Source 1).
*   **Width of layers:** There is a degree of **uncertainty/ambiguity** in the source text, which states the hidden size is **"typically 4096–5120"** without specifying the exact number for this specific 8B model (Source 1).

**2. Number of attention heads**
*   **Query Heads:** The model uses **32 attention heads per layer** (Source 1).
*   **Key-Value Heads:** The model utilizes **Grouped-Query Attention (GQA)** (Source 1, Source 2). However, the exact number of Key-Value (KV) heads is **missing** from the provided sources. 

**3. Sizes of everything**
*   **MLP layers:** The exact dimensions of the intermediate Feed-Forward Network (MLP) layers are **missing** from the sources.
*   **MoE layers:** The model does **not feature Mixture of Experts (MoE) layers**; it is built on a standard **dense decoder-only Transformer** architecture (Source 1).
*   **Attention components:** The specific dimensions for the attention heads (e.g., head dimension size) are **missing** from the provided text.
*   **Embeddings and Unembedding layers:** The exact sizes of the vocabulary and the embedding/unembedding matrices are **missing** from the provided sources.

**4. Position encoding**
*   **Method:** The model handles position encoding using **Rotary Positional Embeddings (RoPE)** (Source 1).
*   **Max position supported:** There is a clear **disagreement** between the sources regarding the maximum context window. Source 1 states the tokenizer supports up to **8,192 input tokens**. Conversely, Source 2 explicitly lists the context length as a massive **128k tokens**.
*   **Hyper-parameters:** The model utilizes a `rope_scaling` configuration to manage its context length. According to user error logs, the exact hyper-parameters involve a **`factor` of 8.0**, a **`low_freq_factor` of 1.0**, a **`high_freq_factor` of 4.0**, an **`original_max_position_embeddings` of 8,192**, and a **`rope_type` set to 'llama3'** (Source 3). Additionally, users noted that depending on the software version, the `type` hyper-parameter sometimes needs to be manually changed to **'dynamic' or 'linear'** to prevent out-of-memory or scaling errors (Source 3).

**5. Activations**
*   **Type of activation:** The model uses **SwiGLU activations** (Source 1).
*   **Consistency:** The sources do not mention mixing different kinds of activations, indicating SwiGLU is used consistently across the Feed-Forward sublayers.

**6. Normalizations**
*   **Components and Placement:** The model uses a **pre-normalization** architecture, applying normalization before each sublayer (Source 1).
*   **Type of normalization:** There is **ambiguity** regarding the exact normalization function used. Source 1 states it uses **"RMSNorm or LayerNorm"** without definitively committing to one over the other for this specific parameter class.

**7. Any other property of interest**
*   **Parameter Count:** The model has approximately **8.03 billion total parameters** (Source 1).
*   **Alignment:** It is an instruction-tuned model trained using **Supervised Fine-Tuning (SFT)** alongside **Direct Preference Optimization (DPO)** or RLHF to improve instruction adherence and reasoning (Source 1).
*   **Environmental Footprint:** Training the 8B model took **1.46 Million GPU hours** on H100-80GB hardware (consuming 700W peak power per GPU), resulting in **420 tons of location-based CO2eq** (Source 2).
*   **Multilingual Capabilities:** The model natively supports **8 languages** (English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai) with a knowledge cutoff of **December 2023** (Source 2).

***

**Indexed List of Sources:**
*   Source 1: <a href="https://www.emergentmind.com/topics/llama-3-1-8b-instruct-679bb9c6-400c-4f7c-b10a-c1048d659134">Llama-3.1-8B-Instruct: Efficient Instruction-Tuned LLM - Emergent Mind</a>
*   Source 2: <a href="https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct">meta-llama/Llama-3.1-8B-Instruct - Hugging Face</a>
*   Source 3: <a href="https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/discussions/15">meta-llama/Llama-3.1-8B-Instruct · ValueError: rope_scaling must be a dictionary with two fields - Hugging Face</a>