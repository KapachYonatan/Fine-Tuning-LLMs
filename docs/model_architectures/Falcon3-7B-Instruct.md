Here is the architectural information for the **tiiuae/Falcon3-7B-Instruct** model, extracted directly from the provided sources.

**1. Number of layers, width of layers**
*   **Number of layers:** The model features **28 decoder blocks** (hidden layers), as confirmed by both Source 1 and Source 2.
*   **Width of layers:** The hidden dimension (width or `hidden_size`) of the model is **3072** (Source 1).

**2. Number of attention heads**
*   **Query Heads:** The model uses **12 attention heads** per layer (Source 1, Source 2).
*   **Key-Value Heads:** The model utilizes Grouped-Query Attention (GQA) with **4 Key-Value (KV) heads** (Source 1, Source 2).

**3. Sizes of everything**
*   **MLP layers:** The Feed-Forward Network (MLP) layers feature an intermediate dimension size of **23,040** (Source 1).
*   **MoE layers:** The sources do **not mention** Mixture of Experts (MoE) layers for this model. It is described as a standard "Transformer Decoder Only Architecture" built on the `LlamaForCausalLM` framework, indicating a dense architecture (Source 1, Source 2).
*   **Attention components:** The model features a notably wider attention head dimension (`head_dim`) of **256** (Source 1, Source 2).
*   **Embeddings and Unembedding layers:** The vocabulary size is **131,072** tokens (Source 1). The model configuration explicitly sets `tie_word_embeddings` to `false`, meaning the input embedding matrix and the final unembedding matrix do not share weights (Source 1).

**4. Position encoding**
*   **Method:** The model utilizes **Rotary Positional Embeddings (RoPE)** (Source 2).
*   **Max position supported:** There is a minor **disagreement/nuance** in the sources regarding the exact maximum context length. Source 1 (the raw `config.json`) sets the `max_position_embeddings` to **32,768** tokens. Meanwhile, Source 2 summarizes the context length more generally as **32,000 tokens** (or 32K). 
*   **Hyper-parameters:** The model uses a highly elevated `rope_theta` value of **1,000,042** to enable its extended context window (Source 1, Source 2). Additionally, the `rope_scaling` parameter is set to **null**, indicating no out-of-the-box dynamic scaling multipliers are applied (Source 1).

**5. Activations**
*   **Type of activation:** The model uses the **SwiGLU** (or `silu`) activation function (Source 1, Source 2).
*   **Consistency:** The configuration lists `"hidden_act": "silu"` without mentioning any mixture of activation types, indicating it is applied consistently across the model's feed-forward layers (Source 1).

**6. Normalizations**
*   **Type of normalization:** The model uses **RMSNorm** with an epsilon (`rms_norm_eps`) of **1e-06** (Source 1, Source 2).
*   **Components and Placement:** The exact placement of the normalizations (e.g., whether the model uses a pre-norm or post-norm configuration) is **missing** from the provided texts.

**7. Any other property of interest**
*   **Architecture Backbone:** Despite being named Falcon, the model's underlying code architecture is explicitly set to **`LlamaForCausalLM`** (Source 1). 
*   **Bias Configuration:** The model explicitly disables both attention biases (`"attention_bias": false`) and MLP biases (`"mlp_bias": false`) (Source 1).
*   **Multilingual Capabilities:** The model supports multilingual inputs and outputs, specifically optimized for **English, French, Spanish, and Portuguese** (Source 2).
*   **Training Data Scale:** The base model was pre-trained on a massive **14 Teratokens** of web, code, STEM, and multilingual data. This instruct version was then post-trained on **1.2 million samples** covering conversations, code, STEM, safety, and function calling (Source 2).
*   **Licensing:** The model is released under a specific **Falcon-LLM-License** rather than a standard open-source license like Apache 2.0 (Source 1, Source 2).

***

**Indexed List of Sources:**
*   Source 1: <a href="https://huggingface.co/tiiuae/Falcon3-7B-Instruct/blob/main/config.json">config.json · tiiuae/Falcon3-7B-Instruct at main - Hugging Face</a>
*   Source 2: <a href="https://build.nvidia.com/tiiuae/falcon3-7b-instruct">falcon3-7b-instruct Model by Tiiuae | NVIDIA NIM</a>