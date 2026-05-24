**1. Number of layers, width of layers**
*   **Number of layers:** The model features **32 layers** (`llama.block_count`) (Source 2).
*   **Width of layers:** The hidden dimension (width or `embedding_length`) is **4096** (Source 2).

**2. Number of attention heads**
*   **Query Heads:** The model uses **32 attention heads** (`head_count`) (Source 2).
*   **Key-Value Heads:** The model utilizes Grouped-Query Attention (GQA) with **8 Key-Value (KV) heads** (`head_count_kv`) (Source 2).

**3. Sizes of everything**
*   **MLP layers:** The Feed-Forward Network (MLP) layers feature an intermediate dimension size of **14,336** (`feed_forward_length`) (Source 2).
*   **MoE layers:** The model does **not** feature Mixture of Experts (MoE) layers; it is a standard dense transformer architecture based on Mistral-7B (Source 2, Source 3).
*   **Attention components:** The model features an attention head dimension (`rope.dimension_count`) of **128** (Source 2). 
*   **Embeddings and Unembedding layers:** The model has a vocabulary size of **33,152** (Source 2). The input embedding matrix (`token_embd.weight`) and the final output unembedding matrix (`output.weight`) both feature dimensions of **4096 x 33,152** (Source 2). 

**4. Position encoding**
*   **Method:** The model utilizes **Rotary Positional Embeddings (RoPE)** (Source 2).
*   **Max position supported:** The model supports a maximum context window of **32,768 (32K) tokens** (Source 1, Source 2).
*   **Hyper-parameters:** The RoPE implementation utilizes a base frequency (`freq_base`) of **10,000** and is applied across **128 dimensions** (Source 2).

**5. Activations**
*   **Type of activation:** The explicit name of the activation function is **missing** from the provided parameter metadata. However, because the model utilizes `ffn_gate`, `ffn_down`, and `ffn_up` matrices characteristic of the Llama/Mistral architectures (Source 2), and is explicitly built upon Mistral-7B-v0.1 (Source 3), it can be inferred that it uses the **SiLU / SwiGLU** activation function.
*   **Consistency:** There is no indication of mixing different activation types, implying it is used consistently across all feed-forward layers.

**6. Normalizations**
*   **Type of normalization:** The model uses **RMSNorm** with an epsilon (`layer_norm_rms_epsilon`) of **1e-05** (Source 2).
*   **Components and Placement:** The model utilizes a **pre-norm** configuration. This is indicated by the presence of `attn_norm.weight` and `ffn_norm.weight` tensors applied before their respective blocks, alongside a final `output_norm.weight` applied at the end of the transformer (Source 2).

**7. Any other property of interest**
*   **Base Architecture & Pretraining:** The model is an instruct-tuned version of `DictaLM-2.0` (Source 1). The base model was built on top of `Mistral-7B-v0.1` and was continually pre-trained on over **190 Billion tokens**, balanced evenly between **50% Hebrew and 50% English** (Source 3).
*   **Hebrew Optimization:** To specialize in the Hebrew language, the tokenizer was extended with **1,000 specifically injected Hebrew tokens**. This dramatically improved the model's Hebrew compression rate from 5.78 tokens per word down to **2.76 tokens per word** (Source 3).

***

**Indexed List of Sources:**
*   Source 1: <a href="https://ollama.com/aminadaven/dictalm2.0-instruct">aminadaven/dictalm2.0-instruct - Ollama</a>
*   Source 2: <a href="https://ollama.com/aminadaven/dictalm2.0-instruct:q4_0">aminadaven/dictalm2.0-instruct:q4_0/model - Ollama</a>
*   Source 3: <a href="https://huggingface.co/dicta-il/dictalm2.0">dicta-il/dictalm2.0 - Hugging Face</a>