Here is the architectural information for the **DeepSeek-V3** model, extracted directly from the provided sources.

**1. Number of layers, width of layers**
*   **Number of layers:** The model features **61 Transformer layers** (Source 2, Source 3).
*   **Width of layers:** The hidden dimension (embedding width) of the model is **7168** (Source 1, Source 3).

**2. Number of attention heads**
*   **Query Heads:** The model uses **128 attention heads** (Source 3).
*   **Key-Value Heads:** DeepSeek-V3 utilizes an innovative architecture called **Multi-Head Latent Attention (MLA)** instead of standard MHA or GQA. As a result, it does not use traditional KV heads. Instead, it relies on a shared "down" matrix that compresses keys and values into a shared latent vector to drastically reduce KV cache memory (Source 3, Source 5).

**3. Sizes of everything**
*   **MLP layers:** The first 3 layers of the model are standard dense Feed-Forward Networks (FFNs), while the remaining 58 layers are Mixture of Experts (MoE) layers (Source 3). 
*   **MoE layers:** The model features **257 total experts** per MoE layer: **256 routed experts** and **1 shared expert** (Source 2, Source 3). For each token, **8 routed experts** are activated alongside the 1 shared expert, meaning **9 experts are activated** per token (Source 2, Source 3). 
*   **Dimensions involved:** The intermediate hidden dimension for *each* expert is **2048** (Source 3). 
*   *Note on Disagreement/Nuance:* Source 1 states the intermediate dimension inside the MLP ($H_{inter\_dim}$) is **18,432**. This is actually the combined dimension of all the activated experts for a given token (9 activated experts $\times$ 2048 = 18,432).
*   **Attention components:** The attention head dimension ($d_h$) is **128**. The MLA KV compression dimension ($d_c$) is **512**, and the query compression dimension ($d_c'$) is **1536** (Source 3).
*   **Embeddings and Unembedding layers:** The model uses a **shared embedding and output head** (Source 3).
*   *Note on Disagreement:* There is a slight disagreement regarding the vocabulary size. Source 3 states the vocabulary size is **128,000**, while Source 1 lists it as **129,280**.

**4. Position encoding**
*   **Method:** The model utilizes **Rotary Positional Embeddings (RoPE)**. Specifically, it uses a technique called **Decoupled RoPE Embeddings**, where the position information is concatenated to the keys and values rather than being applied directly to them, which fixes mathematical issues caused by the MLA compression (Source 3, Source 5).
*   **Max position supported:** The model supports a maximum context window of **128,000 (128K) tokens** (Source 3, Source 4).
*   **Hyper-parameters:** The context length was extended from an initial 4K using the **YaRN** extrapolation technique. The YaRN hyper-parameters include a scale ($s$) of **40**, $\alpha$ of **1**, $\beta$ of **32**, and a scaling factor of $t = 0.1 \ln s + 1$ (Source 3). The decoupled queries and key vectors that carry RoPE have a per-head dimension ($d_h^R$) of **64** (Source 3).

**5. Activations**
*   **Type of activation:** The model uses the **SwiGLU** (Swish-Gated Linear Unit) / **SiLU** activation function (Source 1, Source 2).
*   **Consistency:** The sources indicate this activation is used consistently across the MoE and MLP layers.

**6. Normalizations**
*   **Type of normalization:** The model uses **RMSNorm** (Source 1, Source 2, Source 3).
*   **Components and Placement:** The model utilizes a **pre-norm** configuration (e.g., applying RMSNorm to the hidden states before the attention and FFN blocks). Uniquely, due to its MLA architecture, DeepSeek-V3 also places **additional RMSNorm layers after the compressed latent vectors** (Source 3). 

**7. Any other property of interest**
*   **Parameter Count:** The model is massive, featuring **671 Billion total parameters**. However, due to its MoE architecture, only **37 Billion parameters** (4.8%) are activated for each token (Source 2, Source 3, Source 4, Source 6).
*   **Auxiliary-Loss-Free Load Balancing:** To prevent routing collapse (where the model only sends tokens to a few popular experts), DeepSeek-V3 abandons traditional auxiliary losses. Instead, it uses a dynamic bias term in the router to force a balanced distribution without hurting the model's main task performance (Source 3, Source 6).
*   **Multi-Token Prediction (MTP):** The model was trained using an MTP objective, meaning it predicts the next **two tokens** simultaneously (a depth of $D=1$ additional token) at each position. This densifies training signals and allows for speculative decoding to accelerate inference (Source 3, Source 5).
*   **FP8 Training:** DeepSeek-V3 is the first extremely large-scale model to be natively pre-trained using **FP8 mixed precision**, which radically reduced its training costs. The full training (pre-training, context extension, and post-training) took only 2.788 Million H800 GPU hours, costing roughly **$5.58 Million** (Source 2, Source 3, Source 6).

***

**Indexed List of Sources:**
*   Source 1: <a href="https://www.atlascloud.ai/blog/guides/analyzing-deepseek-v3-model-performance">Analyzing DeepSeek-V3 Model Performance - Atlas Cloud Blog</a>
*   Source 2: <a href="https://fireworks.ai/blog/deepseek-model-architecture">DeepSeek v3 and R1 Model Architecture: Why it's powerful and economical - Fireworks AI</a>
*   Source 3: <a href="https://arxiv.org/abs/2412.19437">DeepSeek-V3 Technical Report - arXiv</a>
*   Source 4: <a href="https://github.com/deepseek-ai/DeepSeek-V3">deepseek-ai/DeepSeek-V3 - GitHub</a>
*   Source 5: <a href="https://mccormickml.com/2025/02/12/the-inner-workings-of-deep-seek-v3/">The Inner Workings of DeepSeek-V3 - Chris McCormick</a>
*   Source 6: <a href="https://www.grandlinux.com/en/blogs/deepseek-moe-architecture.html">What is Mixture of Experts (MoE)? — The Technique That Makes DeepSeek 10x Cheaper Than GPT | Saeree ERP</a>