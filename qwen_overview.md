# Overview of chinese AI models from Alibaba, Baidu and DeepSeek
## Qwen AI family

![Qwen](https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Qwen_logo.svg/960px-Qwen_logo.svg.png)
Qwen(originally launched in China as Tongyi Qianwen) is the large language model and large multimodal model series of the Qwen Team, Alibaba Group. Both language models and multimodal models are pretrained on large-scale multilingual and multimodal data and post-trained on quality data for aligning to human preferences. Qwen is capable of natural language understanding, text generation, vision understanding, audio understanding, tool use, role play, playing as AI agent, etc.​[[1]](https://qwen.ai/qwenchat) Many variants are released under permissive licenses, so developers can download, run, and fine‑tune them on their own infrastructure.[[2]](https://en.wikipedia.org/wiki/Qwen) They offer a unique blend of performance, flexibility, and openness, making them ideal for both enterprise and research applications. Their rapid evolution has kept them at the cutting edge of LLM development.

### Core Capabilities
Qwen models can:
* Understand and generate multilingual text for tasks like chat, translation, summarization, and long‑form writing.[[3]](https://qwen.readthedocs.io/)
* Perform multimodal reasoning, including image and audio understanding, image generation, and in newer versions video-related tasks.[[4]](https://qwen.ai/blog?id=99f0335c4ad9ff6153e517418d48535ab6d8afef&from=research.latest-advancements-list)
* Support long-context inputs (on the order of tens of thousands to over 100,000 tokens), which helps with analyzing large documents and sustained conversations.[[5]](https://www.godofprompt.ai/blog/what-is-qwen-ai)

### Products and interfaces
Qwen Chat is the main user-facing interface, offering chatbot-style interaction plus features like document analysis, web search integration, and creative writing tools. Third-party platforms and tools also expose Qwen models via APIs or hosted playgrounds, targeting businesses, developers, and researchers who need an advanced AI engine.[[6]](https://www.prismetric.com/qwen-2-5-what-it-is-and-how-to-use-it/)

## The Evolution of Qwen: From Qwen 1 to Qwen 3 
### Qwen 1 & Qwen 1.5
* Initial releases focused on robust transformer architectures and multilingual capabilities.[[7]](https://datasciencedojo.com/blog/the-evolution-of-qwen-models/)
* The first public Qwen models included parameter-sizes: 1.8 B, 7 B, 14 B, and 72 B.[[8]](https://github.com/QwenLM/Qwen)
* Context windows up to 32K tokens.
* There were both “base” language models (pretrained for general text) and “Chat” / instruction-tuned variants (for conversation, tool use, generation, etc.).[[9]](https://github.com/QwenLM/Qwen)
* According to the original technical report, Qwen (v1) supported “natural language understanding, text generation, tool-use, code generation / reasoning, math, multilingual text” — even code- and math-specialized variants (e.g. “Code-Qwen”, “Math-Qwen”).[[10]](https://skywork.ai/blog/qwen-tongyi-qianwen-open-weight-ai-model)
* Efficiency via MoE & quantization: The MoE-variant (e.g. Qwen1.5-MoE-A2.7B) lets users get “big-model performance with lower resource usage,” which helps for deployment on limited hardware.[[11]](https://www.reddit.com/r/machinelearningnews/comments/1br5kqa)
* Strong performance in Chinese and English, with growing support for other languages

#### List of Qwen 1.5 models[[12]](https://ollama.com/library/qwen)

| Series  | Model     | Size  | Context | Input |
|---------|-----------|-------|---------|-------|
| Qwen1.5 | Qwen-0.5B | 0.5B  | 32K     | Text  |
|         | Qwen-1.8B | 1.8B  | 32K     | Text  |
|         | Qwen-4B   | 4B    | 32K     | Text  |
|         | Qwen-7B   | 7B    | 32K     | Text  |
|         | Qwen-14B  | 14B   | 32K     | Text  |
|         | Qwen-32B  | 32B   | 32K     | Text  |
|         | Qwen-72B  | 72B   | 32K     | Text  |
|         | Qwen-110B | 100B  | 32K     | Text  |
### Qwen 2 & Qwen 2.5
* Expanded parameter sizes (up to 110B dense, 72B instruct).[[13]](https://datasciencedojo.com/blog/the-evolution-of-qwen-models/)
* Improved training data (up to 18 trillion tokens in Qwen 2.5).
* Enhanced alignment via supervised fine-tuning and Direct Preference Optimization (DPO).
* Specialized models for math, coding, and vision-language tasks.
* All variants share a Transformer-decoder architecture, using modern components (e.g. SwiGLU activations, pre-normalization with RMSNorm) for training stability.[[14]](https://qwen-ai.chat/models/qwen2/?utm_source=chatgpt.com)
* It introduces strong multilingual support, long context windows up to around 128k–131k tokens using dual‑chunk attention and RoPE/YARN scaling, and competitive performance on language, coding, math, and reasoning benchmarks compared with prior open‑weight models.​[[15]](https://www.emergentmind.com/topics/qwen-series-models)
* Both Qwen2 and Qwen2.5 use modern transformer design choices: grouped query attention, RoPE positional encoding, RMSNorm, and SwiGLU, along with sliding‑window plus full attention patterns for efficiency at long context.[[16]](https://huggingface.co/docs/transformers/en/model_doc/qwen2)

#### List of Qwen 2 models[[17]](https://ollama.com/library/qwen2)

| Series | Model      | Size  | Context  | Input |
|--------|------------|-------|----------|-------|
| Qwen2  | Qwen2-0.5B | 0.5B  | 32K      | Text  |
|        | Qwen2-1.5B | 1.5B  | 32K      | Text  |
|        | Qwen2-7B   | 7B    | 128K     | Text  |
|        | Qwen2-72B  | 72B   | 128K     | Text  |


Qwen2 is trained on data in 29 languages, including English and Chinese. It is available in 4 parameter sizes: 0.5B, 1.5B, 7B, 72B. In the 7B and 72B models, context length has been extended to 128k tokens.

| Models         | Qwen2-0.5B | Qwen2-1.5B | Qwen2-7B | Qwen2-72B |
|----------------|------------|------------|----------|-----------|
| Params         | 0.49B      | 1.54B      | 7.07B    | 72.71B    |
| Non-Emb Params | 0.35B      | 1.31B      | 5.98B    | 70.21B    |
| GQA            | True       | True       | True     | True      |
| Tie Embedding  | True       | True       | False    | False     |
| Context Length | 32K        | 32K        | 128K     | 128K      |

#### Supported Languages
| Regions                  | Languages                                                                  |
|--------------------------|----------------------------------------------------------------------------|
| Western Europe           | German, French, Spanish, Portuguese, Italian, Dutch                        |
| Eastern & Central Europe | Czech, Polish, Russian                                                     |
| Middle East              | Arabic, Persian, Hebrew, Turkish                                           |
| Eastern Asia             | Japanese, Korean                                                           |
| South-Eastern Asia       | Vietnamese, Thai, Indonesian, Malay, Lao, Burmese, Cebuano, Khmer, Tagalog |
| Souther Asia             | Hindi, Bengali, Urdu                                                       |
#### Qwen2 Performance
![Qwen2-72B](https://ollama.com/assets/library/qwen2/68b445e3-bf1b-4fff-9621-4e5bbf4a72a2)
![Qwen2-72Bins](https://ollama.com/assets/library/qwen2/72e9bf41-f8d6-4b7a-a7ef-9599ef533af6)
![Qwen2-7Bins](https://ollama.com/assets/library/qwen2/6c978d72-c37c-45a2-b7f4-c06178c0182c)

#### List of Qwen 2.5 models[[18]](https://ollama.com/library/qwen2.5)
| Series  | Model        | Size  | Context | Input |
|---------|--------------|-------|---------|-------|
| Qwen2.5 | Qwen2.5-0.5B | 0.5B  | 32K     | Text  |
|         | Qwen2.5-1.5B | 1.5B  | 32K     | Text  |
|         | Qwen2.5-3B   | 3B    | 32K     | Text  |
|         | Qwen2.5-7B   | 7B    | 32K     | Text  |
|         | Qwen2.5-14B  | 14B   | 32K     | Text  |
|         | Qwen2.5-32B  | 32B   | 32K     | Text  |
|         | Qwen2.5-72B  | 72B   | 32K     | Text  |

### Qwen 3
* Released in 2025, Qwen 3 marks a leap in architecture, scale, and reasoning.
* Qwen3 introduces two internal operating modes: a fast “non‑thinking” mode for everyday conversation and a slower “thinking” mode that allocates more compute for complex reasoning, math, and coding, selectable via prompts or API settings.[[19]](https://qwenlm.github.io/blog/qwen3/)
* The family covers dense models (for example around 0.6B, 1.7B, 4B, 8B, 14B, 32B parameters) and MoE variants such as Qwen3‑30B‑A3B and Qwen3‑235B‑A22B, where only a subset of experts is activated per token for high efficiency.[[20]](https://www.siliconflow.com/articles/en/the-best-qwen3-models-in-2025)
* Qwen3 expands multilingual coverage from a few dozen languages in Qwen2.5 to roughly 100–119 languages and dialects, with significantly improved cross‑lingual understanding and translation quality.[[21]](https://qwenlm.github.io/blog/qwen3/)

#### Qwen 3 models[[22]](https://ollama.com/library/qwen3)

Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models. The flagship model, Qwen3-235B-A22B, achieves competitive results in benchmark evaluations of coding, math, general capabilities, etc., when compared to other top-tier models such as DeepSeek-R1, o1, o3-mini, Grok-3, and Gemini-2.5-Pro. Additionally, the small MoE model, Qwen3-30B-A3B, outcompetes QwQ-32B with 10 times of activated parameters, and even a tiny model like Qwen3-4B can rival the performance of Qwen2.5-72B-Instruct.

| Series | Model      | Size  | Context | Input |
|--------|------------|-------|---------|-------|
| Qwen3  | Qwen3-0.6B | 0.6B | 40K     | Text  |
|        | Qwen3-1.7B | 1.7B | 40K     | Text  |
|        | Qwen3-4B   | 4B   | 256K    | Text  |
|        | Qwen3-8B   | 8B   | 40K     | Text  |
|        | Qwen3-14B  | 14B  | 40K     | Text  |
|        | Qwen3-30B  | 30B  | 256K    | Text  |
|        | Qwen3-32B  | 32B  | 40K     | Text  |
|        | Qwen3-235B | 235B | 256K    | Text  |

#### Qwen3 Performance
![Qwen3-30B](https://ollama.com/assets/library/qwen3/bc0ddfea-95b5-49fc-a36e-c817f98a5de0)
![Qwen3-235B](https://ollama.com/assets/library/qwen3/8426a459-dd88-49cd-ae89-ece442e58ec5)

#### Qwen 3-VL models[[23]](https://ollama.com/library/qwen3-vl)

Qwen3-VL is the most powerful vision-language model in the Qwen family to date.

In this generation, there are improvements to the model in many areas: its understanding and generating text, perceiving and reasoning about visual content, supporting longer context lengths, understanding spatial relationships and dynamic videos, or interacting with AI agents — Qwen3-VL shows clear and significant progress in every area.

| Series   | Model         | Size  | Context | Input       |
|----------|---------------|-------|---------|-------------|
| Qwen3-VL | Qwen3-VL-2B   | 2B    | 256K    | Text, Image |
|          | Qwen3-VL-4B   | 4B    | 256K    | Text, Image |
|          | Qwen3-VL-8B   | 8B    | 256K    | Text, Image |
|          | Qwen3-VL-30B  | 30B   | 256K    | Text, Image |
|          | Qwen3-VL-32B  | 32B   | 256K    | Text, Image |
|          | Qwen3-VL-235B | 235B  | 256K    | Text, Image |

#### Qwen3-VL Performance
![Qwen3-VL-235Bins](https://camo.githubusercontent.com/7b30ae6af5402a71bc63076de6e1b67d5bf94c1ada1c3fb59fddaf30b98fbd8a/68747470733a2f2f7169616e77656e2d7265732e6f73732d616363656c65726174652e616c6979756e63732e636f6d2f5177656e332d564c2f7461626c655f6e6f7468696e6b696e675f766c2e6a7067)
![Qwen3-VL-235Bth](https://camo.githubusercontent.com/33b15065d776936438f83ec82199597884a776239ef6318ee82fe175bb0505f0/68747470733a2f2f7169616e77656e2d7265732e6f73732d636e2d6265696a696e672e616c6979756e63732e636f6d2f5177656e332d564c2f7461626c655f7468696e6b696e675f766c5f2e6a7067)

#### Qwen 3-Coder models[[24]](https://ollama.com/library/qwen3-coder)

Qwen3-Coder is Alibaba's performant long context models for agentic and coding tasks.

| Series      | Model            | Size  | Context | Input |
|-------------|------------------|-------|---------|-------|
| Qwen3-Coder | Qwen3-Coder-30B  | 30B  | 256K    | Text  |
|             | Qwen3-Coder-480B | 480B | 256K    | Text  |

#### Qwen3-Coder Performance
![Qwen3-Coder](https://ollama.com/assets/library/qwen3-coder/52070971-5a66-4947-90a0-5e983a5809e7)

#### Qwen 3-Embedding models[[25]](https://ollama.com/library/qwen3-embedding)

The Qwen3 Embedding model series is specifically designed for text embedding tasks. Building upon the dense foundational models of the Qwen3 series, it provides a comprehensive range of text embeddings models in various sizes (0.6B, 4B, and 8B). This series inherits the exceptional multilingual capabilities, long-text understanding, and reasoning skills of its foundational model. The Qwen3 Embedding series represents significant advancements in multiple text embedding and ranking tasks, including text retrieval, code retrieval, text classification, text clustering, and bitext mining.

Qwen3-Embedding-8B has the following features[[26]](https://qwenlm.github.io/blog/qwen3-embedding/):
* Model Type: Text Embedding
* Supported Languages: 100+ Languages
* Number of Paramaters: 8B
* Context Length: 32k
* Embedding Dimension: Up to 4096, supports user-defined output dimensions ranging from 32 to 4096

| Series          | Model                | Size  | Context | Input | Embedding  | MRL Support | Instruction Aware |
|-----------------|----------------------|-------|---------|-------|------------|-------------|-------------------|
| Qwen3-Embedding | Qwen3-Embedding-0.6B | 0.6B  | 32K     | Text  |1024        | Yes         | Yes  |
|                 | Qwen3-Embedding-4B   | 4B    | 32K     | Text  |2560        | Yes         | Yes  |
|                 | Qwen3-Embedding-8B   | 8B    | 32K     | Text  |4096        | Yes         | Yes  |
| Qwen3-Reranking | Qwen3-Reranking-0.6B | 0.6B  | 32K     | Text  |-           | -           | Yes  |
|                 | Qwen3-Reranking-4B   | 4B    | 32K     | Text  |-           | -           | Yes  |
|                 | Qwen3-Reranking-8B   | 8B    | 32K     | Text  |-           | -           | Yes  |

#### Evaluation results for reranking models
| Model                              | Param | MTEB-R    | CMTEB-R   | MMTEB-R   | MLDR      | MTEB-Code | FollowIR  |
|------------------------------------|-------|-----------|-----------|-----------|-----------|-----------|-----------|
| Qwen3-Embedding-0.6B               | 0.6B  | 61.82     | 71.02     | 64.64     | 50.26     | 75.41     | 5.09      |
| Jina-multilingual-reranker-v2-base | 0.3B  | 58.22     | 63.37     | 63.73     | 39.66     | 58.98     | -0.68     |
| gte-multilingual-reraner-base      | 0.3B  | 59.51     | 74.08     | 59.44     | 66.33     | 54.18     | -1.64     |
| BGE-reranker-v2-m3                 | 0.6B  | 57.03     | 72.16     | 58.36     | 59.51     | 41.38     | -0.01     |
| Qwen3-Reranker-0.6B                | 0.6B  | 65.80     | 71.31     | 66.36     | 67.28     | 73.42     | 5.41      |
| Qwen3-Reranker-4B                  | 4B    | **69.76** | 75.94     | 72.74     | 69.97     | 81.20     | **14.84** |
| Qwen3-Reranker-8B                  | 8B    | 69.02     | **77.45** | **72.94** | **70.19** | **81.22** | 8.05      |