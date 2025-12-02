# Overview of chinese AI models from Alibaba, Baidu and DeepSeek
## Ernie AI family

![Ernie](https://upload.wikimedia.org/wikipedia/en/4/42/Ernie_Bot_Logo.png)

ERNIE (Enhanced Representation through Knowledge Integration) are a series of large language models developed by Baidu, a leading Chinese technology company. They are designed primarily for natural language processing tasks with a strong focus on Chinese language capabilities. The ERNIE models excel in handling Chinese text, including names, places, and policy terms, making them especially valuable for bilingual Chinese-English workflows. They support a variety of tasks such as document processing (extracting data from images and PDFs), content creation, language learning, and business document transformation with high accuracy and reliable structured outputs.[[1]](https://en.wikipedia.org/wiki/Ernie_Bot)

The ERNIE model family includes several versions, with ERNIE 4.0 being a major upgrade offering improved reasoning, language understanding, and generation capabilities. ERNIE 4.5 is a more recent multimodal model released with advanced features for both textual and visual understanding and reasoning, supporting thinking and non-thinking modes, and is optimized for general-purpose language use. It includes a mixture-of-experts architecture, modality-isolated routing, and balanced multimodal training to excel in tasks involving text, images, and cross-modal reasoning. These models are powerful in instruction following, world knowledge, and complex problem solving, competing closely with state-of-the-art models globally.

Baidu also provides an industrial-grade toolkit called ERNIEKit for fine-tuning and deploying these models efficiently. The ERNIE models power Baidu's AI chatbot services, such as Ernie Bot, which rivals models like GPT in Chinese NLP tasks and supports various content generation and problem-solving applications. Baidu released the first version of Ernie Bot in March 2023. At the time, the service was based on an LLM dubbed Ernie 3.0 that the company had started developing in 2019. The model featured 10 billion parameters and was trained on a 4-terabyte dataset. Baidu is committed to open-sourcing its ERNIE models, with plans to release further advanced versions like ERNIE 5 to expand capabilities and foster research and development in AI.[[2]](https://siliconangle.com/2025/02/14/baidu-open-source-ernie-large-language-model-series/)

### List of Ernie Models

| Generation      | Approx. year | Role / type                                                                                                                         |
| --------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| ERNIE 1.0       | 2019         | First knowledge‑enhanced pre‑training model for Chinese NLP.[[3]](https://huggingface.co/papers/1904.09223)                         |
| ERNIE 2.0       | 2019         | Continual pre‑training framework extending ERNIE 1.0.[[4]](https://ojs.aaai.org/index.php/AAAI/article/view/6428)                   |
| ERNIE 3.0       | 2021–2023    | Large knowledge‑enhanced model; base of early Ernie Bot.[[5]](https://huggingface.co/papers/2107.02137)                             |
| ERNIE 3.0 Titan | 2021–2023    | “scaled-up” version of 3.0 (with many more parameters), for advanced tasks.[[6]](https://arxiv.org/abs/2112.12731)                  |
| ERNIE 3.5       | 2023         | Improved general LLM for Ernie Bot with better reasoning.[[7]](https://en.wikipedia.org/wiki/Ernie_Bot)                             |
| ERNIE 4.0       | 2023–2024    | High‑end dense LLM, new flagship model.                                                                                             |
| ERNIE 4.0 Turbo | 2024         | Latency‑ and cost‑optimized 4.0 variant.                                                                                            |
| ERNIE 4.5       | 2025         | Multimodal MoE family (10+ variants, up to 424B total parameters).[[8]](https://ernie.baidu.com/blog/posts/ernie4.5/)               |
| ERNIE 4.5 Turbo | 2025         | Faster/cheaper 4.5 variant for production.                                                                                          |
| ERNIE X1        | 2025         | Specialized reasoning model within ERNIE family.[[9]](https://www.datacamp.com/blog/ernie-4-5-x1)                                   |
| ERNIE X1 Turbo  | 2025         | Optimized reasoning model for low‑latency usage.                                                                                    |
| ERNIE 5 (ann.)  | 2025         | Next‑gen MoE model (planned successor to 4.5).[[10]](https://ernie.baidu.com/blog/posts/ernie-5.0-preview-1120-release-on-lmarena/) |



### Ernie 1.0 and Ernie 2.0[[11]](https://github.com/arita37/ERNIE-1)

Early versions (1.0 - 2.0) were primarily “pre-training / representation learning” models focusing on language understanding (not full multimodal chatbot functionality).

ERNIE 1.0 is language representation learning method enhanced by knowledge masking strategies, which includes entity-level masking and phrase-level masking. Inspired by the masking strategy of BERT [[12]](https://arxiv.org/abs/1810.04805), ERNIE introduced phrase masking and named entity masking and predicts the whole masked phrases or named entities. Phrase-level strategy masks the whole phrase which is a group of words that functions as a conceptual unit. Entity-level strategy masks named entities including persons, locations, organizations, products, etc., which can be denoted with proper names.

ERNIE 2.0 is a continual pre-training framework for language understanding in which pre-training tasks can be incrementally built and learned through multi-task learning. In this framework, different customized tasks can be incrementally introduced at any time. For example, the tasks including named entity prediction, discourse relation recognition, sentence order prediction are leveraged in order to enable the models to learn language representations.

#### Comparison of Ernie 1.0 and Ernie 2.0

| Tasks           | ERNIE 1.0                | ERNIE 2.0(en)                                                                        | ERNIE 2.0(zh)                         |
|-----------------|--------------------------|--------------------------------------------------------------------------------------|---------------------------------------|
| Word-aware      | Knowledge masking        | Knowledge Masking<br>Capitalization Prediction<br>Token-Document Relation Prediction | Knowledge Masking                     |
| Structure-aware | -                        | Sentence Reordering                                                                  | Sentence Reprdering Sentence Distance |
| Semantic-aware  | Next Sentence Prediction | Discourse Relation                                                                   | Discourse Relation IR Relevance       |

### Results

#### GLUE-Dev

| <strong>Dataset</strong> | <strong>CoLA</strong> | <strong>SST-2</strong> | <strong>MRPC</strong> | <strong>STS-B</strong> | <strong>QQP</strong> | <strong>MNLI-m</strong> | <strong>QNLI</strong> | <strong>RTE</strong> |
| --------------------- | --------------------- | ---------------------- | --------------------- | ---------------------- | -------------------- | ----------------------- | --------------------- | -------------------- |
| **metric**            | **matthews corr.**    | **acc**                | **acc**          | **pearson corr.**      | **acc**              | **acc**                 | **acc**               | **acc**              |
| **BERT Large**        | 60.6                  | 93.2                   | 88.0                  | 90.0                   | 91.3                 | 86.6                    | 92.3                  | 70.4                 |
| **XLNet Large**       | 63.6          | 95.6   | 89.2   | 91.8    | 91.8  | 89.8   | 93.9   | 83.8   |
| **ERNIE 2.0 Large**   | 65.4<br/>(**+4.8,+1.8**)   | 96.0<br/>(**+2.8,+0.4**)    | 89.7<br/>(**+1.7,+0.5**)   | 92.3<br/>(**+2.3,+0.5**)    | 92.5<br/>(**+1.2,+0.7**)  | 89.1<br/>(**+2.5,-0.7**)     | 94.3<br/>(**+2.0,+0.4**)   | 85.2<br/>(**+14.8,+1.4**) |

#### GLUE-Test

| <strong>Dataset</strong>                | -                          | <strong>CoLA</strong> | <strong>SST-2</strong> | <strong>MRPC</strong>         | <strong>STS-B</strong>        | <strong>QQP</strong>          | <strong>MNLI-m</strong> | <strong>MNLI-mm</strong> | <strong>QNLI</strong> | <strong>RTE</strong> | <strong>WNLI</strong> | <strong>AX</strong> |
| ------------------- | -------------------------- | --------------------- | ---------------------- | ----------------------------- | ----------------------------- | ----------------------------- | ----------------------- | ------------------------ | --------------------- | -------------------- | --------------------- | ------------------- |
| **Metric**          | **<strong>score</strong>** | **matthews corr.**    | **acc**                | **f1-score/acc**              | **spearman/pearson corr.**    | **f1-score/acc**              | **acc**                 | **acc**                  | **acc**               | **acc**              | **acc**               | **matthews corr.**  |
| **BERT Base**       | 78.3                       | 52.1                  | 93.5                   | 88.9/84.8                     | 85.8/87.1                     | 71.2/89.2                     | 84.6                    | 83.4                     | 90.5                  | 66.4                 | 65.1                  | 34.2                |
| **ERNIE 2.0 Base**  | 80.6<br/>(**+2.3**)        | 55.2<br/>(**+3.1**)   | 95.0<br/>(**+1.5**)    | 89.9/86.1<br/>(**+1.0/+1.3**) | 86.5/87.6<br/>(**+0.7/+0.5**) | 73.2/89.8<br/>(**+2.0/+0.6**) | 86.1<br/>(**+1.5**)     | 85.5<br/>(**+2.1**)      | 92.9<br/>(**+2.4**)   | 74.8<br/>(**+8.4**)  | 65.1                  | 37.4<br/>(**+3.2**) |
| **BERT Large**      | 80.5                       | 60.5                  | 94.9                   | 89.3/85.4                     | 86.5/87.6                     | 72.1/89.3                     | 86.7                    | 85.9                     | 92.7                  | 70.1                 | 65.1                  | 39.6                |
| **ERNIE 2.0 Large** | 83.6<br/>(**+3.1**)        | 63.5<br/>(**+3.0**)   | 95.6<br/>(**+0.7**)    | 90.2/87.4<br/>(**+0.9/+2.0**) | 90.6/91.2<br/>(**+4.1/+3.6**) | 73.8/90.1<br/>(**+1.7/+0.8**) | 88.7<br/>(**+2.0**)     | 88.8<br/>(**+2.9**)      | 94.6<br/>(**+1.9**)   | 80.2<br/>(**+10.1**) | 67.8<br/>(**+2.7**)   | 48.0<br/>(**+8.4**) |

### Results on Chinese Datasets
#### Natural Language Inference

| Dataset                                        | XNLI                   |                        |
|------------------------------------------------|------------------------|------------------------|
|                      Metric                    |           acc          |                        |
|                                                |          dev           |          test          |
|          BERT Base                             | 78.1                   | 77.2                   |
|          ERNIE 1.0 Base                        | 79.9 (+1.8)            | 78.4 (+1.2)            |
|          ERNIE 2.0 Base                        | 81.2 (+3.1)            | 79.7 (+2.5)            |
|          ERNIE 2.0 Large                       | 82.6 (+4.5)            | 81.0 (+3.8)            |

#### Machine Reading Comprehension

| Dataset                                        | DuReader              |                            | CMRC2018              |                                     | DRCD                  |                        |                            |                        |
|------------------------------------------------|-----------------------|----------------------------|-----------------------|-------------------------------------|-----------------------|------------------------|----------------------------|------------------------|
|                      Metric                    |           em          |          f1-score          |          em           |          f1-score                   |          em           |                        |          f1-score          |                        |
|                                                |          dev          |                            |          dev          |                                     |          dev          |          test          |          dev               |          test          |
| BERT Base                                      | 59.5                  | 73.1                       | 66.3                  | 85.9                                | 85.7                  | 84.9                   | 91.6                       | 90.9                   |
| ERNIE 1.0 Base                                 | 57.9 (-1.6)           | 72.1 (-1.0)                | 65.1 (-1.2)           | 85.1 (-0.8)                         | 84.6 (-1.1)           | 84.0 (-0.9)            | 90.9 (-0.7)                | 90.5 (-0.4)            |
| ERNIE 2.0 Base                                 | 61.3 (+1.8)           | 74.9 (+1.8)                | 69.1 (+2.8)           | 88.6 (+2.7)                         | 88.5 (+2.8)           | 88.0 (+3.1)            | 93.8 (+2.2)                | 93.4 (+2.5)            |
| ERNIE 2.0 Large                                | 64.2 (+4.7)           | 77.3 (+4.2)                | 71.5 (+5.2)           | 89.9 (+4.0)                         | 89.7 (+4.0)           | 89.0 (+4.1)            | 94.7 (+3.1)                | 94.2 (+3.3)            |

#### Named Entity Recognition

| Dataset                                        | MSRA-NER (SIGHAN2006)       |                        |
|------------------------------------------------|-----------------------------|------------------------|
|                      Metric                    |           f1-score          |                        |
|                                                |          dev                |          test          |
| BERT Base                                      | 94.0                        | 92.6                   |
| ERNIE 1.0 Base                                 | 95.0 (+1.0)                 | 93.8 (+1.2)            |
| ERNIE 2.0 Base                                 | 95.2 (+1.2)                 | 93.8 (+1.2)            |
| ERNIE 2.0 Large                                | 96.3 (+2.3)                 | 95.0 (+2.4)            |

#### Sentiment Analysis Task

| Dataset                                        | ChnSentiCorp           |                        |
|------------------------------------------------|------------------------|------------------------|
|                      Metric                    |           acc          |                        |
|                                                |          dev           |          test          |
| BERT Base                                      | 94.6                   | 94.3                   |
| ERNIE 1.0 Base                                 | 95.2 (+0.6)            | 95.4 (+1.1)            |
| ERNIE 2.0 Base                                 | 95.7 (+1.1)            | 95.5 (+1.2)            |
| ERNIE 2.0 Large                                | 96.1 (+1.5)            | 95.8 (+1.5)            |

#### Question Answering Task

| Datset                                         | NLPCC2016-DBQA         |                        |                             |                        |
|------------------------------------------------|------------------------|------------------------|-----------------------------|------------------------|
|                      Metric                    |           mrr          |                        |           f1-score          |                        |
|                                                |          dev           |          test          |          dev                |          test          |
| BERT Base                                      | 94.7                   | 94.6                   | 80.7                        | 80.8                   |
| ERNIE 1.0 Base                                 | 95.0 (+0.3)            | 95.1 (+0.5)            | 82.3 (+1.6)                 | 82.7 (+1.9)            |
| ERNIE 2.0 Base                                 | 95.7 (+1.0)            | 95.7 (+1.1)            | 84.7 (+4.0)                 | 85.3 (+4.5)            |
| ERNIE 2.0 Large                                | 95.9 (+1.2)            | 95.8 (+1.2)            | 85.3 (+4.6)                 | 85.8 (+5.0)            |



#### Semantic Similarity

| Dataset                                        | LCQMC                 |                        | BQ Corpus             |                        |
|------------------------------------------------|-----------------------|------------------------|-----------------------|------------------------|
|                      Metric                    |           acc         |                        |           acc         |                        |
|                                                |          dev          |          test          |          dev          |          test          |
| BERT Base                                      | 88.8                  | 87.0                   | 85.9                  | 84.8                   |
| ERNIE 1.0 Base                                 | 89.7 (+0.9)           | 87.4 (+0.4)            | 86.1 (+0.2)           | 84.8                   |
| ERNIE 2.0 Base                                 | 90.9 (+2.1)           | 87.9 (+0.9)            | 86.4 (+0.5)           | 85.0 (+0.2)            |
| ERNIE 2.0 Large                                | 90.9 (+2.1)           | 87.9 (+0.9)            | 86.5 (+0.6)           | 85.2 (+0.4)            |

### Ernie 3.0[[12]](https://arxiv.org/pdf/2107.02137)

ERNIE 3.0 is a large-scale knowledge-enhanced pre-training language model released by Baidu. It combines both auto-regressive and auto-encoding network architectures, allowing it to perform natural language understanding (NLU) and natural language generation (NLG) tasks effectively under zero-shot, few-shot, or fine-tuning settings. The model includes a universal representation module that captures lexical and syntactic information shared across tasks and task-specific modules for specialized semantic representations.

![ernie3.0](https://miro.medium.com/1*1h26QVSrWSq94IF8UpoMaA.png)
<div align="center"><i>Ernie 3.0 framework</i></div>

ERNIE 3.0 has been widely applied in Baidu's AI products like search, newsfeeds, and smart speakers, and is also available through Baidu AI Cloud for industry use. A larger variant called ERNIE 3.0 Titan with 260 billion parameters has been developed, achieving state-of-the-art results across more than 60 NLP tasks and demonstrating strong few-shot learning capabilities. ERNIE 3.0 especially excels at leveraging world knowledge from knowledge graphs combined with text data, enhancing its performance over text-only models like GPT-3.

#### Ernie 3.0 Results

![superglue](https://research.baidu.com/ueditor/upload/20210712/1626069286263990.png)
ERNIE 3.0 was pre-trained on a massive dataset integrating both unstructured text and structured knowledge graph data, which helps it generate more coherent and accurate responses. It achieved state-of-the-art results in 54 Chinese NLP tasks and surpassed human performance on the SuperGLUE English language understanding benchmark with a score of 90.6%. The model architecture uses a Transformer-XL backbone with separate decoder networks for NLU and NLG, enabling strong performance in language understanding and generation. The model contains around 10 billion parameters.[[13]](https://research.baidu.com/Blog/index-view?id=160)

Ernie 3.0 was also tested under zero-shot setting for various tasks. Under these parameters, ERNIE 3.0 achieved good results and outperformed other large-scale language models at the time such as CPM and PanGu.

![cpmpangucomp](https://research.baidu.com/ueditor/upload/20210712/1626069329118575.png)

#### Results on Natural Language Understanding Tasks
![nlut](https://i0.wp.com/syncedreview.com/wp-content/uploads/2021/07/image-71.png?w=790&ssl=1)
![nlut2](https://i0.wp.com/syncedreview.com/wp-content/uploads/2021/07/image-70.png?w=790&ssl=1)

#### Results on Natural Language Generation Tasks
![nlgt](https://miro.medium.com/v2/resize:fit:720/format:webp/1*AYRWNTeEDtzizK4lf9mW5w.png)

#### LUGE Benchmark
![luge](https://miro.medium.com/v2/resize:fit:720/format:webp/1*gn8BCCnMxjhIa2z_XoXnZw.png)

#### Results on Zero-Shot Learning Tasks
![zslt](https://miro.medium.com/v2/resize:fit:720/format:webp/1*ksGWsF4Fz3jcEAwCaNklbg.png)
![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*iShy1hqUHMQSVuFiAwbRlA.png)

These strong benchmark performances reflect ERNIE 3.0’s advanced ability to combine lexical, syntactic, and world knowledge representations effectively across different languages and NLP tasks.

### Ernie 3.5

ERNIE 3.5 is an enhanced version of Baidu's ERNIE 3.0 model, introduced in 2023 as an upgraded foundation large language model. Building on the strengths of ERNIE 3.0, ERNIE 3.5 improves reasoning, language generation, and understanding capabilities. It was designed to better compete with other advanced large language models and notably narrows the performance gap with GPT-4, especially on Chinese language tasks and some multi-turn conversational contexts.

This model benefits from increased training data volume and refined architectural enhancements, making it more adept at complex question answering, summarization, and instruction following. ERNIE 3.5 serves as the core engine behind Baidu’s Ernie Bot chatbot upgrades around mid-2023, where it demonstrated significant improvements in contextual comprehension and more natural interaction.

Although specific parameter counts and architectural details are less publicly detailed than ERNIE 3.0, ERNIE 3.5 is recognized for delivering a substantial qualitative leap in practical applications, particularly for Chinese NLP use cases and multimodal tasks initiated later with ERNIE 4.0 and beyond.

#### Performance[[14]](https://gigazine.net/gsc_news/en/20230628-baidu-ernie-3-5/)

Ernie 3.5 is a Microsoft Research benchmark 'AGIEval' that measures indicators in tasks related to human cognition and ability, and a benchmark 'C-Eval' that measures Chinese performance jointly created by Shanghai Jiao Tong University, Tsinghua University, and Edinburgh University. ', Berkeley University, Columbia University, University of Illinois at Urbana-Champaign, and the University of Chicago jointly released a benchmark 'MMLU' that measures multitasking performance.

In these benchmark tests, it was reported that Ernie 3.5 outperformed GPT-3.5 in AGIEval and C-Eval, and the Chinese performance of AGIEval and C-Eval was GPT, the successor model of GPT-3.5. It was confirmed that it exceeded -4.

![](https://i.gzn.jp/img/2023/06/28/baidu-ernie-3-5/02_m.png)

### Ernie 4.0

ERNIE 4.0 is Baidu's fourth-generation large language model, unveiled in October 2023. It represents a significant upgrade over the previous ERNIE 3.0 and 3.5 versions with drastically improved performance in four core AI capabilities: natural language understanding, text generation, reasoning, and memory. This advancement allows ERNIE 4.0 to interpret complex sentences better, generate more coherent and contextually relevant content in real-time, and solve reasoning tasks such as geometry problems.[[15]](https://kr-asia.com/baidu-claims-its-latest-ai-model-ernie-4-0-is-on-par-with-openais-gpt-4)

One of ERNIE 4.0's standout features is its enhanced memory capability, allowing it to better retain and utilize learned knowledge for more informed responses. It can also understand and interpret images and video content, extending its multimodal capacities beyond just text. ERNIE 4.0 powers Baidu's AI chatbot "Ernie Bot" and is integrated into Baidu's products like Baidu Search, Baidu Maps, and Infoflow, enabling these services to provide richer, AI-driven features such as real-time content generation and intelligent search results summarization.[[16]](https://lucidityinsights.com/news/baidus-ernie-40-unveiled-to-challenge-gpt4)

Baidu has positioned ERNIE 4.0 to compete with other leading AI models like OpenAI's GPT-4, Microsoft’s Turing Series, and Google’s Bard, highlighting its strong capabilities in memory, generative writing, and reasoning. The model is accessible via APIs for enterprise applications, driving AI-native applications and intelligent automation across industries.[[17]](https://aimode.co/model/ernie-4/)

#### Ernie 4.0 Performance

ERNIE 4.0 has been benchmarked against leading large language models including OpenAI’s GPT-4, Microsoft’s Turing models, and Google’s Bard on a variety of tasks emphasizing natural language understanding, generation, reasoning, and multimodal capabilities[[18]](https://medium.com/@gaurav.gupta24/ernie-vs-gpt-4o-a-comprehensive-comparison-79cf8e69ae11). Key benchmark comparisons include:

- On both Chinese and English language tasks, ERNIE 4.0 reportedly matches or slightly exceeds GPT-4’s performance on many standard NLP benchmarks such as reading comprehension, commonsense reasoning, and multi-turn dialogue.
  
- It excels particularly in Chinese language processing, where it outperforms GPT-4 by a notable margin due to better native language optimization.
  
- ERNIE 4.0 demonstrates superior memory retention and reasoning in complex problem solving tasks, including math and logic puzzles when compared to GPT-4 and other competitors.
  
- Multimodal benchmarks involving image and video understanding show ERNIE 4.0 is competitive with the best multimodal models, reflecting its enhanced capabilities for contextual visual reasoning.
  
- While exact scores vary by benchmark, official Baidu claims and independent analyses emphasize that ERNIE 4.0 is positioned as a strong challenger to GPT-4, particularly in Asia-centric language tasks and applications.

Overall, ERNIE 4.0’s benchmarking highlights its capacity to combine advanced knowledge integration, reasoning, and multimodal inputs at performance levels close to or matching those of top Western AI models like GPT-4 on many standardized AI evaluation metrics.[[19]](https://kr-asia.com/baidu-claims-its-latest-ai-model-ernie-4-0-is-on-par-with-openais-gpt-4)

### Ernie 4.5

ERNIE 4.5 is the successor to ERNIE 4.0, released by Baidu in early 2025 as part of its multimodal large language model family[[20]](https://ernie.baidu.com/blog/posts/ernie4.5/). This generation represents a leap forward with several new capabilities and architectural innovations, making ERNIE 4.5 highly versatile in processing and reasoning over multiple modalities like text, images, audio, and video.[[21]](https://yiyan.baidu.com/blog/publication/ERNIE_Technical_Report.pdf)

Key features of ERNIE 4.5 include:

- Multimodal understanding and generation: ERNIE 4.5 excels at integrating and reasoning across visual and textual data, supporting tasks such as image captioning, video summarization, and cross-modal question answering.
  
- Mixture-of-Experts (MoE) architecture: It uses a sophisticated routing mechanism that isolates modalities while balancing computational resources, allowing efficient scaling up to ~424 billion parameters across different variants without prohibitive inference costs.
  
- Strong instruction following and knowledge integration: The model excels in instruction tuning, delivering accurate and context-aware responses for complex queries and multi-step reasoning problems.
  
- Large variant family: ERNIE 4.5 has over ten different model variants, ranging from compact dense models for resource-constrained environments to massive MoE models for high-end industrial applications.
  
- Open-source release: Baidu announced the open-sourcing of ERNIE 4.5 models and the accompanying ERNIEKit toolkit for fine-tuning and deployment, promoting wider adoption and innovation in AI research and applications.
  
- Deployment: ERNIE 4.5 powers enhanced versions of Baidu’s AI products, including Ernie Bot, and supports intelligent content creation, multimodal search, and document processing workloads.

ERNIE 4.5 is seen as a state-of-the-art multimodal AI system competing globally with models like GPT-4.5 and other advanced LLMs, especially excelling in scenarios requiring complex multimodal understanding, knowledge retention, and highly adaptable inference capabilities.

#### Ernie 4.5 Models

| ERNIE 4.5 Models               |                             | Model Information |                 |                  |
|--------------------------------|-----------------------------|-------------------|-----------------|------------------|
| Model Category                 | Model                       | Input Modality    | Output Modality | Context Window   |
| Large Language Models (LLMs)   | ERNIE-4.5-300B-A47B-Base    | Text              | Text            | 128K             |
|                                | ERNIE-4.5-300B-A47B         |                   |                 |                  |
|                                | ERNIE-4.5-21B-A3B-Base      |                   |                 |                  |
|                                | ERNIE-4.5-21B-A3B           |                   |                 |                  |
|  Vision-Language Models (VLMs) | ERNIE-4.5-VL-424B-A47B-Base | Text/Image/Video  | Text            | 128K             |
|                                | ERNIE-4.5-VL-424B-A47B      |                   |                 |                  |
|                                | ERNIE-4.5-VL-28B-A3B-Base   |                   |                 |                  |
|                                | ERNIE-4.5-VL-28B-A3B        |                   |                 |                  |
| Dense Models                   | ERNIE-4.5-0.3B-Base         | Text              | Text            | 128K             |
|                                | ERNIE-4.5-0.3B              |                   |                 |                  |

A more detailed look at Ernie 4.5 models specs.[[22]](https://blogs.novita.ai/ernie-vram-native-needs-high-novita-ai-needs-zero/)

| Model Name                  | Base Parameters | Active Parameters | Model Type | Modality      | Training Type |
|-----------------------------|-----------------|-------------------|------------|---------------|---------------|
| ERNIE 4.5 VL 424B A47B      | 424B            | 47B               | MoE        | Text & Vision | PT            |
| ERNIE 4.5 VL 424B A47B Base | 424B            | 47B               | MoE        | Text & Vision | Base          |
| ERNIE 4.5 VL 28B A3B        | 28B             | 3B                | MoE        | Text & Vision | PT            |
| ERNIE 4.5 VL 28B A3B Base   | 28B             | 3B                | MoE        | Text & Vision | Base          |
| ERNIE 4.5 300B A47B         | 300B            | 47B               | MoE        | Text          | PT            |
| ERNIE 4.5 300B A47B Base    | 300B            | 47B               | MoE        | Text          | Base          |
| ERNIE 4.5 21B A3B           | 21B             | 3B                | MoE        | Text          | PT            |
| ERNIE 4.5 21B A3B Base      | 21B             | 3B                | MoE        | Text          | Base          |
| ERNIE 4.5 0.3B              | 0.3B            | –                 | Dense      | Text          | PT            |
| ERNIE 4.5 0.3B Base         | 0.3B            | –                 | Dense      | Text          | Base          |

#### Ernie 4.5 Performance

ERNIE-4.5-300B-A47B-Base surpasses DeepSeek-V3-671B-A37B-Base on 22 out of 28 benchmarks, demonstrating leading performance across all major capability categories. This underscores the substantial improvements in generalization, reasoning, and knowledge-intensive tasks brought about by scaling up the ERNIE-4.5-Base model relative to other state-of-the-art large models. With a total parameter size of 21B (approximately 70% that of Qwen3-30B), ERNIE-4.5-21B-A3B-Base outperforms Qwen3-30B-A3B-Base on several math and reasoning benchmarks, including BBH and CMATH. ERNIE-4.5-21B-A3B-Base remains highly competitive given its significantly smaller model size, demonstrating notable parameter efficiency and favorable performance trade-offs.[[23]](https://github.com/PaddlePaddle/ERNIE)

#### Performace of ERNIE-4.5 pre-trained models

![](https://camo.githubusercontent.com/ddd95f5cdcdf354f2f7de788742ba684a9db962270c8aabac44bfbebce9a4207/68747470733a2f2f796979616e2e62616964752e636f6d2f626c6f672f706f7374732f65726e6965342e352f626173655f6d6f64656c5f62656e63686d61726b2e706e67)

#### Performance of post-trained model ERNIE-4.5-300B-A47B

![](https://camo.githubusercontent.com/3c98b957a09accc63fd36285a140bd8a474d061a2f24b8a71535309820000742/68747470733a2f2f796979616e2e62616964752e636f6d2f626c6f672f706f7374732f65726e6965342e352f636861745f6d6f64656c5f62656e63686d61726b312e706e67)

#### Performance of post-trained model ERNIE-4.5-21B-A3B

![](https://miro.medium.com/v2/resize:fit:720/format:webp/0*ESNXOvCpcaeANvMe)

#### Performance of post-trained multimodal models in thinking mode

![](https://camo.githubusercontent.com/5ffa942cf3417c6d2a343a74eb70f964f76ed45635fc94cafc7745a10b9ff7d3/68747470733a2f2f796979616e2e62616964752e636f6d2f626c6f672f706f7374732f65726e6965342e352f766c5f6d6f64656c5f7468696e6b696e675f62656e63686d61726b2e706e67)

#### Performance of post-trained multimodal models in non-thinking mode

![](https://i0.wp.com/blogs.novita.ai/wp-content/uploads/2025/07/9fbcfdf4-5858-40f0-8d60-2d497861f42d.jpeg?resize=1024%2C905&ssl=1)

### Ernie X1

ERNIE X1 is Baidu's specialized deep-thinking reasoning model announced in March 2025 as part of its next-generation large language model lineup. It is designed to excel in advanced reasoning, planning, reflection, and evolution tasks, making it distinct from the generalist ERNIE 4.5 model family. ERNIE X1 supports multimodal capabilities, including text, image, audio, and video understanding, and is Baidu’s first AI model capable of tool use, such as advanced search, document question-answering, image understanding, AI image generation, and webpage reading.

The model incorporates several innovative technologies such as "FlashMask" dynamic attention masking and a heterogeneous multimodal mixture-of-experts architecture, boosting its effectiveness and efficiency. ERNIE X1 has demonstrated strong performance in Chinese knowledge Q&A, literary creation, dialogue, logical reasoning, and complex calculations. It delivers performance comparable to leading models like DeepSeek R1 but at roughly half the operational cost.[[24]](https://xpert.digital/en/baidu-reaches-with-its-new-ai-models/)

In September 2025, Baidu released ERNIE X1.1, an upgraded version of X1, which substantially improves factual accuracy (up by 34.8%), instruction following (12.5% better), and agentic capabilities (9.6% enhancement). ERNIE X1.1 matches the performance of top-tier models such as GPT-5 and Gemini 2.5 Pro while being more cost-effective.[[25]](https://overchat.ai/models/ernie-x1-1#:~:text=ERNIE%20X1.1%20is%20Baidu's%20newest%20AI%20model%20released%20in%20September%202025)

### Ernie 5.0

ERNIE 5.0 is the next-generation large language model announced by Baidu in 2025 as the successor to the ERNIE 4.5 family.[[26]](https://global.chinadaily.com.cn/a/202511/13/WS691571bda310d6866eb29500.html) It represents a significant scale-up and architectural advancement, incorporating the latest innovations in mixture-of-experts (MoE) technology and multimodal learning.

Key aspects of ERNIE 5.0 include:
- Extremely large parameter scale, with planned configurations reaching up to approximately 2.4 trillion parameters, making it one of the largest MoE models globally.
  
- Enhanced multimodal capabilities, with improved integration and reasoning across text, images, audio, and video inputs.
  
- Advanced learning algorithms to boost instruction-following, reasoning, memory, and generalization performance beyond prior ERNIE versions.
  
- Designed to further close the gap with or surpass leading international models like GPT-4.5 and Gemini series in benchmarks and real-world applications.
  
- Expected to accelerate Baidu's AI ecosystem development through broader open-source releases and expanded AI Cloud services.

While detailed technical specifications and public benchmarks remain forthcoming, ERNIE 5.0 is positioned as a state-of-the-art AI foundation model for both industrial and research use, continuing Baidu's trajectory toward high-parameter, multimodal large language models with strong reasoning and knowledge capabilities.

### ERNIE-5.0-Preview-1022 LMArena results

In November of 2025 Baidu released Ernie 5.0 preview model. It's an early version released as a glimpse into the capabilities of Baidu’s next-generation large language model. It serves as a technological demonstration to showcase the advancements in scale, multimodal integration, and reasoning introduced with ERNIE 5.0.[[27]](https://ernie.baidu.com/blog/posts/ernie-5.0-preview-1022-release-on-lmarena/)

![](https://ernie.baidu.com/blog/posts/ernie-5.0-preview-1022-release-on-lmarena/ernie-5.0-preview-1022-release-on-lmarena.png)