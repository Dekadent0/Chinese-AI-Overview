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



### Ernie 1.0 and Ernie 2.0

Early versions (1.0 - 2.0) were primarily “pre-training / representation learning” models focusing on language understanding (not full multimodal chatbot functionality).

ERNIE 1.0 is language representation learning method enhanced by knowledge masking strategies, which includes entity-level masking and phrase-level masking. Inspired by the masking strategy of BERT [[11]](https://arxiv.org/abs/1810.04805), ERNIE introduced phrase masking and named entity masking and predicts the whole masked phrases or named entities. Phrase-level strategy masks the whole phrase which is a group of words that functions as a conceptual unit. Entity-level strategy masks named entities including persons, locations, organizations, products, etc., which can be denoted with proper names.

ERNIE 2.0 is a continual pre-training framework for language understanding in which pre-training tasks can be incrementally built and learned through multi-task learning. In this framework, different customized tasks can be incrementally introduced at any time. For example, the tasks including named entity prediction, discourse relation recognition, sentence order prediction are leveraged in order to enable the models to learn language representations.

#### Comparison of Ernie 1.0 and Ernie 2.0

| Tasks           | ERNIE 1.0                | ERNIE 2.0(en)                                                                        | ERNIE 2.0(zh)                         |
|-----------------|--------------------------|--------------------------------------------------------------------------------------|---------------------------------------|
| Word-aware      | Knowledge masking        | Knowledge Masking<br>Capitalization Prediction<br>Token-Document Relation Prediction | Knowledge Masking                     |
| Structure-aware | -                        | Sentence Reordering                                                                  | Sentence Reprdering Sentence Distance |
| Semantic-aware  | Next Sentence Prediction | Discourse Relation                                                                   | Discourse Relation IR Relevance       |

### Performance

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

### Performance on Chinese Datasets
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

### Ernie 3.0 and 3.5

![ernie3.0](https://miro.medium.com/1*1h26QVSrWSq94IF8UpoMaA.png)
<div align="center"><i>Ernie 3.0 framework</i>




