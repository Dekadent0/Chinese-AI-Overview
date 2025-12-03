# Overview of chinese AI models from Alibaba, Baidu and DeepSeek
## DeepSeek AI family
![deepseek](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/DeepSeek_logo.svg/960px-DeepSeek_logo.svg.png)

DeepSeek is a cutting-edge large language model (LLM) developed by a China-based AI company DeepSeek founded in 2023 by Liang Wenfeng.[[1]](https://en.wikipedia.org/wiki/DeepSeek) It is designed for software development, natural language processing, and business automation. DeepSeek stands out with its Mixture-of-Experts (MoE) system, which activates only 37 billion parameters of its total 671 billion parameters per task. This selective activation significantly reduces computational costs while maintaining high performance. It supports up to 128K tokens for long-context tasks, which is beneficial for complex problem-solving and large codebase management. The model is open-source, enabling developers and businesses to access advanced AI tools without heavy infrastructure costs.[[2]](https://daily.dev/blog/deepseek-everything-you-need-to-know-about-this-new-llm-in-one-place)

Use cases for DeepSeek include code generation and debugging automation, business workflow optimization, multilingual customer service chatbots, document summarization, educational tutoring, and rapid AI prototyping. However, its open-source model does carry some risks, such as the potential misuse for generating malware or other malicious activities.

### List of DeepSeek models[[3]](https://deepseeks.live/deepseek-release/)[[4]](https://deepseeks.live/deepseek-release/)[[5]](https://aiwiki.ai/wiki/DeepSeek)

| Series            | Model                           | Year                    | Notes                                                                                                                                          |
|-------------------|---------------------------------|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| DeepSeek-Coder    | DeepSeek-Coder-1.3B-base        | November 2023           | First model focused on code generation and software development tasks                                                                          |
|                   | DeepSeek-Coder-6.7B-base        | November 2023           |                                                                                                                                                |
|                   | DeepSeek-Coder-7B-base          | November 2023           |                                                                                                                                                |
|                   | DeepSeek-Coder-33B-base         | November 2023           |                                                                                                                                                |
|                   | DeepSeek-Coder-1.3B-instruct    | November 2023           |                                                                                                                                                |
|                   | DeepSeek-Coder-6.7B-instruct    | November 2023           |                                                                                                                                                |
|                   | DeepSeek-Coder-7B-instruct      | November 2023           |                                                                                                                                                |
|                   | DeepSeek-Coder-33B-instruct     | November 2023           |                                                                                                                                                |
| DeepSeek-LLM      | DeepSeek-LLM-7B-Base            | November 2023           | General-purpose language model                                                                                                                 |
| 	            | DeepSeek-LLM-67B-Base           | November 2023           | 				                                                                                                                 |
| 	            | DeepSeek-LLM-7B-Chat            | November 2023           | 				                                                                                                                 |
| 	            | DeepSeek-LLM-67B-Chat           | November 2023           | 				                                                                                                                 |
| DeepSeek-MoE      | DeepSeek-MoE-16B-Base           | January 2024            | More efficient and faster performance                                                                                                          |
|                   | DeepSeek-MoE-16B-Chat           | January 2024            |                                                                                                                                                |
| DeepSeek-VL       | DeepSeek-VL1.3B-Base            | March 2024              | Vision-language model that expanded DeepSeek beyond text-only LLMs                                                                             |
|                   | DeepSeek-VL1.3B-chat            | March 2024              |                                                                                                                                                |
|                   | DeepSeek-VL7B-base              | March 2024              |                                                                                                                                                |
|                   | DeepSeek-VL7B-chat              | March 2024              |                                                                                                                                                |
| DeepSeek-Math     | DeepSeek-Math-Base-7B           | April 2024              | AI support for solving mathematical problems designed to help in STEM fields                                                                   |
|                   | DeepSeek-Math-Instruct-7B       | April 2024              |                                                                                                                                                |
|                   | DeepSeek-Math-RL-7B             | April 2024              |                                                                                                                                                |
| DeepSeek-V2       | DeepSeek-V2                     | May 2024                | Major upgrade enhancing knowledge integration and NLP performance                                                                              |
|                   | DeepSeek-V2-Chat                | May 2024                |                                                                                                                                                |
|                   | DeepSeek-V2-Lite                | May 2024                | Smaller 16B variant                                                                                                                            |
|                   | DeepSeek-V2-Lite-Chat           | May 2024                |                                                                                                                                                |
| DeepSeek-Coder-V2 | DeepSeek-Coder-V2-Base          | June 2024               | MoE code language model with better performance, programming language support and context length                                               |
|                   | DeepSeek-Coder-V2-Lite-Base     | June 2024               |                                                                                                                                                |
|                   | DeepSeek-Coder-V2-Instruct      | June 2024               |                                                                                                                                                |
|                   | DeepSeek-Coder-V2-Lite-Instruct | June 2024               |                                                                                                                                                |
| DeepSeek-V2.5     | DeepSeek-V2.5                   | September-December 2024 | Upgraded version that combines DeepSeek-V2-Chat and DeepSeek-Coder-V2-Instruct                                                                 |
| 		    | DeepSeek-V2.5-1210              | September-December 2024 | Improved version of V2.5 										                                         |
| DeepSeek-V3       | DeepSeek-V3                     | December 2024           | Upgraded MoE model with 671B parameters, supporting 128K context, advanced reasoning and speed                                                 |
|                   | DeepSeek-V3-Base                | December 2024           |                                                                                                                                                |
| DeepSeek-VL2      | DeepSeek-VL2                    | December 2024           | An advanced MoE Vision-language model that improved upon DeepSeek-VL                                                                           |
|                   | DeepSeek-VL2-tiny               | December 2024           |                                                                                                                                                |
|                   | DeepSeek-VL2-small              | December 2024           |                                                                                                                                                |
| DeepSeek-R1       | DeepSeek-R1                     | January 2025            | Reasoning-focused model utilizing RL techniques                                                                                                |
|                   | DeepSeek-R1-Zero                | January 2025            | R1 & R1-Zero are trained based on DeepSeek-V3-Base                                                                                             |
|                   | DeepSeek-R1-Distill-Qwen-1.5B   | 2025                    | R1-Distill models are based on open-source models, using samples generated by DeepSeek-R1                                                      |
|                   | DeepSeek-R1-Distill-Qwen-7B     | 2025                    |                                                                                                                                                |
|                   | DeepSeek-R1-Distill-Qwen-14B    | 2025                    |                                                                                                                                                |
|                   | DeepSeek-R1-Distill-Qwen-32B    | 2025                    |                                                                                                                                                |
|                   | DeepSeek-R1-Distill-Llama-8B    | 2025                    |                                                                                                                                                |
|                   | DeepSeek-R1-Distill-Llama-70B   | 2025                    |                                                                                                                                                |
| DeepSeek-V3.1     | DeepSeek-V3.1                   | August 2025             | Hybrid model supporting thinking and non-thinking modes                                                                                        |
|                   | DeepSeek-V3.1-Base              | August 2025             |                                                                                                                                                |
|                   | DeepSeek-V3.1-Terminus          | September 2025          | Update addressing issues with Language consistency and Agent Capabilities                                                                      |
| DeepSeek-V3.2     | DeepSeek-V3.2                   | September 2025          | Model with DSA(DeepSeek Sparse Attention) mechanism, Scalable Reinforcement Learning Framework and Large-Scale Agentic Task Synthesis Pipeline |
|                   | DeepSeek-V3.2-Exp               | September 2025          | Experimental version                                                                                                                           |
|                   | DeepSeek-V3.2-Exp-Base          | September 2025          |                                                                                                                                                |
|                   | DeepSeek-V3.2-Speciale          | September 2025          | High-compute variant                                                                                                                           |


## DeepSeek-Coder

DeepSeek-Coder is an open-source series of code language models developed by DeepSeek AI, released in November 2023 as their first specialized coding model, trained from scratch on 87% code and 13% natural language data in English and Chinese. Each model is pre-trained on project-level code corpus by employing a window size of 16K and an extra fill-in-the-blank task, to support project-level code completion and infilling. For coding capabilities, DeepSeek Coder achieves state-of-the-art performance among open-source code models on multiple programming languages and various benchmarks.[[6]](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct)

![](https://github.com/deepseek-ai/DeepSeek-Coder/raw/main/pictures/result.png)

- Massive Training Data: Trained from scratch on 2T tokens, including 87% code and 13% linguistic data in both English and Chinese languages.
- Highly Flexible & Scalable: Offered in model sizes of 1.3B, 5.7B, 6.7B, and 33B, enabling users to choose the setup most suitable for their requirements.
- Superior Model Performance: State-of-the-art performance among publicly available code models on HumanEval, MultiPL-E, MBPP, DS-1000, and APPS benchmarks.
- Advanced Code Completion Capabilities: A window size of 16K and a fill-in-the-blank task, supporting project-level code completion and infilling tasks.


### List of DeepSeek-Coder models

| Series         | Model                        | Year          | Notes                                                                 |
|----------------|------------------------------|---------------|-----------------------------------------------------------------------|
| DeepSeek-Coder | DeepSeek-Coder-1.3B-base     | November 2023 | First model focused on code generation and software development tasks |
|                | DeepSeek-Coder-6.7B-base     | November 2023 |                                                                       |
|                | DeepSeek-Coder-7B-base       | November 2023 |                                                                       |
|                | DeepSeek-Coder-33B-base      | November 2023 |                                                                       |
|                | DeepSeek-Coder-1.3B-instruct | November 2023 |                                                                       |
|                | DeepSeek-Coder-6.7B-instruct | November 2023 |                                                                       |
|                | DeepSeek-Coder-7B-instruct   | November 2023 |                                                                       |
|                | DeepSeek-Coder-33B-instruct  | November 2023 |                                                                       |

### Performance[[7]](https://github.com/deepseek-ai/DeepSeek-Coder)
![](https://github.com/deepseek-ai/DeepSeek-Coder/raw/main/pictures/table.png)

The result shows that DeepSeek-Coder-Base-33B outperforms open-source code LLMs available at the time. Compared with CodeLlama-34B, it leads by 7.9%, 9.3%, 10.8% and 5.9% respectively on HumanEval Python, HumanEval Multilingual, MBPP and DS-1000. Surprisingly, our DeepSeek-Coder-Base-7B reaches the performance of CodeLlama-34B. The DeepSeek-Coder-Instruct-33B model after instruction tuning outperforms GPT35-turbo on HumanEval and achieves comparable results with GPT35-turbo on MBPP.

#### Multilingual HumanEval Benchmark
![](https://github.com/deepseek-ai/DeepSeek-Coder/raw/main/pictures/HumanEval.png)

#### MBPP Benchmark
![](https://github.com/deepseek-ai/DeepSeek-Coder/raw/main/pictures/MBPP.png)

#### DS-1000 Benchmark
![](https://github.com/deepseek-ai/DeepSeek-Coder/raw/main/pictures/DS-1000.png)

#### Program-Aid Math Reasoning Benchmark
![](https://github.com/deepseek-ai/DeepSeek-Coder/raw/main/pictures/Math.png)

## DeepSeek LLM

DeepSeek-LLM is an open-source family of large language models released in November 2023 by DeepSeek AI, featuring 7B and 67B parameter variants in both Base and Chat configurations, trained from scratch on 2 trillion tokens of English and Chinese data.[[8]](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat)

![](https://github.com/deepseek-ai/DeepSeek-LLM/raw/main/images/llm_radar.png)

- Superior General Capabilities: DeepSeek LLM 67B Base outperforms Llama2 70B Base in areas such as reasoning, coding, math, and Chinese comprehension.
- Proficient in Coding and Math: DeepSeek LLM 67B Chat exhibits outstanding performance in coding (HumanEval Pass@1: 73.78) and mathematics (GSM8K 0-shot: 84.1, Math 0-shot: 32.6). It also demonstrates remarkable generalization abilities, as evidenced by its exceptional score of 65 on the Hungarian National High School Exam.
- Mastery in Chinese Language: Based on our evaluation, DeepSeek LLM 67B Chat surpasses GPT-3.5 in Chinese.

### List of DeepSeek-LLM models

| Series            | Model                           | Year                    | Notes                          |
|-------------------|---------------------------------|-------------------------|--------------------------------|
| DeepSeek-LLM      | DeepSeek-LLM-7B-Base            | November 2023           | General-purpose language model |
| 	                | DeepSeek-LLM-67B-Base           | November 2023           | 				                 |
| 	                | DeepSeek-LLM-7B-Chat            | November 2023           | 				                 |
| 	                | DeepSeek-LLM-67B-Chat           | November 2023           | 				                 |

### Performance[[9]](https://github.com/deepseek-ai/DeepSeek-LLM)
The evaluation results are based on the internal, non-open-source hai-llm evaluation framework. 

| Model                 | Hella Swag | Trivia QA | MMLU   | GSM8K  | Human Eval | BBH    | CEval  | CMMLU  | Chinese QA |
|-----------------------|------------|-----------|--------|--------|------------|--------|--------|--------|------------|
|                       | 0-shot     | 5-shot    | 5-shot | 8-shot | 0-shot     | 3-shot | 5-shot | 5-shot | 5-shot     |
| LLaMA-2 -7B           | 75.6       | 63.8      | 45.8   | 15.5   | 14.6       | 38.5   | 33.9   | 32.6   | 21.5       |
| LLaMA-2 -70B          | 84.0       | 79.5      | 69.0   | 58.4   | 28.7       | 62.9   | 51.4   | 53.1   | 50.2       |
| DeepSeek LLM 7B Base  | 75.4       | 59.7      | 48.2   | 17.4   | 26.2       | 39.5   | 45.0   | 47.2   | 78.0       |
| DeepSeek LLM 67B Base | 84.0       | 78.9      | 71.3   | 63.4   | 42.7       | 68.7   | 66.1   | 70.8   | 87.6       |

#### Hungarian National High-School Exam
![](https://github.com/deepseek-ai/DeepSeek-LLM/raw/main/images/mathexam.png)

#### Instruction Following Evaluation
![](https://github.com/deepseek-ai/DeepSeek-LLM/raw/main/images/if_eval.png)

#### LeetCode Weekly Contest
Weekly Contest 351-372, Bi-Weekly Contest 108-117, from July 2023 to Nov 2023
![](https://github.com/deepseek-ai/DeepSeek-LLM/raw/main/images/leetcode.png)

#### Standard Benchmark

| Model                 | TriviaQA | MMLU | GSM8K | HumanEval | BBH  | C-Eval | CMMLU | ChineseQA |
|-----------------------|----------|------|-------|-----------|------|--------|-------|-----------|
| DeepSeek LLM 7B  Base | 59.7     | 48.2 | 17.4  | 26.2      | 39.5 | 45.0   | 47.2  | 78.0      |
| DeepSeek LLM 67B Base | 78.9     | 71.3 | 63.4  | 42.7      | 68.7 | 66.1   | 70.8  | 87.6      |
| DeepSeek LLM 7B Chat  | 57.9     | 49.4 | 62.6  | 48.2      | 42.3 | 47.0   | 49.7  | 75.0      |
| DeepSeek LLM 67B Chat | 81.5     | 71.1 | 84.1  | 73.8      | 71.7 | 65.2   | 67.8  | 85.1      |

Revisited Multi-Choice Question Benchmarks

 By incorporating multi-choice questions from Chinese exams, some results were improved, as depicted in the table below:

| Model                     | MMLU | C-Eval | CMMLU |
|---------------------------|------|--------|-------|
| DeepSeek LLM 7B Chat      | 49.4 | 47.0   | 49.7  |
| DeepSeek LLM 7B Chat + MC | 60.9 | 71.3   | 73.8  |

## DeepSeek MoE[[10]](https://huggingface.co/collections/deepseek-ai/deepseek-moe)

DeepSeekMoE 16B is a Mixture-of-Experts (MoE) language model with 16.4B parameters. It employs an innovative MoE architecture, which involves two principal strategies: fine-grained expert segmentation and shared experts isolation. It is trained from scratch on 2T English and Chinese tokens, and exhibits comparable performance with DeekSeek 7B and LLaMA2 7B, with only about 40% of computations.

### List of DeepSeek-MoE models

| Series       | Model                 | Year         | Notes                                 |
|--------------|-----------------------|--------------|---------------------------------------|
| DeepSeek-MoE | DeepSeek-MoE-16B-Base | January 2024 | More efficient and faster performance |
|              | DeepSeek-MoE-16B-Chat | January 2024 |                                       |

### Performance[[11]](https://github.com/deepseek-ai/DeepSeek-MoE)
### DeepSeek-MoE-16B-Base

![](https://github.com/deepseek-ai/DeepSeek-MoE/raw/main/images/evaluation_deepseekmoe16b_base_openllm.jpg)
![](https://github.com/deepseek-ai/DeepSeek-MoE/raw/main/images/evaluation_deepseekmoe16b_base_1.jpg)
![](https://github.com/deepseek-ai/DeepSeek-MoE/raw/main/images/evaluation_deepseekmoe16b_base_2.jpg)

### DeepSeek-MoE-16B-Chat

![](https://github.com/deepseek-ai/DeepSeek-MoE/raw/main/images/evaluation_deepseekmoe16b_chat.jpg)

## DeepSeek VL[[12]](https://huggingface.co/collections/deepseek-ai/deepseek-vl)

DeepSeek-VL, is an open-source Vision-Language (VL) Model designed for real-world vision and language understanding applications. DeepSeek-VL possesses general multimodal understanding capabilities, capable of processing logical diagrams, web pages, formula recognition, scientific literature, natural images, and embodied intelligence in complex scenarios.

### List of DeepSeek-VL models

| Series      | Model                | Year       | Notes                                                              |
|-------------|----------------------|------------|--------------------------------------------------------------------|
| DeepSeek-VL | DeepSeek-VL1.3B-Base | March 2024 | Vision-language model that expanded DeepSeek beyond text-only LLMs |
|             | DeepSeek-VL1.3B-chat | March 2024 |                                                                    |
|             | DeepSeek-VL7B-base   | March 2024 |                                                                    |
|             | DeepSeek-VL7B-chat   | March 2024 |                                                                    |

## DeepSeek-Math[[13]](https://huggingface.co/collections/deepseek-ai/deepseek-math)

DeepSeek-Math is an open-source series of large language models from DeepSeek AI, first released in April 2024, specializing in mathematical reasoning, problem-solving, and theorem proving through advanced reinforcement learning techniques like GRPO. It's initialized with DeepSeek-Coder-v1.5 7B and continues pre-training on math-related tokens sourced from Common Crawl, together with natural language and code data for 500B tokens.

### List of DeepSeek Math models

| Series        | Model                     | Year       | Notes                                                                        |
|---------------|---------------------------|------------|------------------------------------------------------------------------------|
| DeepSeek-Math | DeepSeek-Math-Base-7B     | April 2024 | AI support for solving mathematical problems designed to help in STEM fields |
|               | DeepSeek-Math-Instruct-7B | April 2024 |                                                                              |
|               | DeepSeek-Math-RL-7B       | April 2024 |                                                                              |

### Performance[[14]](https://github.com/deepseek-ai/DeepSeek-Math)
#### MATH benchmark

![](https://github.com/deepseek-ai/DeepSeek-Math/raw/main/images/math.png)

### DeepSeekMath-Base-7B
#### Mathematical problem solving with step-by-step reasoning
![](https://github.com/deepseek-ai/DeepSeek-Math/raw/main/images/base_results_1.png)

#### Mathematical problem solving with tool use
![](https://github.com/deepseek-ai/DeepSeek-Math/raw/main/images/base_results_2.png)

#### Natural Language Understanding, Reasoning, and Code
![](https://github.com/deepseek-ai/DeepSeek-Math/raw/main/images/base_results_3.png)

### DeepSeekMath-Instruct and -RL-7B
![](https://github.com/deepseek-ai/DeepSeek-Math/raw/main/images/instruct_results.png)

## DeepSeek V2

DeepSeek-V2 is an advanced open-source large language model series from DeepSeek AI, released in May 2024, featuring a Mixture-of-Experts (MoE) architecture with 236B total parameters but only 21B active per token for high efficiency.[[15]](https://huggingface.co/collections/deepseek-ai/deepseek-v2)

### List of DeepSeek-V2 models

| Series      | Model                 | Year     | Notes                                                             |
|-------------|-----------------------|----------|-------------------------------------------------------------------|
| DeepSeek-V2 | DeepSeek-V2           | May 2024 | Major upgrade enhancing knowledge integration and NLP performance |
|             | DeepSeek-V2-Chat      | May 2024 |                                                                   |
|             | DeepSeek-V2-Lite      | May 2024 | Smaller 16B variant                                               |
|             | DeepSeek-V2-Lite-Chat | May 2024 |                                                                   |

### Performance[[16]](https://github.com/deepseek-ai/DeepSeek-V2)

Compared with DeepSeek 67B, DeepSeek-V2 achieves stronger performance, and meanwhile saves 42.5% of training costs, reduces the KV cache by 93.3%, and boosts the maximum generation throughput to 5.76 times.

![](https://github.com/deepseek-ai/DeepSeek-V2/raw/main/figures/activationparameters.png?raw=true)
![](https://github.com/deepseek-ai/DeepSeek-V2/raw/main/figures/trainingcost.png?raw=true)

DeepSeek-V2 was pretrained on a diverse and high-quality corpus comprising 8.1 trillion tokens. This comprehensive pretraining was followed by a process of Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) to fully unleash the model's capabilities.

### Base Model
#### Standard Benchmark (Models larger than 67B)

| Benchmark | Domain  | LLaMA3 70B | Mixtral 8x22B | DeepSeek-V1 (Dense-67B) | DeepSeek-V2 (MoE-236B) |
|-----------|---------|------------|---------------|-------------------------|------------------------|
| MMLU      | English | 78.9       | 77.6          | 71.3                    | 78.5                   |
| BBH       | English | 81.0       | 78.9          | 68.7                    | 78.9                   |
| C-Eval    | Chinese | 67.5       | 58.6          | 66.1                    | 81.7                   |
| CMMLU     | Chinese | 69.3       | 60.0          | 70.8                    | 84.0                   |
| HumanEval | Code    | 48.2       | 53.1          | 45.1                    | 48.8                   |
| MBPP      | Code    | 68.6       | 64.2          | 57.4                    | 66.6                   |
| GSM8K     | Math    | 83.0       | 80.3          | 63.4                    | 79.2                   |
| Math      | Math    | 42.2       | 42.5          | 18.7                    | 43.6                   |

#### Standard Benchmark (Models smaller than 16B)

| Benchmark    | Domain  | DeepSeek 7B (Dense) | DeepSeekMoE 16B | DeepSeek-V2-Lite (MoE-16B) |
|--------------|---------|---------------------|-----------------|----------------------------|
| Architecture | -       | MHA+Dense           | MHA+MoE         | MLA+MoE                    |
| MMLU         | English | 48.2                | 45.0            | 58.3                       |
| BBH          | English | 39.5                | 38.9            | 44.1                       |
| C-Eval       | Chinese | 45.0                | 40.6            | 60.3                       |
| CMMLU        | Chinese | 47.2                | 42.5            | 64.3                       |
| HumanEval    | Code    | 26.2                | 26.8            | 29.9                       |
| MBPP         | Code    | 39.0                | 39.2            | 43.2                       |
| GSM8K        | Math    | 17.4                | 18.8            | 41.1                       |
| Math         | Math    | 3.3                 | 4.3             | 17.1                       |

### Chat Model
#### Standard Benchmark (Models larger than 67B)

| Benchmark                  | Domain  | QWen1.5 72B Chat | Mixtral 8x22B | LLaMA3 70B Instruct | DeepSeek-V1 Chat (SFT) | DeepSeek-V2 Chat (SFT) | DeepSeek-V2 Chat (RL) |
|----------------------------|---------|------------------|---------------|---------------------|------------------------|------------------------|-----------------------|
| MMLU                       | English | 76.2             | 77.8          | 80.3                | 71.1                   | 78.4                   | 77.8                  |
| BBH                        | English | 65.9             | 78.4          | 80.1                | 71.7                   | 81.3                   | 79.7                  |
| C-Eval                     | Chinese | 82.2             | 60.0          | 67.9                | 65.2                   | 80.9                   | 78.0                  |
| CMMLU                      | Chinese | 82.9             | 61.0          | 70.7                | 67.8                   | 82.4                   | 81.6                  |
| HumanEval                  | Code    | 68.9             | 75.0          | 76.2                | 73.8                   | 76.8                   | 81.1                  |
| MBPP                       | Code    | 52.2             | 64.4          | 69.8                | 61.4                   | 70.4                   | 72.0                  |
| LiveCodeBench  (0901-0401) | Code    | 18.8             | 25.0          | 30.5                | 18.3                   | 28.7                   | 32.5                  |
| GSM8K                      | Math    | 81.9             | 87.9          | 93.2                | 84.1                   | 90.8                   | 92.2                  |
| Math                       | Math    | 40.6             | 49.8          | 48.5                | 32.6                   | 52.7                   | 53.9                  |

#### Standard Benchmark (Models smaller than 16B)

| Benchmark | Domain  | DeepSeek 7B Chat (SFT) | DeepSeekMoE 16B Chat (SFT) | DeepSeek-V2-Lite 16B Chat (SFT) |
|-----------|---------|------------------------|----------------------------|---------------------------------|
| MMLU      | English | 49.7                   | 47.2                       | 55.7                            |
| BBH       | English | 43.1                   | 42.2                       | 48.1                            |
| C-Eval    | Chinese | 44.7                   | 40.0                       | 60.1                            |
| CMMLU     | Chinese | 51.2                   | 49.3                       | 62.5                            |
| HumanEval | Code    | 45.1                   | 45.7                       | 57.3                            |
| MBPP      | Code    | 39.0                   | 46.2                       | 45.8                            |
| GSM8K     | Math    | 62.6                   | 62.2                       | 72.0                            |
| Math      | Math    | 14.7                   | 15.2                       | 27.9                            |

#### English Open Ended Generation Evaluation
![](https://github.com/deepseek-ai/DeepSeek-V2/raw/main/figures/mtbench.png?raw=true)

#### Chinese Open Ended Generation Evaluation

| 模型                              | 开源/闭源  | 总分 | 中文推理  | 中文语言  |
|-----------------------------------|-----------|------|----------|----------|
| gpt-4-1106-preview                | 闭源      | 8.01 | 7.73     | 8.29     |
| DeepSeek-V2 Chat (RL)             | 开源      | 7.91 | 7.45     | 8.36     |
| erniebot-4.0-202404 (文心一言)     | 闭源      | 7.89 | 7.61     | 8.17     |
| DeepSeek-V2 Chat (SFT)            | 开源      | 7.74 | 7.30     | 8.17     |
| gpt-4-0613                        | 闭源      | 7.53 | 7.47     | 7.59     |
| erniebot-4.0-202312 (文心一言)     | 闭源      | 7.36 | 6.84     | 7.88     |
| moonshot-v1-32k-202404 (月之暗面)  | 闭源      | 7.22 | 6.42     | 8.02     |
| Qwen1.5-72B-Chat (通义千问)        | 开源      | 7.19 | 6.45     | 7.93     |
| DeepSeek-67B-Chat                 | 开源      | 6.43 | 5.75     | 7.11     |
| Yi-34B-Chat (零一万物)             | 开源      | 6.12 | 4.86     | 7.38     |
| gpt-3.5-turbo-0613                | 闭源      | 6.08 | 5.35     | 6.71     |
| DeepSeek-V2-Lite 16B Chat         | 开源      | 6.01 | 4.71     | 7.32     |

#### Coding Benchmarks
![](https://github.com/deepseek-ai/DeepSeek-V2/raw/main/figures/code_benchmarks.png?raw=true)

## DeepSeek-Coder V2

DeepSeek-Coder-V2 is an open-source Mixture-of-Experts (MoE) code language model from DeepSeek AI, released in 2024 as an upgrade to the original DeepSeek-Coder, with 236B total parameters (21B active in instruct variant) or a lighter 16B version (2.4B active).[[17]](https://huggingface.co/collections/deepseek-ai/deepseekcoder-v2)

### List of DeepSeek-Coder-V2 models

| Series            | Model                           | Year      | Notes                                                                                            |
|-------------------|---------------------------------|-----------|--------------------------------------------------------------------------------------------------|
| DeepSeek-Coder-V2 | DeepSeek-Coder-V2-Base          | June 2024 | MoE code language model with better performance, programming language support and context length |
|                   | DeepSeek-Coder-V2-Lite-Base     | June 2024 |                                                                                                  |
|                   | DeepSeek-Coder-V2-Instruct      | June 2024 |                                                                                                  |
|                   | DeepSeek-Coder-V2-Lite-Instruct | June 2024 |                                                                                                  |

### Performance[[18]](https://github.com/deepseek-ai/DeepSeek-Coder-V2)

#### Code Generation

|                                 | #TP  | #AP  | HumanEval | MBPP+ | LiveCodeBench | USACO |
|---------------------------------|------|------|-----------|-------|---------------|-------|
| Closed-Source Models            |      |      |           |       |               |       |
| Gemini-1.5-Pro                  | -    | -    | 83.5      | 74.6  | 34.1          | 4.9   |
| Claude-3-Opus                   | -    | -    | 84.2      | 72.0  | 34.6          | 7.8   |
| GPT-4-Turbo-1106                | -    | -    | 87.8      | 69.3  | 37.1          | 11.1  |
| GPT-4-Turbo-0409                | -    | -    | 88.2      | 72.2  | 45.7          | 12.3  |
| GPT-4o-0513                     | -    | -    | 91.0      | 73.5  | 43.4          | 18.8  |
| Open-Source Models              |      |      |           |       |               |       |
| CodeStral                       | 22B  | 22B  | 78.1      | 68.2  | 31.0          | 4.6   |
| DeepSeek-Coder-Instruct         | 33B  | 33B  | 79.3      | 70.1  | 22.5          | 4.2   |
| Llama3-Instruct                 | 70B  | 70B  | 81.1      | 68.8  | 28.7          | 3.3   |
| DeepSeek-Coder-V2-Lite-Instruct | 16B  | 2.4B | 81.1      | 68.8  | 24.3          | 6.5   |
| DeepSeek-Coder-V2-Instruct      | 236B | 21B  | 90.2      | 76.2  | 43.4          | 12.1  |

#### Code Completion

| Model                       | #TP | #AP  | RepoBench (Python) | RepoBench (Java) | HumanEval FIM |
|-----------------------------|-----|------|--------------------|------------------|---------------|
| CodeStral                   | 22B | 22B  | 46.1               | 45.7             | 83.0          |
| DeepSeek-Coder-Base         | 7B  | 7B   | 36.2               | 43.3             | 86.1          |
| DeepSeek-Coder-Base         | 33B | 33B  | 39.1               | 44.8             | 86.4          |
| DeepSeek-Coder-V2-Lite-Base | 16B | 2.4B | 38.9               | 43.3             | 86.4          |

#### Code Fixing

|                                 | #TP  | #AP  | Defects4J | SWE-Bench | Aider |
|---------------------------------|------|------|-----------|-----------|-------|
| Closed-Source Models            |      |      |           |           |       |
| Gemini-1.5-Pro                  | -    | -    | 18.6      | 19.3      | 57.1  |
| Claude-3-Opus                   | -    | -    | 25.5      | 11.7      | 68.4  |
| GPT-4-Turbo-1106                | -    | -    | 22.8      | 22.7      | 65.4  |
| GPT-4-Turbo-0409                | -    | -    | 24.3      | 18.3      | 63.9  |
| GPT-4o-0513                     | -    | -    | 26.1      | 26.7      | 72.9  |
| Open-Source Models              |      |      |           |           |       |
| CodeStral                       | 22B  | 22B  | 17.8      | 2.7       | 51.1  |
| DeepSeek-Coder-Instruct         | 33B  | 33B  | 11.3      | 0.0       | 54.5  |
| Llama3-Instruct                 | 70B  | 70B  | 16.2      | -         | 49.2  |
| DeepSeek-Coder-V2-Lite-Instruct | 16B  | 2.4B | 9.2       | 0.0       | 44.4  |
| DeepSeek-Coder-V2-Instruct      | 236B | 21B  | 21.0      | 12.7      | 73.7  |

#### Mathematical Reasoning

|                                 | #TP  | #AP  | Defects4J | SWE-Bench | Aider |
|---------------------------------|------|------|-----------|-----------|-------|
| Closed-Source Models            |      |      |           |           |       |
| Gemini-1.5-Pro                  | -    | -    | 18.6      | 19.3      | 57.1  |
| Claude-3-Opus                   | -    | -    | 25.5      | 11.7      | 68.4  |
| GPT-4-Turbo-1106                | -    | -    | 22.8      | 22.7      | 65.4  |
| GPT-4-Turbo-0409                | -    | -    | 24.3      | 18.3      | 63.9  |
| GPT-4o-0513                     | -    | -    | 26.1      | 26.7      | 72.9  |
| Open-Source Models              |      |      |           |           |       |
| CodeStral                       | 22B  | 22B  | 17.8      | 2.7       | 51.1  |
| DeepSeek-Coder-Instruct         | 33B  | 33B  | 11.3      | 0.0       | 54.5  |
| Llama3-Instruct                 | 70B  | 70B  | 16.2      | -         | 49.2  |
| DeepSeek-Coder-V2-Lite-Instruct | 16B  | 2.4B | 9.2       | 0.0       | 44.4  |
| DeepSeek-Coder-V2-Instruct      | 236B | 21B  | 21.0      | 12.7      | 73.7  |

#### General Natural Language

| Benchmark        | Domain  | DeepSeek-V2-Lite Chat | DeepSeek-Coder-V2-Lite Instruct | DeepSeek-V2 Chat | DeepSeek-Coder-V2 Instruct |
|------------------|---------|-----------------------|---------------------------------|------------------|----------------------------|
| BBH              | English | 48.1                  | 61.2                            | 79.7             | 83.9                       |
| MMLU             | English | 55.7                  | 60.1                            | 78.1             | 79.2                       |
| ARC-Easy         | English | 86.1                  | 88.9                            | 98.1             | 97.4                       |
| ARC-Challenge    | English | 73.4                  | 77.4                            | 92.3             | 92.8                       |
| TriviaQA         | English | 65.2                  | 59.5                            | 86.7             | 82.3                       |
| NaturalQuestions | English | 35.5                  | 30.8                            | 53.4             | 47.5                       |
| AGIEval          | English | 42.8                  | 28.7                            | 61.4             | 60                         |
| CLUEWSC          | Chinese | 80.0                  | 76.5                            | 89.9             | 85.9                       |
| C-Eval           | Chinese | 60.1                  | 61.6                            | 78.0             | 79.4                       |
| CMMLU            | Chinese | 62.5                  | 62.7                            | 81.6             | 80.9                       |
| Arena-Hard       | -       | 11.4                  | 38.1                            | 41.6             | 65.0                       |
| AlpaceEval 2.0   | -       | 16.9                  | 17.7                            | 38.9             | 36.9                       |
| MT-Bench         | -       | 7.37                  | 7.81                            | 8.97             | 8.77                       |
| Alignbench       | -       | 6.02                  | 6.83                            | 7.91             | 7.84                       |

#### Context Window

Evaluation results on the Needle In A Haystack (NIAH) tests. DeepSeek-Coder-V2 performs well across all context window lengths up to 128K.
![](https://github.com/deepseek-ai/DeepSeek-Coder-V2/raw/main/figures/long_context.png)

## DeepSeek V2.5

DeepSeek-V2.5 is an upgraded version that combines DeepSeek-V2-Chat and DeepSeek-Coder-V2-Instruct. The new model integrates the general and coding abilities of the two previous versions.[[19]](https://huggingface.co/deepseek-ai/DeepSeek-V2.5) 

### List of DeepSeek-V2.5 models

| Series        | Model              | Year                    | Notes                                                                          |
|---------------|--------------------|-------------------------|--------------------------------------------------------------------------------|
| DeepSeek-V2.5 | DeepSeek-V2.5      | September-December 2024 | Upgraded version that combines DeepSeek-V2-Chat and DeepSeek-Coder-V2-Instruct |
|               | DeepSeek-V2.5-1210 | September-December 2024 | Improved version of V2.5                                                       |

### Performance

| Metric               | DeepSeek-V2-0628 | DeepSeek-Coder-V2-0724 | DeepSeek-V2.5 |
|----------------------|------------------|------------------------|---------------|
| AlpacaEval 2.0       | 46.6             | 44.5                   | 50.5          |
| ArenaHard            | 68.3             | 66.3                   | 76.2          |
| AlignBench           | 7.88             | 7.91                   | 8.04          |
| MT-Bench             | 8.85             | 8.91                   | 9.02          |
| HumanEval python     | 84.5             | 87.2                   | 89            |
| HumanEval Multi      | 73.8             | 74.8                   | 73.8          |
| LiveCodeBench(01-09) | 36.6             | 39.7                   | 41.8          |
| Aider                | 69.9             | 72.9                   | 72.2          |
| SWE-verified         | N/A              | 19                     | 16.8          |
| DS-FIM-Eval          | N/A              | 73.2                   | 78.3          |
| DS-Arena-Code        | N/A              | 49.5                   | 63.1          |

## DeepSeek V3

DeepSeek-V3 is an open-source Mixture-of-Experts (MoE) large language model from DeepSeek AI, released on December 26, 2024, with 671 billion total parameters (37 billion active per token) for efficient scaling. To achieve efficient inference and cost-effective training, DeepSeek-V3 adopts Multi-head Latent Attention (MLA) and DeepSeekMoE architectures, which were thoroughly validated in DeepSeek-V2. Furthermore, DeepSeek-V3 pioneers an auxiliary-loss-free strategy for load balancing and sets a multi-token prediction training objective for stronger performance. DeepSeek-V3 was pretrained on 14.8 trillion diverse and high-quality tokens, followed by Supervised Fine-Tuning and Reinforcement Learning stages to fully harness its capabilities.

### List of DeepSeek-V3 models[[20]](https://huggingface.co/collections/deepseek-ai/deepseek-v3)

| Series      | Model            | Year          | Notes                                                                                          |
|-------------|------------------|---------------|------------------------------------------------------------------------------------------------|
| DeepSeek-V3 | DeepSeek-V3      | December 2024 | Upgraded MoE model with 671B parameters, supporting 128K context, advanced reasoning and speed |
|             | DeepSeek-V3-Base | December 2024 |                                                                                                |

### Performance[[21]](https://github.com/deepseek-ai/DeepSeek-V3)

Comprehensive evaluations reveal that DeepSeek-V3 outperforms other open-source models and achieves performance comparable to leading closed-source models. Despite its excellent performance, DeepSeek-V3 requires only 2.788M H800 GPU hours for its full training. In addition, its training process is remarkably stable. Throughout the entire training process, we did not experience any irrecoverable loss spikes or perform any rollbacks.

![](https://github.com/deepseek-ai/DeepSeek-V3/raw/main/figures/benchmark.png)

### Base Model
#### Standart benchmarks

|              | Benchmark (Metric)          | # Shots | DeepSeek-V2 | Qwen2.5 72B | LLaMA3.1 405B | DeepSeek-V3 |
|--------------|-----------------------------|---------|-------------|-------------|---------------|-------------|
|              | Architecture                | -       | MoE         | Dense       | Dense         | MoE         |
|              | # Activated Params          | -       | 21B         | 72B         | 405B          | 37B         |
|              | # Total Params              | -       | 236B        | 72B         | 405B          | 671B        |
| English      | Pile-test (BPB)             | -       | 0.606       | 0.638       | 0.542         | 0.548       |
|              | BBH (EM)                    | 3-shot  | 78.8        | 79.8        | 82.9          | 87.5        |
|              | MMLU (Acc.)                 | 5-shot  | 78.4        | 85.0        | 84.4          | 87.1        |
|              | MMLU-Redux (Acc.)           | 5-shot  | 75.6        | 83.2        | 81.3          | 86.2        |
|              | MMLU-Pro (Acc.)             | 5-shot  | 51.4        | 58.3        | 52.8          | 64.4        |
|              | DROP (F1)                   | 3-shot  | 80.4        | 80.6        | 86.0          | 89.0        |
|              | ARC-Easy (Acc.)             | 25-shot | 97.6        | 98.4        | 98.4          | 98.9        |
|              | ARC-Challenge (Acc.)        | 25-shot | 92.2        | 94.5        | 95.3          | 95.3        |
|              | HellaSwag (Acc.)            | 10-shot | 87.1        | 84.8        | 89.2          | 88.9        |
|              | PIQA (Acc.)                 | 0-shot  | 83.9        | 82.6        | 85.9          | 84.7        |
|              | WinoGrande (Acc.)           | 5-shot  | 86.3        | 82.3        | 85.2          | 84.9        |
|              | RACE-Middle (Acc.)          | 5-shot  | 73.1        | 68.1        | 74.2          | 67.1        |
|              | RACE-High (Acc.)            | 5-shot  | 52.6        | 50.3        | 56.8          | 51.3        |
|              | TriviaQA (EM)               | 5-shot  | 80.0        | 71.9        | 82.7          | 82.9        |
|              | NaturalQuestions (EM)       | 5-shot  | 38.6        | 33.2        | 41.5          | 40.0        |
|              | AGIEval (Acc.)              | 0-shot  | 57.5        | 75.8        | 60.6          | 79.6        |
| Code         | HumanEval (Pass@1)          | 0-shot  | 43.3        | 53.0        | 54.9          | 65.2        |
|              | MBPP (Pass@1)               | 3-shot  | 65.0        | 72.6        | 68.4          | 75.4        |
|              | LiveCodeBench-Base (Pass@1) | 3-shot  | 11.6        | 12.9        | 15.5          | 19.4        |
|              | CRUXEval-I (Acc.)           | 2-shot  | 52.5        | 59.1        | 58.5          | 67.3        |
|              | CRUXEval-O (Acc.)           | 2-shot  | 49.8        | 59.9        | 59.9          | 69.8        |
| Math         | GSM8K (EM)                  | 8-shot  | 81.6        | 88.3        | 83.5          | 89.3        |
|              | MATH (EM)                   | 4-shot  | 43.4        | 54.4        | 49.0          | 61.6        |
|              | MGSM (EM)                   | 8-shot  | 63.6        | 76.2        | 69.9          | 79.8        |
|              | CMath (EM)                  | 3-shot  | 78.7        | 84.5        | 77.3          | 90.7        |
| Chinese      | CLUEWSC (EM)                | 5-shot  | 82.0        | 82.5        | 83.0          | 82.7        |
|              | C-Eval (Acc.)               | 5-shot  | 81.4        | 89.2        | 72.5          | 90.1        |
|              | CMMLU (Acc.)                | 5-shot  | 84.0        | 89.5        | 73.7          | 88.8        |
|              | CMRC (EM)                   | 1-shot  | 77.4        | 75.8        | 76.0          | 76.3        |
|              | C3 (Acc.)                   | 0-shot  | 77.4        | 76.7        | 79.7          | 78.6        |
|              | CCPM (Acc.)                 | 0-shot  | 93.0        | 88.5        | 78.6          | 92.0        |
| Multilingual | MMMLU-non-English (Acc.)    | 5-shot  | 64.0        | 74.8        | 73.8          | 79.4        |

#### Context Window
![](https://github.com/deepseek-ai/DeepSeek-V3/raw/main/figures/niah.png)

### Chat Model
#### Standart benchmarks

|         | Benchmark (Metric)         | DeepSeek V2-0506 | DeepSeek V2.5-0905 | Qwen2.5 72B-Inst. | Llama3.1 405B-Inst. | Claude-3.5-Sonnet-1022 | GPT-4o 0513 | DeepSeek V3 |
|---------|----------------------------|------------------|--------------------|-------------------|---------------------|------------------------|-------------|-------------|
|         | Architecture               | MoE              | MoE                | Dense             | Dense               | -                      | -           | MoE         |
|         | # Activated Params         | 21B              | 21B                | 72B               | 405B                | -                      | -           | 37B         |
|         | # Total Params             | 236B             | 236B               | 72B               | 405B                | -                      | -           | 671B        |
| English | MMLU (EM)                  | 78.2             | 80.6               | 85.3              | 88.6                | 88.3                   | 87.2        | 88.5        |
|         | MMLU-Redux (EM)            | 77.9             | 80.3               | 85.6              | 86.2                | 88.9                   | 88.0        | 89.1        |
|         | MMLU-Pro (EM)              | 58.5             | 66.2               | 71.6              | 73.3                | 78.0                   | 72.6        | 75.9        |
|         | DROP (3-shot F1)           | 83.0             | 87.8               | 76.7              | 88.7                | 88.3                   | 83.7        | 91.6        |
|         | IF-Eval (Prompt Strict)    | 57.7             | 80.6               | 84.1              | 86.0                | 86.5                   | 84.3        | 86.1        |
|         | GPQA-Diamond (Pass@1)      | 35.3             | 41.3               | 49.0              | 51.1                | 65.0                   | 49.9        | 59.1        |
|         | SimpleQA (Correct)         | 9.0              | 10.2               | 9.1               | 17.1                | 28.4                   | 38.2        | 24.9        |
|         | FRAMES (Acc.)              | 66.9             | 65.4               | 69.8              | 70.0                | 72.5                   | 80.5        | 73.3        |
|         | LongBench v2 (Acc.)        | 31.6             | 35.4               | 39.4              | 36.1                | 41.0                   | 48.1        | 48.7        |
| Code    | HumanEval-Mul (Pass@1)     | 69.3             | 77.4               | 77.3              | 77.2                | 81.7                   | 80.5        | 82.6        |
|         | LiveCodeBench (Pass@1-COT) | 18.8             | 29.2               | 31.1              | 28.4                | 36.3                   | 33.4        | 40.5        |
|         | LiveCodeBench (Pass@1)     | 20.3             | 28.4               | 28.7              | 30.1                | 32.8                   | 34.2        | 37.6        |
|         | Codeforces (Percentile)    | 17.5             | 35.6               | 24.8              | 25.3                | 20.3                   | 23.6        | 51.6        |
|         | SWE Verified (Resolved)    | -                | 22.6               | 23.8              | 24.5                | 50.8                   | 38.8        | 42.0        |
|         | Aider-Edit (Acc.)          | 60.3             | 71.6               | 65.4              | 63.9                | 84.2                   | 72.9        | 79.7        |
|         | Aider-Polyglot (Acc.)      | -                | 18.2               | 7.6               | 5.8                 | 45.3                   | 16.0        | 49.6        |
| Math    | AIME 2024 (Pass@1)         | 4.6              | 16.7               | 23.3              | 23.3                | 16.0                   | 9.3         | 39.2        |
|         | MATH-500 (EM)              | 56.3             | 74.7               | 80.0              | 73.8                | 78.3                   | 74.6        | 90.2        |
|         | CNMO 2024 (Pass@1)         | 2.8              | 10.8               | 15.9              | 6.8                 | 13.1                   | 10.8        | 43.2        |
| Chinese | CLUEWSC (EM)               | 89.9             | 90.4               | 91.4              | 84.7                | 85.4                   | 87.9        | 90.9        |
|         | C-Eval (EM)                | 78.6             | 79.5               | 86.1              | 61.5                | 76.7                   | 76.0        | 86.5        |
|         | C-SimpleQA (Correct)       | 48.5             | 54.1               | 48.4              | 50.4                | 51.3                   | 59.3        | 64.8        |

#### Open Ended Generation Evaluation

| Model                  | Arena-Hard | AlpacaEval 2.0 |
|------------------------|------------|----------------|
| DeepSeek-V2.5-0905     | 76.2       | 50.5           |
| Qwen2.5-72B-Instruct   | 81.2       | 49.1           |
| LLaMA-3.1 405B         | 69.3       | 40.5           |
| GPT-4o-0513            | 80.4       | 51.1           |
| Claude-Sonnet-3.5-1022 | 85.2       | 52.0           |
| DeepSeek-V3            | 85.5       | 70.0           |

## DeepSeek V3.1[[22]](https://huggingface.co/deepseek-ai/DeepSeek-V3.1)

DeepSeek-V3.1 is a hybrid model that supports both thinking mode and non-thinking mode. Compared to the previous version, this upgrade brings improvements in multiple aspects:

- Hybrid thinking mode: One model supports both thinking mode and non-thinking mode by changing the chat template.
- Smarter tool calling: Through post-training optimization, the model's performance in tool usage and agent tasks has significantly improved.
- Higher thinking efficiency: DeepSeek-V3.1-Think achieves comparable answer quality to DeepSeek-R1-0528, while responding more quickly.
  
DeepSeek-V3.1 is post-trained on the top of DeepSeek-V3.1-Base, which is built upon the original V3 base checkpoint through a two-phase long context extension approach. The 32K extension phase has been increased 10-fold to 630B tokens, while the 128K extension phase has been extended by 3.3x to 209B tokens. Additionally, DeepSeek-V3.1 is trained using the UE8M0 FP8 scale data format on both model weights and activations to ensure compatibility with microscaling data formats.

### List of DeepSeek-V3.1 models

| Series        | Model                  | Year           | Notes                                                                     |
|---------------|------------------------|----------------|---------------------------------------------------------------------------|
| DeepSeek-V3.1 | DeepSeek-V3.1          | August 2025    | Hybrid model supporting thinking and non-thinking modes                   |
|               | DeepSeek-V3.1-Base     | August 2025    |                                                                           |
|               | DeepSeek-V3.1-Terminus | September 2025 | Update addressing issues with Language consistency and Agent Capabilities |

## DeepSeek V3.2[[23]](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)

DeepSeek-V3.2, a model that harmonizes high computational efficiency with superior reasoning and agent performance. It's approach is built upon three key technical breakthroughs:

- DeepSeek Sparse Attention (DSA): An efficient attention mechanism that substantially reduces computational complexity while preserving model performance, specifically optimized for long-context scenarios.
- Scalable Reinforcement Learning Framework: By implementing a robust RL protocol and scaling post-training compute, DeepSeek-V3.2 performs comparably to GPT-5. Notably, our high-compute variant, DeepSeek-V3.2-Speciale, surpasses GPT-5 and exhibits reasoning proficiency on par with Gemini-3.0-Pro.
- Large-Scale Agentic Task Synthesis Pipeline: To integrate reasoning into tool-use scenarios, novel synthesis pipeline was developed that systematically generates training data at scale. This facilitates scalable agentic post-training, improving compliance and generalization in complex interactive environments.

### List of DeepSeek-V3.2 models

| Series        | Model                  | Year           | Notes                                                                                                                                          |
|---------------|------------------------|----------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| DeepSeek-V3.2 | DeepSeek-V3.2          | September 2025 | Model with DSA(DeepSeek Sparse Attention) mechanism, Scalable Reinforcement Learning Framework and Large-Scale Agentic Task Synthesis Pipeline |
|               | DeepSeek-V3.2-Exp      | September 2025 | Experimental version                                                                                                                           |
|               | DeepSeek-V3.2-Exp-Base | September 2025 |                                                                                                                                                |
|               | DeepSeek-V3.2-Speciale | September 2025 | High-compute variant                                                                                                                           |

### Performance

![](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/raw/main/cost.jpg)

| Benchmark                   | DeepSeek-V3.1-Terminus | DeepSeek-V3.2-Exp |
|-----------------------------|------------------------|-------------------|
| Reasoning Mode w/o Tool Use |                        |                   |
| MMLU-Pro                    | 85.0                   | 85.0              |
| GPQA-Diamond                | 80.7                   | 79.9              |
| Humanity's Last Exam        | 21.7                   | 19.8              |
| LiveCodeBench               | 74.9                   | 74.1              |
| AIME 2025                   | 88.4                   | 89.3              |
| HMMT 2025                   | 86.1                   | 83.6              |
| Codeforces                  | 2046                   | 2121              |
| Aider-Polyglot              | 76.1                   | 74.5              |
| **Agentic Tool Use**        |                        |                   |
| BrowseComp                  | 38.5                   | 40.1              |
| BrowseComp-zh               | 45.0                   | 47.9              |
| SimpleQA                    | 96.8                   | 97.1              |
| SWE Verified                | 68.4                   | 67.8              |
| SWE-bench Multilingual      | 57.8                   | 57.9              |
| Terminal-bench              | 36.7                   | 37.7              |

## DeepSeek-VL2

 DeepSeek-VL2, is an advanced series of large Mixture-of-Experts (MoE) Vision-Language Models that significantly improves upon its predecessor, DeepSeek-VL. DeepSeek-VL2 demonstrates superior capabilities across various tasks, including but not limited to visual question answering, optical character recognition, document/table/chart understanding, and visual grounding.[[24]](https://huggingface.co/collections/deepseek-ai/deepseek-vl2)

 ### List of DeepSeek-VL2 models

| Series       | Model              | Year          | Notes                                                                |
|--------------|--------------------|---------------|----------------------------------------------------------------------|
| DeepSeek-VL2 | DeepSeek-VL2       | December 2024 | An advanced MoE Vision-language model that improved upon DeepSeek-VL |
|              | DeepSeek-VL2-tiny  | December 2024 |                                                                      |
|              | DeepSeek-VL2-small | December 2024 |                                                                      |

### Performance

DeepSeek-VL2 achieves competitive or state-of-the-art performance with similar or fewer activated parameters compared to existing open-source dense and MoE-based models.[[25]](https://github.com/deepseek-ai/DeepSeek-VL2)

![](https://github.com/deepseek-ai/DeepSeek-VL2/raw/main/images/vl2_teaser.jpeg)

## DeepSeek R1 and R1-Zero

DeepSeek's first-generation reasoning models, DeepSeek-R1-Zero and DeepSeek-R1. DeepSeek-R1-Zero, a model trained via large-scale reinforcement learning (RL) without supervised fine-tuning (SFT) as a preliminary step, demonstrated remarkable performance on reasoning. With RL, DeepSeek-R1-Zero naturally emerged with numerous powerful and interesting reasoning behaviors. However, DeepSeek-R1-Zero encounters challenges such as endless repetition, poor readability, and language mixing. To address these issues and further enhance reasoning performance, DeepSeek-R1 was introduced, which incorporates cold-start data before RL. DeepSeek-R1 achieves performance comparable to OpenAI-o1 across math, code, and reasoning tasks.[[26]](https://huggingface.co/deepseek-ai/DeepSeek-R1)

![](https://github.com/deepseek-ai/DeepSeek-R1/raw/main/figures/benchmark.jpg)

### List of DeepSeek-R1 models

| Series      | Model                         | Year         | Notes                                                                                     |
|-------------|-------------------------------|--------------|-------------------------------------------------------------------------------------------|
| DeepSeek-R1 | DeepSeek-R1                   | January 2025 | Reasoning-focused model utilizing RL techniques                                           |
|             | DeepSeek-R1-Zero              | January 2025 | R1 & R1-Zero are trained based on DeepSeek-V3-Base                                        |
|             | DeepSeek-R1-Distill-Qwen-1.5B | 2025         | R1-Distill models are based on open-source models, using samples generated by DeepSeek-R1 |
|             | DeepSeek-R1-Distill-Qwen-7B   | 2025         |                                                                                           |
|             | DeepSeek-R1-Distill-Qwen-14B  | 2025         |                                                                                           |
|             | DeepSeek-R1-Distill-Qwen-32B  | 2025         |                                                                                           |
|             | DeepSeek-R1-Distill-Llama-8B  | 2025         |                                                                                           |
|             | DeepSeek-R1-Distill-Llama-70B | 2025         |                                                                                           |

Using the reasoning data generated by DeepSeek-R1, several dense models were fine-tuned that are widely used in the research community. The evaluation results demonstrate that the distilled smaller dense models perform exceptionally well on benchmarks.

### Performance[[27]](https://github.com/deepseek-ai/DeepSeek-R1)
### DeepSeek-R1 Evaluation

| Category | Benchmark (Metric)         | Claude-3.5-Sonnet-1022 | GPT-4o 0513 | DeepSeek V3 | OpenAI o1-mini | OpenAI o1-1217 | DeepSeek R1 |
|----------|----------------------------|------------------------|-------------|-------------|----------------|----------------|-------------|
|          | Architecture               | -                      | -           | MoE         | -              | -              | MoE         |
|          | # Activated Params         | -                      | -           | 37B         | -              | -              | 37B         |
|          | # Total Params             | -                      | -           | 671B        | -              | -              | 671B        |
| English  | MMLU (Pass@1)              | 88.3                   | 87.2        | 88.5        | 85.2           | 91.8           | 90.8        |
|          | MMLU-Redux (EM)            | 88.9                   | 88.0        | 89.1        | 86.7           | -              | 92.9        |
|          | MMLU-Pro (EM)              | 78.0                   | 72.6        | 75.9        | 80.3           | -              | 84.0        |
|          | DROP (3-shot F1)           | 88.3                   | 83.7        | 91.6        | 83.9           | 90.2           | 92.2        |
|          | IF-Eval (Prompt Strict)    | 86.5                   | 84.3        | 86.1        | 84.8           | -              | 83.3        |
|          | GPQA-Diamond (Pass@1)      | 65.0                   | 49.9        | 59.1        | 60.0           | 75.7           | 71.5        |
|          | SimpleQA (Correct)         | 28.4                   | 38.2        | 24.9        | 7.0            | 47.0           | 30.1        |
|          | FRAMES (Acc.)              | 72.5                   | 80.5        | 73.3        | 76.9           | -              | 82.5        |
|          | AlpacaEval2.0 (LC-winrate) | 52.0                   | 51.1        | 70.0        | 57.8           | -              | 87.6        |
|          | ArenaHard (GPT-4-1106)     | 85.2                   | 80.4        | 85.5        | 92.0           | -              | 92.3        |
| Code     | LiveCodeBench (Pass@1-COT) | 33.8                   | 34.2        | -           | 53.8           | 63.4           | 65.9        |
|          | Codeforces (Percentile)    | 20.3                   | 23.6        | 58.7        | 93.4           | 96.6           | 96.3        |
|          | Codeforces (Rating)        | 717                    | 759         | 1134        | 1820           | 2061           | 2029        |
|          | SWE Verified (Resolved)    | 50.8                   | 38.8        | 42.0        | 41.6           | 48.9           | 49.2        |
|          | Aider-Polyglot (Acc.)      | 45.3                   | 16.0        | 49.6        | 32.9           | 61.7           | 53.3        |
| Math     | AIME 2024 (Pass@1)         | 16.0                   | 9.3         | 39.2        | 63.6           | 79.2           | 79.8        |
|          | MATH-500 (Pass@1)          | 78.3                   | 74.6        | 90.2        | 90.0           | 96.4           | 97.3        |
|          | CNMO 2024 (Pass@1)         | 13.1                   | 10.8        | 43.2        | 67.6           | -              | 78.8        |
| Chinese  | CLUEWSC (EM)               | 85.4                   | 87.9        | 90.9        | 89.9           | -              | 92.8        |
|          | C-Eval (EM)                | 76.7                   | 76.0        | 86.5        | 68.9           | -              | 91.8        |
|          | C-SimpleQA (Correct)       | 55.4                   | 58.7        | 68.0        | 40.3           | -              | 63.7        |

### Distilled Models Evaluation

| Model                         | AIME 2024 pass@1 | AIME 2024 cons@64 | MATH-500 pass@1 | GPQA Diamond pass@1 | LiveCodeBench pass@1 | CodeForces rating |
|-------------------------------|------------------|-------------------|-----------------|---------------------|----------------------|-------------------|
| GPT-4o-0513                   | 9.3              | 13.4              | 74.6            | 49.9                | 32.9                 | 759               |
| Claude-3.5-Sonnet-1022        | 16.0             | 26.7              | 78.3            | 65.0                | 38.9                 | 717               |
| o1-mini                       | 63.6             | 80.0              | 90.0            | 60.0                | 53.8                 | 1820              |
| QwQ-32B-Preview               | 44.0             | 60.0              | 90.6            | 54.5                | 41.9                 | 1316              |
| DeepSeek-R1-Distill-Qwen-1.5B | 28.9             | 52.7              | 83.9            | 33.8                | 16.9                 | 954               |
| DeepSeek-R1-Distill-Qwen-7B   | 55.5             | 83.3              | 92.8            | 49.1                | 37.6                 | 1189              |
| DeepSeek-R1-Distill-Qwen-14B  | 69.7             | 80.0              | 93.9            | 59.1                | 53.1                 | 1481              |
| DeepSeek-R1-Distill-Qwen-32B  | 72.6             | 83.3              | 94.3            | 62.1                | 57.2                 | 1691              |
| DeepSeek-R1-Distill-Llama-8B  | 50.4             | 80.0              | 89.1            | 49.0                | 39.6                 | 1205              |
| DeepSeek-R1-Distill-Llama-70B | 70.0             | 86.7              | 94.5            | 65.2                | 57.5                 | 1633              |

