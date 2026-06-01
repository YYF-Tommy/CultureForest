# CultureForest
The repository for "CultureForest: Understanding and Evaluating Cultural Norm Grounded Reasoning in LLMs"

<p align="center">
  <b>CultureForest</b> is a benchmark for evaluating <i>Cultural Norm Grounded Reasoning</i> in Large Language Models (LLMs).
</p>

---


Dataset: https://huggingface.co/datasets/TommyYe/CultureForest

C-Verifier: https://huggingface.co/TommyYe/C-Verifier

## Overview

Existing cultural benchmarks primarily evaluate whether LLMs can **recall cultural knowledge**.
However, real-world cultural intelligence requires more than memorization: models must be able to **effectively utilize cultural knowledge under contextual and compositional scenarios**.

To study this problem, we introduce **CultureForest**, a benchmark grounded in explicit cultural norms and designed to progressively evaluate models from closed-form selection to open-ended generation.

### Key Features

- 🌍 **Broad Cultural Coverage**
  - 53 countries/regions
  - 8 cultural domains
  - Diverse geographic and cultural settings

- 🧠 **Norm-Grounded Reasoning**
  - Each question is grounded in **3 atomic cultural norms**
  - Requires joint reasoning rather than shallow cue matching

- 📈 **Progressive Difficulty**
  - Easy: Multiple-choice
  - Medium: Reduced guidance
  - Hard: Open-ended generation

- 🔍 **Attributable Evaluation**
  - Explicit norm grounding enables disentangling:
    - cultural knowledge acquisition
    - cultural reasoning ability

- ⚖️ **Open-ended Evaluation**
  - Includes scalable verifier-based evaluation for generated responses

---

## Repository Structure

```text
.
├── evaluation/
│   ├── CulturalNLI.txt
│   ├── eval_easy.py
│   ├── eval_judge.py
│   ├── eval_open_step1.py
│   ├── eval_open_step2_scale.py
│   └── norms_both_acc.json 
│
├── inference/
│   ├── run_easy_vllm.py
│   ├── run_open_vllm.py
│   ├── run_judge1_vllm.py
│   ├── run_judge2_vllm.py
│   ├── run_judge3_vllm.py
│   └── run_judge4_vllm.py
│
├── qa_generation_prompts/
│   ├── 1_Scenario.txt
│   ├── 2_QA.txt
│   ├── 3_Alignment_Checker_A.txt
│   ├── 3_Alignment_Checker_B.txt
│   ├── 3_Alignment_Checker_C.txt
│   ├── 3_Alignment_Checker_D.txt
│   ├── 4_Upgrade_Context_Question.txt
│   ├── 4_Upgrade_Options_step1.txt
│   ├── 4_Upgrade_Options_step2.txt
│   ├── 4_Upgrade_Options_step3.txt
│   └── 4_Upgrade_Options_step4.txt
│
└── README.md
```

---

## Running Inference

### Environment

```
transformers==5.5.3
vllm==0.19.0
torch==2.10.0
jsonlines
```


### Easy Mode (Multiple Choice Question)

```bash
python inference/run_easy_vllm.py --model XXX --gpu gpu_num --think
```

### Medium Mode (Binary Judgement)

```bash
python inference/run_judge1_vllm.py --model XXX --gpu gpu_num --think
python inference/run_judge2_vllm.py --model XXX --gpu gpu_num --think
python inference/run_judge3_vllm.py --model XXX --gpu gpu_num --think
python inference/run_judge4_vllm.py --model XXX --gpu gpu_num --think
```

### Hard Mode (Open-ended Generation)

```bash
python inference/run_open_vllm.py --model XXX --gpu gpu_num --think
```

---

## Evaluation

### Easy Evaluation

```bash
./evaluation/eval_easy
```

### Medium Evaluation

```bash
./evaluation/eval_judge
```

### Open-ended Evaluation

```bash
./evaluation/eval_open_step1
./evaluation/eval_open_step2
```