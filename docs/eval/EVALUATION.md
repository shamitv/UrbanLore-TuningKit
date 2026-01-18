# Model Evaluation Process (V2)

This document describes how the fine-tuned UrbanLore models are evaluated, including methodology, metrics, and an analysis of the latest training run (Qwen fine-tune).

## Overview

The evaluation pipeline (`eval/evaluate.py`) tests the fine-tuned model against a held-out test dataset (`dataset/test.jsonl`). It generates responses for a subset of test examples and compares them against the ground truth reference answers using standard NLP metrics.

## Methodology

### 1. Data Loading
- **Test Set**: `dataset/test.jsonl`
- **Sampling**: By default, the evaluator processes the first **100 examples** from the test set to ensure timely feedback.

### 2. Prompt Processing
The evaluator intelligently splits the raw training examples into a "Prompt" (input) and "Reference" (expected output) to simulate real inference.
It looks for specific separators in the following order:
1.  **QA Format**: Splits at `Answer:` -> Prompt includes the question, Reference is the answer.
2.  **Instruction Format**: Splits at `### Response:` -> Prompt includes instruction/context, Reference is the response.
3.  **Fallback**: If no separators are found, it splits the text in half.

### 3. Generation Parameters
- **Max New Tokens**: 256
- **Temperature**: 0.7
- **Top P**: 0.9

## Metrics: ROUGE Score

We use **ROUGE** to measure the overlap between the generated text and the reference text.

| Metric | Score (Latest Run) | Interpretation |
| :--- | :--- | :--- |
| **ROUGE-1** | **0.238** | Low. The model struggles to reproduce the exact key terms from the ground truth. |
| **ROUGE-2** | **0.084** | Very Low. Phrase-level usage is inconsistent with the reference. |
| **ROUGE-L** | **0.160** | Low. Sentence structure differs significantly from the target. |

## Latest Run Analysis

The latest evaluation reveals significant challenges in the model's performance. Below are examples and observations.

### 1. Correctness & Hallucination
The model frequently hallucinates generic information instead of retrieving specific UrbanLore facts.

**Example (Incorrect / Hallucination):**
> **Question:** Describe the climate and microclimate features of Meridian Reach.
> **Reference:** "Meridian Reach has a temperate climate. Fog forms at dawn, especially near the harbor edge..."
> **Prediction:** "Meridian Reach is a coastal region located in the northwestern part of the United States... The coastal waters provide a steady supply of moisture..."
>
> **Analysis:** The model fails to recall the fictional setting and instead defaults to generic US geography descriptions. It also exhibits severe repetition.

### 2. Instruction Leakage
There are instances where the model outputs meta-instructions, suggesting issues with the training data formatting or prompt template.

**Example (Meta-Text):**
> **Prediction:** "...You may need to adjust the answer according to the question. For example, the answer could include the specific term..."
>
> **Analysis:** The model is outputting the "internal thought" or "instruction" that likely existed in the synthetic data generation pipeline, rather than the final answer.

### 3. Formatting & Repetition
The model struggles with stop tokens and often repeats itself or formatting markers.

**Example (Repetition):**
> **Prediction:** "...The coastal winds bring warm, moist air, creating a gentle sea breeze... The coastal winds bring warm, moist air... ### Response: The exhibit presents..."
>
> **Analysis:** The model loops and fails to terminate generation, sometimes regurgitating the prompt format (`### Response:`).

## Data Quality & Next Steps

The qualitative analysis suggests several issues with the current data or tuning process:
1.  **Data Quality**: The presence of meta-instructions ("You may need to adjust...") in the output implies they exist in the training data. The dataset needs to be scrubbed of these synthetic artifacts.
2.  **Prompt Mismatch**: Similarly, the leakage of `### Response:` markers suggests the inference prompt template may not match the training format perfectly, or the model has not learned to stop.
3.  **Factuality**: The model prioritizes generic, smooth-sounding text over the specific fictional facts of the corpus. We may need to increase the weight of the loss on factual tokens or reduce temperature during evaluation to favor exact retrieval.
