# Model Evaluation Process

This document describes how the fine-tuned UrbanLore models are evaluated and how performance scores are calculated.

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
3.  **Fallback**: If no separators are found, it splits the text in half (first 50% as prompt, last 50% as reference).

### 3. Generation Parameters
The model generates responses using the following configuration:
- **Max New Tokens**: 256
- **Temperature**: 0.7 (adds slight creativity)
- **Top P**: 0.9 (nucleus sampling)
- **Do Sample**: True

## Metrics: ROUGE Score

We use **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) to measure the overlap between the generated text and the reference text.

| Metric | What it Measures | Importance |
| :--- | :--- | :--- |
| **ROUGE-1** | Overlap of unigrams (single words). | Measures content coverage and basic word choice accuracy. |
| **ROUGE-2** | Overlap of bigrams (two adjacent words). | Measures fluency and phrase coherence. |
| **ROUGE-L** | Longest Common Subsequence. | Measures sentence-level structure similarity. |

### Calculation Details
- **Library**: `rouge_score`
- **Stemming**: Enabled (`use_stemmer=True`) to treat word variants (e.g., "walk", "walking") as the same.
- **Aggregation**: Scores are averaged across all evaluated samples.

## Outputs

Results are saved to `eval/results/`:

1.  **`evaluation_results.json`**: Contains the aggregated ROUGE scores.
    ```json
    {
      "metrics": {
        "rouge1": 0.45,
        "rouge2": 0.22,
        "rougeL": 0.41
      }
    }
    ```

2.  **`sample_predictions.json`**: Contains the first 10 examples showing:
    - `example`: Original full text.
    - `prediction`: Model's generated output.
    - `reference`: Ground truth output.
