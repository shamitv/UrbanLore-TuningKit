#!/usr/bin/env python3
"""
Model Evaluation
Evaluates the fine-tuned model on test dataset
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import jsonlines
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from rouge_score import rouge_scorer

load_dotenv()


def load_model_and_tokenizer(model_dir: str):
    """Load the fine-tuned model and tokenizer"""
    print(f"Loading model from {model_dir}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the model (it should already have LoRA weights merged or be a PEFT model)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    except Exception as e:
        print(f"Error loading as full model: {e}")
        print("Attempting to load as PEFT model...")
        # Try loading as PEFT model with base model
        base_model_path = os.getenv("BASE_MODEL", "microsoft/phi-2")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        model = PeftModel.from_pretrained(base_model, model_dir)
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate a response from the model"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode only the generated part (exclude the input prompt)
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated_text


def calculate_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate ROUGE scores for generated responses"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        "rouge1": sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0,
        "rouge2": sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0,
        "rougeL": sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0
    }


def evaluate_model(model_dir: str = "finetune/models/final",
                   test_file: str = "dataset/test.jsonl",
                   output_dir: str = "eval/results") -> str:
    """
    Evaluate the fine-tuned model on test dataset
    
    Args:
        model_dir: Directory containing the fine-tuned model
        test_file: Path to test dataset JSONL file
        output_dir: Directory to save evaluation results
    
    Returns:
        Path to the evaluation results file
    """
    print("Starting model evaluation...")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_dir)
    model.eval()
    
    # Load test dataset
    print(f"Loading test dataset from {test_file}...")
    test_examples = []
    with jsonlines.open(test_file) as reader:
        test_examples = list(reader)
    
    print(f"Test dataset loaded: {len(test_examples)} examples")
    
    # Limit to a reasonable number for evaluation
    max_eval_examples = min(len(test_examples), 100)
    test_examples = test_examples[:max_eval_examples]
    
    # Generate predictions
    print(f"Generating predictions for {len(test_examples)} examples...")
    predictions = []
    references = []
    
    for example in tqdm(test_examples):
        text = example.get("text", "")
        
        # Split text to get prompt and expected response
        if "Question:" in text:
            parts = text.split("Answer:")
            prompt = parts[0] + "Answer:"
            reference = parts[1].strip() if len(parts) > 1 else ""
        elif "### Response:" in text:
            parts = text.split("### Response:")
            prompt = parts[0] + "### Response:"
            reference = parts[1].strip() if len(parts) > 1 else ""
        else:
            # Fallback: use first half as prompt, second half as reference
            mid = len(text) // 2
            prompt = text[:mid]
            reference = text[mid:].strip()
        
        # Generate response
        generated = generate_response(model, tokenizer, prompt)
        
        predictions.append(generated)
        references.append(reference)
    
    # Calculate metrics
    print("Calculating evaluation metrics...")
    rouge_scores = calculate_rouge_scores(predictions, references)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results = {
        "model_dir": model_dir,
        "test_file": test_file,
        "num_examples": len(test_examples),
        "metrics": {
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "rougeL": rouge_scores["rougeL"]
        }
    }
    
    results_file = output_path / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save sample predictions
    samples = []
    for i in range(min(10, len(predictions))):
        samples.append({
            "example": test_examples[i].get("text", ""),
            "prediction": predictions[i],
            "reference": references[i]
        })
    
    samples_file = output_path / "sample_predictions.json"
    with open(samples_file, "w") as f:
        json.dump(samples, f, indent=2)
    
    print(f"\n✓ Evaluation complete!")
    print(f"✓ Results saved to {results_file}")
    print(f"✓ Sample predictions saved to {samples_file}")
    print(f"\nMetrics:")
    print(f"  ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"  ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"  ROUGE-L: {rouge_scores['rougeL']:.4f}")
    
    return str(results_file)


if __name__ == "__main__":
    evaluate_model()
