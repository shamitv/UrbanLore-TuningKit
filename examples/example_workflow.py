#!/usr/bin/env python3
"""
Example workflow for UrbanLore-TuningKit
Demonstrates the complete pipeline from corpus generation to model evaluation
"""
from dotenv import load_dotenv
from agents.generator import generate_city_corpus
from agents.extractor import extract_facts_from_corpus
from agents.qa_generator import generate_qa_dataset
from finetune.train import train_model
from eval.evaluate import evaluate_model

# Load environment variables
load_dotenv()


def main():
    """Run the complete UrbanLore pipeline"""
    print("=" * 80)
    print("UrbanLore-TuningKit - Complete Pipeline Example")
    print("=" * 80)
    
    # Step 1: Generate corpus
    print("\n[Step 1/5] Generating fictional city corpus...")
    corpus_file = generate_city_corpus(
        target_words=50000,  # Smaller for example
        output_dir="examples/output/corpus"
    )
    
    # Step 2: Extract facts
    print("\n[Step 2/5] Extracting structured facts...")
    facts_file = extract_facts_from_corpus(
        corpus_file=corpus_file,
        output_dir="examples/output/corpus"
    )
    
    # Step 3: Generate QA dataset
    print("\n[Step 3/5] Generating QA/instruction dataset...")
    dataset_files = generate_qa_dataset(
        facts_file=facts_file,
        corpus_file=corpus_file,
        output_dir="examples/output/dataset",
        num_qa=100,  # Smaller for example
        num_instructions=50
    )
    
    # Step 4: Fine-tune model
    print("\n[Step 4/5] Fine-tuning model with LoRA/QLoRA...")
    model_dir = train_model(
        dataset_file=dataset_files["train"],
        base_model="microsoft/phi-2",
        output_dir="examples/output/finetune",
        use_qlora=True,
        epochs=1  # Fewer epochs for example
    )
    
    # Step 5: Evaluate model
    print("\n[Step 5/5] Evaluating fine-tuned model...")
    results_file = evaluate_model(
        model_dir=model_dir,
        test_file=dataset_files["test"],
        output_dir="examples/output/eval"
    )
    
    print("\n" + "=" * 80)
    print("Pipeline complete!")
    print("=" * 80)
    print(f"\nResults:")
    print(f"  Corpus: {corpus_file}")
    print(f"  Facts: {facts_file}")
    print(f"  Dataset: {dataset_files['train']}")
    print(f"  Model: {model_dir}")
    print(f"  Evaluation: {results_file}")


if __name__ == "__main__":
    main()
