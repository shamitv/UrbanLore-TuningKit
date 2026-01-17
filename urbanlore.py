#!/usr/bin/env python3
"""
UrbanLore-TuningKit CLI
Main entrypoint for the UrbanLore pipeline
"""
import os
from pathlib import Path

import click
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def file_exists_and_not_empty(path: str) -> bool:
    """Check if file exists and is not empty."""
    p = Path(path)
    return p.exists() and p.stat().st_size > 0


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """UrbanLore-TuningKit: Generate synthetic city corpus and fine-tune LLMs"""
    pass


@cli.command()
@click.option("--target-words", default=200000, help="Target word count for corpus")
@click.option("--output-dir", default="corpus", help="Output directory for corpus")
@click.option("--force", is_flag=True, default=False, help="Regenerate even if output exists")
def generate_corpus(target_words, output_dir, force):
    """Generate fictional city corpus (~200k words)"""
    corpus_file = os.path.join(output_dir, "city_corpus.txt")
    
    if not force and file_exists_and_not_empty(corpus_file):
        click.echo(f"⏭  Corpus already exists at {corpus_file}, skipping (use --force to regenerate)")
        return
    
    from agents.generator import generate_city_corpus
    
    click.echo(f"Generating corpus with target {target_words} words...")
    generate_city_corpus(target_words=target_words, output_dir=output_dir)
    click.echo(f"✓ Corpus generated in {output_dir}/")


@cli.command()
@click.option("--corpus-file", default="corpus/city_corpus.txt", help="Input corpus file")
@click.option("--output-dir", default="corpus", help="Output directory for facts")
@click.option("--force", is_flag=True, default=False, help="Regenerate even if output exists")
def extract_facts(corpus_file, output_dir, force):
    """Extract structured facts from corpus"""
    facts_file = os.path.join(output_dir, "facts.json")
    
    if not force and file_exists_and_not_empty(facts_file):
        click.echo(f"⏭  Facts already exist at {facts_file}, skipping (use --force to regenerate)")
        return
    
    from agents.extractor import extract_facts_from_corpus
    
    click.echo(f"Extracting facts from {corpus_file}...")
    extract_facts_from_corpus(corpus_file=corpus_file, output_dir=output_dir)
    click.echo(f"✓ Facts extracted to {output_dir}/")


@cli.command()
@click.option("--facts-file", default="corpus/facts.json", help="Input facts file")
@click.option("--corpus-file", default="corpus/city_corpus.txt", help="Input corpus file")
@click.option("--output-dir", default="dataset", help="Output directory for dataset")
@click.option("--num-qa", default=1000, help="Number of QA pairs to generate")
@click.option("--num-instructions", default=500, help="Number of instruction pairs to generate")
@click.option("--force", is_flag=True, default=False, help="Regenerate even if output exists")
def generate_qa(facts_file, corpus_file, output_dir, num_qa, num_instructions, force):
    """Generate QA/instruction dataset (JSONL)"""
    train_file = os.path.join(output_dir, "train.jsonl")
    
    if not force and file_exists_and_not_empty(train_file):
        click.echo(f"⏭  Dataset already exists at {train_file}, skipping (use --force to regenerate)")
        return
    
    from agents.qa_generator import generate_qa_dataset
    
    click.echo(f"Generating QA dataset...")
    generate_qa_dataset(
        facts_file=facts_file,
        corpus_file=corpus_file,
        output_dir=output_dir,
        num_qa=num_qa,
        num_instructions=num_instructions
    )
    click.echo(f"✓ Dataset generated in {output_dir}/")


@cli.command()
@click.option("--dataset-file", default="dataset/train.jsonl", help="Training dataset JSONL file")
@click.option("--base-model", default="microsoft/phi-2", help="Base model to fine-tune")
@click.option("--output-dir", default="finetune/models", help="Output directory for fine-tuned model")
@click.option("--use-qlora", is_flag=True, default=True, help="Use QLoRA for training")
@click.option("--epochs", default=3, help="Number of training epochs")
def finetune(dataset_file, base_model, output_dir, use_qlora, epochs):
    """Fine-tune model with LoRA/QLoRA"""
    from finetune.train import train_model
    
    click.echo(f"Fine-tuning {base_model}...")
    train_model(
        dataset_file=dataset_file,
        base_model=base_model,
        output_dir=output_dir,
        use_qlora=use_qlora,
        epochs=epochs
    )
    click.echo(f"✓ Model fine-tuned and saved to {output_dir}/")


@cli.command()
@click.option("--model-dir", default="finetune/models/final", help="Fine-tuned model directory")
@click.option("--test-file", default="dataset/test.jsonl", help="Test dataset JSONL file")
@click.option("--output-dir", default="eval/results", help="Output directory for results")
def evaluate(model_dir, test_file, output_dir):
    """Run model evaluation"""
    from eval.evaluate import evaluate_model
    
    click.echo(f"Evaluating model from {model_dir}...")
    evaluate_model(
        model_dir=model_dir,
        test_file=test_file,
        output_dir=output_dir
    )
    click.echo(f"✓ Evaluation results saved to {output_dir}/")


@cli.command()
@click.option("--target-words", default=200000, help="Target word count for corpus")
@click.option("--force", is_flag=True, default=False, help="Regenerate all steps even if outputs exist")
def run_all(target_words, force):
    """Run complete pipeline (corpus -> QA -> finetune -> eval)"""
    click.echo("Starting complete UrbanLore pipeline...")
    
    # Run each step
    from click import Context
    ctx = Context(cli)
    
    ctx.invoke(generate_corpus, target_words=target_words, output_dir="corpus", force=force)
    ctx.invoke(extract_facts, corpus_file="corpus/city_corpus.txt", output_dir="corpus", force=force)
    ctx.invoke(generate_qa, facts_file="corpus/facts.json", corpus_file="corpus/city_corpus.txt", 
               output_dir="dataset", num_qa=1000, num_instructions=500, force=force)
    ctx.invoke(finetune, dataset_file="dataset/train.jsonl", base_model="microsoft/phi-2", 
               output_dir="finetune/models", use_qlora=True, epochs=3)
    ctx.invoke(evaluate, model_dir="finetune/models/final", test_file="dataset/test.jsonl", 
               output_dir="eval/results")
    
    click.echo("✓ Complete pipeline finished!")


if __name__ == "__main__":
    cli()
