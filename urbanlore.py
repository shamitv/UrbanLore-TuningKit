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
@click.option("--base-model", default="Qwen/Qwen3-0.6B", help="Base model to fine-tune")
@click.option("--output-dir", default="finetune/models", help="Output directory for fine-tuned model")
@click.option("--use-qlora", is_flag=True, default=True, help="Use QLoRA for training")
@click.option("--epochs", default=3, help="Number of training epochs")
@click.option("--force", is_flag=True, default=False, help="Regenerate even if output exists")
def finetune(dataset_file, base_model, output_dir, use_qlora, epochs, force):
    """Fine-tune model with LoRA/QLoRA"""
    if not base_model:
        base_model = os.getenv("BASE_MODEL", "Qwen/Qwen3-0.6B")

    final_model_dir = os.path.join(output_dir, "final")
    metadata_file = os.path.join(final_model_dir, "training_metadata.json")

    if not force and file_exists_and_not_empty(metadata_file):
        click.echo(f"⏭  Fine-tuned model already exists at {final_model_dir}, skipping (use --force to retrain)")
        return

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
@click.option("--force", is_flag=True, default=False, help="Regenerate even if output exists")
def evaluate(model_dir, test_file, output_dir, force):
    """Run model evaluation"""
    results_file = os.path.join(output_dir, "evaluation_results.json")

    if not force and file_exists_and_not_empty(results_file):
        click.echo(f"⏭  Evaluation results already exist at {results_file}, skipping (use --force to re-evaluate)")
        return

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
@click.option("--corpus-dir", default="corpus", help="Output directory for corpus/facts")
@click.option("--dataset-dir", default="dataset", help="Output directory for dataset")
@click.option("--model-dir", default="finetune/models", help="Output directory for fine-tuned model")
@click.option("--base-model", default="Qwen/Qwen3-0.6B", help="Base model to fine-tune")
@click.option("--eval-dir", default="eval/results", help="Output directory for evaluation results")
@click.option("--num-qa", default=1000, help="Number of QA pairs to generate")
@click.option("--num-instructions", default=500, help="Number of instruction pairs to generate")
@click.option("--use-qlora", default=None, help="Use QLoRA for training (true/false)")
@click.option("--force", is_flag=True, default=False, help="Regenerate all steps even if outputs exist")
def run_all(target_words, corpus_dir, dataset_dir, model_dir, eval_dir, base_model, num_qa, num_instructions, use_qlora, force):
    """Run complete pipeline (corpus -> QA -> finetune -> eval)"""
    click.echo("Starting complete UrbanLore pipeline...")
    
    # Run each step
    from click import Context
    ctx = Context(cli)
    
    corpus_file = os.path.join(corpus_dir, "city_corpus.txt")
    facts_file = os.path.join(corpus_dir, "facts.json")
    train_file = os.path.join(dataset_dir, "train.jsonl")
    test_file = os.path.join(dataset_dir, "test.jsonl")
    final_model_dir = os.path.join(model_dir, "final")
    model_metadata = os.path.join(final_model_dir, "training_metadata.json")
    eval_results = os.path.join(eval_dir, "evaluation_results.json")

    ctx.invoke(generate_corpus, target_words=target_words, output_dir=corpus_dir, force=force)
    ctx.invoke(extract_facts, corpus_file=corpus_file, output_dir=corpus_dir, force=force)
    ctx.invoke(generate_qa, facts_file=facts_file, corpus_file=corpus_file, 
               output_dir=dataset_dir, num_qa=num_qa, num_instructions=num_instructions, force=force)

    if force or not file_exists_and_not_empty(model_metadata):
        if not base_model:
            base_model = os.getenv("BASE_MODEL", "Qwen/Qwen3-0.6B")

        if use_qlora is None:
            use_qlora_env = os.getenv("USE_QLORA", "true").strip().lower()
            use_qlora = use_qlora_env in {"1", "true", "yes", "y"}

        ctx.invoke(finetune, dataset_file=train_file, base_model=base_model, 
                   output_dir=model_dir, use_qlora=use_qlora, epochs=3, force=force)
    else:
        click.echo(f"⏭  Fine-tuned model already exists at {final_model_dir}, skipping (use --force to retrain)")

    if force or not file_exists_and_not_empty(eval_results):
        ctx.invoke(evaluate, model_dir=final_model_dir, test_file=test_file, 
                   output_dir=eval_dir, force=force)
    else:
        click.echo(f"⏭  Evaluation results already exist at {eval_results}, skipping (use --force to re-evaluate)")
    
    click.echo("✓ Complete pipeline finished!")


if __name__ == "__main__":
    cli()
