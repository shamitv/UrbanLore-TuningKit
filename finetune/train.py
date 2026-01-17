#!/usr/bin/env python3
"""
Model Fine-tuning with LoRA/QLoRA
Fine-tunes a small HuggingFace model on the generated dataset
"""
import os
import json
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

load_dotenv()


def load_training_dataset(dataset_file: str):
    """Load the JSONL training dataset"""
    dataset = load_dataset("json", data_files=dataset_file, split="train")
    return dataset


def prepare_model_and_tokenizer(base_model: str, use_qlora: bool = True):
    """
    Prepare model and tokenizer with optional QLoRA quantization
    
    Args:
        base_model: HuggingFace model name or path
        use_qlora: Whether to use QLoRA (4-bit quantization)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {base_model}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure quantization for QLoRA
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
    else:
        # Standard LoRA without quantization
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            trust_remote_code=True
        )
    
    return model, tokenizer


def setup_lora_config():
    """Setup LoRA configuration"""
    lora_config = LoraConfig(
        r=int(os.getenv("LORA_R", "16")),
        lora_alpha=int(os.getenv("LORA_ALPHA", "32")),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Common attention modules
        lora_dropout=float(os.getenv("LORA_DROPOUT", "0.05")),
        bias="none",
        task_type="CAUSAL_LM"
    )
    return lora_config


def train_model(dataset_file: str = "dataset/train.jsonl",
                base_model: str = "microsoft/phi-2",
                output_dir: str = "finetune/models",
                use_qlora: bool = True,
                epochs: int = 3) -> str:
    """
    Fine-tune a model with LoRA/QLoRA
    
    Args:
        dataset_file: Path to training dataset JSONL file
        base_model: Base model name from HuggingFace
        output_dir: Directory to save the fine-tuned model
        use_qlora: Whether to use QLoRA (4-bit quantization)
        epochs: Number of training epochs
    
    Returns:
        Path to the saved model
    """
    print(f"Starting fine-tuning with {'QLoRA' if use_qlora else 'LoRA'}...")
    
    # Load dataset
    print(f"Loading dataset from {dataset_file}...")
    dataset = load_training_dataset(dataset_file)
    print(f"Dataset loaded: {len(dataset)} examples")
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer(base_model, use_qlora)
    
    # Setup LoRA
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_path / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=int(os.getenv("BATCH_SIZE", "4")),
        gradient_accumulation_steps=int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4")),
        learning_rate=float(os.getenv("LEARNING_RATE", "2e-4")),
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        fp16=True if torch.cuda.is_available() else False,
        optim="paged_adamw_8bit" if use_qlora else "adamw_torch",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to=["none"],  # Disable wandb/tensorboard for simplicity
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=int(os.getenv("MAX_SEQ_LENGTH", "512")),
        packing=False,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    final_model_dir = output_path / "final"
    final_model_dir.mkdir(exist_ok=True)
    
    print(f"Saving model to {final_model_dir}...")
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    
    # Save training metadata
    metadata = {
        "base_model": base_model,
        "use_qlora": use_qlora,
        "epochs": epochs,
        "dataset_file": dataset_file,
        "num_examples": len(dataset),
        "lora_config": {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout
        }
    }
    
    metadata_file = final_model_dir / "training_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"✓ Model saved to {final_model_dir}")
    print(f"✓ Metadata saved to {metadata_file}")
    
    return str(final_model_dir)


if __name__ == "__main__":
    train_model()
