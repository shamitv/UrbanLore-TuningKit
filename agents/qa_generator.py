#!/usr/bin/env python3
"""
QA/Instruction Dataset Generator
Generates QA pairs and instruction-following examples from corpus and facts
"""
import os
import json
import random
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import jsonlines

load_dotenv()


def generate_qa_from_facts(facts: List[Dict[str, Any]], llm: ChatOpenAI, num_pairs: int = 100) -> List[Dict[str, str]]:
    """Generate QA pairs from extracted facts"""
    qa_pairs = []
    
    # Sample facts if we have more than needed
    sampled_facts = random.sample(facts, min(len(facts), num_pairs))
    
    for fact in sampled_facts:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are creating question-answer pairs for training a language model. Create a natural question and detailed answer based on the fact."),
            ("user", """Based on this fact, create a Q&A pair:
Entity: {entity}
Attribute: {attribute}
Value: {value}
Context: {context}

Return as JSON with 'question' and 'answer' fields.""")
        ])
        
        chain = prompt | llm
        response = chain.invoke(fact)
        
        try:
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                qa_pair = json.loads(json_match.group())
                qa_pairs.append(qa_pair)
        except:
            # Fallback: create simple QA
            qa_pairs.append({
                "question": f"What is the {fact.get('attribute', 'characteristic')} of {fact.get('entity', 'it')}?",
                "answer": f"The {fact.get('attribute', 'characteristic')} is {fact.get('value', 'unknown')}. {fact.get('context', '')}"
            })
    
    return qa_pairs


def generate_instructions_from_corpus(corpus_text: str, llm: ChatOpenAI, num_instructions: int = 100) -> List[Dict[str, str]]:
    """Generate instruction-following examples from corpus"""
    instructions = []
    
    # Split corpus into chunks
    paragraphs = [p.strip() for p in corpus_text.split("\n\n") if len(p.strip()) > 100]
    sampled_paragraphs = random.sample(paragraphs, min(len(paragraphs), num_instructions))
    
    instruction_templates = [
        "Summarize the following passage:",
        "Extract key information from this text:",
        "What are the main points discussed in this passage?",
        "Describe the content of this text:",
        "Provide a detailed explanation based on this passage:",
    ]
    
    for paragraph in sampled_paragraphs:
        instruction = random.choice(instruction_templates)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are responding to instructions about a text passage. Provide a helpful, detailed response."),
            ("user", "{instruction}\n\nText: {text}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({"instruction": instruction, "text": paragraph[:1000]})  # Limit text length
        
        instructions.append({
            "instruction": instruction,
            "input": paragraph[:500],  # Include part of the text as input
            "output": response.content
        })
    
    return instructions


def generate_qa_dataset(facts_file: str = "corpus/facts.json",
                        corpus_file: str = "corpus/city_corpus.txt",
                        output_dir: str = "dataset",
                        num_qa: int = 1000,
                        num_instructions: int = 500) -> Dict[str, str]:
    """
    Generate QA and instruction dataset in JSONL format
    
    Args:
        facts_file: Path to the facts JSON file
        corpus_file: Path to the corpus file
        output_dir: Directory to save the dataset
        num_qa: Number of QA pairs to generate
        num_instructions: Number of instruction pairs to generate
    
    Returns:
        Dict with paths to train/test files
    """
    print(f"Generating QA/instruction dataset...")
    
    # Load facts
    facts_path = Path(facts_file)
    if not facts_path.exists():
        raise FileNotFoundError(f"Facts file not found: {facts_file}")
    
    with open(facts_path, "r", encoding="utf-8") as f:
        facts_data = json.load(f)
        facts = facts_data.get("facts", [])
    
    # Load corpus
    corpus_path = Path(corpus_file)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_text = f.read()
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        temperature=float(os.getenv("TEMPERATURE", "0.7"))
    )
    
    # Generate QA pairs
    print(f"Generating {num_qa} QA pairs from facts...")
    qa_pairs = generate_qa_from_facts(facts, llm, num_qa)
    
    # Generate instruction pairs
    print(f"Generating {num_instructions} instruction pairs from corpus...")
    instruction_pairs = generate_instructions_from_corpus(corpus_text, llm, num_instructions)
    
    # Combine and convert to training format
    all_examples = []
    
    # QA format
    for qa in qa_pairs:
        all_examples.append({
            "text": f"Question: {qa.get('question', '')}\nAnswer: {qa.get('answer', '')}",
            "type": "qa"
        })
    
    # Instruction format
    for inst in instruction_pairs:
        all_examples.append({
            "text": f"### Instruction:\n{inst.get('instruction', '')}\n\n### Input:\n{inst.get('input', '')}\n\n### Response:\n{inst.get('output', '')}",
            "type": "instruction"
        })
    
    # Shuffle and split
    random.shuffle(all_examples)
    split_idx = int(len(all_examples) * 0.9)  # 90% train, 10% test
    train_examples = all_examples[:split_idx]
    test_examples = all_examples[split_idx:]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save train set
    train_file = output_path / "train.jsonl"
    with jsonlines.open(train_file, "w") as writer:
        writer.write_all(train_examples)
    
    # Save test set
    test_file = output_path / "test.jsonl"
    with jsonlines.open(test_file, "w") as writer:
        writer.write_all(test_examples)
    
    # Save metadata
    metadata = {
        "num_qa_pairs": len(qa_pairs),
        "num_instruction_pairs": len(instruction_pairs),
        "total_examples": len(all_examples),
        "train_examples": len(train_examples),
        "test_examples": len(test_examples)
    }
    
    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Generated {len(all_examples)} total examples")
    print(f"✓ Train set: {len(train_examples)} examples -> {train_file}")
    print(f"✓ Test set: {len(test_examples)} examples -> {test_file}")
    print(f"✓ Metadata saved to {metadata_file}")
    
    return {
        "train": str(train_file),
        "test": str(test_file),
        "metadata": str(metadata_file)
    }


if __name__ == "__main__":
    generate_qa_dataset()
