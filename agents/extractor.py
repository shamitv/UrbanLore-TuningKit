#!/usr/bin/env python3
"""
Fact Extractor Agent
Extracts structured facts from the generated corpus
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def extract_facts_from_text(text: str, llm: ChatOpenAI) -> List[Dict[str, Any]]:
    """Extract structured facts from a text chunk"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a fact extraction expert. Extract specific, verifiable facts from the text.
        Each fact should include:
        - entity: The main subject (person, place, thing, organization)
        - attribute: The property or characteristic
        - value: The specific value or description
        - context: Brief context or time period
        
        Return as a JSON array of fact objects."""),
        ("user", "Extract all facts from this text:\n\n{text}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"text": text})
    
    # Parse JSON response
    try:
        import re
        json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
        if json_match:
            facts = json.loads(json_match.group())
            return facts
        else:
            # If no JSON array found, return empty list
            return []
    except Exception as e:
        print(f"Warning: Could not parse facts: {e}")
        return []


def extract_facts_from_corpus(corpus_file: str = "corpus/city_corpus.txt", 
                               output_dir: str = "corpus") -> str:
    """
    Extract structured facts from the corpus
    
    Args:
        corpus_file: Path to the corpus file
        output_dir: Directory to save the extracted facts
    
    Returns:
        Path to the facts JSON file
    """
    print(f"Extracting facts from {corpus_file}...")
    
    # Load corpus
    corpus_path = Path(corpus_file)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_text = f.read()
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-5-nano"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        temperature=float(os.getenv("TEMPERATURE", "0.1"))
    )
    
    # Split corpus into sections for processing
    sections = corpus_text.split("##")
    all_facts = []
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Intermediate storage
    intermediate_dir = output_path / "facts_intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    
    for i, section in enumerate(sections):
        if not section.strip():
            continue
            
        section_file = intermediate_dir / f"section_{i}.json"
        
        if section_file.exists():
            print(f"Loading cached facts for section {i+1}/{len(sections)}...")
            try:
                with open(section_file, "r", encoding="utf-8") as f:
                    section_facts = json.load(f)
                    all_facts.extend(section_facts)
                continue
            except Exception as e:
                print(f"Error loading cache, reprocessing: {e}")

        print(f"Processing section {i+1}/{len(sections)}...")
        
        # Extract facts from this section
        section_facts = extract_facts_from_text(section, llm)
        all_facts.extend(section_facts)
        
        # Save intermediate result
        with open(section_file, "w", encoding="utf-8") as f:
            json.dump(section_facts, f, indent=2)
    
    # Save combined facts
    facts_file = output_path / "facts.json"
    with open(facts_file, "w", encoding="utf-8") as f:
        json.dump({
            "source_corpus": str(corpus_file),
            "num_facts": len(all_facts),
            "facts": all_facts
        }, f, indent=2)
    
    print(f"\n✓ Extracted {len(all_facts)} facts")
    print(f"✓ Facts saved to {facts_file}")
    
    return str(facts_file)


if __name__ == "__main__":
    extract_facts_from_corpus()
