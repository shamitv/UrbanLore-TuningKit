#!/usr/bin/env python3
"""
Custom corpus generation example
Demonstrates how to customize the corpus generation process
"""
import os
from dotenv import load_dotenv
from agents.generator import generate_city_corpus

load_dotenv()


def generate_custom_corpus():
    """Generate a custom corpus with specific parameters"""
    
    # Override environment variables for this run
    os.environ["OPENAI_MODEL"] = "gpt-5-nano"
    os.environ["TEMPERATURE"] = "0.8"
    
    # Generate corpus with custom parameters
    corpus_file = generate_city_corpus(
        target_words=100000,  # Smaller corpus
        output_dir="examples/output/custom_corpus"
    )
    
    print(f"\nCustom corpus generated: {corpus_file}")
    
    # You can also modify the generation by:
    # 1. Editing the sections in the generator.py file
    # 2. Adjusting prompts for different styles
    # 3. Adding new section types


def generate_multiple_cities():
    """Generate multiple city corpora for variety"""
    cities = []
    
    for i in range(3):
        print(f"\nGenerating city {i+1}/3...")
        corpus_file = generate_city_corpus(
            target_words=30000,
            output_dir=f"examples/output/multi_cities/city_{i+1}"
        )
        cities.append(corpus_file)
    
    print(f"\nGenerated {len(cities)} cities:")
    for city_file in cities:
        print(f"  - {city_file}")


if __name__ == "__main__":
    print("Custom Corpus Generation Examples")
    print("=" * 80)
    
    # Example 1: Custom parameters
    print("\n[Example 1] Custom parameters")
    generate_custom_corpus()
    
    # Example 2: Multiple cities (commented out by default)
    # print("\n[Example 2] Multiple cities")
    # generate_multiple_cities()
