#!/usr/bin/env python3
"""
Corpus Generator Agent
Generates a ~200k-word fictional city corpus using LangGraph multi-agent pipeline
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

load_dotenv()


# State definition for the graph
class CorpusState(dict):
    """State for corpus generation pipeline"""
    city_name: str
    sections: List[str]
    current_section: str
    generated_content: Dict[str, str]
    word_count: int
    target_words: int


def create_city_concept(state: CorpusState) -> CorpusState:
    """Create the initial city concept"""
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        temperature=float(os.getenv("TEMPERATURE", "0.7"))
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a creative world-builder specializing in fictional cities."),
        ("user", "Create a unique fictional city. Provide: 1) City name, 2) Brief description (50 words), 3) Key characteristics. Format as JSON.")
    ])
    
    chain = prompt | llm
    response = chain.invoke({})
    
    # Parse response to get city name
    content = response.content
    try:
        # Try to parse as JSON
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            city_data = json.loads(json_match.group())
            state["city_name"] = city_data.get("city_name", "Neo-Haven")
        else:
            state["city_name"] = "Neo-Haven"
    except:
        state["city_name"] = "Neo-Haven"
    
    state["generated_content"]["concept"] = content
    state["word_count"] = len(content.split())
    
    print(f"✓ Created city concept: {state['city_name']}")
    return state


def generate_section(state: CorpusState) -> CorpusState:
    """Generate content for the current section"""
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        temperature=float(os.getenv("TEMPERATURE", "0.7"))
    )
    
    section = state["current_section"]
    city_name = state.get("city_name", "Neo-Haven")
    
    section_prompts = {
        "history": f"Write a detailed 3000-word history of {city_name}, including founding, major events, conflicts, and evolution.",
        "geography": f"Describe the geography and layout of {city_name} in 2500 words, including districts, landmarks, and natural features.",
        "culture": f"Write 3000 words about the culture, traditions, festivals, and social dynamics of {city_name}.",
        "economy": f"Describe the economy of {city_name} in 2500 words, including industries, trade, major businesses, and economic systems.",
        "government": f"Write 2500 words about the government structure, laws, political factions, and leadership of {city_name}.",
        "daily_life": f"Describe daily life in {city_name} in 3000 words, including typical residents, routines, education, and social life.",
        "notable_figures": f"Write 2500 words about notable figures, heroes, villains, and influential people in {city_name}'s history and present.",
        "mysteries": f"Describe 2000 words worth of mysteries, legends, unexplained phenomena, and secrets of {city_name}.",
    }
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a creative writer crafting a detailed fictional city world. Be specific, creative, and include many concrete details, names, dates, and facts."),
        ("user", "{section_prompt}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"section_prompt": section_prompts.get(section, f"Write about {section} of {city_name}")})
    
    content = response.content
    state["generated_content"][section] = content
    state["word_count"] += len(content.split())
    
    print(f"✓ Generated {section} section ({len(content.split())} words, total: {state['word_count']})")
    return state


def route_next_section(state: CorpusState) -> str:
    """Determine next section to generate"""
    if state["word_count"] >= state["target_words"]:
        return "finalize"
    
    if not state["sections"]:
        return "finalize"
    
    state["current_section"] = state["sections"].pop(0)
    return "generate"


def finalize_corpus(state: CorpusState) -> CorpusState:
    """Finalize and save the corpus"""
    print(f"✓ Corpus generation complete. Total words: {state['word_count']}")
    return state


def build_corpus_graph() -> CompiledStateGraph:
    """Build the LangGraph workflow for corpus generation"""
    workflow = StateGraph(CorpusState)
    
    # Add nodes
    workflow.add_node("create_concept", create_city_concept)
    workflow.add_node("generate_section", generate_section)
    workflow.add_node("finalize", finalize_corpus)
    
    # Add edges
    workflow.set_entry_point("create_concept")
    workflow.add_edge("create_concept", "generate_section")
    workflow.add_conditional_edges(
        "generate_section",
        route_next_section,
        {
            "generate": "generate_section",
            "finalize": "finalize"
        }
    )
    workflow.add_edge("finalize", END)
    
    return workflow.compile()


def generate_city_corpus(target_words: int = 200000, output_dir: str = "corpus") -> str:
    """
    Generate a fictional city corpus using multi-agent LangGraph pipeline
    
    Args:
        target_words: Target word count for the corpus
        output_dir: Directory to save the corpus
    
    Returns:
        Path to the generated corpus file
    """
    print(f"Starting corpus generation (target: {target_words} words)...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize state
    initial_state = CorpusState(
        city_name="",
        sections=["history", "geography", "culture", "economy", "government", "daily_life", "notable_figures", "mysteries"],
        current_section="",
        generated_content={},
        word_count=0,
        target_words=target_words
    )
    
    # Build and run the graph
    graph = build_corpus_graph()
    final_state = graph.invoke(initial_state)
    
    # Compile full corpus
    city_name = final_state.get("city_name", "Neo-Haven")
    full_corpus = f"# {city_name}: A Fictional City\n\n"
    
    for section, content in final_state["generated_content"].items():
        full_corpus += f"\n## {section.replace('_', ' ').title()}\n\n"
        full_corpus += content + "\n\n"
    
    # Save corpus
    corpus_file = output_path / "city_corpus.txt"
    with open(corpus_file, "w", encoding="utf-8") as f:
        f.write(full_corpus)
    
    # Save metadata
    metadata = {
        "city_name": city_name,
        "word_count": final_state["word_count"],
        "target_words": target_words,
        "sections": list(final_state["generated_content"].keys())
    }
    
    metadata_file = output_path / "corpus_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Corpus saved to {corpus_file}")
    print(f"✓ Metadata saved to {metadata_file}")
    print(f"✓ Final word count: {final_state['word_count']}")
    
    return str(corpus_file)


if __name__ == "__main__":
    generate_city_corpus()
