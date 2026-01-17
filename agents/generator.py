#!/usr/bin/env python3
"""
Corpus Generator Agent
Generates a ~200k-word fictional city corpus using LangGraph multi-agent pipeline
Refactored to match 'World Bible' prompt structure and use fractal expansion.
"""
import os
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

# Import the parser
from agents.prompt_parser import parse_world_bible_prompt

load_dotenv()


# State definition for the graph
class CorpusState(dict):
    """State for corpus generation pipeline"""
    city_name: str
    # sections is now a list of dicts: {"name": str, "prompt": str, "type": "category"|"sub_section", ...}
    sections: List[Dict[str, Any]] 
    current_section: Dict[str, Any]
    generated_content: Dict[str, str] # Keyed by section name
    word_count: int
    target_words: int
    concept_data: str # Store the raw concept text
    start_time: float # Epoch time when generation started


def get_llm():
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-5-nano"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        temperature=float(os.getenv("TEMPERATURE", "0.7"))
    )


def create_city_concept(state: CorpusState) -> CorpusState:
    """Create the initial city concept"""
    print("\n--- Step 1: Creating City Concept ---")
    
    # 1. Parse the prompt file
    # Assuming the file is in a fixed location relative to this script
    script_dir = Path(__file__).parent.parent
    prompt_path = script_dir / "docs" / "agents" / "sample_prompt.md"
    
    if not prompt_path.exists():
        # Fallback if path is wrong (e.g. running from different cwd)
        prompt_path = Path("docs/agents/sample_prompt.md")
    
    print(f"Loading prompt from: {prompt_path}")
    parsed_prompt = parse_world_bible_prompt(str(prompt_path))
    
    core_inputs = parsed_prompt["core_inputs"]
    categories = parsed_prompt["categories"]
    
    # 2. Generate Concept using the Core Inputs
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a creative world-builder specializing in rich, immersive fictional cities."),
        ("user", 
         """
         Using the following Core Inputs, create a high-level concept for a fictional city.
         
         CORE INPUTS:
         {core_inputs}
         
         Provide:
         1. City Name
         2. High Concept Pitch (one sentence)
         3. The "Vibe" (sensory details)
         4. The Main Conflict
         
         Format the output as a JSON object with keys: "city_name", "concept_text".
         """)
    ])
    
    chain = prompt | llm
    formatted_input = core_inputs if core_inputs else "A metropolis that defies physics, built on the back of a giant sleeping god."
    response = chain.invoke({"core_inputs": formatted_input})
    
    content = response.content
    city_name = "Neo-Haven"
    concept_text = content
    
    try:
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            city_name = data.get("city_name", "Neo-Haven")
            concept_text = data.get("concept_text", content)
    except:
        pass
        
    state["city_name"] = city_name
    state["concept_data"] = concept_text
    state["generated_content"]["_CONCEPT"] = concept_text
    state["word_count"] = len(concept_text.split())
    
    # 3. Initialize Sections from Parsed Categories
    # We mark them as type="category" so they get expanded later
    initial_sections = []
    for cat in categories:
        initial_sections.append({
            "name": cat["name"],
            "prompt": cat["full_prompt"],
            "type": "category",
            "id": cat["id"]
        })
        
    state["sections"] = initial_sections
    
    print(f"[OK] Created concept for '{city_name}'")
    print(f"[OK] Loaded {len(initial_sections)} base categories from prompt.")
    return state


def expand_category(state: CorpusState, category: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expands a high-level category into 5-8 detailed sub-chapters/scenes.
    """
    llm = get_llm()
    print(f"  > Expanding category: {category['name']}...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert editor outlining a massive lore book. Break this topic into detailed sub-chapters."),
        ("user", 
         """
         City: {city_name}
         Concept: {concept}
         
         Target Category: {category_name}
         Category Prompt info: 
         {category_prompt}
         
         Task: Break this category into 6 distinct, detailed sub-chapters (scenes, essays, or descriptions) that we can write about. Each should cover a specific aspect found in the prompt or implied by it.
         
         Return a JSON list of objects: {{"title": "...", "description": "..."}}
         """)
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "city_name": state["city_name"],
        "concept": state["concept_data"],
        "category_name": category["name"],
        "category_prompt": category["prompt"]
    })
    
    sub_sections = []
    try:
        json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
        if json_match:
            items = json.loads(json_match.group())
            for item in items:
                sub_sections.append({
                    "name": f"{category['name']}: {item['title']}",
                    "prompt": item['description'],
                    "type": "sub_section",
                    "parent_context": category['prompt']
                })
    except Exception as e:
        print(f"    ! Error parsing expansion: {e}")
        # Fallback: just use the category itself as a single section if expansion fails
        sub_sections.append({
            "name": category["name"],
            "prompt": category["prompt"],
            "type": "sub_section",
            "parent_context": ""
        })
        
    print(f"    -> Created {len(sub_sections)} sub-sections.")
    return sub_sections

def format_duration(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{int(h)}h {int(m)}m {int(s)}s"
    return f"{int(m)}m {int(s)}s"


def generate_section_node(state: CorpusState) -> CorpusState:
    """
    Main generation node.
    - Pop next section from queue.
    - If current section is 'category', expand it and enqueue sub-sections.
    - If current section is 'sub_section', generate the text.
    """
    if not state["sections"]:
        return state
        
    section = state["sections"].pop(0)
    state["current_section"] = section
    
    if section.get("type", "UNKNOWN") == "category":
        # EXPANSION PHASE
        new_sub_sections = expand_category(state, section)
        # Prepend to sections (Depth-First) to keep context freshness
        state["sections"] = new_sub_sections + state["sections"]
        # No 'generated_content' update here, just state manipulation
        return state
        
    else:
        # WRITING PHASE (sub_section)
        llm = get_llm()
        section_start = time.time()
        print(f"  > Writing: '{section['name']}'...", flush=True)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a lead writer for an immersive fictional city encyclopedia. Write in a vivid, 'Show, Don't Tell' style."),
            ("user", 
             """
             City: {city_name}
             
             Topic: {name}
             Prompt/Context: {prompt}
             Parent Category Context: {parent_context}
             
             Write a detailed, immersive entry for this topic. 
             - Aim for ~2000-2500 words.
             - Include specific dialogue snippets, 'in-world' documents, or sensory descriptions.
             - Connect it to the core city concept: {concept_snippet}...
             
             BEGIN ENTRY:
             """)
        ])
        
        # Truncate concept to save tokens if needed
        concept_snippet = state["concept_data"][:500]
        
        chain = prompt | llm
        response = chain.invoke({
            "city_name": state["city_name"],
            "name": section["name"],
            "prompt": section["prompt"],
            "parent_context": section.get("parent_context", ""),
            "concept_snippet": concept_snippet
        })
        
        content = response.content
        word_count = len(content.split())
        
        state["generated_content"][section["name"]] = content
        state["word_count"] += word_count
        
        duration = time.time() - section_start
        total_elapsed = time.time() - state["start_time"]
        
        print(f"[OK] Completed '{section['name']}'")
        print(f"     Stats: {word_count} words | Time: {format_duration(duration)} | Total Elapsed: {format_duration(total_elapsed)} | Total Words: {state['word_count']}")
        return state


def route_next_section(state: CorpusState) -> str:
    """Determine next step"""
    # Check if we hit target (optional soft limit)
    if state["word_count"] >= state["target_words"]:
        return "finalize" 
    
    if not state["sections"]:
        return "finalize"
    
    return "generate"


def finalize_corpus(state: CorpusState) -> CorpusState:
    """Finalize and save"""
    print(f"\n[OK] Corpus generation complete. Total words: {state['word_count']}")
    return state


def build_corpus_graph() -> CompiledStateGraph:
    workflow = StateGraph(CorpusState)
    
    workflow.add_node("create_concept", create_city_concept)
    workflow.add_node("generate_section", generate_section_node)
    workflow.add_node("finalize", finalize_corpus)
    
    workflow.set_entry_point("create_concept")
    workflow.add_edge("create_concept", "generate_section") # This logic needs the first route
    
    # Actually, create_concept doesn't pop. We need a router effectively or manual pop.
    # The standard pattern: Node -> Router -> Node
    # Let's add a router edge out of 'create_concept' too? 
    # Easier: Make 'create_concept' NOT set current_section, but just setup queue.
    # Then go to a router or direct to 'generate_section' which assumes 'current_section' is set?
    # Let's fix the graph flow.
    # Entry -> create_concept -> router -> generate -> router ...
    
    workflow.add_conditional_edges(
        "create_concept",
        route_next_section,
        {
            "generate": "generate_section",
            "finalize": "finalize"
        }
    )
    
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
    print(f"Starting Fractal Corpus Generation (Target: {target_words})...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    initial_state = CorpusState(
        city_name="",
        sections=[],
        current_section={},
        generated_content={},
        word_count=0,
        target_words=target_words,
        concept_data="",
        start_time=time.time()
    )
    
    graph = build_corpus_graph()
    # Recursion limit needs to be high because we loop many times (22 categories * 6 subsections = 132 steps)
    final_state = graph.invoke(initial_state, {"recursion_limit": 200})
    
    # Compile
    city_name = final_state.get("city_name", "Unknown City")
    full_corpus = f"# {city_name}: World Bible\n\n"
    full_corpus += f"{final_state.get('concept_data', '')}\n\n"
    
    # Sort keys? Or keep generation order? Generation order is safer if we want logical flow.
    # But generated_content is a dict... Python 3.7+ keeps insertion order.
    for section_name, content in final_state["generated_content"].items():
        if section_name == "_CONCEPT": continue
        full_corpus += f"\n## {section_name}\n\n"
        full_corpus += content + "\n\n"
        
    corpus_file = output_path / "city_corpus.txt"
    with open(corpus_file, "w", encoding="utf-8") as f:
        f.write(full_corpus)
        
    metadata = {
        "city_name": city_name,
        "word_count": final_state["word_count"],
        "sections": list(final_state["generated_content"].keys())
    }
    with open(output_path / "corpus_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Saved: {corpus_file}")
    return str(corpus_file)

if __name__ == "__main__":
    generate_city_corpus()
