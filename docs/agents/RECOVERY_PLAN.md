# Recovery Plan for Generator Agent

This document outlines how to make the corpus generator agent resilient to failures by persisting intermediate state and enabling resumption from checkpoints.

---

## Problem Statement

The generator runs a long multi-step pipeline:

1. Create city concept
2. Expand each category into sub-sections
3. Write each sub-section (~2000 words)

With 22 categories × ~6 sub-sections = ~132 LLM calls, the full run takes hours and is vulnerable to:

- Network failures / API timeouts
- Rate limits
- Process crashes
- Machine restarts

Currently, if the process fails mid-run, all progress is lost.

---

## Goals

1. **Persist state after each step** so the agent can resume from the last successful checkpoint.
2. **Save metadata immediately** after the concept is generated (city name, config).
3. **Write sections incrementally** to disk as they are generated.
4. **Detect and resume** from an existing checkpoint directory.

---

## Proposed Checkpoint Structure

```
corpus/
├── checkpoint.json          # Current pipeline state (serialized CorpusState)
├── city_corpus.txt          # Incrementally appended corpus (final output)
├── corpus_metadata.json     # Metadata (updated incrementally)
├── sections/                # Individual section files
│   ├── _CONCEPT.md
│   ├── Geography__The_Coastal_Districts.md
│   ├── History__The_Founding.md
│   └── ...
└── expansion_cache/         # Cached category expansions
    ├── Geography.json
    ├── History.json
    └── ...
```

---

## State Serialization

### Fields to persist in `checkpoint.json`

```json
{
  "version": 1,
  "city_name": "Yorkbori",
  "target_words": 200000,
  "word_count": 45230,
  "start_time": 1737100000.0,
  "concept_data": "...",
  "sections_queue": [
    {"name": "Economy: Trade Routes", "prompt": "...", "type": "sub_section", "parent_context": "..."}
  ],
  "current_section": null,
  "completed_sections": ["_CONCEPT", "Geography: Coastal Districts", "..."],
  "phase": "generating"  
}
```

### Phases

| Phase        | Description                                      |
|--------------|--------------------------------------------------|
| `init`       | Not started                                      |
| `concept`    | Concept generation in progress                   |
| `generating` | Section generation loop                          |
| `finalize`   | All sections done, compiling final output        |
| `done`       | Pipeline complete                                |

---

## Recovery Logic

### On startup

```python
def generate_city_corpus(target_words: int, output_dir: str, resume: bool = True) -> str:
    checkpoint_path = Path(output_dir) / "checkpoint.json"
    
    if resume and checkpoint_path.exists():
        state = load_checkpoint(checkpoint_path)
        print(f"Resuming from checkpoint. Phase: {state['phase']}, Words: {state['word_count']}")
    else:
        state = create_initial_state(target_words)
    
    # Continue pipeline from current phase
    ...
```

### After each step

```python
def generate_section_node(state: CorpusState) -> CorpusState:
    # ... generate content ...
    
    # Persist immediately
    save_section_to_disk(state, section_name, content)
    save_checkpoint(state)
    
    return state
```

---

## Incremental Writes

### Section files

Write each section to `sections/{safe_filename}.md` immediately after generation:

```python
def save_section_to_disk(state: CorpusState, section_name: str, content: str):
    sections_dir = Path(state["output_dir"]) / "sections"
    sections_dir.mkdir(exist_ok=True)
    
    safe_name = section_name.replace(":", "_").replace(" ", "_")
    path = sections_dir / f"{safe_name}.md"
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {section_name}\n\n")
        f.write(content)
```

### Expansion cache

Cache category expansions to avoid re-calling LLM on resume:

```python
def expand_category(state: CorpusState, category: Dict) -> List[Dict]:
    cache_dir = Path(state["output_dir"]) / "expansion_cache"
    cache_file = cache_dir / f"{category['id']}.json"
    
    if cache_file.exists():
        print(f"  > Loading cached expansion for {category['name']}")
        return json.loads(cache_file.read_text())
    
    # ... call LLM ...
    
    cache_dir.mkdir(exist_ok=True)
    cache_file.write_text(json.dumps(sub_sections, indent=2))
    return sub_sections
```

---

## Checkpoint Save/Load

```python
def save_checkpoint(state: CorpusState):
    checkpoint = {
        "version": 1,
        "city_name": state["city_name"],
        "target_words": state["target_words"],
        "word_count": state["word_count"],
        "start_time": state["start_time"],
        "concept_data": state["concept_data"],
        "sections_queue": state["sections"],
        "current_section": state.get("current_section"),
        "completed_sections": list(state["generated_content"].keys()),
        "phase": state.get("phase", "generating"),
    }
    
    path = Path(state["output_dir"]) / "checkpoint.json"
    # Atomic write
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(checkpoint, indent=2))
    tmp.replace(path)


def load_checkpoint(path: Path) -> CorpusState:
    data = json.loads(path.read_text())
    
    output_dir = path.parent
    
    # Rebuild generated_content from section files
    generated_content = {}
    sections_dir = output_dir / "sections"
    for section_name in data["completed_sections"]:
        safe_name = section_name.replace(":", "_").replace(" ", "_")
        section_file = sections_dir / f"{safe_name}.md"
        if section_file.exists():
            # Strip the markdown header we added
            text = section_file.read_text()
            # Remove first line (# Title)
            lines = text.split("\n", 2)
            generated_content[section_name] = lines[2] if len(lines) > 2 else ""
    
    return CorpusState(
        city_name=data["city_name"],
        sections=data["sections_queue"],
        current_section=data.get("current_section") or {},
        generated_content=generated_content,
        word_count=data["word_count"],
        target_words=data["target_words"],
        concept_data=data["concept_data"],
        start_time=data["start_time"],
        output_dir=str(output_dir),
        phase=data.get("phase", "generating"),
    )
```

---

## Updated CorpusState

Add fields:

```python
class CorpusState(dict):
    city_name: str
    sections: List[Dict[str, Any]]
    current_section: Dict[str, Any]
    generated_content: Dict[str, str]
    word_count: int
    target_words: int
    concept_data: str
    start_time: float
    # NEW
    output_dir: str      # Path to output directory
    phase: str           # "init" | "concept" | "generating" | "finalize" | "done"
```

---

## Graph Modifications

The LangGraph workflow needs minimal changes:

1. Pass `output_dir` in initial state.
2. Call `save_checkpoint(state)` at the end of each node.
3. On resume, skip to the appropriate node based on `phase`.

For simplicity, we can keep the existing graph structure and just wrap the entry point:

```python
def generate_city_corpus(..., resume: bool = True):
    checkpoint_path = output_path / "checkpoint.json"
    
    if resume and checkpoint_path.exists():
        initial_state = load_checkpoint(checkpoint_path)
        # If phase == "generating", invoke graph at generate_section
        # If phase == "concept", invoke at create_concept
        # etc.
    else:
        initial_state = CorpusState(...)
        initial_state["phase"] = "init"
    
    graph = build_corpus_graph()
    final_state = graph.invoke(initial_state, {"recursion_limit": 200})
```

Or, more robustly, use LangGraph's built-in checkpointing if available.

---

## CLI Flags

Add CLI options:

```bash
# Fresh start (ignore existing checkpoint)
python urbanlore.py generate-corpus --no-resume

# Resume from checkpoint (default)
python urbanlore.py generate-corpus

# Show checkpoint status
python urbanlore.py checkpoint-status --output-dir corpus/
```

---

## Error Handling

Wrap LLM calls with retry logic:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=60))
def call_llm_with_retry(chain, inputs):
    return chain.invoke(inputs)
```

On persistent failure, save checkpoint and exit gracefully:

```python
except Exception as e:
    print(f"[ERROR] Failed after retries: {e}")
    save_checkpoint(state)
    raise SystemExit(1)
```

---

## Summary of Changes

| File                  | Change                                                      |
|-----------------------|-------------------------------------------------------------|
| `agents/generator.py` | Add checkpoint save/load, incremental writes, phase tracking |
| `urbanlore.py`        | Add `--no-resume` flag, `checkpoint-status` command         |
| `CorpusState`         | Add `output_dir`, `phase` fields                            |

---

## Implementation Order

1. Add `output_dir` and `phase` to `CorpusState`.
2. Implement `save_checkpoint()` and `load_checkpoint()`.
3. Implement `save_section_to_disk()`.
4. Add expansion cache in `expand_category()`.
5. Call save functions at end of each node.
6. Update `generate_city_corpus()` to detect and resume from checkpoint.
7. Add CLI flags.
8. Add retry logic with tenacity.
9. Update docs.

---

## Open Questions

1. **LangGraph native checkpointing**: LangGraph has checkpointing support via `MemorySaver` / `SqliteSaver`. Should we use that instead of manual checkpointing? Pros: less code. Cons: less control over file format.

2. **Partial section recovery**: If the process crashes mid-section write, should we retry that section or skip it? Current plan: retry (section file won't exist until write completes).

3. **Concept regeneration**: If concept phase completes but no sections are generated, should resume regenerate the concept? Current plan: no, use cached concept.
