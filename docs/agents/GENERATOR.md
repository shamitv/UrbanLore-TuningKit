# Generator agent

The generator builds a fictional city corpus using a LangGraph state machine. It creates a city concept, then iteratively generates section narratives until it hits the target word count or exhausts all sections.

---

## State diagram

```
                        ┌──────────────────┐
                        │      START       │
                        └────────┬─────────┘
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │   create_concept     │
                      │  (generate city idea)│
                      └──────────┬───────────┘
                                 │
                                 ▼
                      ┌──────────────────────┐
              ┌──────▶│   generate_section   │
              │       │ (write current section)│
              │       └──────────┬───────────┘
              │                  │
              │                  ▼
              │       ┌──────────────────────┐
              │       │  route_next_section  │
              │       │   (conditional edge) │
              │       └──────────┬───────────┘
              │                  │
              │       ┌──────────┴───────────┐
              │       │                      │
              │  word_count < target   word_count >= target
              │  AND sections left      OR sections empty
              │       │                      │
              │       ▼                      ▼
              │   "generate"            "finalize"
              │       │                      │
              └───────┘                      ▼
                                  ┌──────────────────────┐
                                  │       finalize       │
                                  │  (log completion)    │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                                  ┌──────────────────────┐
                                  │         END          │
                                  └──────────────────────┘
```

---

## How it works

### 1) State model (`CorpusState`)

Defined at [agents/generator.py#L21-L28](../agents/generator.py#L21-L28).

| Field               | Type                | Description                                       |
|---------------------|---------------------|---------------------------------------------------|
| `city_name`         | `str`               | Generated name of the fictional city              |
| `sections`          | `List[str]`         | Queue of section keys still to generate           |
| `current_section`   | `str`               | The section currently being generated             |
| `generated_content` | `Dict[str, str]`    | Map of section key → generated text               |
| `word_count`        | `int`               | Running total of words generated                  |
| `target_words`      | `int`               | Word count target (default 200 000)               |

### 2) Graph nodes

Built in `build_corpus_graph()` at [agents/generator.py#L124-L147](../agents/generator.py#L124-L147).

| Node               | Function             | Purpose                                                                                   |
|--------------------|----------------------|-------------------------------------------------------------------------------------------|
| `create_concept`   | `create_city_concept`| Calls LLM to invent city name & concept; stores result in `generated_content["concept"]` |
| `generate_section` | `generate_section`   | Generates prose for `current_section`; appends to `generated_content`; updates word count|
| `finalize`         | `finalize_corpus`    | Logs completion; no file I/O (writing happens in `generate_city_corpus`)                 |

### 3) Edges

| From               | To                 | Condition                                                    |
|--------------------|--------------------|--------------------------------------------------------------|
| START              | `create_concept`   | Entry point                                                  |
| `create_concept`   | `generate_section` | Unconditional                                                |
| `generate_section` | (conditional)      | Routed by `route_next_section`                               |
|                    | → `generate_section` | `word_count < target_words` **and** `sections` not empty   |
|                    | → `finalize`       | Otherwise                                                    |
| `finalize`         | END                | Unconditional                                                |

### 4) Routing logic

`route_next_section()` at [agents/generator.py#L106-L116](../agents/generator.py#L106-L116):

1. If `word_count >= target_words` → return `"finalize"`.
2. Else if `sections` list is empty → return `"finalize"`.
3. Else pop the first item from `sections` into `current_section` → return `"generate"`.

### 5) Section queue

The section order is hard-coded when `CorpusState` is initialised in `generate_city_corpus()` at [agents/generator.py#L170](../agents/generator.py#L170):

```python
sections=["history", "geography", "culture", "economy",
          "government", "daily_life", "notable_figures", "mysteries"]
```

Each section has a target word count baked into `section_prompts` inside `generate_section()`:

| Section          | Target words |
|------------------|--------------|
| history          | 3000         |
| geography        | 2500         |
| culture          | 3000         |
| economy          | 2500         |
| government       | 2500         |
| daily_life       | 3000         |
| notable_figures  | 2500         |
| mysteries        | 2000         |

---

## Output artifacts

`generate_city_corpus()` writes two files after the graph completes:

| File                         | Contents                                                   |
|------------------------------|------------------------------------------------------------|
| `corpus/city_corpus.txt`     | Full corpus with markdown section headers                  |
| `corpus/corpus_metadata.json`| JSON with `city_name`, `word_count`, `target_words`, `sections` |

---

## Configuration

Environment variables (loaded via `load_dotenv()`):

| Variable          | Default                         | Used for                        |
|-------------------|---------------------------------|---------------------------------|
| `OPENAI_MODEL`    | `gpt-4`                         | LLM model name                  |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1`     | API endpoint                    |
| `TEMPERATURE`     | `0.7`                           | Sampling temperature            |

---

## Running the generator

```bash
# As a module
python -m agents.generator

# Via CLI
python urbanlore.py generate-corpus --target-words 200000 --output-dir corpus
```

---

## Notes

- The pipeline stops as soon as `word_count >= target_words` **or** all sections are exhausted, so the final corpus may be slightly above or below the target.
- `city_name` is extracted from the LLM response as JSON; if parsing fails, it falls back to `"Neo-Haven"`.
