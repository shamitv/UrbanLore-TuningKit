# UrbanLore Pipeline Recoverability Strategy

## Status
**Implemented and Verified** (Jan 2026)
- ✅ Extractor persistence
- ✅ Generator persistence (Concept, Plans, Sections)
- ✅ QA Generator persistence (Batches)
- ✅ Verification passed

## Overview
The pipeline has been enhanced to be "recoverable". This means if the long-running generation or extraction processes are interrupted (e.g., by a system restart or crash), they can be resumed from where they left off without reprocessing completed work.

## Persistence Architecture

### 1. Corpus Generator (`generate-corpus`)
The generator now uses a 3-layer persistence strategy to ensure deterministic resumption:

1.  **City Concept** (`corpus/concept.json`): 
    - Stores the high-level city name, concept, and "vibe".
    - Ensures the core identity of the city doesn't change on restart.
    
2.  **Expansion Plans** (`corpus/plans/*.json`):
    - Stores the breakdown of each high-level category into sub-sections/chapters.
    - **Why?** The LLM breakdown is non-deterministic. By saving the plan, we ensure that subsequent runs look for the exact same sub-section filenames.

3.  **Section Content** (`corpus/sections/*.txt`):
    - Stores the actual generated text for each sub-section (approx. 2000 words each).
    - If a section file exists, the LLM generation is skipped.

**Flow:**
`Start` -> `Load Concept` -> `Load/Generate Plan` -> `Load/Generate Section` -> `Assemble Corpus`.

### 2. Fact Extractor (`extract-facts`)
The extractor processes the corpus sequentially in chunks (sections).

1.  **Intermediate Facts** (`corpus/facts_intermediate/section_N.json`):
    - Stores the facts extracted from a specific section index.
    - The extractor checks for this file before processing a section.

**Flow:**
`Start` -> `Split Corpus` -> `Loop Sections` -> `Check Intermediate File` -> `Skip or Extract` -> `Assemble Final JSON`.

### 3. QA Generator (`generate-qa`)
The QA generator processes requests in batches (batch size: 50).

1.  **Intermediate Batches**:
    - `dataset/qa_intermediate/batch_N.json` (QA pairs)
    - `dataset/instructions_intermediate/batch_N.json` (Instructions)
    
**Flow:**
`Start` -> `Sample Data` -> `Split to Batches` -> `Loop Batches` -> `Check Batch File` -> `Skip or Generate` -> `Assemble Final JSONL`.

## How to Resume
Simply run the same command again.
```bash
python urbanlore.py run-all ...
```
The system will automatically detect existing files and print `> Loading cached...` or `> Found cached section...`.

## How to Force Regeneration
To regenerate specific parts or the whole pipeline, you can:
1.  Use the `--force` flag (regenerates everything).
2.  Manually delete the specific intermediate folders (`corpus/plans`, `corpus/sections`, etc.) to trigger regeneration of those parts.
