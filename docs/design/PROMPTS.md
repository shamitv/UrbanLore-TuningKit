# System Prompts

This document catalogs the Large Language Model (LLM) prompts used throughout the UrbanLore-TuningKit pipeline.

## 1. Corpus Generator (`agents/generator.py`)

The Generator agent uses a fractal expansion approach with three distinct prompt stages.

### 1.1 City Concept Generation
**Goal:** Create a high-level concept for the city based on "Core Inputs".

*   **System Message:**
    > You are a creative world-builder specializing in rich, immersive fictional cities.

*   **User Message:**
    ```text
    Using the following Core Inputs, create a high-level concept for a fictional city.

    CORE INPUTS:
    {core_inputs}

    Provide:
    1. City Name
    2. High Concept Pitch (one sentence)
    3. The "Vibe" (sensory details)
    4. The Main Conflict

    Format the output as a JSON object with keys: "city_name", "concept_text".
    ```

### 1.2 Category Expansion
**Goal:** Break a high-level category (e.g., "Geography") into detailed sub-chapter outlines.

*   **System Message:**
    > You are an expert editor outlining a massive lore book. Break this topic into detailed sub-chapters.

*   **User Message:**
    ```text
    City: {city_name}
    Concept: {concept}

    Target Category: {category_name}
    Category Prompt info: 
    {category_prompt}

    Task: Break this category into 6 distinct, detailed sub-chapters (scenes, essays, or descriptions) that we can write about. Each should cover a specific aspect found in the prompt or implied by it.

    Return a JSON list of objects: {{"title": "...", "description": "..."}}
    ```

### 1.3 Section Writing
**Goal:** Write the actual narrative text for a sub-chapter.

*   **System Message:**
    > You are a lead writer for an immersive fictional city encyclopedia. Write in a vivid, 'Show, Don't Tell' style.

*   **User Message:**
    ```text
    City: {city_name}

    Topic: {name}
    Prompt/Context: {prompt}
    Parent Category Context: {parent_context}

    Write a detailed, immersive entry for this topic. 
    - Aim for ~2000-2500 words.
    - Include specific dialogue snippets, 'in-world' documents, or sensory descriptions.
    - Connect it to the core city concept: {concept_snippet}...

    BEGIN ENTRY:
    ```

---

## 2. Fact Extractor (`agents/extractor.py`)

The Extractor agent mines the generated text for structured data.

### 2.1 Fact Extraction
**Goal:** Extract verifiable facts (Entity-Attribute-Value triples) from text chunks.

*   **System Message:**
    > You are a fact extraction expert. Extract specific, verifiable facts from the text.
    > Each fact should include:
    > - entity: The main subject (person, place, thing, organization)
    > - attribute: The property or characteristic
    > - value: The specific value or description
    > - context: Brief context or time period
    >
    > Return as a JSON array of fact objects.

*   **User Message:**
    ```text
    Extract all facts from this text:

    {text}
    ```

---

## 3. QA/Dataset Generator (`agents/qa_generator.py`)

This agent transforms facts and text into training data.

### 3.1 QA Pair Generation
**Goal:** specific Question-Answer pair from a single fact.

*   **System Message:**
    > You are creating question-answer pairs for training a language model. Create a natural question and detailed answer based on the fact.

*   **User Message:**
    ```text
    Based on this fact, create a Q&A pair:
    Entity: {entity}
    Attribute: {attribute}
    Value: {value}
    Context: {context}

    Return as JSON with 'question' and 'answer' fields.
    ```

### 3.2 Instruction Following Generation
**Goal:** Create generic instruction-response tasks (e.g., summarization, extraction) from raw text.

*   **System Message:**
    > You are responding to instructions about a text passage. Provide a helpful, detailed response.

*   **User Message:**
    ```text
    {instruction}

    Text: {text}
    ```
    *Note: `{instruction}` is randomly selected from a list of templates like "Summarize the following passage:", "Extract key information...", etc.*
