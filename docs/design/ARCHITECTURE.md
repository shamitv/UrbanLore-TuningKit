# UrbanLore-TuningKit Architecture

## System Overview

UrbanLore-TuningKit is a complete pipeline for generating synthetic training data and fine-tuning language models.

## Component Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Configuration Layer                       │
│  .env, config/default_config.yaml, environment variables      │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                      CLI Layer (urbanlore.py)                 │
│  Main entrypoint - Click-based CLI with 6 commands            │
└───────────────┬────────────────────────────────┬─────────────┘
                │                                │
    ┌───────────┴────────┐          ┌────────────┴──────────┐
    ▼                    ▼          ▼                       ▼
┌─────────┐      ┌──────────┐  ┌────────┐        ┌──────────┐
│ Agents  │      │ Finetune │  │  Eval  │        │ Examples │
│ Module  │      │  Module  │  │ Module │        │          │
└─────────┘      └──────────┘  └────────┘        └──────────┘
```

## Data Flow Pipeline

```
1. CORPUS GENERATION
   ┌────────────────────────────────┐
   │  agents/generator.py           │
   │  - LangGraph workflow          │
   │  - Multi-agent system          │
   │  - OpenAI API calls            │
   └────────────┬───────────────────┘
                │ Produces
                ▼
   ┌────────────────────────────────┐
   │  corpus/city_corpus.txt        │
   │  corpus/corpus_metadata.json   │
   │  (~200k words)                 │
   └────────────┬───────────────────┘
                │
                │
2. FACT EXTRACTION
   ┌────────────────────────────────┐
   │  agents/extractor.py           │
   │  - LLM-based fact extraction   │
   │  - Structured data creation    │
   └────────────┬───────────────────┘
                │ Produces
                ▼
   ┌────────────────────────────────┐
   │  corpus/facts.json             │
   │  (structured fact database)    │
   └────────────┬───────────────────┘
                │
                │
3. DATASET GENERATION
   ┌────────────────────────────────┐
   │  agents/qa_generator.py        │
   │  - QA pair generation          │
   │  - Instruction examples        │
   └────────────┬───────────────────┘
                │ Produces
                ▼
   ┌────────────────────────────────┐
   │  dataset/train.jsonl           │
   │  dataset/test.jsonl            │
   │  dataset/dataset_metadata.json │
   │  (JSONL format, ready for SFT) │
   └────────────┬───────────────────┘
                │
                │
4. FINE-TUNING
   ┌────────────────────────────────┐
   │  finetune/train.py             │
   │  - LoRA/QLoRA configuration    │
   │  - HF Transformers + PEFT      │
   │  - TRL SFTTrainer              │
   └────────────┬───────────────────┘
                │ Produces
                ▼
   ┌────────────────────────────────┐
   │  finetune/models/final/        │
   │  - Fine-tuned model            │
   │  - Tokenizer                   │
   │  - Training metadata           │
   └────────────┬───────────────────┘
                │
                │
5. EVALUATION
   ┌────────────────────────────────┐
   │  eval/evaluate.py              │
   │  - Model inference             │
   │  - ROUGE metrics               │
   │  - Sample predictions          │
   └────────────┬───────────────────┘
                │ Produces
                ▼
   ┌────────────────────────────────┐
   │  eval/results/                 │
   │  - evaluation_results.json     │
   │  - sample_predictions.json     │
   └────────────────────────────────┘
```

## Module Details

### Agents Module (`agents/`)

**Purpose**: Generate corpus and datasets using LLM agents

**Components**:
- `generator.py`: LangGraph-based multi-agent corpus generation
  - Creates city concept
  - Generates multiple sections (history, geography, culture, etc.)
  - Uses state management for workflow
  
- `extractor.py`: Fact extraction from corpus
  - Parses generated text
  - Extracts structured facts
  - Creates JSON database
  
- `qa_generator.py`: Dataset creation
  - Generates QA pairs from facts
  - Creates instruction-following examples
  - Splits into train/test sets

**Key Technologies**: LangChain, LangGraph, OpenAI API

### Finetune Module (`finetune/`)

**Purpose**: Fine-tune models with efficient parameter updates

**Components**:
- `train.py`: LoRA/QLoRA training
  - Supports 4-bit quantization (QLoRA)
  - Configurable LoRA parameters
  - HuggingFace Transformers integration
  
**Key Technologies**: Transformers, PEFT, BitsAndBytes, TRL, PyTorch

### Eval Module (`eval/`)

**Purpose**: Evaluate fine-tuned models

**Components**:
- `evaluate.py`: Model evaluation
  - Loads fine-tuned models
  - Generates predictions
  - Calculates ROUGE scores
  
**Key Technologies**: Transformers, ROUGE, scikit-learn

### Config Module (`config/`)

**Purpose**: Centralized configuration

**Components**:
- `default_config.yaml`: Default settings for all pipeline stages

### Examples Module (`examples/`)

**Purpose**: Demonstration and templates

**Components**:
- `example_workflow.py`: Complete pipeline example
- `custom_generation.py`: Custom parameter examples
- Sample data files

## Configuration System

### Environment Variables (.env)

Primary configuration mechanism using python-dotenv:
- OpenAI API settings
- Model parameters
- Training hyperparameters
- Directory paths

### YAML Configuration (config/default_config.yaml)

Structured configuration for advanced users:
- Nested settings
- Type-safe parameters
- Documentation inline

### CLI Arguments

Override configuration at runtime:
- Per-command options
- Command-specific parameters
- Flexible execution

## Extension Points

### Adding New Corpus Sections

Modify `agents/generator.py`:
- Add section to `sections` list
- Add prompt to `section_prompts` dict

### Custom Dataset Formats

Modify `agents/qa_generator.py`:
- Change JSONL structure
- Add new example types
- Modify train/test split

### Different Models

Change in `.env`:
```
BASE_MODEL=different/model-name
```

### Custom Evaluation Metrics

Extend `eval/evaluate.py`:
- Import new metrics
- Add to evaluation loop
- Include in results

## Dependencies

### Core Dependencies
- `langgraph`: Multi-agent workflows
- `langchain`: LLM orchestration
- `transformers`: Model loading and training
- `peft`: LoRA/QLoRA implementation
- `click`: CLI framework

### Optional Dependencies
- `bitsandbytes`: Quantization support
- `accelerate`: Distributed training
- `rouge-score`: Evaluation metrics

## Testing Strategy

### Unit Tests
- Module imports
- Configuration validation
- File structure

### Integration Tests
- End-to-end pipeline (with small data)
- API mocking for reproducibility

### Manual Testing
- Example workflows
- Documentation validation

## Performance Considerations

### Memory Management
- QLoRA for 4-bit quantization
- Gradient checkpointing available
- Batch size configuration

### Scalability
- Chunked corpus processing
- Streaming dataset loading
- Checkpoint-based training

### Cost Optimization
- Configurable word counts
- Adjustable example counts
- Local model fine-tuning

## Security

### API Key Management
- Environment variables only
- .env excluded from git
- .env.example for templates

### Data Privacy
- Local processing
- No external data sharing
- Configurable API endpoints

## Future Enhancements

Potential areas for expansion:
1. Support for additional base models
2. Alternative fact extraction methods
3. More evaluation metrics
4. Distributed training support
5. Web UI for pipeline management
6. Dataset versioning
7. Model registry integration
