# UrbanLore-TuningKit

LangGraph multi-agent pipeline that generates a ~200k-word fictional city corpus, extracts structured facts and QA/instruction SFT dataset (JSONL), then fine-tunes a small HuggingFace model with LoRA/QLoRA and runs evaluations.

## 🎯 Overview

UrbanLore-TuningKit is a comprehensive toolkit for:

1. **Corpus Generation**: Multi-agent LangGraph workflow to generate rich, detailed fictional city lore
2. **Fact Extraction**: Automated extraction of structured facts from the generated corpus
3. **Dataset Creation**: Generation of QA pairs and instruction-following examples in JSONL format
4. **Fine-tuning**: LoRA/QLoRA-based fine-tuning of small language models
5. **Evaluation**: Comprehensive evaluation with ROUGE metrics and sample predictions

## 📁 Project Structure

```
UrbanLore-TuningKit/
├── agents/              # Multi-agent corpus and dataset generation
│   ├── generator.py     # Corpus generation using LangGraph
│   ├── extractor.py     # Fact extraction from corpus
│   └── qa_generator.py  # QA/instruction dataset generation
├── corpus/              # Generated corpus and facts (created at runtime)
├── dataset/             # Generated JSONL datasets (created at runtime)
├── finetune/            # Fine-tuning scripts and models
│   └── train.py         # LoRA/QLoRA training script
├── eval/                # Evaluation scripts and results
│   └── evaluate.py      # Model evaluation with metrics
├── config/              # Configuration files
│   └── default_config.yaml
├── examples/            # Example scripts and sample data
│   ├── example_workflow.py
│   ├── custom_generation.py
│   ├── sample_corpus.txt
│   └── sample_dataset.jsonl
├── urbanlore.py         # Main CLI entrypoint
├── Makefile             # Convenient commands
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variables template
└── README.md            # This file
```

## 🚀 Quick Start

## TL;DR

```bash
python urbanlore.py run-all --target-words 200000 --corpus-dir corpus --dataset-dir dataset --model-dir finetune/models --base-model Qwen/Qwen3-0.6B --eval-dir eval/results --num-qa 1000 --num-instructions 500 --use-qlora true
```

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/shamitv/UrbanLore-TuningKit.git
cd UrbanLore-TuningKit

# Install dependencies and setup
make setup
```

For a quick, script-based setup that creates required folders and initializes .env (if missing), see [docs/infra/INIT_SETUP.md](docs/infra/INIT_SETUP.md) and the [docs/QUICKSTART.md](docs/QUICKSTART.md).

### 2. Configuration

Copy `.env.example` to `.env` and configure your settings:

```bash
cp .env.example .env
# Edit .env with your OpenAI API key and preferences
```

**Key Configuration Variables:**
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_BASE_URL`: OpenAI API base URL (default: https://api.openai.com/v1)
- `OPENAI_MODEL`: Model to use for generation (default: gpt-5-nano)
- `BASE_MODEL`: HuggingFace model for fine-tuning (default: Qwen/Qwen3-0.6B)
- `USE_QLORA`: Enable QLoRA quantization (default: true)

### 3. Run the Pipeline

#### Using CLI Commands

```bash
# Generate corpus (~200k words)
python urbanlore.py generate-corpus

# Extract facts from corpus
python urbanlore.py extract-facts

# Generate QA/instruction dataset
python urbanlore.py generate-qa

# Fine-tune model with LoRA/QLoRA
python urbanlore.py finetune

# Evaluate the fine-tuned model
python urbanlore.py evaluate

# Or run everything at once
python urbanlore.py run-all
```

#### Using Makefile

```bash
# Individual steps
make generate-corpus
make extract-facts
make generate-qa
make finetune
make evaluate

# Run complete pipeline
make all
```

## 📖 Detailed Usage

### Corpus Generation

Generate a fictional city corpus with rich details:

```bash
python urbanlore.py generate-corpus --target-words 200000 --output-dir corpus
```

This creates:
- `corpus/city_corpus.txt`: Full text corpus
- `corpus/corpus_metadata.json`: Generation metadata

### Fact Extraction

Extract structured facts from the corpus:

```bash
python urbanlore.py extract-facts --corpus-file corpus/city_corpus.txt --output-dir corpus
```

Creates:
- `corpus/facts.json`: Extracted facts in structured format

### QA Dataset Generation

Generate QA and instruction-following examples:

```bash
python urbanlore.py generate-qa \
  --facts-file corpus/facts.json \
  --corpus-file corpus/city_corpus.txt \
  --num-qa 1000 \
  --num-instructions 500 \
  --output-dir dataset
```

Creates:
- `dataset/train.jsonl`: Training dataset
- `dataset/test.jsonl`: Test dataset
- `dataset/dataset_metadata.json`: Dataset statistics

### Fine-tuning

Fine-tune a model using LoRA/QLoRA:

```bash
python urbanlore.py finetune \
  --dataset-file dataset/train.jsonl \
  --base-model Qwen/Qwen3-0.6B \
  --use-qlora \
  --epochs 3 \
  --output-dir finetune/models
```

Configuration options:
- `--base-model`: HuggingFace model name
- `--use-qlora`: Enable 4-bit quantization
- `--epochs`: Number of training epochs

Creates:
- `finetune/models/<org>/<model>/final/`: Fine-tuned model
- `finetune/models/<org>/<model>/checkpoints/`: Training checkpoints
- `finetune/models/<org>/<model>/final/training_metadata.json`: Training info

### Evaluation

Evaluate the fine-tuned model:

```bash
python urbanlore.py evaluate \
  --model-dir finetune/models/Qwen/Qwen3-0.6B/final \
  --test-file dataset/test.jsonl \
  --output-dir eval/results
```

Creates:
- `eval/results/<org>/<model>/evaluation_results.json`: ROUGE scores and metrics
- `eval/results/<org>/<model>/sample_predictions.json`: Example predictions

For detailed information on the evaluation metrics and methodology, see [docs/eval/EVALUATION.md](docs/eval/EVALUATION.md).

## 🧪 Test Runs

See the recorded pipeline run summary at [docs/runs/test_pipeline_2026-01-17.md](docs/runs/test_pipeline_2026-01-17.md).

## 📚 Documentation

Comprehensive documentation for UrbanLore-TuningKit:

### Setup & Getting Started
- **[QUICKSTART.md](docs/QUICKSTART.md)**: Fastest way to get the pipeline running.
- **[INIT_SETUP.md](docs/infra/INIT_SETUP.md)**: Automated environment initialization.
- **[PYTORCH_SETUP.md](docs/infra/pytorch_rtx5080_windows_setup.md)**: Detailed CUDA/GPU configuration for RTX 50-series.

### Architecture & Design
- **[ARCHITECTURE.md](docs/design/ARCHITECTURE.md)**: System design and multi-agent workflow details.
- **[GENERATOR.md](docs/agents/GENERATOR.md)**: Technical details on the corpus generation agent.

### Evaluation & Results
- **[EVALUATION.md](docs/eval/EVALUATION.md)**: Explanation of ROUGE metrics and scoring logic.
- **[Test Runs](docs/runs/test_pipeline_2026-01-17.md)**: Evidence of successful pipeline execution.

### Project & Contribution
- **[CONTRIBUTING.md](docs/project/CONTRIBUTING.md)**: Guidelines for contributing to the toolkit.
- **[PROJECT_SUMMARY.md](docs/project/PROJECT_SUMMARY.md)**: High-level overview of goals and features.


## 🔧 Environment Variables

The `.env` file controls all aspects of the pipeline. Key variables:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-5-nano

# Generation Parameters
CORPUS_TARGET_WORDS=200000
TEMPERATURE=0.7

# Fine-tuning Configuration
BASE_MODEL=Qwen/Qwen3-0.6B
LORA_R=16
LORA_ALPHA=32
USE_QLORA=true
BATCH_SIZE=4
NUM_EPOCHS=3
LEARNING_RATE=2e-4
```

## 📚 Examples

See the `examples/` directory for:

- `example_workflow.py`: Complete pipeline example
- `custom_generation.py`: Custom generation parameters
- `sample_corpus.txt`: Example corpus excerpt
- `sample_dataset.jsonl`: Example QA/instruction data

Run examples:

```bash
python examples/example_workflow.py
python examples/custom_generation.py
```

## 🛠️ Development

### Testing

```bash
make test
```

### Code Formatting

```bash
make format
make lint
```

### Cleaning Generated Files

```bash
make clean
```

## 📊 Pipeline Architecture

```
┌─────────────────────┐
│   Corpus Generator  │  LangGraph multi-agent workflow
│   (agents/          │  → Generates ~200k word city lore
│    generator.py)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Fact Extractor     │  LLM-based fact extraction
│  (agents/          │  → Structured fact database
│   extractor.py)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  QA Generator       │  Generate training examples
│  (agents/          │  → JSONL format datasets
│   qa_generator.py) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Fine-tuner         │  LoRA/QLoRA training
│  (finetune/        │  → Fine-tuned model
│   train.py)        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Evaluator          │  ROUGE metrics & samples
│  (eval/            │  → Performance report
│   evaluate.py)     │
└─────────────────────┘
```

For a detailed technical deep-dive into how the dataset is created and how we ensure chat format alignment, see [docs/design/PIPELINE_DETAILS.md](docs/design/PIPELINE_DETAILS.md).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) for multi-agent orchestration
- Uses [HuggingFace Transformers](https://github.com/huggingface/transformers) for model fine-tuning
- Powered by [PEFT](https://github.com/huggingface/peft) for efficient LoRA/QLoRA training

## 📧 Contact

For questions and support, please open an issue on GitHub.
