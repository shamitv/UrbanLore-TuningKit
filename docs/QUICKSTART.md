# UrbanLore-TuningKit Quickstart Guide

Get started with UrbanLore-TuningKit in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- OpenAI API key (for corpus generation)
- 8GB+ RAM (16GB recommended for fine-tuning)
- GPU recommended (but not required)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/shamitv/UrbanLore-TuningKit.git
cd UrbanLore-TuningKit
```

### 2. Install Dependencies

```bash
make setup
```

Or manually:

```bash
pip install -r requirements.txt
cp .env.example .env
```

### 3. Configure Your Environment

Edit `.env` file and add your OpenAI API key:

```bash
OPENAI_API_KEY=sk-your-api-key-here
```

Optional configurations:
- `OPENAI_MODEL=gpt-4` (or gpt-3.5-turbo for lower cost)
- `BASE_MODEL=microsoft/phi-2` (model to fine-tune)
- `CORPUS_TARGET_WORDS=200000` (adjust for faster testing)

## Quick Test Run

### Test with Smaller Parameters

For a quick test (5-10 minutes), create a test configuration:

```bash
# Generate small corpus
python urbanlore.py generate-corpus --target-words 5000 --output-dir test_output/corpus

# Extract facts
python urbanlore.py extract-facts \
  --corpus-file test_output/corpus/city_corpus.txt \
  --output-dir test_output/corpus

# Generate QA dataset
python urbanlore.py generate-qa \
  --facts-file test_output/corpus/facts.json \
  --corpus-file test_output/corpus/city_corpus.txt \
  --num-qa 50 \
  --num-instructions 25 \
  --output-dir test_output/dataset
```

## Full Pipeline Run

### Option 1: Use the All-in-One Command

```bash
python urbanlore.py run-all
```

This runs the complete pipeline:
1. Generate ~200k word corpus
2. Extract structured facts
3. Generate ~1500 QA/instruction pairs
4. Fine-tune model with LoRA
5. Evaluate the fine-tuned model

**Expected Duration**: 2-4 hours (depending on API speed and hardware)

### Option 2: Step-by-Step Execution

#### Step 1: Generate Corpus (~30-60 minutes)

```bash
make generate-corpus
# or
python urbanlore.py generate-corpus
```

Output:
- `corpus/city_corpus.txt` - Full text corpus
- `corpus/corpus_metadata.json` - Metadata

#### Step 2: Extract Facts (~10-20 minutes)

```bash
make extract-facts
# or
python urbanlore.py extract-facts
```

Output:
- `corpus/facts.json` - Structured facts

#### Step 3: Generate QA Dataset (~20-40 minutes)

```bash
make generate-qa
# or
python urbanlore.py generate-qa
```

Output:
- `dataset/train.jsonl` - Training data
- `dataset/test.jsonl` - Test data

#### Step 4: Fine-tune Model (~30-90 minutes)

```bash
make finetune
# or
python urbanlore.py finetune
```

Output:
- `finetune/models/final/` - Fine-tuned model

#### Step 5: Evaluate Model (~5-10 minutes)

```bash
make evaluate
# or
python urbanlore.py evaluate
```

Output:
- `eval/results/evaluation_results.json` - Metrics
- `eval/results/sample_predictions.json` - Examples

## Using the Makefile

The Makefile provides convenient shortcuts:

```bash
# See all available commands
make help

# Install dependencies
make install

# Setup (install + create .env)
make setup

# Run individual steps
make generate-corpus
make extract-facts
make generate-qa
make finetune
make evaluate

# Run everything
make all

# Clean generated files
make clean

# Run tests
make test

# Format code
make format

# Lint code
make lint
```

## Customization Examples

### Use a Different Model

Edit `.env`:
```bash
OPENAI_MODEL=gpt-3.5-turbo  # Cheaper option
BASE_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0  # Smaller model
```

### Adjust Corpus Size

```bash
python urbanlore.py generate-corpus --target-words 100000
```

### Change Dataset Size

```bash
python urbanlore.py generate-qa --num-qa 500 --num-instructions 250
```

### Modify Training Parameters

Edit `.env`:
```bash
NUM_EPOCHS=5
LEARNING_RATE=1e-4
BATCH_SIZE=8
```

## Verify Your Installation

Run the test suite:

```bash
make test
# or
pytest -v
```

Check CLI is working:

```bash
python urbanlore.py --help
```

## Example Outputs

### Sample Corpus Snippet

```
# Neo-Haven: A Fictional City

## History

In the year 1847, explorer Marcus Thornwood discovered a natural 
harbor along the eastern seaboard...
```

### Sample QA Pair

```json
{
  "text": "Question: When was Neo-Haven founded?\nAnswer: Neo-Haven was officially founded on March 15, 1848...",
  "type": "qa"
}
```

### Sample Evaluation Results

```json
{
  "metrics": {
    "rouge1": 0.4521,
    "rouge2": 0.2134,
    "rougeL": 0.3891
  }
}
```

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError`
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**Issue**: `OpenAI API key not found`
```bash
# Solution: Set your API key in .env
echo "OPENAI_API_KEY=sk-your-key" >> .env
```

**Issue**: Out of memory during fine-tuning
```bash
# Solution: Reduce batch size in .env
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=8
```

**Issue**: Slow corpus generation
```bash
# Solution: Use smaller target or faster model
CORPUS_TARGET_WORDS=50000
OPENAI_MODEL=gpt-3.5-turbo
```

### GPU Issues

If you don't have a GPU:
```bash
# The pipeline will automatically use CPU
# Fine-tuning will be slower but will work
```

If you have a GPU but it's not detected:
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

## Next Steps

1. **Explore Examples**: Check `examples/` directory
   ```bash
   python examples/example_workflow.py
   ```

2. **Read Documentation**: 
   - `ARCHITECTURE.md` - System design
   - `CONTRIBUTING.md` - Contribution guide
   - `README.md` - Full documentation

3. **Customize Pipeline**: 
   - Modify prompts in `agents/generator.py`
   - Add new sections to corpus
   - Create custom evaluation metrics

4. **Share Results**: 
   - Fine-tune different models
   - Compare evaluation metrics
   - Experiment with different corpus sizes

## Getting Help

- **Issues**: https://github.com/shamitv/UrbanLore-TuningKit/issues
- **Discussions**: https://github.com/shamitv/UrbanLore-TuningKit/discussions
- **Documentation**: See README.md and ARCHITECTURE.md

## Cost Estimation

Using OpenAI API (GPT-4):
- Corpus generation (200k words): ~$10-20
- Fact extraction: ~$5-10
- QA generation: ~$5-10
- **Total API cost**: ~$20-40

Using OpenAI API (GPT-3.5-Turbo):
- **Total API cost**: ~$2-5

Fine-tuning (local):
- Free (uses your hardware)

## Tips for Success

1. **Start Small**: Test with small parameters first
2. **Monitor Costs**: Watch your OpenAI API usage
3. **Use QLoRA**: Enable for memory efficiency
4. **Save Checkpoints**: Training saves progress automatically
5. **Experiment**: Try different models and parameters
6. **Read Logs**: Check output for errors or warnings

Happy fine-tuning! 🚀
