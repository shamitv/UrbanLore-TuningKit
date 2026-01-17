.PHONY: help install setup clean generate-corpus extract-facts generate-qa finetune evaluate all test lint format

help:
	@echo "UrbanLore-TuningKit - Makefile Commands"
	@echo "========================================"
	@echo "setup              - Install dependencies and create .env file"
	@echo "install            - Install Python dependencies"
	@echo "generate-corpus    - Generate fictional city corpus (~200k words)"
	@echo "extract-facts      - Extract structured facts from corpus"
	@echo "generate-qa        - Generate QA/instruction dataset (JSONL)"
	@echo "finetune           - Fine-tune model with LoRA/QLoRA"
	@echo "evaluate           - Run model evaluation"
	@echo "all                - Run complete pipeline (corpus -> QA -> finetune -> eval)"
	@echo "clean              - Clean generated files"
	@echo "test               - Run tests"
	@echo "lint               - Run code linting"
	@echo "format             - Format code with black"

install:
	pip install -r requirements.txt

setup: install
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file. Please edit it with your configuration."; \
	else \
		echo ".env file already exists."; \
	fi
	@mkdir -p corpus dataset finetune/models eval/results

clean:
	rm -rf corpus/*.txt corpus/*.json
	rm -rf dataset/*.jsonl dataset/*.json
	rm -rf finetune/models/*
	rm -rf eval/results/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

generate-corpus:
	python -m agents.generator

extract-facts:
	python -m agents.extractor

generate-qa:
	python -m agents.qa_generator

finetune:
	python -m finetune.train

evaluate:
	python -m eval.evaluate

all: generate-corpus extract-facts generate-qa finetune evaluate
	@echo "Complete pipeline finished!"

test:
	pytest -v

lint:
	ruff check .

format:
	black .
