"""
Basic tests for the UrbanLore-TuningKit scaffold
"""
import os
import pytest
from pathlib import Path


def test_project_structure():
    """Test that all required directories exist"""
    base_dir = Path(__file__).parent.parent
    
    required_dirs = [
        "agents",
        "corpus",
        "dataset",
        "finetune",
        "eval",
        "config",
        "examples",
    ]
    
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        assert dir_path.exists(), f"Directory {dir_name} should exist"
        assert dir_path.is_dir(), f"{dir_name} should be a directory"


def test_required_files():
    """Test that all required files exist"""
    base_dir = Path(__file__).parent.parent
    
    required_files = [
        "urbanlore.py",
        "Makefile",
        "requirements.txt",
        ".env.example",
        "README.md",
        "agents/__init__.py",
        "agents/generator.py",
        "agents/extractor.py",
        "agents/qa_generator.py",
        "finetune/__init__.py",
        "finetune/train.py",
        "eval/__init__.py",
        "eval/evaluate.py",
        "config/default_config.yaml",
    ]
    
    for file_name in required_files:
        file_path = base_dir / file_name
        assert file_path.exists(), f"File {file_name} should exist"
        assert file_path.is_file(), f"{file_name} should be a file"


def test_env_example_format():
    """Test that .env.example has required variables"""
    base_dir = Path(__file__).parent.parent
    env_file = base_dir / ".env.example"
    
    with open(env_file, "r") as f:
        content = f.read()
    
    required_vars = [
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "BASE_MODEL",
        "LORA_R",
        "LORA_ALPHA",
    ]
    
    for var in required_vars:
        assert var in content, f"Environment variable {var} should be in .env.example"


def test_makefile_targets():
    """Test that Makefile has required targets"""
    base_dir = Path(__file__).parent.parent
    makefile = base_dir / "Makefile"
    
    with open(makefile, "r") as f:
        content = f.read()
    
    required_targets = [
        "help",
        "install",
        "setup",
        "generate-corpus",
        "extract-facts",
        "generate-qa",
        "finetune",
        "evaluate",
        "all",
        "clean",
    ]
    
    for target in required_targets:
        assert f"{target}:" in content, f"Makefile should have {target} target"


def test_imports():
    """Test that modules can be imported (skipped if dependencies not installed)"""
    try:
        # Test agents imports
        from agents import generate_city_corpus, extract_facts_from_corpus, generate_qa_dataset
        assert callable(generate_city_corpus)
        assert callable(extract_facts_from_corpus)
        assert callable(generate_qa_dataset)
        
        # Test finetune imports
        from finetune import train_model
        assert callable(train_model)
        
        # Test eval imports
        from eval import evaluate_model
        assert callable(evaluate_model)
    except ImportError as e:
        # Skip if dependencies not installed
        pytest.skip(f"Skipping import test due to missing dependencies: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
