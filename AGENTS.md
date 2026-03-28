# AGENTS.md - AI Coding Agent Guidelines

## Project Overview

Python-based ML project for QLoRA fine-tuning of LLMs using Unsloth. Trains models on RunPod GPU instances and exports to GGUF format for local inference on Mac M2.

## Build/Run Commands

```bash
# Setup environment (first time)
bash setup.sh
source venv/bin/activate

# Run training
python train.py

# Generate dataset with Bonito
python create_dataset.py --input raw_data.jsonl --output dataset.jsonl

# Install dependencies
uv pip install -r requirements.txt  # if exists
```

## Test Commands

```bash
# Run Python syntax check
python -m py_compile train.py
python -m py_compile create_dataset.py

# Type checking (if mypy installed)
mypy train.py --ignore-missing-imports
mypy create_dataset.py --ignore-missing-imports

# Run single test (if pytest available)
pytest test_file.py::test_function -v

# Run all tests
pytest -v
```

## Code Style Guidelines

### Imports
- Group imports: stdlib → third-party → local
- Use absolute imports
- Sort alphabetically within groups
- Example:
  ```python
  import argparse
  import json
  import os
  from typing import Dict, List, Optional
  from pathlib import Path
  ```

### Formatting
- 4 spaces indentation
- Max line length: 88-100 characters
- Use double quotes for strings
- Use single quotes for dict keys only if needed

### Naming Conventions
- Constants: `UPPER_CASE` (e.g., `MAX_SEQ_LENGTH`, `BONITO_MODEL`)
- Functions/variables: `snake_case` (e.g., `formatting_prompts_func`, `load_raw_texts`)
- Classes: `PascalCase` (e.g., `CustomTrainer`)
- Private functions: `_leading_underscore`

### Types
- Add type hints for function signatures when possible
- Use `Optional`, `List`, `Dict` from `typing` module
- Use Python 3.9+ built-in generics (`list[str]`, `dict[str, int]`) where available
- Return type hints for all functions

### Error Handling
- Use try/except blocks with specific exceptions
- Always log errors with context
- Use `set -e` in bash scripts for strict mode
- Prefer `raise from` for exception chaining

### Configuration
- Keep all tunable parameters as constants at file top
- Use environment variables for secrets (not hardcoded)
- Document config variables with comments
- Group related constants with separators

### Documentation
- Spanish comments are OK (project uses Spanish)
- Docstrings for all functions using triple quotes
- Keep README.md in sync with code changes
- Include Args, Returns, Raises sections in docstrings
- Use emoji prefixes for section headers in output

## Project Structure

```
/
├── train.py              # Main training script (QLoRA fine-tuning)
├── create_dataset.py     # Dataset generation pipeline using Bonito
├── setup.sh              # Environment setup script
├── dataset.jsonl         # Training data (instruction/input/output)
├── raw_data.jsonl        # Raw input data for Bonito generation
├── AGENTS.md             # This file
└── lora_model/           # Output directory (generated)
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── modelo_custom_m2.gguf
```

## Important Notes

- Requires CUDA-capable GPU for training
- Virtual environment must be activated before running
- Dataset must be in JSONL format with instruction/input/output fields
- Training outputs GGUF files for Mac M2 inference
- RunPod specific: Use `tmux` for long-running processes to prevent disconnection issues
- Bonito generation requires transformers and torch installed

## Git Workflow

- Never commit generated files (lora_model/, dataset.jsonl, raw_data.jsonl)
- Never commit secrets or API keys
- Write meaningful commit messages in Spanish or English
- Keep commits atomic and focused
- Use single-line commit messages (no multi-line descriptions)

## Environment Variables

No environment variables are currently required, but if adding:
- Use `.env` file for local development
- Never commit `.env` files
- Document all variables in README.md
