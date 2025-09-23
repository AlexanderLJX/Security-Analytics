# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a phishing email detection system using reasoning distillation from GPT-5 to fine-tune a Qwen model for production deployment. The system combines large language model reasoning with efficient deployment for security operations.

## Key Development Commands

### Setup and Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Install with GPU support (Unsloth)
pip install -e .[gpu]

# Install development dependencies
pip install -e .[dev]

# Create environment configuration
cp .env.template .env
# Edit .env with actual API keys
```

### Testing
```bash
# Run all tests
pytest

# Run specific test types
pytest -m unit
pytest -m integration
pytest -m "not slow"
pytest -m "not gpu"

# Run tests with coverage
pytest --cov=data --cov=models --cov=training --cov=evaluation --cov=inference --cov=utils

# Run single test file
pytest tests/test_dataset_loader.py

# Run specific test function
pytest tests/test_dataset_loader.py::TestPhishingDatasetLoader::test_load_base_dataset
```

### Linting and Code Quality
```bash
# Format code
black . --line-length 100

# Sort imports
isort .

# Lint code
flake8

# Type checking
mypy .
```

### Model Training and Evaluation
```bash
# Create sample dataset
python main.py create-sample --output-path ./data/sample.csv --num-samples 2000

# Train without reasoning generation
python main.py train

# Train with GPT-5 reasoning generation
python main.py train --generate-reasoning

# Resume training from checkpoint
python main.py train --resume

# Evaluate model
python main.py evaluate --model-path ./checkpoints/phishing_detector/best

# Analyze single email
python main.py analyze --model-path ./checkpoints/phishing_detector/best --email-text "Sample email"
```

### Production Server
```bash
# Start inference server
python main.py server --model-path ./checkpoints/phishing_detector/best

# Health check
curl http://localhost:5000/health

# Analyze email via API
curl -X POST http://localhost:5000/analyze -H "Content-Type: application/json" -d '{"email": "Email text"}'
```

## Architecture Overview

### Core Components
- **Teacher Model**: GPT-5 for generating detailed security reasoning chains
- **Student Model**: Qwen model fine-tuned with LoRA for efficient production inference
- **Reasoning Distillation**: Process that transfers teacher model knowledge to student
- **SIEM Integration**: Kafka-based alerting and Flask API for security operations

### Data Flow
1. Email input → Feature extraction (URLs, urgency, patterns)
2. Teacher model generates detailed security reasoning (if training)
3. Student model processes email with reasoning guidance
4. Output: Classification, confidence, risk score, indicators, recommended action

### Key Configuration Files
- `config.py`: Centralized configuration with dataclasses for all components
- `.env.template`: Environment variables template (copy to `.env`)
- `pyproject.toml`: Project metadata, dependencies, and tool configurations

### Module Structure
- `data/`: Dataset loading, preprocessing, feature extraction
- `models/`: Teacher (GPT-5) and student (Qwen) model implementations
- `training/`: Training pipeline and reasoning distillation logic
- `evaluation/`: Metrics calculation and performance analysis
- `inference/`: Production detector and SIEM integration
- `utils/`: Helper functions and utilities
- `tests/`: Test suite with fixtures and multiple test categories

### Key Design Patterns
- **Configuration Management**: Dataclass-based configs in `config.py` with environment variable overrides
- **Modular Architecture**: Clear separation between data processing, training, and inference
- **Caching Strategy**: Reasoning cache to avoid re-generating expensive teacher model outputs
- **API Design**: Teacher model abstraction supports both OpenAI and Anthropic APIs
- **Production Ready**: Quantization, GGUF export, batch processing, and monitoring support

### Testing Architecture
- **Fixtures**: Comprehensive test data in `tests/conftest.py`
- **Test Categories**: Unit, integration, slow, and GPU-specific test markers
- **Mocking**: Sample datasets and API responses for isolated testing
- **Coverage**: Configured to track coverage across all main modules

### Model Configuration
- Teacher model is configurable via `config.py` and currently set to `gpt-5`
- Student model uses Unsloth Qwen implementation with LoRA fine-tuning
- Both quantization (4-bit/8-bit) and GGUF export supported for deployment

### SIEM Integration
- Kafka producer for real-time security alerts
- MITRE ATT&CK framework compliance for alert formatting
- RESTful API with health checks and batch processing capabilities
- Configurable risk thresholds and recommended actions

### Important Notes for Development
- Always use the centralized configuration system in `config.py`
- Test GPU-dependent code with appropriate markers
- Use the reasoning cache to avoid expensive API calls during development
- Follow the established dataclass patterns for new configuration additions
- Maintain the modular structure when adding new components
- Remember to use UV instead. the venv is at .venv
- Remember i created a .venv with UV, so when you install packages you need to activate the venv at .venv first