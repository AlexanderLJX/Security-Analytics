# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a phishing email detection system using reasoning distillation from GPT-5 to fine-tune a Qwen model for production deployment. The system combines large language model reasoning with efficient deployment for security operations. Features structured JSON output, multiple quantization levels (Q2_K to Q8_0), and enterprise SIEM integration including Splunk HEC.

## Key Development Commands

### Setup and Installation
```bash
# Create and activate virtual environment (using UV)
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# For GPU support, ensure CUDA is available
uv pip install torch --index-url https://download.pytorch.org/whl/cu118

# Create environment configuration
cp .env.template .env
# Edit .env with actual API keys (OPENAI_API_KEY, SPLUNK_TOKEN, etc.)
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

# Analyze single email with HuggingFace model
python main.py analyze --model-path ./checkpoints/phishing_detector/best --email-text "Sample email" --show-reasoning

# Analyze with GGUF model for CPU inference
python main.py analyze --model-path ./models/qwen_q4km/model-unsloth.Q4_K_M.gguf --email-file ./test_email.txt --device cpu

# Export LoRA checkpoint to GGUF with different quantization levels
python main.py export --model-path ./checkpoints/phishing_detector/best --output-path ./models/qwen_q4km --format gguf --quantization q4_k_m
python main.py export --model-path ./checkpoints/phishing_detector/best --output-path ./models/qwen_q3km --format gguf --quantization q3_k_m
python main.py export --model-path ./checkpoints/phishing_detector/best --output-path ./models/qwen_q2k --format gguf --quantization q2_k
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
1. Email input → LLM-based analysis (no manual feature extraction)
2. Teacher model generates detailed security reasoning (if training)
3. Student model processes email with structured JSON prompting
4. Output: Classification, confidence, risk score, indicators, recommended action, reasoning
5. SIEM integration → Kafka/Splunk alerts with MITRE ATT&CK mappings

### Key Configuration Files
- `config.py`: Centralized configuration with dataclasses for all components
- `.env.template`: Environment variables template (copy to `.env`)
- `pyproject.toml`: Project metadata, dependencies, and tool configurations

### Module Structure
- `data/`: Dataset loading, basic email preprocessing (simplified)
- `models/`: Teacher (GPT-5) and student (Qwen) model implementations with GGUF support
- `training/`: Training pipeline and reasoning distillation logic
- `evaluation/`: Metrics calculation and performance analysis
- `inference/`: Production detector with structured JSON output and SIEM integration
- `utils/`: Helper functions and utilities
- `tests/`: Test suite with fixtures and multiple test categories

### Key Design Patterns
- **Configuration Management**: Dataclass-based configs in `config.py` with environment variable overrides
- **Modular Architecture**: Clear separation between data processing, training, and inference
- **Caching Strategy**: Reasoning cache to avoid re-generating expensive teacher model outputs
- **API Design**: Teacher model abstraction supports both OpenAI and Anthropic APIs
- **Production Ready**: Multiple quantization levels (Q2_K to Q8_0), GGUF export, CPU/GPU inference, batch processing
- **Structured Output**: JSON prompting with reliable parsing (no manual text pattern matching)
- **Thinking Mode**: Preserved for higher quality reasoning while extracting structured output

### Testing Architecture
- **Fixtures**: Comprehensive test data in `tests/conftest.py`
- **Test Categories**: Unit, integration, slow, and GPU-specific test markers
- **Mocking**: Sample datasets and API responses for isolated testing
- **Coverage**: Configured to track coverage across all main modules

### Model Configuration
- Teacher model is configurable via `config.py` and currently set to `gpt-5`
- Student model uses Unsloth Qwen3-4B-Thinking with LoRA fine-tuning
- GGUF export with multiple quantization levels: Q2_K (2.5GB), Q3_K_M (3.5GB), Q4_K_M (5GB), Q8_0 (8GB)
- Supports both HuggingFace transformers and llama-cpp-python backends
- Automatic device detection (GPU/CPU) with optimized settings for each

### SIEM Integration
- **Kafka**: Producer for real-time security alerts with structured events
- **Splunk HEC**: HTTP Event Collector integration with proper indexing
- **MITRE ATT&CK**: Framework compliance with technique mappings (T1566.002, T1204.002)
- **RESTful API**: Health checks, single/batch processing capabilities
- **Threat Intelligence**: Automatic extraction of domains, URLs, email addresses, social engineering tactics
- **Risk Scoring**: Dynamic risk calculation based on classification, confidence, and indicators
- **Automated Actions**: Email quarantine, sender blocking, SOC alerting based on thresholds

### Important Notes for Development
- Always use the centralized configuration system in `config.py`
- Test GPU-dependent code with appropriate markers
- Use the reasoning cache to avoid expensive API calls during development
- Follow the established dataclass patterns for new configuration additions
- Maintain the modular structure when adding new components
- **Environment**: Use UV package manager with .venv virtual environment
- **Package Installation**: Always activate .venv first: `source .venv/bin/activate`
- **Model Inference**: Both QwenPhishingModel and PhishingDetector now use structured JSON output
- **GGUF Models**: Use llama-cpp-python backend, detected by .gguf file extension
- **Quantization**: Q3_K_M recommended for best balance of size/quality
- **SIEM**: Configure SPLUNK_HEC_URL and SPLUNK_TOKEN for Splunk integration
- **Debugging**: Model prompts and responses are logged for debugging structured output