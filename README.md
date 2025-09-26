# Phishing Detection with Reasoning Distillation

A comprehensive phishing email detection system using reasoning distillation from large language models (GPT-5) to fine-tune a smaller Qwen model for production deployment. Features structured JSON output, GGUF quantization for CPU inference, and enterprise SIEM integration.

## Features

- **Reasoning Distillation**: Uses GPT-5 as teacher model to generate detailed security reasoning
- **Production-Ready**: Optimized Qwen model with LoRA fine-tuning for fast inference
- **Multiple Inference Formats**: HuggingFace transformers, GGUF quantized models for CPU
- **Structured Output**: Reliable JSON format with confidence scores and risk indicators
- **SIEM Integration**: Kafka, Splunk HEC, and REST API for security operations
- **Comprehensive Evaluation**: Detailed metrics and performance analysis
- **Quantization Support**: Q2_K to Q8_0 quantization levels for different deployment needs

## Project Structure

```
phishing_detector/
├── main.py                    # Main entry point
├── config.py                  # Configuration settings
├── requirements.txt           # Dependencies
├── data/                      # Data processing modules
│   ├── dataset_loader.py      # Dataset loading and management
│   └── preprocessor.py        # Email preprocessing and features
├── models/                    # Model implementations
│   ├── qwen_model.py          # Qwen student model
│   └── teacher_model.py       # GPT-5 teacher model
├── training/                  # Training pipeline
│   ├── trainer.py             # Model training logic
│   └── reasoning_distillation.py # Reasoning generation
├── evaluation/                # Evaluation and metrics
│   └── metrics.py             # Performance metrics
├── inference/                 # Production inference
│   ├── detector.py            # Phishing detector
│   └── siem_integration.py    # SIEM alerts and API
└── utils/                     # Utilities
    └── helpers.py             # Helper functions
```

## Quick Start

### 1. Installation

```bash
# Create and activate virtual environment (using UV)
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# Create environment file
cp .env.template .env
# Edit .env with your API keys (OPENAI_API_KEY, SPLUNK_TOKEN, etc.)
```

### 2. Create Sample Dataset

```bash
python main.py create-sample --output-path ./data/phishing_emails.csv --num-samples 2000
```

### 3. Train Model

```bash
# Basic training (without reasoning generation)
python main.py train

# With reasoning generation from teacher model
python main.py train --generate-reasoning

# Resume from checkpoint
python main.py train --resume

# Save in GGUF format for llama.cpp
python main.py train --save-gguf
```

### 4. Evaluate Model

```bash
python main.py evaluate --model-path ./checkpoints/phishing_detector/best --save-results ./results/
```

### 5. Run Inference Server

```bash
python main.py server --model-path ./checkpoints/phishing_detector/best
```

### 6. Analyze Single Email

```bash
# Using HuggingFace model (GPU/CPU)
python main.py analyze --model-path ./checkpoints/phishing_detector/best --email-text "URGENT: Your account will be suspended!" --show-reasoning

# Using GGUF model for CPU inference
python main.py analyze --model-path ./models/qwen_q4km/model-unsloth.Q4_K_M.gguf --email-file ./test_email.txt --device cpu --show-reasoning

# Test base model without fine-tuning
python main.py analyze --email-text "Test email content" --device cpu --show-reasoning
```

### 7. Export Models

```bash
# Export LoRA checkpoint to merged HuggingFace format
python main.py export --model-path ./checkpoints/phishing_detector/best --output-path ./models/merged --format merged_16bit

# Export to GGUF for CPU inference (various quantization levels)
python main.py export --model-path ./checkpoints/phishing_detector/best --output-path ./models/qwen_q4km --format gguf --quantization q4_k_m

# Smaller quantized models
python main.py export --model-path ./checkpoints/phishing_detector/best --output-path ./models/qwen_q3km --format gguf --quantization q3_k_m
python main.py export --model-path ./checkpoints/phishing_detector/best --output-path ./models/qwen_q2k --format gguf --quantization q2_k
```

## Configuration

Edit `config.py` or use environment variables:

### Model Configuration
- `student_model`: Qwen model to fine-tune
- `teacher_model`: GPT-5 model for reasoning
- `max_seq_length`: Maximum sequence length
- `lora_r`: LoRA rank for efficient fine-tuning

### API Configuration
- `OPENAI_API_KEY`: OpenAI API key for GPT-5
- `ANTHROPIC_API_KEY`: Anthropic API key for Claude
- `api_provider`: "openai" or "anthropic"

### Training Configuration
- `num_epochs`: Training epochs
- `batch_size`: Training batch size
- `learning_rate`: Learning rate

### SIEM Configuration
- `KAFKA_BROKER`: Kafka broker for alerts
- `kafka_topic`: Topic for security alerts
- `SPLUNK_HEC_URL`: Splunk HTTP Event Collector URL
- `SPLUNK_TOKEN`: Splunk HEC authentication token
- `SPLUNK_INDEX`: Splunk index for security events
- `alert_threshold`: Risk score threshold for alerts

## API Usage

### Health Check
```bash
curl http://localhost:5000/health
```

### Analyze Email
```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"email": "URGENT: Click here to verify your account!"}'
```

### Batch Analysis
```bash
curl -X POST http://localhost:5000/batch \
  -H "Content-Type: application/json" \
  -d '{"emails": ["Email 1 text", "Email 2 text"]}'
```

## Analysis Capabilities

The system uses advanced LLM reasoning to analyze emails for:

- **Social Engineering**: Urgency tactics, authority impersonation, psychological manipulation
- **URL Analysis**: Suspicious domains, shortened links, phishing redirects
- **Sender Authentication**: Domain spoofing, Gmail impersonation of businesses
- **Information Harvesting**: PII requests, credential phishing, ID document requests
- **Financial Fraud**: Payment requests, fee scams, fake refunds
- **Content Analysis**: Grammar patterns, suspicious attachments, generic greetings
- **Behavioral Patterns**: Deviation from legitimate business communication

## Example Output

### Structured JSON Response
```json
{
  "classification": "PHISHING",
  "confidence": 0.95,
  "risk_score": 1.0,
  "recommended_action": "BLOCK",
  "risk_indicators": [
    "Suspicious email address (gmail domain)",
    "Request for sensitive information (ID scan)",
    "Urgent action required (48 hours)",
    "Fake airport agent claim",
    "Payment request for non-existent inspection fee"
  ],
  "reasoning": "Email uses a fake Gmail address for an airport agent, requests sensitive ID scans, and creates urgency with a 48-hour deadline for an abandoned package. Classic phishing scam designed to steal personal information and funds through fake urgency and impersonation.",
  "processing_time": 133.514
}
```

### Model Performance
- **Processing Time**: 2-3 minutes CPU (GGUF Q4_K_M), 30-60 seconds GPU
- **Accuracy**: 95%+ confidence on clear phishing attempts
- **Model Size**: 2.5GB (Q2_K) to 8GB (Q8_0) quantized versions
- **Inference**: Supports both thinking mode (better quality) and direct output

## SIEM Integration

### Supported Platforms
- **Kafka**: Real-time streaming alerts
- **Splunk**: HTTP Event Collector with structured indexing
- **REST API**: For custom integrations

### Splunk Event Format
```json
{
  "timestamp": "2025-09-25T20:04:03.354Z",
  "source": "phishing-detector-ai",
  "sourcetype": "phishing_detection",
  "host": "phishing-ai-worker-01",
  "index": "security",
  "event": {
    "alert_id": "phish-20250925-200403-001",
    "severity": "high",
    "email": {
      "sender": "inquiry.officeoo12@gmail.com",
      "subject": "Re: Your Package for Delivery"
    },
    "detection": {
      "classification": "PHISHING",
      "confidence": 0.95,
      "risk_score": 1.0,
      "recommended_action": "BLOCK"
    },
    "mitre_attack": {
      "tactics": ["T1566", "T1204"],
      "techniques": ["T1566.002", "T1204.002"]
    },
    "actions_taken": {
      "email_quarantined": true,
      "sender_blocked": true,
      "soc_alerted": true
    }
  }
}
```

### Splunk Searches
```splunk
# High-risk phishing attempts
index=security sourcetype=phishing_detection detection.risk_score>0.8

# Monitor blocked senders
index=security sourcetype=phishing_detection actions_taken.sender_blocked=true

# Track social engineering campaigns
index=security sourcetype=phishing_detection analysis.threat_indicators.social_engineering_tactics="urgency"
```

## Advanced Usage

### Custom Dataset

Replace the sample dataset with your own CSV file containing columns:
- `text`: Email content
- `type`: "phishing" or "legitimate"

### Model Optimization

- Use LoRA for efficient fine-tuning
- Quantization support (4-bit, 8-bit)
- GGUF export for llama.cpp deployment
- Gradient checkpointing for memory efficiency

### Production Deployment

- Docker containerization support
- Horizontal scaling with multiple workers
- Prometheus metrics integration
- Health checks and monitoring

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in config.py
2. **API Rate Limits**: Increase delays in reasoning generation
3. **Missing Dependencies**: Run `pip install -r requirements.txt`
4. **Model Loading Issues**: Check model path and permissions

### Logs

Set logging level in config:
```python
LOG_LEVEL=DEBUG
```

Enable file logging:
```bash
python main.py train --log-file training.log
```

## License

This project is for educational and defensive security purposes only.