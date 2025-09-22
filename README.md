# Phishing Detection with Reasoning Distillation

A comprehensive phishing email detection system using reasoning distillation from large language models (GPT-4/Claude) to fine-tune a smaller Qwen model for production deployment.

## Features

- **Reasoning Distillation**: Uses GPT-4/Claude as teacher models to generate detailed security reasoning
- **Production-Ready**: Optimized Qwen model for fast inference
- **SIEM Integration**: Kafka-based alerting and Flask API for security operations
- **Comprehensive Evaluation**: Detailed metrics and visualizations
- **Feature Extraction**: Advanced email preprocessing and security feature detection

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
│   └── teacher_model.py       # GPT-4/Claude teacher model
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
# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.template .env
# Edit .env with your API keys
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
# From command line
python main.py analyze --model-path ./checkpoints/phishing_detector/best --email-text "URGENT: Your account will be suspended!"

# From file
python main.py analyze --model-path ./checkpoints/phishing_detector/best --email-file ./test_email.txt --show-reasoning
```

## Configuration

Edit `config.py` or use environment variables:

### Model Configuration
- `student_model`: Qwen model to fine-tune
- `teacher_model`: GPT-4/Claude model for reasoning
- `max_seq_length`: Maximum sequence length
- `lora_r`: LoRA rank for efficient fine-tuning

### API Configuration
- `OPENAI_API_KEY`: OpenAI API key for GPT-4
- `ANTHROPIC_API_KEY`: Anthropic API key for Claude
- `api_provider`: "openai" or "anthropic"

### Training Configuration
- `num_epochs`: Training epochs
- `batch_size`: Training batch size
- `learning_rate`: Learning rate

### SIEM Configuration
- `KAFKA_BROKER`: Kafka broker for alerts
- `kafka_topic`: Topic for security alerts
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

## Security Features

The system analyzes emails for:

- **Sender Authentication**: SPF, DKIM, domain spoofing
- **URL Analysis**: Suspicious links, shorteners, redirects
- **Social Engineering**: Urgency, fear appeals, authority abuse
- **Technical Indicators**: Grammar errors, formatting issues
- **Content Analysis**: Credential requests, attachments
- **Behavioral Patterns**: Deviation from normal communication

## Example Output

```json
{
  "classification": "PHISHING",
  "confidence": 0.95,
  "risk_score": 0.87,
  "recommended_action": "BLOCK",
  "risk_indicators": [
    "urgency",
    "credential_request",
    "suspicious_url"
  ],
  "reasoning": "This email exhibits multiple phishing indicators including urgent language, credential harvesting attempts, and suspicious URLs...",
  "processing_time": 0.234
}
```

## SIEM Integration

Alerts are sent to Kafka in MITRE ATT&CK format:

```json
{
  "alert_id": "PHISH_20240101120000_abc12345",
  "severity": "HIGH",
  "event_type": "EMAIL_SECURITY",
  "mitre_attack": {
    "technique": "T1566",
    "name": "Phishing",
    "tactic": "Initial Access"
  },
  "risk_score": 0.87,
  "recommended_action": "BLOCK"
}
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