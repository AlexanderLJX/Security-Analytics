"""
Helper utilities for phishing detection system
"""
import logging
import sys
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
import json
import random

def setup_logging(log_file: Optional[str] = None, level: str = "INFO"):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    if log_file:
        logging.basicConfig(
            level=numeric_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(
            level=numeric_level,
            format=log_format,
            handlers=[logging.StreamHandler(sys.stdout)]
        )

    return logging.getLogger(__name__)

def create_sample_dataset(output_path: str, num_samples: int = 1000):
    """Create a sample phishing dataset for testing"""

    # Sample phishing emails
    phishing_samples = [
        "URGENT: Your account will be suspended! Click here to verify your identity immediately.",
        "Congratulations! You've won $10,000! Click this link to claim your prize now.",
        "Security Alert: Unusual activity detected. Please verify your login credentials.",
        "Your payment has failed. Update your credit card information to avoid service interruption.",
        "Final Notice: Your subscription expires today. Renew now to avoid losing access.",
        "Bank Alert: Your account has been compromised. Click here to secure your account.",
        "You have received a new message. Click here to view your secure document.",
        "IRS Notice: You owe back taxes. Pay immediately to avoid legal action.",
        "Your package could not be delivered. Click here to reschedule delivery.",
        "LinkedIn: Someone viewed your profile. Click here to see who."
    ]

    # Sample legitimate emails
    legitimate_samples = [
        "Thank you for your recent purchase. Your order #12345 has been shipped.",
        "Your monthly statement is now available in your online banking portal.",
        "Reminder: Your appointment with Dr. Smith is scheduled for tomorrow at 2 PM.",
        "Your subscription to our newsletter has been confirmed. Welcome!",
        "Meeting reminder: Team standup at 9 AM in conference room B.",
        "Your flight check-in is now available. Flight AA123 departs at 3:45 PM.",
        "Welcome to our service! Here's how to get started with your new account.",
        "Your order has been delivered. Please rate your experience.",
        "System maintenance is scheduled for this weekend. Services may be unavailable.",
        "Your password was successfully changed. If this wasn't you, please contact support."
    ]

    # Generate dataset
    data = []

    for i in range(num_samples):
        if i % 2 == 0:  # Generate phishing sample
            base_text = random.choice(phishing_samples)
            # Add some variation
            variations = [
                f"Dear Customer, {base_text}",
                f"Important: {base_text}",
                f"Action Required: {base_text}",
                base_text,
                f"{base_text} Please act within 24 hours."
            ]
            text = random.choice(variations)
            label = "phishing"
        else:  # Generate legitimate sample
            base_text = random.choice(legitimate_samples)
            variations = [
                f"Hello, {base_text}",
                f"Hi there, {base_text}",
                base_text,
                f"{base_text} Best regards, Customer Service",
                f"{base_text} Thank you for your business."
            ]
            text = random.choice(variations)
            label = "legitimate"

        data.append({
            'text': text,
            'type': label,
            'id': i
        })

    # Create DataFrame and save
    df = pd.DataFrame(data)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Created sample dataset with {num_samples} emails at {output_path}")
    print(f"Phishing emails: {len(df[df['type'] == 'phishing'])}")
    print(f"Legitimate emails: {len(df[df['type'] == 'legitimate'])}")

    return df

def validate_dataset(dataset_path: str) -> Dict:
    """Validate dataset format and content"""
    try:
        df = pd.read_csv(dataset_path)

        # Check required columns
        required_columns = ['text', 'type']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            return {
                'valid': False,
                'error': f"Missing required columns: {missing_columns}",
                'columns': list(df.columns)
            }

        # Check for empty values
        empty_text = df['text'].isna().sum()
        empty_type = df['type'].isna().sum()

        # Check label distribution
        label_counts = df['type'].value_counts().to_dict()

        # Check text length statistics
        text_lengths = df['text'].str.len()

        return {
            'valid': True,
            'total_samples': len(df),
            'empty_text': empty_text,
            'empty_type': empty_type,
            'label_distribution': label_counts,
            'text_length_stats': {
                'min': text_lengths.min(),
                'max': text_lengths.max(),
                'mean': text_lengths.mean(),
                'median': text_lengths.median()
            },
            'unique_labels': df['type'].unique().tolist()
        }

    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }

def create_env_template(output_path: str = ".env.template"):
    """Create environment variable template file"""
    template_content = """# API Configuration
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Kafka Configuration
KAFKA_BROKER=localhost:9092

# Model Configuration
MODEL_PATH=./checkpoints/phishing_detector/best

# Logging
LOG_LEVEL=INFO
LOG_FILE=phishing_detector.log

# Data Paths
DATASET_PATH=./data/phishing_emails.csv
REASONING_CACHE_PATH=./data/reasoning_cache.json

# Training Configuration
BATCH_SIZE=2
LEARNING_RATE=2e-4
NUM_EPOCHS=3

# Inference Configuration
API_HOST=0.0.0.0
API_PORT=5000
ALERT_THRESHOLD=0.7
"""

    with open(output_path, 'w') as f:
        f.write(template_content)

    print(f"Created environment template at {output_path}")
    print("Please copy this to .env and fill in your actual values")

def check_dependencies():
    """Check if required dependencies are installed"""
    # Map package names to their import names
    package_imports = {
        'torch': 'torch',
        'transformers': 'transformers',
        'unsloth': 'unsloth',
        'trl': 'trl',
        'datasets': 'datasets',
        'openai': 'openai',
        'anthropic': 'anthropic',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'kafka-python': 'kafka',
        'flask': 'flask'
    }

    missing_packages = []

    for package_name, import_name in package_imports.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)

    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print("pip install -r requirements.txt")
        return False
    else:
        print("All required packages are installed!")
        return True

def format_size(bytes):
    """Format bytes as human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} PB"

def get_model_info(model_path: str) -> Dict:
    """Get information about a saved model"""
    model_path = Path(model_path)

    if not model_path.exists():
        return {'exists': False, 'error': 'Model path does not exist'}

    info = {'exists': True, 'path': str(model_path)}

    # Check for model files
    config_file = model_path / 'config.json'
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            info['model_type'] = config.get('model_type', 'unknown')
            info['vocab_size'] = config.get('vocab_size', 'unknown')
        except:
            info['config_error'] = 'Could not read config.json'

    # Calculate total size
    total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
    info['total_size'] = format_size(total_size)

    # List files
    info['files'] = [f.name for f in model_path.iterdir() if f.is_file()]

    return info