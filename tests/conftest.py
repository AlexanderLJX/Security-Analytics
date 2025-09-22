"""
Pytest configuration and shared fixtures
"""
import pytest
import tempfile
import os
import pandas as pd
from pathlib import Path

@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing"""
    data = pd.DataFrame({
        'text': [
            'URGENT: Your account will be suspended! Click here now.',
            'Thank you for your recent purchase. Your order will ship soon.',
            'Congratulations! You have won $10,000. Claim your prize now!',
            'Your meeting is scheduled for tomorrow at 2 PM.',
            'Security alert: Unusual activity detected. Verify immediately.',
            'Your subscription has been renewed successfully.'
        ],
        'type': ['phishing', 'legitimate', 'phishing', 'legitimate', 'phishing', 'legitimate']
    })
    return data

@pytest.fixture
def temp_dataset_file(sample_dataset):
    """Create a temporary dataset file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_dataset.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def sample_phishing_email():
    """Sample phishing email for testing"""
    return """
    URGENT SECURITY ALERT!

    Dear Customer,

    We have detected suspicious activity on your account. Your account will be
    SUSPENDED within 24 hours unless you verify your identity immediately.

    Click here to verify: http://suspicious-bank-site.com/verify

    Provide the following information:
    - Username and Password
    - Social Security Number
    - Credit Card Details

    Failure to respond will result in permanent account closure.

    Security Team
    Bank of America
    """

@pytest.fixture
def sample_legitimate_email():
    """Sample legitimate email for testing"""
    return """
    Order Confirmation #12345

    Dear Valued Customer,

    Thank you for your recent purchase from our store. Your order has been
    confirmed and will be processed within 2-3 business days.

    Order Details:
    - Item: Wireless Headphones
    - Quantity: 1
    - Total: $89.99

    You can track your order status at our official website.

    If you have any questions, please contact our customer service team.

    Best regards,
    Customer Service Team
    Electronics Store
    """

@pytest.fixture
def mock_reasoning_response():
    """Mock reasoning response from teacher model"""
    return {
        'email': 'test email',
        'true_label': 'phishing',
        'initial_reasoning': 'This email shows signs of phishing...',
        'initial_prediction': 'phishing',
        'initial_correct': True,
        'corrected_reasoning': None,
        'adversarial_perspective': 'Attacker perspective...',
        'final_reasoning': 'Complete reasoning analysis...'
    }

@pytest.fixture
def sample_analysis_result():
    """Sample analysis result from detector"""
    return {
        'classification': 'PHISHING',
        'confidence': 0.92,
        'risk_score': 0.85,
        'risk_indicators': ['urgency', 'credential_request', 'suspicious_url'],
        'recommended_action': 'BLOCK',
        'reasoning': 'This email exhibits multiple phishing indicators...',
        'processing_time': 0.234,
        'features': {
            'urgency_score': 0.8,
            'suspicious_score': 0.7,
            'num_urls': 1,
            'has_shortened_url': False,
            'has_attachments': False
        }
    }

@pytest.fixture
def temp_config_file():
    """Create temporary configuration file"""
    config_content = """
    # Test configuration
    OPENAI_API_KEY=test_key
    ANTHROPIC_API_KEY=test_key
    KAFKA_BROKER=localhost:9092
    """

    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write(config_content)
        yield f.name
    os.unlink(f.name)

# Test markers
def pytest_configure(config):
    """Configure custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )

# Skip GPU tests if CUDA not available
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle GPU tests"""
    import torch

    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)