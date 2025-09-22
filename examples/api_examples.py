"""
API usage examples for phishing detection system
"""
import requests
import json
from typing import Dict, List

# API base URL
BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test the health check endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check Response:")
    print(json.dumps(response.json(), indent=2))
    return response.json()

def analyze_single_email(email_text: str) -> Dict:
    """Analyze a single email"""
    payload = {"email": email_text}

    response = requests.post(
        f"{BASE_URL}/analyze",
        headers={"Content-Type": "application/json"},
        json=payload
    )

    result = response.json()
    print("Analysis Result:")
    print(json.dumps(result, indent=2))
    return result

def analyze_batch_emails(emails: List[str]) -> Dict:
    """Analyze multiple emails in batch"""
    payload = {"emails": emails}

    response = requests.post(
        f"{BASE_URL}/batch",
        headers={"Content-Type": "application/json"},
        json=payload
    )

    result = response.json()
    print("Batch Analysis Results:")
    print(json.dumps(result, indent=2))
    return result

def main():
    """Run example API calls"""

    # Sample phishing email
    phishing_email = """
    URGENT: Your account will be suspended!

    Dear Customer,

    We have detected suspicious activity on your account. Click here immediately
    to verify your identity: http://suspicious-link.com/verify

    Failure to act within 24 hours will result in permanent account closure.

    Security Team
    """

    # Sample legitimate email
    legitimate_email = """
    Thank you for your recent purchase!

    Dear Valued Customer,

    Your order #12345 has been confirmed and will be shipped within 2-3 business days.

    You can track your order at our official website.

    Best regards,
    Customer Service Team
    """

    print("=== Health Check ===")
    test_health_check()

    print("\n=== Single Email Analysis ===")
    print("Analyzing phishing email...")
    analyze_single_email(phishing_email)

    print("\nAnalyzing legitimate email...")
    analyze_single_email(legitimate_email)

    print("\n=== Batch Email Analysis ===")
    analyze_batch_emails([phishing_email, legitimate_email])

if __name__ == "__main__":
    main()