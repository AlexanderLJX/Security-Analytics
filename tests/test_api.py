"""
Test cases for API functionality
"""
import pytest
import json
from unittest.mock import Mock, patch
from flask import Flask

# Mock the heavy dependencies
with patch.dict('sys.modules', {
    'torch': Mock(),
    'transformers': Mock(),
    'unsloth': Mock(),
    'kafka': Mock(),
}):
    from inference.siem_integration import PhishingAPI, SIEMIntegration

class MockDetector:
    """Mock detector for testing"""

    def analyze_email(self, email_text):
        # Return mock analysis result
        return {
            'classification': 'PHISHING' if 'urgent' in email_text.lower() else 'LEGITIMATE',
            'confidence': 0.85,
            'risk_score': 0.7,
            'risk_indicators': ['urgency'],
            'recommended_action': 'BLOCK',
            'reasoning': 'Mock reasoning for testing',
            'processing_time': 0.1
        }

class MockSIEMConfig:
    """Mock SIEM configuration"""
    kafka_broker = "localhost:9092"
    kafka_topic = "test-alerts"
    alert_threshold = 0.5
    api_host = "0.0.0.0"
    api_port = 5000

class TestPhishingAPI:
    """Test suite for Phishing API"""

    def setup_method(self):
        """Setup test client"""
        self.mock_detector = MockDetector()
        self.mock_config = MockSIEMConfig()

        # Mock the SIEM integration
        with patch('inference.siem_integration.KafkaProducer'):
            self.siem = SIEMIntegration(self.mock_detector, self.mock_config)
            self.api = PhishingAPI(self.siem)
            self.client = self.api.app.test_client()

    def test_health_check(self):
        """Test health check endpoint"""
        response = self.client.get('/health')

        assert response.status_code == 200

        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'stats' in data

    def test_analyze_single_email(self):
        """Test single email analysis endpoint"""
        email_data = {
            "email": "URGENT: Click here to verify your account"
        }

        response = self.client.post(
            '/analyze',
            data=json.dumps(email_data),
            content_type='application/json'
        )

        assert response.status_code == 200

        data = json.loads(response.data)
        assert 'analysis' in data
        assert 'alert' in data

        analysis = data['analysis']
        assert analysis['classification'] == 'PHISHING'
        assert 'confidence' in analysis
        assert 'risk_score' in analysis

    def test_analyze_legitimate_email(self):
        """Test analysis of legitimate email"""
        email_data = {
            "email": "Thank you for your recent purchase. Your order will ship soon."
        }

        response = self.client.post(
            '/analyze',
            data=json.dumps(email_data),
            content_type='application/json'
        )

        assert response.status_code == 200

        data = json.loads(response.data)
        analysis = data['analysis']
        assert analysis['classification'] == 'LEGITIMATE'

    def test_analyze_missing_email(self):
        """Test error handling for missing email"""
        response = self.client.post(
            '/analyze',
            data=json.dumps({}),
            content_type='application/json'
        )

        assert response.status_code == 400

        data = json.loads(response.data)
        assert 'error' in data

    def test_batch_analysis(self):
        """Test batch email analysis endpoint"""
        emails_data = {
            "emails": [
                "URGENT: Verify your account now!",
                "Thank you for your purchase."
            ]
        }

        response = self.client.post(
            '/batch',
            data=json.dumps(emails_data),
            content_type='application/json'
        )

        assert response.status_code == 200

        data = json.loads(response.data)
        assert 'results' in data
        assert len(data['results']) == 2

        # Check first email (phishing)
        first_result = data['results'][0]
        assert first_result['analysis']['classification'] == 'PHISHING'

        # Check second email (legitimate)
        second_result = data['results'][1]
        assert second_result['analysis']['classification'] == 'LEGITIMATE'

    def test_batch_analysis_missing_emails(self):
        """Test error handling for missing emails in batch"""
        response = self.client.post(
            '/batch',
            data=json.dumps({}),
            content_type='application/json'
        )

        assert response.status_code == 400

        data = json.loads(response.data)
        assert 'error' in data

    def test_invalid_json(self):
        """Test error handling for invalid JSON"""
        response = self.client.post(
            '/analyze',
            data="invalid json",
            content_type='application/json'
        )

        assert response.status_code == 400

class TestSIEMIntegration:
    """Test suite for SIEM Integration"""

    def setup_method(self):
        """Setup test instance"""
        self.mock_detector = MockDetector()
        self.mock_config = MockSIEMConfig()

    def test_create_alert(self):
        """Test alert creation"""
        with patch('inference.siem_integration.KafkaProducer'):
            siem = SIEMIntegration(self.mock_detector, self.mock_config)

            email_text = "URGENT: Verify your account"
            analysis_result = {
                'classification': 'PHISHING',
                'confidence': 0.9,
                'risk_score': 0.8,
                'risk_indicators': ['urgency', 'credential_request'],
                'recommended_action': 'BLOCK',
                'reasoning': 'Test reasoning',
                'processing_time': 0.1
            }

            alert = siem.create_alert(email_text, analysis_result)

            assert 'alert_id' in alert
            assert alert['event_type'] == 'EMAIL_SECURITY'
            assert alert['classification'] == 'PHISHING'
            assert alert['severity'] in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            assert 'mitre_attack' in alert

    def test_determine_severity(self):
        """Test severity determination"""
        with patch('inference.siem_integration.KafkaProducer'):
            siem = SIEMIntegration(self.mock_detector, self.mock_config)

            # Test critical severity
            critical_result = {'risk_score': 0.95}
            assert siem._determine_severity(critical_result) == 'CRITICAL'

            # Test high severity
            high_result = {'risk_score': 0.8}
            assert siem._determine_severity(high_result) == 'HIGH'

            # Test medium severity
            medium_result = {'risk_score': 0.6}
            assert siem._determine_severity(medium_result) == 'MEDIUM'

            # Test low severity
            low_result = {'risk_score': 0.4}
            assert siem._determine_severity(low_result) == 'LOW'

            # Test info severity
            info_result = {'risk_score': 0.2}
            assert siem._determine_severity(info_result) == 'INFO'

    def test_process_email(self):
        """Test email processing workflow"""
        with patch('inference.siem_integration.KafkaProducer'):
            siem = SIEMIntegration(self.mock_detector, self.mock_config)

            email_text = "URGENT: Click here now!"
            result = siem.process_email(email_text)

            assert 'analysis' in result
            assert 'alert' in result

            analysis = result['analysis']
            assert analysis['classification'] == 'PHISHING'

if __name__ == "__main__":
    pytest.main([__file__])