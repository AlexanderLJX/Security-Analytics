"""
Test cases for email preprocessor functionality
"""
import pytest
from data.preprocessor import EmailPreprocessor

class TestEmailPreprocessor:
    """Test suite for EmailPreprocessor"""

    def setup_method(self):
        """Setup test instance"""
        self.preprocessor = EmailPreprocessor()

    def test_extract_urls(self):
        """Test URL extraction from text"""
        text = "Click here http://example.com or visit https://test.org"
        urls = self.preprocessor.extract_urls(text)

        assert len(urls) == 2
        assert "http://example.com" in urls
        assert "https://test.org" in urls

    def test_extract_features_basic(self):
        """Test basic feature extraction"""
        email_text = "URGENT: Click here http://bit.ly/test to verify your account!"
        features = self.preprocessor.extract_features(email_text)

        # Check basic features
        assert 'length' in features
        assert 'num_words' in features
        assert 'num_urls' in features
        assert features['num_urls'] == 1

        # Check urgency detection
        assert features['urgency_score'] > 0
        assert features['num_urgency_words'] > 0

        # Check shortened URL detection
        assert features['has_shortened_url'] == True

    def test_urgency_detection(self):
        """Test urgency word detection"""
        urgent_email = "URGENT action required immediate response"
        normal_email = "Thank you for your purchase"

        urgent_features = self.preprocessor.extract_features(urgent_email)
        normal_features = self.preprocessor.extract_features(normal_email)

        assert urgent_features['urgency_score'] > normal_features['urgency_score']
        assert urgent_features['num_urgency_words'] > 0
        assert normal_features['num_urgency_words'] == 0

    def test_suspicious_phrase_detection(self):
        """Test suspicious phrase detection"""
        suspicious_email = "Click here to verify your account"
        normal_email = "Your order has been processed"

        suspicious_features = self.preprocessor.extract_features(suspicious_email)
        normal_features = self.preprocessor.extract_features(normal_email)

        assert suspicious_features['suspicious_score'] > normal_features['suspicious_score']
        assert suspicious_features['num_suspicious_phrases'] > 0

    def test_credential_request_detection(self):
        """Test credential request detection"""
        credential_email = "Please provide your password and username"
        normal_email = "Thank you for contacting us"

        credential_features = self.preprocessor.extract_features(credential_email)
        normal_features = self.preprocessor.extract_features(normal_email)

        assert credential_features['credential_request_score'] > normal_features['credential_request_score']
        assert credential_features['num_credential_words'] > 0

    def test_shortened_url_detection(self):
        """Test shortened URL detection"""
        emails_with_short_urls = [
            "Visit http://bit.ly/test",
            "Check out https://tinyurl.com/abc123",
            "Click https://goo.gl/xyz789"
        ]

        normal_email = "Visit https://legitimate-company.com/page"

        for email in emails_with_short_urls:
            features = self.preprocessor.extract_features(email)
            assert features['has_shortened_url'] == True

        normal_features = self.preprocessor.extract_features(normal_email)
        assert normal_features['has_shortened_url'] == False

    def test_sender_domain_extraction(self):
        """Test sender domain extraction"""
        sender1 = "user@example.com"
        sender2 = "John Doe <john@company.org>"
        sender3 = "noreply@service.net"

        domain1 = self.preprocessor._extract_sender_domain(sender1)
        domain2 = self.preprocessor._extract_sender_domain(sender2)
        domain3 = self.preprocessor._extract_sender_domain(sender3)

        assert domain1 == "example.com"
        assert domain2 == "company.org"
        assert domain3 == "service.net"

    def test_suspicious_sender_detection(self):
        """Test suspicious sender detection"""
        suspicious_senders = [
            "noreply@example.com",
            "no-reply@test.org",
            "user12345@domain.com",
            "admin@192.168.1.1"
        ]

        legitimate_sender = "support@legitimate-company.com"

        for sender in suspicious_senders:
            is_suspicious = self.preprocessor._check_suspicious_sender(sender)
            assert is_suspicious == True

        is_legitimate = self.preprocessor._check_suspicious_sender(legitimate_sender)
        assert is_legitimate == False

    def test_suspicious_attachments(self):
        """Test suspicious attachment detection"""
        suspicious_attachments = [
            "document.exe",
            "file.scr",
            "script.vbs",
            "archive.zip"
        ]

        safe_attachments = [
            "document.pdf",
            "image.jpg",
            "data.csv"
        ]

        is_suspicious = self.preprocessor._check_suspicious_attachments(suspicious_attachments)
        assert is_suspicious == True

        is_safe = self.preprocessor._check_suspicious_attachments(safe_attachments)
        assert is_safe == False

    def test_enhanced_prompt_creation(self):
        """Test enhanced prompt creation with features"""
        email_text = "Test email content"
        features = {
            'urgency_score': 0.5,
            'suspicious_score': 0.3,
            'credential_request_score': 0.1,
            'num_urls': 2,
            'has_shortened_url': True,
            'has_attachments': False,
            'has_suspicious_sender': True,
            'spf_pass': False,
            'dkim_pass': True
        }

        prompt = self.preprocessor.create_enhanced_prompt(email_text, features)

        assert email_text in prompt
        assert "Urgency Score: 0.50" in prompt
        assert "Suspicious Content Score: 0.30" in prompt
        assert "Number of URLs: 2" in prompt
        assert "Has Shortened URLs: True" in prompt

    def test_parse_email_simple_text(self):
        """Test parsing simple email text"""
        email_text = "This is a simple email message"
        parsed = self.preprocessor.parse_email(email_text)

        assert 'body' in parsed
        assert parsed['body'] == email_text
        assert 'subject' in parsed
        assert 'sender' in parsed

    def test_grammar_score_calculation(self):
        """Test grammar score calculation"""
        good_grammar = "This is a well-written email. Thank you for your time."
        poor_grammar = "this  is bad grammar.no spaces here"

        good_score = self.preprocessor._calculate_grammar_score(good_grammar)
        poor_score = self.preprocessor._calculate_grammar_score(poor_grammar)

        assert good_score > poor_score
        assert 0 <= good_score <= 1
        assert 0 <= poor_score <= 1

if __name__ == "__main__":
    pytest.main([__file__])