"""
Email preprocessing and feature extraction
"""
import re
import email
from email import policy
from email.parser import Parser
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class EmailPreprocessor:
    """Preprocess and extract features from emails"""

    def __init__(self):
        self.parser = Parser(policy=policy.default)

        # Phishing indicators
        self.urgency_words = [
            'urgent', 'immediate', 'expire', 'suspend', 'verify now',
            'act now', 'limited time', 'deadline', 'final notice'
        ]

        self.suspicious_phrases = [
            'click here', 'verify your account', 'suspended account',
            'confirm your identity', 'update payment', 'security alert',
            'unusual activity', 'prize', 'winner', 'congratulations'
        ]

        self.credential_requests = [
            'password', 'username', 'pin', 'ssn', 'social security',
            'credit card', 'bank account', 'login', 'credentials'
        ]

    def parse_email(self, email_text: str) -> Dict:
        """Parse email into structured components"""
        try:
            if email_text.startswith('From '):
                # Unix mbox format
                msg = self.parser.parsestr(email_text)
            else:
                # Assume it's just the body
                msg = email.message.EmailMessage()
                msg.set_content(email_text)

            return {
                'subject': msg.get('Subject', ''),
                'sender': msg.get('From', ''),
                'recipients': msg.get('To', ''),
                'date': msg.get('Date', ''),
                'body': self.extract_body(msg),
                'headers': dict(msg.items()),
                'attachments': self.check_attachments(msg)
            }
        except Exception as e:
            logger.warning(f"Failed to parse email: {e}")
            return {
                'subject': '',
                'sender': '',
                'recipients': '',
                'date': '',
                'body': email_text,
                'headers': {},
                'attachments': []
            }

    def extract_body(self, msg) -> str:
        """Extract email body from message object"""
        body = ""

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    try:
                        body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    except:
                        body += str(part.get_payload())
                elif content_type == "text/html":
                    try:
                        html_content = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        body += BeautifulSoup(html_content, 'html.parser').get_text()
                    except:
                        body += str(part.get_payload())
        else:
            try:
                body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            except:
                body = str(msg.get_payload())

        return body.strip()

    def check_attachments(self, msg) -> List[str]:
        """Check for attachments in email"""
        attachments = []

        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_disposition() == 'attachment':
                    filename = part.get_filename()
                    if filename:
                        attachments.append(filename)

        return attachments

    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        return urls

    def extract_features(self, email_text: str) -> Dict:
        """Extract security-relevant features from email"""
        # Parse email
        parsed = self.parse_email(email_text)
        body = parsed['body'].lower()

        features = {
            # Basic features
            'length': len(body),
            'num_words': len(body.split()),
            'num_sentences': len(re.split(r'[.!?]', body)),

            # URL features
            'urls': self.extract_urls(body),
            'num_urls': len(self.extract_urls(body)),
            'has_shortened_url': self._has_shortened_url(body),

            # Urgency and psychological triggers
            'urgency_score': self._calculate_urgency_score(body),
            'num_urgency_words': sum(1 for word in self.urgency_words if word in body),

            # Suspicious content
            'suspicious_score': self._calculate_suspicious_score(body),
            'num_suspicious_phrases': sum(1 for phrase in self.suspicious_phrases if phrase in body),

            # Credential requests
            'credential_request_score': self._calculate_credential_score(body),
            'num_credential_words': sum(1 for word in self.credential_requests if word in body),

            # Grammar and spelling
            'has_spelling_errors': self._check_spelling_errors(body),
            'grammar_score': self._calculate_grammar_score(body),

            # Sender features
            'sender_domain': self._extract_sender_domain(parsed['sender']),
            'has_suspicious_sender': self._check_suspicious_sender(parsed['sender']),

            # Attachment features
            'has_attachments': len(parsed['attachments']) > 0,
            'num_attachments': len(parsed['attachments']),
            'suspicious_attachments': self._check_suspicious_attachments(parsed['attachments']),

            # Headers
            'spf_pass': self._check_spf(parsed['headers']),
            'dkim_pass': self._check_dkim(parsed['headers'])
        }

        return features

    def _has_shortened_url(self, text: str) -> bool:
        """Check for shortened URLs"""
        shorteners = ['bit.ly', 'tinyurl.com', 'goo.gl', 'ow.ly', 't.co', 'short.link']
        return any(shortener in text for shortener in shorteners)

    def _calculate_urgency_score(self, text: str) -> float:
        """Calculate urgency score based on presence of urgency words"""
        score = sum(text.count(word) for word in self.urgency_words)
        return min(score / 10.0, 1.0)  # Normalize to 0-1

    def _calculate_suspicious_score(self, text: str) -> float:
        """Calculate suspicious content score"""
        score = sum(text.count(phrase) for phrase in self.suspicious_phrases)
        return min(score / 10.0, 1.0)

    def _calculate_credential_score(self, text: str) -> float:
        """Calculate credential request score"""
        score = sum(text.count(word) for word in self.credential_requests)
        return min(score / 5.0, 1.0)

    def _check_spelling_errors(self, text: str) -> bool:
        """Simple check for common spelling errors"""
        # Simplified - in production, use a proper spell checker
        common_errors = ['recieve', 'occured', 'seperate', 'untill', 'occassion']
        return any(error in text for error in common_errors)

    def _calculate_grammar_score(self, text: str) -> float:
        """Calculate grammar score (simplified)"""
        # Check for basic grammar issues
        issues = 0

        # Multiple spaces
        if '  ' in text:
            issues += 1

        # Missing spaces after punctuation
        if re.search(r'[.!?][a-z]', text):
            issues += 1

        # Improper capitalization
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences:
            if sentence.strip() and not sentence.strip()[0].isupper():
                issues += 1

        return max(0, 1 - (issues / 10.0))

    def _extract_sender_domain(self, sender: str) -> str:
        """Extract domain from sender email"""
        match = re.search(r'@([a-zA-Z0-9.-]+)', sender)
        return match.group(1) if match else ''

    def _check_suspicious_sender(self, sender: str) -> bool:
        """Check if sender appears suspicious"""
        suspicious_patterns = [
            r'noreply',
            r'no-reply',
            r'donotreply',
            r'[0-9]{5,}',  # Many numbers
            r'@[0-9]+\.',  # IP address
        ]

        sender_lower = sender.lower()
        return any(re.search(pattern, sender_lower) for pattern in suspicious_patterns)

    def _check_suspicious_attachments(self, attachments: List[str]) -> bool:
        """Check for suspicious attachment types"""
        suspicious_extensions = [
            '.exe', '.scr', '.vbs', '.js', '.zip', '.rar',
            '.bat', '.cmd', '.com', '.pif', '.lnk'
        ]

        for attachment in attachments:
            if any(attachment.lower().endswith(ext) for ext in suspicious_extensions):
                return True
        return False

    def _check_spf(self, headers: Dict) -> bool:
        """Check SPF authentication (simplified)"""
        auth_results = headers.get('Authentication-Results', '')
        return 'spf=pass' in auth_results.lower()

    def _check_dkim(self, headers: Dict) -> bool:
        """Check DKIM authentication (simplified)"""
        auth_results = headers.get('Authentication-Results', '')
        return 'dkim=pass' in auth_results.lower()

    def create_enhanced_prompt(self, email_text: str, features: Dict) -> str:
        """Create enhanced prompt with extracted features for LLM"""
        prompt = f"""Analyze this email for phishing indicators:

Email Content:
{email_text}

Security Features Detected:
- Urgency Score: {features['urgency_score']:.2f}
- Suspicious Content Score: {features['suspicious_score']:.2f}
- Credential Request Score: {features['credential_request_score']:.2f}
- Number of URLs: {features['num_urls']}
- Has Shortened URLs: {features['has_shortened_url']}
- Has Attachments: {features['has_attachments']}
- Suspicious Sender: {features['has_suspicious_sender']}
- SPF Pass: {features['spf_pass']}
- DKIM Pass: {features['dkim_pass']}

Provide detailed security analysis and classification."""

        return prompt