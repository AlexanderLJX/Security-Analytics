"""
Email preprocessing - simplified for LLM-based analysis
"""
import email
from email import policy
from email.parser import Parser
from bs4 import BeautifulSoup
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class EmailPreprocessor:
    """Basic email preprocessing for LLM analysis"""

    def __init__(self):
        self.parser = Parser(policy=policy.default)

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

    def clean_email_text(self, email_text: str) -> str:
        """Basic email text cleanup for LLM processing"""
        # Parse the email to extract just the body if it's in full format
        parsed = self.parse_email(email_text)

        # If we have a body, use that, otherwise use the original text
        cleaned_text = parsed['body'] if parsed['body'] else email_text

        # Basic cleanup - remove excessive whitespace
        cleaned_text = ' '.join(cleaned_text.split())

        return cleaned_text