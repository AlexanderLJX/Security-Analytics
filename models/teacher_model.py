"""
Teacher model for reasoning generation (GPT-4/Claude)
"""
import openai
import anthropic
import json
import time
from typing import Dict, List, Optional
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class TeacherModel:
    """Teacher model for generating reasoning chains"""

    def __init__(self, config):
        self.config = config
        self.api_provider = config.api_provider

        if self.api_provider == "openai":
            openai.api_key = config.api_key
            self.client = openai.OpenAI(api_key=config.api_key)
        elif self.api_provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=config.api_key)
        else:
            raise ValueError(f"Unsupported API provider: {self.api_provider}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_reasoning(self,
                          email_text: str,
                          true_label: str) -> Dict:
        """Generate security reasoning for an email"""

        # Generate initial reasoning
        initial_reasoning = self._generate_initial_reasoning(email_text)
        initial_prediction = self._extract_prediction(initial_reasoning)

        result = {
            'email': email_text,
            'true_label': true_label,
            'initial_reasoning': initial_reasoning,
            'initial_prediction': initial_prediction,
            'initial_correct': initial_prediction.lower() == true_label.lower(),
            'corrected_reasoning': None,
            'adversarial_perspective': None,
            'final_reasoning': initial_reasoning
        }

        # Generate corrective feedback if wrong
        if not result['initial_correct'] and self.config.use_corrective_feedback:
            logger.info("Initial prediction incorrect, generating corrective feedback")
            result['corrected_reasoning'] = self._generate_corrected_reasoning(
                email_text, true_label, initial_reasoning
            )
            result['final_reasoning'] = result['corrected_reasoning']

        # Generate adversarial perspective if enabled
        if self.config.include_adversarial:
            result['adversarial_perspective'] = self._generate_adversarial_perspective(
                email_text, true_label
            )
            if result['adversarial_perspective']:
                result['final_reasoning'] += f"\n\n{result['adversarial_perspective']}"

        return result

    def _generate_initial_reasoning(self, email_text: str) -> str:
        """Generate initial security analysis"""
        prompt = f"""You are a cybersecurity expert analyzing emails for phishing attempts.

Analyze this email step-by-step:

Email: {email_text}

Provide detailed security analysis examining:
1. **Sender Authentication**: Check sender legitimacy, domain spoofing, email headers
2. **URL Analysis**: Identify suspicious links, URL shorteners, homoglyphs, redirects
3. **Social Engineering Tactics**: Urgency, fear appeals, authority abuse, emotional manipulation
4. **Technical Indicators**: Grammar errors, generic greetings, formatting issues, unusual patterns
5. **Content Analysis**: Requests for sensitive information, unusual attachments, cryptocurrency mentions
6. **Behavioral Patterns**: Deviation from normal communication, time-based attacks

Think through this step-by-step:
- First, identify any immediate red flags
- Then analyze the sender and authentication
- Check all URLs and links carefully
- Evaluate the psychological tactics used
- Consider the overall context and intent

Based on your analysis, classify as: PHISHING or LEGITIMATE

Provide your reasoning in a clear, educational format that explains the security principles."""

        return self._call_api(prompt)

    def _generate_corrected_reasoning(self,
                                     email_text: str,
                                     true_label: str,
                                     initial_reasoning: str) -> str:
        """Generate corrected reasoning when initial was wrong"""
        prompt = f"""Your initial analysis was incorrect. This email is actually {true_label.upper()}.

Original Email:
{email_text}

Your Initial Analysis:
{initial_reasoning}

Generate a CORRECTED security analysis that:
1. Identifies what security indicators you missed or misinterpreted
2. Explains why certain features are deceptive or were overlooked
3. Provides the correct reasoning with specific evidence from the email
4. Includes important lessons for future detection
5. Highlights the sophisticated techniques used (if phishing) or legitimate patterns (if safe)

Focus on security-specific insights that would help detect similar cases.
Be thorough and educational in your explanation."""

        return self._call_api(prompt)

    def _generate_adversarial_perspective(self, email_text: str, true_label: str) -> str:
        """Generate adversarial analysis for robustness"""
        prompt = f"""Provide adversarial security analysis for this {true_label} email:

Email: {email_text}

"""

        if true_label.lower() == "phishing":
            prompt += """As this is a PHISHING email:
1. Explain the attacker's strategy and social engineering techniques
2. Identify how they tried to evade detection systems
3. Analyze what makes this phishing attempt effective or ineffective
4. Suggest how attackers might improve this to be more convincing
5. Identify detection-resistant features used"""
        else:
            prompt += """As this is a LEGITIMATE email:
1. Explain what features might cause false positive detection
2. Identify patterns that could be mistaken for phishing
3. Analyze why automated systems might flag this incorrectly
4. Suggest how to avoid false positives for similar legitimate emails
5. Highlight distinguishing features that confirm legitimacy"""

        prompt += "\n\nProvide insights that improve detection robustness."

        return self._call_api(prompt)

    def _call_api(self, prompt: str) -> str:
        """Call the appropriate API"""
        try:
            if self.api_provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.config.teacher_model,
                    messages=[
                        {"role": "system", "content": "You are a cybersecurity expert specializing in email security and phishing detection."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    top_p=self.config.top_p
                )
                return response.choices[0].message.content

            elif self.api_provider == "anthropic":
                response = self.client.messages.create(
                    model=self.config.teacher_model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
                return response.content[0].text

        except Exception as e:
            logger.error(f"API call failed: {e}")
            return "Error generating reasoning"

    def _extract_prediction(self, reasoning: str) -> str:
        """Extract prediction from reasoning text"""
        reasoning_lower = reasoning.lower()

        # Look for explicit classification
        if "classification:" in reasoning_lower:
            after_class = reasoning_lower.split("classification:")[1][:50]
            if "phishing" in after_class:
                return "phishing"
            elif "legitimate" in after_class:
                return "legitimate"

        # Count occurrences in first part of response
        first_part = reasoning_lower[:500]
        phishing_count = first_part.count("phishing")
        legitimate_count = first_part.count("legitimate")

        if phishing_count > legitimate_count:
            return "phishing"
        elif legitimate_count > phishing_count:
            return "legitimate"

        # Fallback to full text
        full_phishing = reasoning_lower.count("phishing")
        full_legitimate = reasoning_lower.count("legitimate")

        return "phishing" if full_phishing > full_legitimate else "legitimate"

    def generate_batch_reasoning(self, emails: List[Dict]) -> List[Dict]:
        """Generate reasoning for a batch of emails"""
        results = []

        for email_data in emails:
            try:
                result = self.generate_reasoning(
                    email_data['text'],
                    email_data['label']
                )
                results.append(result)

                # Rate limiting
                time.sleep(1)

            except Exception as e:
                logger.error(f"Failed to generate reasoning: {e}")
                results.append({
                    'email': email_data['text'],
                    'true_label': email_data['label'],
                    'error': str(e)
                })

        return results