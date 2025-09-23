#!/usr/bin/env python3
"""
Generate reasoning using GPT-5 teacher model only (minimal imports)
"""
import sys
import os
import pandas as pd
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
from tqdm import tqdm
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SimpleReasoningConfig:
    """Simple reasoning configuration"""
    api_provider: str = "openai"
    api_key: str = os.getenv("OPENAI_API_KEY")
    teacher_model: str = "gpt-5"
    temperature: float = 1.0  # GPT-5 only supports default temperature
    max_tokens: int = 1000
    top_p: float = 1.0  # Use default for GPT-5
    use_corrective_feedback: bool = True
    max_correction_attempts: int = 2
    include_adversarial: bool = True
    batch_size: int = 10
    save_frequency: int = 5

class SimpleTeacherModel:
    """Simplified teacher model for reasoning generation"""

    def __init__(self, config):
        self.config = config
        self.client = openai.OpenAI(api_key=config.api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_reasoning(self, email_text: str, true_label: str) -> Dict:
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

    def _generate_corrected_reasoning(self, email_text: str, true_label: str, initial_reasoning: str) -> str:
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
        """Call the OpenAI API using GPT-5 responses format"""
        try:
            if self.config.teacher_model == "gpt-5":
                # Use GPT-5 responses API
                system_prompt = "You are a cybersecurity expert specializing in email security and phishing detection."

                response = self.client.responses.create(
                    model="gpt-5",
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    text={
                        "format": {
                            "type": "text"
                        },
                        "verbosity": "high"
                    },
                    reasoning={
                        "effort": "high"
                    },
                    tools=[],
                    store=True,
                    include=[
                        "reasoning.encrypted_content",
                        "web_search_call.action.sources"
                    ]
                )

                # Extract content from GPT-5 response format
                if hasattr(response, 'output_text') and response.output_text:
                    content = response.output_text
                else:
                    content = str(response)

            else:
                # Use standard chat completions for other models
                response = self.client.chat.completions.create(
                    model=self.config.teacher_model,
                    messages=[
                        {"role": "system", "content": "You are a cybersecurity expert specializing in email security and phishing detection."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.config.max_tokens,
                    temperature=0.3
                )
                content = response.choices[0].message.content

            logger.info(f"API response length: {len(content) if content else 0}")

            if content is None:
                logger.warning("API returned None content")
                return "No content returned from API"

            if not content.strip():
                logger.warning("API returned empty content")
                return "Empty response from API"

            return content

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

def main():
    """Generate reasoning for the sampled dataset"""
    config = SimpleReasoningConfig()

    if not config.api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        return

    # Load dataset
    dataset_path = "./combined-dataset-sample-1000x.parquet"
    if not Path(dataset_path).exists():
        logger.error(f"Dataset not found at {dataset_path}")
        return

    logger.info(f"Loading dataset from {dataset_path}")
    df = pd.read_parquet(dataset_path)
    logger.info(f"Loaded {len(df)} emails")

    # Initialize teacher model
    logger.info(f"Initializing teacher model: {config.teacher_model}")
    teacher = SimpleTeacherModel(config)

    # Generate reasoning
    logger.info("Starting reasoning generation with GPT-5...")
    reasoning_cache = {}
    cache_path = Path("./reasoning_cache_simple.json")

    # Load existing cache if it exists
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            reasoning_cache = json.load(f)
        logger.info(f"Loaded {len(reasoning_cache)} cached reasoning entries")

    # Process emails
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating reasoning"):
        email_hash = str(hash(row['text']))

        if email_hash not in reasoning_cache:
            try:
                # Convert 'not phishing' to 'legitimate' for consistency
                label = 'legitimate' if row['type'] == 'not phishing' else row['type']

                reasoning_result = teacher.generate_reasoning(row['text'], label)
                reasoning_cache[email_hash] = reasoning_result

                # Rate limiting
                time.sleep(0.5)

                # Save periodically
                if (idx + 1) % config.save_frequency == 0:
                    with open(cache_path, 'w') as f:
                        json.dump(reasoning_cache, f, indent=2)
                    logger.info(f"Saved reasoning cache after {idx + 1} emails")

            except Exception as e:
                logger.error(f"Failed to generate reasoning for email {idx}: {e}")
                reasoning_cache[email_hash] = {
                    'email': row['text'],
                    'true_label': label,
                    'error': str(e),
                    'initial_reasoning': None,
                    'corrected_reasoning': None,
                    'final_reasoning': None
                }

    # Final save
    with open(cache_path, 'w') as f:
        json.dump(reasoning_cache, f, indent=2)
    logger.info(f"Final save: {len(reasoning_cache)} reasoning entries saved")

    # Add reasoning to dataframe
    logger.info("Adding reasoning to dataframe...")
    reasoning_data = []
    for idx, row in df.iterrows():
        email_hash = str(hash(row['text']))
        if email_hash in reasoning_cache:
            reasoning = reasoning_cache[email_hash]
        else:
            reasoning = {'final_reasoning': None}
        reasoning_data.append(reasoning)

    df['reasoning'] = [r.get('final_reasoning') for r in reasoning_data]
    df['has_reasoning'] = df['reasoning'].notna()

    # Save enhanced dataset
    output_path = "./combined-dataset-with-reasoning.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Enhanced dataset saved to {output_path}")

    # Print statistics
    logger.info(f"Reasoning generation completed!")
    logger.info(f"Total emails: {len(df)}")
    logger.info(f"Emails with reasoning: {df['has_reasoning'].sum()}")
    logger.info(f"Success rate: {df['has_reasoning'].sum() / len(df) * 100:.1f}%")

if __name__ == "__main__":
    main()