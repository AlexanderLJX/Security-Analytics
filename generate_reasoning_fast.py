#!/usr/bin/env python3
"""
Fast async version of reasoning generation using GPT-5
"""
import sys
import os
import pandas as pd
import json
import time
import logging
import asyncio
# import aiofiles  # Not needed, using sync file operations
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
from tqdm.asyncio import tqdm
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FastReasoningConfig:
    """Fast reasoning configuration"""
    api_provider: str = "openai"
    api_key: str = os.getenv("OPENAI_API_KEY")
    teacher_model: str = "gpt-5"
    temperature: float = 1.0
    max_tokens: int = 1000
    top_p: float = 1.0
    use_corrective_feedback: bool = False  # Disabled for speed
    max_correction_attempts: int = 0
    include_adversarial: bool = False  # Disabled for speed
    batch_size: int = 50  # Increased batch size
    save_frequency: int = 10
    max_concurrent: int = 20  # Concurrent API calls
    rate_limit_delay: float = 0.05  # Reduced delay

class FastTeacherModel:
    """Fast async teacher model for reasoning generation"""

    def __init__(self, config):
        self.config = config
        self.client = openai.AsyncOpenAI(api_key=config.api_key)
        self.semaphore = asyncio.Semaphore(config.max_concurrent)

    async def generate_reasoning_batch(self, emails_and_labels: List[tuple]) -> List[Dict]:
        """Generate reasoning for a batch of emails concurrently"""
        tasks = []
        for email_text, true_label in emails_and_labels:
            task = self.generate_reasoning_single(email_text, true_label)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                email_text, true_label = emails_and_labels[i]
                logger.error(f"Failed to generate reasoning: {result}")
                processed_results.append({
                    'email': email_text,
                    'true_label': true_label,
                    'error': str(result),
                    'final_reasoning': None
                })
            else:
                processed_results.append(result)

        return processed_results

    async def generate_reasoning_single(self, email_text: str, true_label: str) -> Dict:
        """Generate security reasoning for a single email"""
        async with self.semaphore:
            try:
                # Generate simplified reasoning (no corrections/adversarial for speed)
                reasoning = await self._generate_initial_reasoning(email_text)
                prediction = self._extract_prediction(reasoning)

                result = {
                    'email': email_text,
                    'true_label': true_label,
                    'initial_reasoning': reasoning,
                    'initial_prediction': prediction,
                    'initial_correct': prediction.lower() == true_label.lower(),
                    'final_reasoning': reasoning
                }

                # Small delay for rate limiting
                await asyncio.sleep(self.config.rate_limit_delay)
                return result

            except Exception as e:
                logger.error(f"Error generating reasoning: {e}")
                return {
                    'email': email_text,
                    'true_label': true_label,
                    'error': str(e),
                    'final_reasoning': None
                }

    async def _generate_initial_reasoning(self, email_text: str) -> str:
        """Generate initial security analysis"""
        prompt = f"""You are a cybersecurity expert analyzing emails for phishing attempts.

Analyze this email step-by-step for security threats:

Email: {email_text}

Provide concise security analysis examining:
1. **Sender Authentication**: Domain legitimacy, spoofing indicators
2. **URL Analysis**: Suspicious links, redirects, homoglyphs
3. **Social Engineering**: Urgency, fear appeals, authority abuse
4. **Technical Indicators**: Grammar, formatting, patterns
5. **Content Analysis**: Info requests, attachments, crypto mentions

Classification: PHISHING or LEGITIMATE

Provide clear reasoning explaining your decision."""

        return await self._call_api_async(prompt)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def _call_api_async(self, prompt: str) -> str:
        """Call the OpenAI API asynchronously using GPT-5 responses format"""
        try:
            if self.config.teacher_model == "gpt-5":
                # Use GPT-5 responses API
                system_prompt = "You are a cybersecurity expert specializing in email security and phishing detection."

                response = await self.client.responses.create(
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
                        "effort": "medium"  # Reduced effort for speed
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
                response = await self.client.chat.completions.create(
                    model=self.config.teacher_model,
                    messages=[
                        {"role": "system", "content": "You are a cybersecurity expert specializing in email security and phishing detection."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.config.max_tokens,
                    temperature=0.3
                )
                content = response.choices[0].message.content

            if content is None or not content.strip():
                logger.warning("API returned empty content")
                return "No content returned from API"

            return content

        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise

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

def save_cache_sync(cache_path: Path, reasoning_cache: dict):
    """Save cache synchronously"""
    with open(cache_path, 'w') as f:
        json.dump(reasoning_cache, f, indent=2)

def load_cache_sync(cache_path: Path) -> dict:
    """Load cache synchronously"""
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            return json.load(f)
    return {}

async def main():
    """Generate reasoning for the sampled dataset with async processing"""
    config = FastReasoningConfig()

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
    teacher = FastTeacherModel(config)

    # Load existing cache
    cache_path = Path("./reasoning_cache_fast.json")
    reasoning_cache = load_cache_sync(cache_path)
    logger.info(f"Loaded {len(reasoning_cache)} cached reasoning entries")

    # Prepare emails for processing
    emails_to_process = []
    email_hashes = []

    for idx, row in df.iterrows():
        email_hash = str(hash(row['text']))
        email_hashes.append(email_hash)

        if email_hash not in reasoning_cache:
            label = 'legitimate' if row['type'] == 'not phishing' else row['type']
            emails_to_process.append((row['text'], label))

    logger.info(f"Processing {len(emails_to_process)} new emails (skipping {len(df) - len(emails_to_process)} cached)")

    if emails_to_process:
        # Process in batches
        logger.info("Starting fast reasoning generation with GPT-5...")

        batch_size = config.batch_size
        total_batches = (len(emails_to_process) + batch_size - 1) // batch_size

        processed_count = 0
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(emails_to_process))
            batch = emails_to_process[start_idx:end_idx]

            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch)} emails)")

            # Process batch concurrently
            batch_results = await teacher.generate_reasoning_batch(batch)

            # Update cache
            for i, result in enumerate(batch_results):
                email_text, _ = batch[i]
                email_hash = str(hash(email_text))
                reasoning_cache[email_hash] = result

            processed_count += len(batch)

            # Save periodically
            if (batch_idx + 1) % config.save_frequency == 0 or batch_idx == total_batches - 1:
                save_cache_sync(cache_path, reasoning_cache)
                logger.info(f"Saved cache after processing {processed_count} emails")

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
    output_path = "./combined-dataset-with-reasoning-fast.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Enhanced dataset saved to {output_path}")

    # Print statistics
    logger.info(f"Fast reasoning generation completed!")
    logger.info(f"Total emails: {len(df)}")
    logger.info(f"Emails with reasoning: {df['has_reasoning'].sum()}")
    logger.info(f"Success rate: {df['has_reasoning'].sum() / len(df) * 100:.1f}%")

if __name__ == "__main__":
    asyncio.run(main())