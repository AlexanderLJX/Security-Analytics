"""
Reasoning distillation pipeline
"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class ReasoningDistillation:
    """Handle reasoning generation and distillation process"""

    def __init__(self, teacher_model, dataset_loader, config):
        self.teacher = teacher_model
        self.dataset_loader = dataset_loader
        self.config = config
        self.reasoning_cache = {}
        self._load_cache()

    def _load_cache(self):
        """Load existing reasoning cache"""
        cache_path = Path(self.config.reasoning_cache_path)
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                self.reasoning_cache = json.load(f)
            logger.info(f"Loaded {len(self.reasoning_cache)} cached reasoning entries")

    def _save_cache(self):
        """Save reasoning cache to disk"""
        cache_path = Path(self.config.reasoning_cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(self.reasoning_cache, f, indent=2)
        logger.info(f"Saved {len(self.reasoning_cache)} reasoning entries to cache")

    def generate_reasoning_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate reasoning for entire dataset"""
        logger.info(f"Generating reasoning for {len(df)} emails")

        # Filter emails that need reasoning
        emails_to_process = []
        for idx, row in df.iterrows():
            email_hash = self.dataset_loader.create_email_hash(row['text'])
            if email_hash not in self.reasoning_cache:
                emails_to_process.append({
                    'index': idx,
                    'hash': email_hash,
                    'text': row['text'],
                    'label': 'phishing' if row['label'] == 1 else 'legitimate'
                })

        logger.info(f"Need to generate reasoning for {len(emails_to_process)} emails")

        if emails_to_process:
            # Process in batches
            batch_size = self.config.batch_size
            for i in tqdm(range(0, len(emails_to_process), batch_size), desc="Generating reasoning"):
                batch = emails_to_process[i:i+batch_size]
                self._process_batch(batch)

                # Save periodically
                if (i // batch_size) % self.config.save_frequency == 0:
                    self._save_cache()

        # Final save
        self._save_cache()

        # Add reasoning to dataframe
        return self._add_reasoning_to_dataframe(df)

    def _process_batch(self, batch: List[Dict]):
        """Process a batch of emails for reasoning generation"""
        for email_data in batch:
            try:
                # Generate reasoning
                reasoning_result = self.teacher.generate_reasoning(
                    email_data['text'],
                    email_data['label']
                )

                # Store in cache
                self.reasoning_cache[email_data['hash']] = reasoning_result

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Failed to generate reasoning for email {email_data['index']}: {e}")

                # Store error result
                self.reasoning_cache[email_data['hash']] = {
                    'email': email_data['text'],
                    'true_label': email_data['label'],
                    'error': str(e),
                    'initial_reasoning': None,
                    'corrected_reasoning': None,
                    'final_reasoning': None
                }

    def _add_reasoning_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cached reasoning to dataframe"""
        reasoning_columns = []

        for idx, row in df.iterrows():
            email_hash = self.dataset_loader.create_email_hash(row['text'])

            if email_hash in self.reasoning_cache:
                reasoning = self.reasoning_cache[email_hash]
            else:
                reasoning = {
                    'initial_reasoning': None,
                    'corrected_reasoning': None,
                    'adversarial_perspective': None,
                    'final_reasoning': None,
                    'initial_correct': None
                }

            reasoning_columns.append(reasoning)

        # Add columns to dataframe
        df['initial_reasoning'] = [r.get('initial_reasoning') for r in reasoning_columns]
        df['corrected_reasoning'] = [r.get('corrected_reasoning') for r in reasoning_columns]
        df['adversarial_perspective'] = [r.get('adversarial_perspective') for r in reasoning_columns]
        df['final_reasoning'] = [r.get('final_reasoning') for r in reasoning_columns]
        df['initial_correct'] = [r.get('initial_correct') for r in reasoning_columns]
        df['has_reasoning'] = df['final_reasoning'].notna()

        logger.info(f"Added reasoning to {df['has_reasoning'].sum()}/{len(df)} emails")

        return df

    def analyze_reasoning_quality(self, df: pd.DataFrame) -> Dict:
        """Analyze quality of generated reasoning"""
        stats = {
            'total_emails': len(df),
            'emails_with_reasoning': df['has_reasoning'].sum(),
            'initial_accuracy': df['initial_correct'].sum() / df['initial_correct'].notna().sum() if df['initial_correct'].notna().sum() > 0 else 0,
            'corrected_count': df['corrected_reasoning'].notna().sum(),
            'adversarial_count': df['adversarial_perspective'].notna().sum(),
            'average_reasoning_length': df[df['has_reasoning']]['final_reasoning'].str.len().mean() if df['has_reasoning'].sum() > 0 else 0
        }

        # Analyze by label
        for label in [0, 1]:
            label_name = 'legitimate' if label == 0 else 'phishing'
            label_df = df[df['label'] == label]

            stats[f'{label_name}_count'] = len(label_df)
            stats[f'{label_name}_with_reasoning'] = label_df['has_reasoning'].sum()
            stats[f'{label_name}_initial_accuracy'] = (
                label_df['initial_correct'].sum() / label_df['initial_correct'].notna().sum()
                if label_df['initial_correct'].notna().sum() > 0 else 0
            )

        return stats

    def export_reasoning_dataset(self, df: pd.DataFrame, output_path: str):
        """Export the reasoning dataset to various formats"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as CSV
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved reasoning dataset to {csv_path}")

        # Save as JSON for better preservation of reasoning text
        json_path = output_path.with_suffix('.json')
        df.to_json(json_path, orient='records', indent=2)
        logger.info(f"Saved reasoning dataset to {json_path}")

        # Save statistics
        stats = self.analyze_reasoning_quality(df)
        stats_path = output_path.parent / f"{output_path.stem}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved statistics to {stats_path}")