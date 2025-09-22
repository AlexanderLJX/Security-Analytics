"""
Dataset loading and management for phishing detection
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import hashlib
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class PhishingDatasetLoader:
    """Load and manage phishing email datasets"""

    def __init__(self, config):
        self.config = config
        self.dataset_path = Path(config.dataset_path)
        self.reasoning_cache = self._load_reasoning_cache()

    def _load_reasoning_cache(self) -> Dict:
        """Load cached reasoning data if exists"""
        cache_path = Path(self.config.reasoning_cache_path)
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                return json.load(f)
        return {}

    def save_reasoning_cache(self):
        """Save reasoning cache to disk"""
        cache_path = Path(self.config.reasoning_cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(self.reasoning_cache, f, indent=2)

    def load_base_dataset(self) -> pd.DataFrame:
        """Load the base phishing dataset"""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")

        # Load based on file extension
        if self.dataset_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(self.dataset_path)
        elif self.dataset_path.suffix.lower() == '.csv':
            df = pd.read_csv(self.dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {self.dataset_path.suffix}")

        # Validate required columns
        required_columns = ['text', 'type']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Dataset must contain columns: {required_columns}")

        # Clean and preprocess
        df = df.dropna(subset=['text'])
        df = df[df['text'].str.len() >= self.config.min_email_length]
        df = df[df['text'].str.len() <= self.config.max_email_length]

        # Standardize labels for your dataset format
        df['label'] = df['type'].map({
            'phishing': 1,
            'not phishing': 0,
            'legitimate': 0,
            'ham': 0,
            'spam': 1  # Treating spam as phishing for this task
        })

        # Handle any unmapped labels
        if df['label'].isna().any():
            logger.warning(f"Found {df['label'].isna().sum()} rows with unmapped labels")
            df = df.dropna(subset=['label'])

        logger.info(f"Loaded {len(df)} emails from {self.dataset_path}")
        logger.info(f"Phishing emails: {df['label'].sum()}, Legitimate: {len(df) - df['label'].sum()}")

        return df

    def create_email_hash(self, email_text: str) -> str:
        """Create unique hash for email text"""
        return hashlib.sha256(email_text.encode()).hexdigest()

    def add_reasoning_to_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add reasoning data to dataset"""
        reasoning_data = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Adding reasoning"):
            email_hash = self.create_email_hash(row['text'])

            if email_hash in self.reasoning_cache:
                reasoning = self.reasoning_cache[email_hash]
            else:
                reasoning = {
                    'initial_reasoning': None,
                    'corrected_reasoning': None,
                    'adversarial_perspective': None,
                    'final_reasoning': None,
                    'has_reasoning': False
                }

            reasoning_data.append(reasoning)

        # Add reasoning columns
        for key in ['initial_reasoning', 'corrected_reasoning',
                   'adversarial_perspective', 'final_reasoning', 'has_reasoning']:
            df[key] = [r[key] for r in reasoning_data]

        logger.info(f"Added reasoning to {df['has_reasoning'].sum()} emails")
        return df

    def create_conversation_format(self, df: pd.DataFrame) -> List[Dict]:
        """Convert dataset to conversation format for training"""
        conversations = []

        for idx, row in df.iterrows():
            # Create user prompt
            user_content = (
                "Analyze this email for phishing indicators and provide detailed "
                "security reasoning:\n\n"
                f"Email: {row['text']}"
            )

            # Create assistant response with reasoning
            label = "PHISHING" if row['label'] == 1 else "LEGITIMATE"

            if row.get('has_reasoning', False) and row.get('final_reasoning'):
                assistant_content = (
                    f"Classification: {label}\n\n"
                    f"Security Analysis:\n{row['final_reasoning']}"
                )
            else:
                # Fallback for emails without reasoning
                assistant_content = (
                    f"Classification: {label}\n\n"
                    "Security Analysis:\nEmail classified based on content analysis."
                )

            conversation = {
                'conversations': [
                    {'role': 'user', 'content': user_content},
                    {'role': 'assistant', 'content': assistant_content}
                ],
                'email_hash': self.create_email_hash(row['text']),
                'label': row['label'],
                'has_reasoning': row.get('has_reasoning', False)
            }

            conversations.append(conversation)

        return conversations

    def prepare_dataset_splits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train, validation, and test sets"""
        # Stratified split to maintain class balance
        X = df.drop('label', axis=1)
        y = df['label']

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config.test_split,
            stratify=y,
            random_state=self.config.seed
        )

        # Second split: train vs val
        val_size = self.config.val_split / (1 - self.config.test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            stratify=y_temp,
            random_state=self.config.seed
        )

        # Reconstruct DataFrames
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        logger.info(f"Dataset splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df

    def create_hf_dataset(self, train_df: pd.DataFrame,
                          val_df: Optional[pd.DataFrame] = None,
                          test_df: Optional[pd.DataFrame] = None) -> DatasetDict:
        """Create HuggingFace dataset from pandas DataFrames"""
        # Convert to conversation format
        train_conversations = self.create_conversation_format(train_df)

        datasets_dict = {
            'train': Dataset.from_list(train_conversations)
        }

        if val_df is not None:
            val_conversations = self.create_conversation_format(val_df)
            datasets_dict['validation'] = Dataset.from_list(val_conversations)

        if test_df is not None:
            test_conversations = self.create_conversation_format(test_df)
            datasets_dict['test'] = Dataset.from_list(test_conversations)

        return DatasetDict(datasets_dict)

    def load_additional_datasets(self) -> pd.DataFrame:
        """Load additional phishing datasets if available"""
        additional_dfs = []

        # Check for other common dataset formats
        dataset_paths = [
            Path("./data/enron_spam.csv"),
            Path("./data/nazario_phishing.csv"),
            Path("./data/iwspa_ap.csv")
        ]

        for path in dataset_paths:
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    logger.info(f"Loaded additional dataset from {path}")
                    additional_dfs.append(df)
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")

        if additional_dfs:
            return pd.concat(additional_dfs, ignore_index=True)

        return pd.DataFrame()