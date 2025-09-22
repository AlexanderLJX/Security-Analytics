"""
Test cases for dataset loader functionality
"""
import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path

from data.dataset_loader import PhishingDatasetLoader
from config import DataConfig

class TestDataConfig:
    """Mock config for testing"""
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.reasoning_cache_path = "./test_cache.json"
        self.min_email_length = 10
        self.max_email_length = 2000
        self.train_split = 0.8
        self.val_split = 0.1
        self.test_split = 0.1
        self.seed = 42

class TestPhishingDatasetLoader:
    """Test suite for PhishingDatasetLoader"""

    def setup_method(self):
        """Setup test data"""
        self.test_data = pd.DataFrame({
            'text': [
                'This is a phishing email click here',
                'This is a legitimate email thank you',
                'Urgent action required verify account',
                'Your order has been shipped successfully',
                'Winner of prize claim now',
                'Meeting scheduled for tomorrow at 2pm'
            ],
            'type': ['phishing', 'legitimate', 'phishing', 'legitimate', 'phishing', 'legitimate']
        })

    def test_load_base_dataset(self):
        """Test loading base dataset"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            config = TestDataConfig(f.name)
            loader = PhishingDatasetLoader(config)

            df = loader.load_base_dataset()

            assert len(df) == 6
            assert 'label' in df.columns
            assert df['label'].sum() == 3  # 3 phishing emails
            assert (df['label'] == 0).sum() == 3  # 3 legitimate emails

        os.unlink(f.name)

    def test_create_email_hash(self):
        """Test email hash creation"""
        config = TestDataConfig("dummy.csv")
        loader = PhishingDatasetLoader(config)

        email1 = "This is a test email"
        email2 = "This is a different email"

        hash1 = loader.create_email_hash(email1)
        hash2 = loader.create_email_hash(email2)

        assert hash1 != hash2
        assert len(hash1) == 64  # SHA256 produces 64 character hex string
        assert loader.create_email_hash(email1) == hash1  # Consistent hashing

    def test_prepare_dataset_splits(self):
        """Test dataset splitting"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            config = TestDataConfig(f.name)
            loader = PhishingDatasetLoader(config)

            df = loader.load_base_dataset()
            train_df, val_df, test_df = loader.prepare_dataset_splits(df)

            # Check split sizes
            total_size = len(train_df) + len(val_df) + len(test_df)
            assert total_size == len(df)

            # Check no overlap
            train_indices = set(train_df.index)
            val_indices = set(val_df.index)
            test_indices = set(test_df.index)

            assert len(train_indices & val_indices) == 0
            assert len(train_indices & test_indices) == 0
            assert len(val_indices & test_indices) == 0

        os.unlink(f.name)

    def test_create_conversation_format(self):
        """Test conversation format creation"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            config = TestDataConfig(f.name)
            loader = PhishingDatasetLoader(config)

            df = loader.load_base_dataset()
            conversations = loader.create_conversation_format(df)

            assert len(conversations) == len(df)

            # Check conversation structure
            for conv in conversations:
                assert 'conversations' in conv
                assert len(conv['conversations']) == 2
                assert conv['conversations'][0]['role'] == 'user'
                assert conv['conversations'][1]['role'] == 'assistant'
                assert 'email_hash' in conv
                assert 'label' in conv

        os.unlink(f.name)

    def test_invalid_dataset_format(self):
        """Test handling of invalid dataset format"""
        invalid_data = pd.DataFrame({
            'content': ['email 1', 'email 2'],  # Wrong column name
            'label': ['spam', 'ham']  # Wrong column name
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            invalid_data.to_csv(f.name, index=False)
            config = TestDataConfig(f.name)
            loader = PhishingDatasetLoader(config)

            with pytest.raises(ValueError):
                loader.load_base_dataset()

        os.unlink(f.name)

    def test_missing_dataset_file(self):
        """Test handling of missing dataset file"""
        config = TestDataConfig("nonexistent_file.csv")
        loader = PhishingDatasetLoader(config)

        with pytest.raises(FileNotFoundError):
            loader.load_base_dataset()

if __name__ == "__main__":
    pytest.main([__file__])