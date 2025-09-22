"""
Integration tests for the complete phishing detection pipeline
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Mock heavy dependencies for testing
torch_mock = Mock()
transformers_mock = Mock()
unsloth_mock = Mock()

with patch.dict('sys.modules', {
    'torch': torch_mock,
    'transformers': transformers_mock,
    'unsloth': unsloth_mock,
    'kafka': Mock(),
    'wandb': Mock(),
    'tensorboard': Mock(),
}):
    from data.dataset_loader import PhishingDatasetLoader
    from data.preprocessor import EmailPreprocessor
    from utils.helpers import create_sample_dataset, validate_dataset

@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests for the complete pipeline"""

    def test_dataset_creation_and_loading(self, tmp_path):
        """Test creating and loading a dataset"""
        dataset_path = tmp_path / "test_dataset.csv"

        # Create sample dataset
        df = create_sample_dataset(str(dataset_path), num_samples=100)

        # Validate dataset
        validation_result = validate_dataset(str(dataset_path))

        assert validation_result['valid'] == True
        assert validation_result['total_samples'] == 100
        assert len(validation_result['unique_labels']) == 2

        # Test loading with PhishingDatasetLoader
        class MockConfig:
            dataset_path = str(dataset_path)
            reasoning_cache_path = str(tmp_path / "cache.json")
            min_email_length = 10
            max_email_length = 2000
            train_split = 0.8
            val_split = 0.1
            test_split = 0.1
            seed = 42

        loader = PhishingDatasetLoader(MockConfig())
        loaded_df = loader.load_base_dataset()

        assert len(loaded_df) == 100
        assert 'label' in loaded_df.columns
        assert set(loaded_df['label'].unique()) == {0, 1}

    def test_preprocessing_pipeline(self, sample_phishing_email, sample_legitimate_email):
        """Test preprocessing pipeline on sample emails"""
        preprocessor = EmailPreprocessor()

        # Test phishing email
        phishing_features = preprocessor.extract_features(sample_phishing_email)

        # Should detect phishing indicators
        assert phishing_features['urgency_score'] > 0
        assert phishing_features['suspicious_score'] > 0
        assert phishing_features['credential_request_score'] > 0
        assert phishing_features['num_urls'] > 0

        # Test legitimate email
        legitimate_features = preprocessor.extract_features(sample_legitimate_email)

        # Should have lower risk scores
        assert legitimate_features['urgency_score'] < phishing_features['urgency_score']
        assert legitimate_features['suspicious_score'] < phishing_features['suspicious_score']
        assert legitimate_features['credential_request_score'] < phishing_features['credential_request_score']

    def test_dataset_splitting_consistency(self, tmp_path):
        """Test that dataset splitting is consistent and maintains balance"""
        dataset_path = tmp_path / "test_dataset.csv"
        create_sample_dataset(str(dataset_path), num_samples=1000)

        class MockConfig:
            dataset_path = str(dataset_path)
            reasoning_cache_path = str(tmp_path / "cache.json")
            min_email_length = 10
            max_email_length = 2000
            train_split = 0.8
            val_split = 0.1
            test_split = 0.1
            seed = 42

        loader = PhishingDatasetLoader(MockConfig())
        df = loader.load_base_dataset()

        # Split dataset multiple times with same seed
        train_df1, val_df1, test_df1 = loader.prepare_dataset_splits(df)
        train_df2, val_df2, test_df2 = loader.prepare_dataset_splits(df)

        # Should be identical
        assert len(train_df1) == len(train_df2)
        assert len(val_df1) == len(val_df2)
        assert len(test_df1) == len(test_df2)

        # Check class balance in splits
        total_phishing = df['label'].sum()
        total_legitimate = len(df) - total_phishing

        train_phishing_ratio = train_df1['label'].sum() / len(train_df1)
        val_phishing_ratio = val_df1['label'].sum() / len(val_df1)
        test_phishing_ratio = test_df1['label'].sum() / len(test_df1)

        # Ratios should be similar (within 10% tolerance)
        overall_ratio = total_phishing / len(df)
        assert abs(train_phishing_ratio - overall_ratio) < 0.1
        assert abs(val_phishing_ratio - overall_ratio) < 0.1
        assert abs(test_phishing_ratio - overall_ratio) < 0.1

    def test_conversation_format_creation(self, tmp_path):
        """Test conversation format creation for training"""
        dataset_path = tmp_path / "test_dataset.csv"
        create_sample_dataset(str(dataset_path), num_samples=50)

        class MockConfig:
            dataset_path = str(dataset_path)
            reasoning_cache_path = str(tmp_path / "cache.json")
            min_email_length = 10
            max_email_length = 2000
            train_split = 0.8
            val_split = 0.1
            test_split = 0.1
            seed = 42

        loader = PhishingDatasetLoader(MockConfig())
        df = loader.load_base_dataset()

        # Add mock reasoning data
        df['has_reasoning'] = True
        df['final_reasoning'] = "Mock security analysis reasoning"

        conversations = loader.create_conversation_format(df)

        assert len(conversations) == len(df)

        for conv in conversations:
            assert 'conversations' in conv
            assert len(conv['conversations']) == 2

            user_msg = conv['conversations'][0]
            assistant_msg = conv['conversations'][1]

            assert user_msg['role'] == 'user'
            assert 'Analyze this email' in user_msg['content']

            assert assistant_msg['role'] == 'assistant'
            assert 'Classification:' in assistant_msg['content']
            assert 'Security Analysis:' in assistant_msg['content']

    @pytest.mark.slow
    def test_end_to_end_small_dataset(self, tmp_path):
        """Test end-to-end pipeline with small dataset"""
        dataset_path = tmp_path / "small_dataset.csv"
        create_sample_dataset(str(dataset_path), num_samples=20)

        class MockConfig:
            dataset_path = str(dataset_path)
            reasoning_cache_path = str(tmp_path / "cache.json")
            min_email_length = 10
            max_email_length = 2000
            train_split = 0.7
            val_split = 0.15
            test_split = 0.15
            seed = 42

        # Load and preprocess data
        loader = PhishingDatasetLoader(MockConfig())
        df = loader.load_base_dataset()

        # Split data
        train_df, val_df, test_df = loader.prepare_dataset_splits(df)

        # Verify splits
        assert len(train_df) + len(val_df) + len(test_df) == len(df)
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0

        # Test preprocessing on each split
        preprocessor = EmailPreprocessor()

        for split_df in [train_df, val_df, test_df]:
            for idx, row in split_df.iterrows():
                features = preprocessor.extract_features(row['text'])
                assert isinstance(features, dict)
                assert 'length' in features
                assert 'urgency_score' in features
                assert features['length'] > 0

    def test_feature_extraction_consistency(self):
        """Test that feature extraction is consistent"""
        preprocessor = EmailPreprocessor()

        test_email = "URGENT: Click here http://bit.ly/test to verify account!"

        # Extract features multiple times
        features1 = preprocessor.extract_features(test_email)
        features2 = preprocessor.extract_features(test_email)

        # Should be identical
        assert features1 == features2

        # Test enhanced prompt creation
        prompt1 = preprocessor.create_enhanced_prompt(test_email, features1)
        prompt2 = preprocessor.create_enhanced_prompt(test_email, features2)

        assert prompt1 == prompt2
        assert test_email in prompt1
        assert "Security Features Detected" in prompt1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])