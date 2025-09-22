#!/usr/bin/env python3
"""
Analyze the parquet dataset and show statistics
"""
import sys
sys.path.append('.')

from data.dataset_loader import PhishingDatasetLoader
from data.preprocessor import EmailPreprocessor
from config import data_config
import pandas as pd

def main():
    print("Analyzing your dataset...")

    # Load dataset
    loader = PhishingDatasetLoader(data_config)
    df = loader.load_base_dataset()

    print(f"\nDataset Statistics:")
    print(f"Total emails: {len(df):,}")
    print(f"Phishing emails: {df['label'].sum():,} ({df['label'].mean()*100:.1f}%)")
    print(f"Legitimate emails: {(df['label']==0).sum():,} ({(1-df['label'].mean())*100:.1f}%)")

    # Show original columns
    print(f"\nAvailable columns: {list(df.columns)}")

    # Text length statistics
    text_lengths = df['text'].str.len()
    print(f"\nText Length Statistics:")
    print(f"Min length: {text_lengths.min():,} characters")
    print(f"Max length: {text_lengths.max():,} characters")
    print(f"Mean length: {text_lengths.mean():.0f} characters")
    print(f"Median length: {text_lengths.median():.0f} characters")

    # Show data splits
    train_df, val_df, test_df = loader.prepare_dataset_splits(df)
    print(f"\nData Splits:")
    print(f"Training: {len(train_df):,} emails")
    print(f"Validation: {len(val_df):,} emails")
    print(f"Test: {len(test_df):,} emails")

    # Sample feature extraction
    print(f"\nTesting feature extraction...")
    preprocessor = EmailPreprocessor()

    sample_phishing = df[df['label']==1]['text'].iloc[0]
    sample_legitimate = df[df['label']==0]['text'].iloc[0]

    phishing_features = preprocessor.extract_features(sample_phishing)
    legitimate_features = preprocessor.extract_features(sample_legitimate)

    print(f"\nPhishing email features:")
    print(f"  Urgency score: {phishing_features['urgency_score']:.3f}")
    print(f"  Suspicious score: {phishing_features['suspicious_score']:.3f}")
    print(f"  URLs found: {phishing_features['num_urls']}")
    print(f"  Has shortened URLs: {phishing_features['has_shortened_url']}")

    print(f"\nLegitimate email features:")
    print(f"  Urgency score: {legitimate_features['urgency_score']:.3f}")
    print(f"  Suspicious score: {legitimate_features['suspicious_score']:.3f}")
    print(f"  URLs found: {legitimate_features['num_urls']}")
    print(f"  Has shortened URLs: {legitimate_features['has_shortened_url']}")

    # Show domain distribution if available
    if 'domain' in df.columns:
        print(f"\nTop 10 domains:")
        domain_counts = df['domain'].value_counts().head(10)
        for domain, count in domain_counts.items():
            print(f"  {domain}: {count:,}")

    # Show source distribution if available
    if 'source' in df.columns:
        print(f"\nSource distribution:")
        source_counts = df['source'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count:,}")

    print(f"\nDataset analysis complete!")
    print(f"\nYour dataset is ready for training. You can now run:")
    print(f"  python main.py train --generate-reasoning")

if __name__ == "__main__":
    main()