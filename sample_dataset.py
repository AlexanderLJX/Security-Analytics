#!/usr/bin/env python3
"""
Sample the dataset to 1/100th of its size to reduce GPT-5 API costs
"""
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def sample_dataset(input_path: str, output_path: str, sample_fraction: float = 0.01, random_state: int = 42):
    """
    Sample a fraction of the dataset while maintaining label distribution

    Args:
        input_path: Path to input parquet file
        output_path: Path to save sampled dataset
        sample_fraction: Fraction of data to sample (default 0.01 = 1%)
        random_state: Random seed for reproducible sampling
    """
    print(f"Loading dataset from {input_path}")

    # Load the dataset
    df = pd.read_parquet(input_path)

    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Check if we have a label column
    if 'type' in df.columns:
        label_col = 'type'
    elif 'label' in df.columns:
        label_col = 'label'
    else:
        # If no clear label column, just do random sampling
        print("No clear label column found, doing random sampling")
        sampled_df = df.sample(frac=sample_fraction, random_state=random_state)
        sampled_df.to_parquet(output_path, index=False)
        print(f"Sampled dataset saved to {output_path}")
        print(f"New dataset shape: {sampled_df.shape}")
        return sampled_df

    # Check label distribution
    print(f"\nOriginal label distribution:")
    print(df[label_col].value_counts())

    # Stratified sampling to maintain label distribution
    sampled_df = df.groupby(label_col, group_keys=False).apply(
        lambda x: x.sample(frac=sample_fraction, random_state=random_state)
    ).reset_index(drop=True)

    print(f"\nSampled dataset shape: {sampled_df.shape}")
    print(f"Sampled label distribution:")
    print(sampled_df[label_col].value_counts())

    # Calculate reduction
    reduction_factor = len(df) / len(sampled_df)
    print(f"\nDataset reduced by factor of {reduction_factor:.1f}x")
    print(f"This will reduce API costs by approximately {reduction_factor:.1f}x")

    # Save the sampled dataset
    sampled_df.to_parquet(output_path, index=False)
    print(f"\nSampled dataset saved to {output_path}")

    return sampled_df

def main():
    parser = argparse.ArgumentParser(description="Sample dataset to reduce API costs")
    parser.add_argument("--input", default="combined-datasetwithoutemptys.parquet",
                       help="Input parquet file path")
    parser.add_argument("--output", default="combined-dataset-sample.parquet",
                       help="Output parquet file path")
    parser.add_argument("--fraction", type=float, default=0.01,
                       help="Fraction to sample (default: 0.01 = 1%)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible sampling")

    args = parser.parse_args()

    # Ensure input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} not found")
        return

    # Create sample
    sample_dataset(args.input, args.output, args.fraction, args.seed)

if __name__ == "__main__":
    main()