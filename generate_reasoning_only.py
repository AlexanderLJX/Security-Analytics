#!/usr/bin/env python3
"""
Generate reasoning using GPT-5 teacher model only
"""
import pandas as pd
import json
from pathlib import Path
from config import data_config, reasoning_config
from models.teacher_model import TeacherModel
from data.dataset_loader import PhishingDatasetLoader
from training.reasoning_distillation import ReasoningDistillation
from utils.helpers import setup_logging

def main():
    """Generate reasoning for the sampled dataset"""
    logger = setup_logging()
    logger.info("Starting GPT-5 reasoning generation")

    # Load dataset
    logger.info("Loading dataset...")
    dataset_loader = PhishingDatasetLoader(data_config)

    if not Path(data_config.dataset_path).exists():
        logger.error(f"Dataset not found at {data_config.dataset_path}")
        return

    base_df = dataset_loader.load_base_dataset()
    logger.info(f"Loaded {len(base_df)} emails")

    # Initialize teacher model
    logger.info(f"Initializing teacher model: {reasoning_config.teacher_model}")
    teacher = TeacherModel(reasoning_config)

    # Initialize reasoning distillation
    distillation = ReasoningDistillation(teacher, dataset_loader, reasoning_config)

    # Generate reasoning
    logger.info("Starting reasoning generation with GPT-5...")
    df_with_reasoning = distillation.generate_reasoning_dataset(base_df)

    # Export results
    output_path = Path(data_config.dataset_path).parent / "reasoning_dataset"
    distillation.export_reasoning_dataset(df_with_reasoning, output_path)

    # Print statistics
    stats = distillation.analyze_reasoning_quality(df_with_reasoning)
    logger.info("Reasoning generation completed!")
    logger.info(f"Statistics: {json.dumps(stats, indent=2)}")

if __name__ == "__main__":
    main()