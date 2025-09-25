"""
Main entry point for phishing detection system
"""
import argparse
import logging
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

# Import configurations
from config import (
    model_config, training_config, data_config,
    reasoning_config, inference_config, siem_config
)

# Import modules
from data.dataset_loader import PhishingDatasetLoader
# from data.preprocessor import EmailPreprocessor  # No longer needed
from models.qwen_model import QwenPhishingModel
from models.teacher_model import TeacherModel
from training.trainer import PhishingTrainer
from training.reasoning_distillation import ReasoningDistillation
from evaluation.metrics import PhishingMetrics
from inference.detector import PhishingDetector
from inference.siem_integration import SIEMIntegration, PhishingAPI
from utils.helpers import setup_logging, create_sample_dataset, check_dependencies

def train_model(args):
    """Train the phishing detection model"""
    logger = setup_logging(args.log_file)
    logger.info("Starting training pipeline")

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Initialize model and tokenizer first, as tokenizer is needed for data prep
    logger.info("Initializing model...")
    qwen_model = QwenPhishingModel(model_config)
    qwen_model.load_model()
    qwen_model.setup_lora()
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset_loader = PhishingDatasetLoader(data_config)

    # Check if dataset exists, create sample if not
    if not Path(data_config.dataset_path).exists():
        logger.warning(f"Dataset not found at {data_config.dataset_path}")
        logger.info("Creating sample dataset...")
        # Create CSV sample if parquet dataset is missing
        sample_path = data_config.dataset_path.replace('.parquet', '.csv')
        create_sample_dataset(sample_path, num_samples=2000)
        logger.info(f"Sample dataset created at {sample_path}. Please update config to use this file or provide the correct dataset.")

    base_df = dataset_loader.load_base_dataset()

    # Generate reasoning if needed
    if args.generate_reasoning:
        logger.info("Generating reasoning dataset...")
        teacher = TeacherModel(reasoning_config)
        distillation = ReasoningDistillation(teacher, dataset_loader, reasoning_config)
        base_df = distillation.generate_reasoning_dataset(base_df)

        distillation.export_reasoning_dataset(
            base_df,
            Path(data_config.dataset_path).parent / "reasoning_dataset"
        )
    else:
        # Load existing reasoning
        base_df = dataset_loader.add_reasoning_to_dataset(base_df)

    # Split dataset
    train_df, val_df, test_df = dataset_loader.prepare_dataset_splits(base_df)

    # Create HF datasets
    datasets = dataset_loader.create_hf_dataset(train_df, val_df, test_df)

    # FIX: Use the tokenizer's chat template for correct and robust formatting.
    # Manual string formatting is error-prone and may not match the model's expected input.
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [qwen_model.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    # Apply formatting to all splits
    formatted_datasets = {}
    for split_name, split_data in datasets.items():
        formatted_datasets[split_name] = split_data.map(formatting_prompts_func, batched=True)

    # Apply formatting to all splits
    formatted_datasets = {}
    for split_name, split_data in datasets.items():
        formatted_datasets[split_name] = split_data.map(formatting_prompts_func, batched=True)

    # ============ ADD THIS DEBUGGING SECTION ============
    print("\n" + "="*60)
    print("DEBUGGING: Checking formatted data structure")
    print("="*60)

    # Check the first example in detail
    if len(formatted_datasets['train']) > 0:
        first_example = formatted_datasets['train'][0]['text']
        
        # Show the first 1000 characters
        print("\nFirst 1000 chars of formatted text:")
        print(repr(first_example[:1000]))
        
        # Check for key markers
        print("\nChecking for key markers:")
        print(f"  - Contains '<|im_start|>user': {('<|im_start|>user' in first_example)}")
        print(f"  - Contains '<|im_start|>assistant': {('<|im_start|>assistant' in first_example)}")
        print(f"  - Contains '<think>': {('<think>' in first_example)}")
        
        # Find where assistant response starts
        assistant_idx = first_example.find('<|im_start|>assistant')
        if assistant_idx != -1:
            print(f"\nAssistant response starts at position {assistant_idx}:")
            print("Next 100 chars after '<|im_start|>assistant':")
            print(repr(first_example[assistant_idx:assistant_idx+100]))
        
    print("="*60 + "\n")
    # ============ END OF DEBUGGING SECTION ============

    print("================= DEBUGGING DATASET =================")
    print("This is the first training example the model will see:")
    print(repr(formatted_datasets['train'][0]['text']))
    print("=====================================================")
    # END DEBUGGING BLOCK

    # Setup trainer
    logger.info("Setting up trainer...")
    trainer = PhishingTrainer(
        qwen_model.model,
        qwen_model.tokenizer,
        training_config
    )
    trainer.setup_trainer(
        formatted_datasets['train'],
        formatted_datasets.get('validation')
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=args.resume)

    # Evaluate
    if 'test' in formatted_datasets:
        logger.info("Evaluating on test set...")
        
        # Tokenize test dataset before evaluation
        test_dataset = formatted_datasets['test']
        
        # Apply the same tokenization as training data
        def tokenize_function(examples):
            return qwen_model.tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=qwen_model.model.config.max_length,
            )
        
        test_dataset = test_dataset.map(
            tokenize_function,
            batched=True,
            desc="Tokenizing test dataset"
        )
        
        eval_results = trainer.evaluate(test_dataset)

    # Save model
    save_path = trainer.save_model()
    logger.info(f"Model saved to {save_path}")

    # Save GGUF if requested
    if args.save_gguf:
        qwen_model.save_gguf(save_path / "gguf", quantization_method="q4_k_m")

    logger.info("Training complete!")

# NOTE: The other functions (evaluate_model, run_inference_server, etc.) were already
# well-structured and will benefit from the fixes in the config and QwenPhishingModel classes.
# I've left them as they were.

def evaluate_model(args):
    """Evaluate the trained model"""
    logger = setup_logging(args.log_file)
    logger.info("Starting evaluation")

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    qwen_model = QwenPhishingModel(inference_config)
    qwen_model.load_model(args.model_path)

    # Load test data
    dataset_loader = PhishingDatasetLoader(data_config)

    if not Path(data_config.dataset_path).exists():
        logger.warning("Dataset not found, creating sample dataset...")
        create_sample_dataset(data_config.dataset_path, num_samples=200)

    base_df = dataset_loader.load_base_dataset()
    _, _, test_df = dataset_loader.prepare_dataset_splits(base_df)

    # Initialize detector
    preprocessor = EmailPreprocessor()
    detector = PhishingDetector(
        qwen_model.model,
        qwen_model.tokenizer,
        preprocessor,
        inference_config
    )

    # Run predictions
    logger.info("Running predictions...")
    predictions, y_true, y_pred = [], [], []

    for idx, row in test_df.iterrows():
        if idx >= 50:  # Limit for demo purposes
            break
        try:
            result = detector.analyze_email(row['text'])
            predictions.append(result)
            y_true.append(row['label'])
            y_pred.append(1 if result['classification'] == 'PHISHING' else 0)
        except Exception as e:
            logger.error(f"Failed to analyze email {idx}: {e}")

    # Calculate metrics
    metrics_calculator = PhishingMetrics()
    metrics = metrics_calculator.calculate_metrics(np.array(y_true), np.array(y_pred))

    metrics_calculator.print_metrics(metrics, "Test Set Evaluation")

    if args.save_results:
        output_path = Path(args.save_results)
        output_path.mkdir(parents=True, exist_ok=True)
        metrics_calculator.export_results(output_path / "metrics.csv")
        metrics_calculator.plot_confusion_matrix(metrics, output_path / "confusion_matrix.png")

    logger.info("Evaluation complete!")

def run_inference_server(args):
    """Run the inference server with SIEM integration"""
    logger = setup_logging(args.log_file)
    logger.info("Starting inference server")

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    qwen_model = QwenPhishingModel(inference_config)
    qwen_model.load_model(args.model_path)

    # Initialize components
    preprocessor = EmailPreprocessor()
    detector = PhishingDetector(
        qwen_model.model,
        qwen_model.tokenizer,
        preprocessor,
        inference_config
    )

    siem = SIEMIntegration(detector, siem_config)
    api = PhishingAPI(siem)
    logger.info(f"Starting API server on {siem_config.api_host}:{siem_config.api_port}")
    api.run()

def analyze_single_email(args):
    """Analyze a single email for testing"""
    logger = setup_logging(args.log_file)

    # Override device if specified
    if args.device:
        inference_config.device = args.device

    # Load model
    if args.model_path:
        logger.info(f"Loading model from {args.model_path}")
    else:
        logger.info("Loading base model (no checkpoint specified)")
    logger.info(f"Using device: {inference_config.device}")
    qwen_model = QwenPhishingModel(inference_config)
    qwen_model.load_model(args.model_path)

    # Get email text
    email_text = args.email_text
    if args.email_file:
        with open(args.email_file, 'r') as f:
            email_text = f.read()

    # Analyze email
    if qwen_model.is_gguf:
        # For GGUF models, use the QwenPhishingModel directly
        result = qwen_model.analyze_email(email_text)
    else:
        # For HuggingFace models, use the detector
        detector = PhishingDetector(
            qwen_model.model,
            qwen_model.tokenizer,
            None,  # No preprocessor needed anymore
            inference_config
        )
        result = detector.analyze_email(email_text)

    # Print results
    print(f"\n{'='*60}\nEMAIL ANALYSIS RESULTS\n{'='*60}")
    print(f"Classification: {result['classification']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Risk Score: {result['risk_score']:.3f}")
    print(f"Recommended Action: {result['recommended_action']}")
    print(f"Processing Time: {result['processing_time']:.3f} seconds")

    if result['risk_indicators']:
        print(f"\nRisk Indicators:")
        for indicator in result['risk_indicators']:
            print(f"  - {indicator}")

    if args.show_reasoning and result.get('reasoning'):
        print(f"\nSecurity Analysis:")
        print(result['reasoning'])

    print(f"{'='*60}\n")

def create_sample_data(args):
    """Create sample dataset for testing"""
    logger = setup_logging()
    logger.info(f"Creating sample dataset with {args.num_samples} samples")

    df = create_sample_dataset(args.output_path, args.num_samples)
    logger.info(f"Sample dataset created at {args.output_path}")

def export_merged_model(args):
    """Export merged LoRA model in various formats"""
    logger = setup_logging(args.log_file)

    logger.info(f"Exporting merged model from checkpoint: {args.model_path}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Export format: {args.format}")

    # Load model with LoRA
    qwen_model = QwenPhishingModel(inference_config)
    qwen_model.load_model(args.model_path)

    # Export in specified format
    if args.format == "gguf":
        qwen_model.export_merged_gguf(args.output_path, args.quantization)
    else:
        qwen_model.export_merged_model(args.output_path, args.format)

    logger.info("Model export completed successfully!")
    print(f"Merged model exported to: {args.output_path}")

    if args.format == "gguf":
        print("You can now use this GGUF file in LMStudio or other llama.cpp applications.")
    else:
        print("You can load this merged model with standard HuggingFace transformers.")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Phishing Detection with Reasoning Distillation")
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    # Training command
    train_parser = subparsers.add_parser('train', help='Train the phishing detection model')
    train_parser.add_argument('--generate-reasoning', action='store_true', help='Generate reasoning using teacher model')
    train_parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    train_parser.add_argument('--save-gguf', action='store_true', help='Save model in GGUF format')
    train_parser.add_argument('--log-file', type=str, help='Log file path')

    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    eval_parser.add_argument('--save-results', type=str, help='Directory to save evaluation results')
    eval_parser.add_argument('--log-file', type=str, help='Log file path')

    # Server command
    server_parser = subparsers.add_parser('server', help='Run inference server')
    server_parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    server_parser.add_argument('--log-file', type=str, help='Log file path')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze single email')
    analyze_parser.add_argument('--model-path', type=str, help='Path to trained model (optional, uses base model if not provided)')
    analyze_parser.add_argument('--email-text', type=str, help='Email text to analyze')
    analyze_parser.add_argument('--email-file', type=str, help='File containing email text')
    analyze_parser.add_argument('--show-reasoning', action='store_true', help='Show detailed reasoning')
    analyze_parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], help='Device for inference')
    analyze_parser.add_argument('--log-file', type=str, help='Log file path')

    # Export merged model command
    export_parser = subparsers.add_parser('export', help='Export merged model in various formats')
    export_parser.add_argument('--model-path', type=str, required=True, help='Path to LoRA checkpoint')
    export_parser.add_argument('--output-path', type=str, required=True, help='Output path for merged model')
    export_parser.add_argument('--format', type=str, default='merged_16bit',
                              choices=['merged_16bit', 'merged_4bit', 'merged_4bit_forced', 'gguf'],
                              help='Export format')
    export_parser.add_argument('--quantization', type=str, default='q4_k_m',
                              choices=['q2_k', 'q3_k_s', 'q3_k_m', 'q3_k_l', 'q4_0', 'q4_1', 'q4_k_s', 'q4_k_m', 'q5_0', 'q5_1', 'q5_k_s', 'q5_k_m', 'q6_k', 'q8_0'],
                              help='GGUF quantization method (only applies when format=gguf)')
    export_parser.add_argument('--log-file', type=str, help='Log file path')

    # Create sample data command
    sample_parser = subparsers.add_parser('create-sample', help='Create sample dataset')
    sample_parser.add_argument('--output-path', type=str, required=True, help='Output path for sample dataset')
    sample_parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples to generate')

    args = parser.parse_args()

    if args.command == 'train':
        train_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    elif args.command == 'server':
        run_inference_server(args)
    elif args.command == 'analyze':
        if not args.email_text and not args.email_file:
            print("Error: Must provide either --email-text or --email-file", file=sys.stderr)
            sys.exit(1)
        analyze_single_email(args)
    elif args.command == 'create-sample':
        create_sample_data(args)
    elif args.command == 'export':
        export_merged_model(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()