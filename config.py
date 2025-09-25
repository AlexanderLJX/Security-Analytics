"""
Configuration settings for phishing detection system
"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ModelConfig:
    """Model configuration parameters"""
    # Model selection
    teacher_model: str = "gpt-5"
    student_model: str = "unsloth/Qwen3-4B-Thinking-2507"

    # Model parameters
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    load_in_8bit: bool = False

    # LoRA configuration
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_target_modules: list = None

    # CHANGE: Set device to "auto" to use GPU when available. "cpu" is too slow for training.
    device: str = "auto"
    seed: int = 3407

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    # Training parameters
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_steps: int = 20
    weight_decay: float = 0.01

    # Optimization
    optimizer: str = "adamw_8bit"
    lr_scheduler: str = "cosine"

    # Checkpointing
    output_dir: str = "./checkpoints/phishing_detector"
    save_steps: int = 100
    eval_steps: int = 50
    logging_steps: int = 10

    # CHANGE: Set device to "auto"
    device: str = "auto"
    seed: int = 3407

    # Tracking
    report_to: str = "tensorboard"  # or "wandb"
    project_name: str = "phishing-reasoning-distillation"

@dataclass
class DataConfig:
    """Data configuration parameters"""
    # Datasets
    dataset_path: str = "./combined-dataset-with-reasoning-fast.parquet"
    reasoning_cache_path: str = "./data/reasoning_cache.json"

    # Data splits
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # Preprocessing
    max_email_length: int = 2048
    min_email_length: int = 10

    # Augmentation
    use_augmentation: bool = True
    augmentation_factor: int = 2

    # Additional
    seed: int = 3407

@dataclass
class ReasoningConfig:
    """Reasoning distillation configuration"""
    # API settings
    api_provider: str = "openai"  # or "anthropic"
    api_key: str = os.getenv("OPENAI_API_KEY")
    teacher_model: str = "gpt-5"

    # Generation parameters
    temperature: float = 0.3
    max_tokens: int = 1000
    top_p: float = 0.9

    # Reasoning settings
    use_corrective_feedback: bool = True
    max_correction_attempts: int = 2
    include_adversarial: bool = True

    # Batch processing
    batch_size: int = 10
    save_frequency: int = 50

    # Cache settings
    reasoning_cache_path: str = "./reasoning_cache_fast.json"

@dataclass
class InferenceConfig:
    """Inference configuration parameters"""
    # Model settings
    model_path: str = "./checkpoints/phishing_detector/best"
    use_quantization: bool = True
    student_model: str = "unsloth/Qwen3-4B-Thinking-2507"

    # Model loading parameters (needed for Qwen model)
    load_in_4bit: bool = True
    load_in_8bit: bool = False

    # Inference parameters
    max_new_tokens: int = 2000
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 20
    do_sample: bool = True

    # Performance
    batch_size: int = 8
    max_concurrent_requests: int = 100

    # Device: "auto" for GPU, "cpu" for CPU inference
    device: str = "auto"
    max_seq_length: int = 4096

@dataclass
class SIEMConfig:
    """SIEM integration configuration"""
    # Kafka settings
    kafka_broker: str = os.getenv("KAFKA_BROKER", "localhost:9092")
    kafka_topic: str = "security-alerts"

    # Splunk HTTP Event Collector settings
    splunk_hec_url: str = os.getenv("SPLUNK_HEC_URL", "")
    splunk_token: str = os.getenv("SPLUNK_TOKEN", "")
    splunk_index: str = os.getenv("SPLUNK_INDEX", "security")

    # Alert settings
    alert_threshold: float = 0.7
    high_risk_threshold: float = 0.9

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 5000

# Create global config instances
model_config = ModelConfig()
training_config = TrainingConfig()
data_config = DataConfig()
reasoning_config = ReasoningConfig()
inference_config = InferenceConfig()
siem_config = SIEMConfig()