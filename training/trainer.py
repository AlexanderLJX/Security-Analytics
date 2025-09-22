"""
Main training logic for the phishing detector
"""
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only
import torch
from pathlib import Path
from typing import Optional, Dict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class PhishingTrainer:
    """Trainer for phishing detection model"""

    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.trainer = None

    def prepare_training_args(self) -> SFTConfig:
        """Prepare training arguments"""
        return SFTConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            learning_rate=self.config.learning_rate,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            optim=self.config.optimizer,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler,
            seed=self.config.seed,
            report_to=self.config.report_to,
            dataset_text_field="text",
            max_seq_length=self.model.config.max_seq_length,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": True},
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
        )

    def setup_trainer(self, train_dataset, eval_dataset=None):
        """Setup the SFT trainer"""
        training_args = self.prepare_training_args()

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            packing=False,
            dataset_num_proc=4,
        )

        # Train only on responses
        self.trainer = train_on_responses_only(
            self.trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

        logger.info("Trainer setup complete")

    def train(self, resume_from_checkpoint: bool = False):
        """Run training"""
        logger.info("Starting training...")

        # Log GPU memory before training
        if torch.cuda.is_available():
            gpu_stats = torch.cuda.get_device_properties(0)
            start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
            logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
            logger.info(f"{start_gpu_memory} GB of memory reserved.")

        # Train
        train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Log training stats
        self._log_training_stats(train_result)

        return train_result

    def _log_training_stats(self, train_result):
        """Log training statistics"""
        if torch.cuda.is_available():
            gpu_stats = torch.cuda.get_device_properties(0)
            max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
            used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            used_percentage = round(used_memory / max_memory * 100, 3)

            logger.info(f"Training completed in {train_result.metrics['train_runtime']:.2f} seconds")
            logger.info(f"Peak memory usage: {used_memory} GB ({used_percentage}%)")

        # Log metrics
        for key, value in train_result.metrics.items():
            logger.info(f"{key}: {value}")

    def evaluate(self, eval_dataset=None):
        """Run evaluation"""
        if eval_dataset is None and self.trainer.eval_dataset is None:
            logger.warning("No evaluation dataset provided")
            return None

        logger.info("Running evaluation...")
        eval_results = self.trainer.evaluate(eval_dataset=eval_dataset)

        # Log evaluation results
        for key, value in eval_results.items():
            logger.info(f"{key}: {value}")

        return eval_results

    def save_model(self, save_path: Optional[str] = None):
        """Save the trained model"""
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = Path(self.config.output_dir) / f"checkpoint_{timestamp}"

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to {save_path}")
        self.trainer.save_model(str(save_path))

        return save_path

    def push_to_hub(self, repo_name: str, private: bool = True):
        """Push model to Hugging Face Hub"""
        logger.info(f"Pushing model to hub: {repo_name}")
        self.trainer.push_to_hub(
            repo_name=repo_name,
            private=private,
            tags=["phishing-detection", "reasoning-distillation", "qwen3"]
        )