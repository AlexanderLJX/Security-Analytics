"""
Training module for phishing detection system
"""

from .trainer import PhishingTrainer
from .reasoning_distillation import ReasoningDistillation

__all__ = ["PhishingTrainer", "ReasoningDistillation"]