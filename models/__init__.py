"""
Models module for phishing detection system
"""

from .qwen_model import QwenPhishingModel
from .teacher_model import TeacherModel

__all__ = ["QwenPhishingModel", "TeacherModel"]