"""
Data module for phishing detection system
"""

from .dataset_loader import PhishingDatasetLoader
from .preprocessor import EmailPreprocessor

__all__ = ["PhishingDatasetLoader", "EmailPreprocessor"]