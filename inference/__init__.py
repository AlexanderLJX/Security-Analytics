"""
Inference module for phishing detection system
"""

from .detector import PhishingDetector
from .siem_integration import SIEMIntegration, PhishingAPI

__all__ = ["PhishingDetector", "SIEMIntegration", "PhishingAPI"]