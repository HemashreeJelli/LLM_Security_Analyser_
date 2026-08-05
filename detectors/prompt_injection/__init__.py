"""
PromptInjectionDetector — PRD Detector #1.

Integrates Layers 1, 2, 2.5, and 3 into a single object implementing BaseDetector.
"""

from .detector import PromptInjectionDetector, get_prompt_injection_detector

__all__ = ["PromptInjectionDetector", "get_prompt_injection_detector"]
