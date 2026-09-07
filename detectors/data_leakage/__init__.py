"""
DataLeakageDetector — PRD Detector #3.

Integrates:
  Layer 1: Regex + Checksum (Secrets, PII, Indian IDs)
  Layer 2: Presidio NER (Names, Addresses, Orgs)
  Layer 3: System Prompt Leak Checker (N-gram + Fuzzy)
"""

from .detector import DataLeakageDetector, get_data_leakage_detector

__all__ = ["DataLeakageDetector", "get_data_leakage_detector"]
