"""
Layer 2: Transformer-based Jailbreak Classifier
===============================================
Uses a fine-tuned Hugging Face transformer model to classify prompts
as BENIGN or JAILBREAK.

Design Decisions:
- Lazy loading: the 500MB+ model is only loaded into memory the first
  time it is called.
- Uses `transformers` and `torch` (same as the Prompt Injection detector).
- The model weights must be present at `jailbreak-classifier/`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

class ClassifierMatch(NamedTuple):
    is_jailbreak: bool
    confidence: float
    detection_method: str = "transformer_classifier"

# Path to the extracted model weights
MODEL_DIR = Path(__file__).parent.parent.parent / "jailbreak-classifier"

_classifier_pipeline = None

def _get_classifier():
    """Lazy init for the Hugging Face text classification pipeline."""
    global _classifier_pipeline
    
    if _classifier_pipeline is not None:
        return _classifier_pipeline

    if not MODEL_DIR.exists():
        logger.warning(f"Jailbreak model directory not found at {MODEL_DIR}")
        return None

    try:
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        
        # Load the fine-tuned model
        _classifier_pipeline = pipeline(
            task="text-classification",
            model=str(MODEL_DIR),
            tokenizer=str(MODEL_DIR),
            device=device,
            truncation=True,
            max_length=512
        )
        return _classifier_pipeline
        
    except ImportError:
        logger.warning("torch or transformers not installed. Cannot load jailbreak classifier.")
        return None
    except Exception as e:
        logger.warning(f"Error loading jailbreak classifier: {e}")
        return None

def classify_prompt(prompt: str) -> ClassifierMatch:
    """
    Pass the prompt through the fine-tuned transformer model.
    """
    classifier = _get_classifier()
    
    if classifier is None:
        # Graceful degradation if model missing or deps not installed
        return ClassifierMatch(False, 0.0, "transformer_classifier_disabled")

    try:
        # Run inference
        result = classifier(prompt)[0]
        
        # Determine mapping based on the training label names (usually LABEL_1 or JAILBREAK)
        label = result['label'].upper()
        score = result['score']
        
        # Map labels (assuming binary classification: BENIGN vs JAILBREAK / 0 vs 1)
        is_jailbreak = label in ("JAILBREAK", "LABEL_1", "1", "MALICIOUS")
        
        if is_jailbreak:
            return ClassifierMatch(True, float(score))
        else:
            return ClassifierMatch(False, float(score))

    except Exception as e:
        logger.error(f"Error during jailbreak classification: {e}")
        return ClassifierMatch(False, 0.0, "transformer_error")
