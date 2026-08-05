"""
Layer 2 — DeBERTa Sequence Classifier
"""

from __future__ import annotations

import os
from typing import TypedDict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, DebertaV2ForSequenceClassification


class ClassifierResult(TypedDict):
    label: str
    confidence: float
    injection_score: float


_DEFAULT_LABEL_MAP = {0: "BENIGN", 1: "INJECTION"}


class InjectionClassifier:
    def __init__(
        self,
        model_dir: str,
        threshold: float = 0.5,
        label_map: dict[int, str] | None = None,
        device: str = "auto",
    ) -> None:
        self.threshold = threshold
        self.label_map = label_map or _DEFAULT_LABEL_MAP
        self._injection_idx = next(
            k for k, v in self.label_map.items() if v == "INJECTION"
        )

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = DebertaV2ForSequenceClassification.from_pretrained(
            model_dir,
            local_files_only=True,
            num_labels=len(self.label_map),
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> ClassifierResult:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = F.softmax(logits, dim=-1).squeeze(0)
        injection_score = float(probs[self._injection_idx])

        if injection_score >= self.threshold:
            label = "INJECTION"
            confidence = injection_score
        else:
            label = "BENIGN"
            confidence = 1.0 - injection_score

        return ClassifierResult(
            label=label,
            confidence=round(confidence, 4),
            injection_score=round(injection_score, 4),
        )


_singleton: InjectionClassifier | None = None


def get_classifier(
    model_dir: str | None = None,
    threshold: float = 0.5,
    **kwargs,
) -> InjectionClassifier:
    global _singleton
    if _singleton is None:
        resolved_dir = model_dir or os.environ.get("MODEL_DIR") or "model_tmp"
        _singleton = InjectionClassifier(resolved_dir, threshold=threshold, **kwargs)
    return _singleton
