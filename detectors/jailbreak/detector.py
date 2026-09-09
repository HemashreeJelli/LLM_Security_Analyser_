"""
JailbreakDetector — PRD Detector #2
===================================
Scans user prompts and conversation histories for jailbreak attempts:
  Layer 1: Pattern Library (regex + keyword templates)
  Layer 2: Embedding Similarity (SentenceTransformers + Cosine Similarity)
  Layer 3: Multi-turn Escalation Tracking

Implements the BaseDetector Protocol.

Design decisions:
  - Strict Mode: Intentionally requires `sentence-transformers` and `scikit-learn`
    to be installed.
  - Sub-score formula: max(confidence across Layers 1, 2, and 3).
"""

from typing import Any

from detectors.base import DetectorResult
from llm_judge.backends.base import LLMBackend
from .patterns import scan_patterns
from .classifier import classify_prompt
from .multi_turn import check_multi_turn_escalation
from .llm_judge import evaluate_jailbreak

class JailbreakDetector:
    """
    Detector #2: Jailbreak Attempts.
    """
    detector_name: str = "jailbreak"

    def __init__(self, backend: LLMBackend | None = None, route_all_to_judge: bool = False, judge_threshold: float = 0.8, **kwargs) -> None:
        """
        Args:
            backend: The LLMBackend to use for gray-zone confirmation (human-in-loop proxy).
            route_all_to_judge: If True, bypasses threshold logic and evaluates ALL prompts via the LLM judge.
            judge_threshold: If route_all_to_judge is False, routes prompts to the judge if ML confidence is < this value.
        """
        self.backend = backend
        self.route_all_to_judge = route_all_to_judge
        self.judge_threshold = judge_threshold

    def detect(
        self,
        prompt: str,
        response: str | None = None,
        context: list[str] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> DetectorResult:
        """
        Scan prompt and history for jailbreak attempts.
        """
        evidence = {
            "pattern_matches": [],
            "ml_classifier": None,
            "multi_turn_escalation": None,
            "llm_judge": None
        }
        
        confidences = []

        # ── Layer 1: Pattern Matches ──────────────────────────────────────────
        pattern_hits = scan_patterns(prompt)
        for hit in pattern_hits:
            evidence["pattern_matches"].append({
                "technique": hit.technique,
                "confidence": hit.confidence,
                "matched_text": hit.matched_text
            })
            confidences.append(hit.confidence)

        # ── Layer 2: ML Classifier ────────────────────────────────────────────
        ml_hit = classify_prompt(prompt)
        if ml_hit.is_jailbreak:
            evidence["ml_classifier"] = {
                "confidence": ml_hit.confidence,
                "detection_method": ml_hit.detection_method
            }
            confidences.append(ml_hit.confidence)
        elif ml_hit.confidence > 0:
            evidence["ml_classifier"] = {
                "confidence": ml_hit.confidence,
                "detection_method": ml_hit.detection_method,
                "note": "Classified as benign"
            }

        # ── Layer 3: Multi-turn Escalation ────────────────────────────────────
        if conversation_history:
            esc_hit = check_multi_turn_escalation(conversation_history)
            if esc_hit.is_escalating:
                evidence["multi_turn_escalation"] = {
                    "confidence": esc_hit.confidence,
                    "reason": esc_hit.escalation_reason
                }
                confidences.append(esc_hit.confidence)

        # ── Layer 4: LLM Judge (Human-in-the-Loop) ────────────────────────────
        # Configurable routing based on flags
        needs_judge = False
        if self.route_all_to_judge:
            needs_judge = True
        elif ml_hit.confidence < self.judge_threshold:
            needs_judge = True
            
        if needs_judge and self.backend:
            judge_hit = evaluate_jailbreak(prompt, self.backend, conversation_history)
            evidence["llm_judge"] = {
                "is_jailbreak": judge_hit.is_jailbreak,
                "confidence": judge_hit.confidence,
                "reason": judge_hit.reason
            }
            if judge_hit.is_jailbreak:
                confidences.append(judge_hit.confidence)

        # ── Aggregate results ────────────────────────────────────────────────
        is_flagged = len(confidences) > 0
        sub_score = max(confidences) if confidences else 0.0
        
        # Overall confidence mapped directly from the max sub_score
        confidence = sub_score if is_flagged else 1.0 - (evidence.get("ml_classifier", {}).get("confidence", 0.0) if evidence.get("ml_classifier") else 0.0)

        return DetectorResult(
            detector_name=self.detector_name,
            is_flagged=is_flagged,
            confidence=round(confidence, 4),
            sub_score=round(sub_score, 4),
            evidence=evidence
        )

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_singleton: JailbreakDetector | None = None

def get_jailbreak_detector(
    backend: LLMBackend | None = None,
    route_all_to_judge: bool = False,
    judge_threshold: float = 0.8,
    **kwargs
) -> JailbreakDetector:
    global _singleton
    if _singleton is None:
        _singleton = JailbreakDetector(
            backend=backend,
            route_all_to_judge=route_all_to_judge,
            judge_threshold=judge_threshold
        )
    return _singleton
