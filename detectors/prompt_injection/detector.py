"""
PromptInjectionDetector — Top-level Detector #1 implementation satisfying BaseDetector.
"""

from __future__ import annotations

from detectors.base import BaseDetector, DetectorResult
from llm_judge.backends.base import LLMBackend
from .classifier import get_classifier
from .intent_filter import Routing, get_filter
from .llm_judge import LLMJudge
from .preprocessing import preprocess


class PromptInjectionDetector:
    """
    Detector #1: Prompt Injection.
    Integrates Layer 1 (pre-process), Layer 2 (DeBERTa), Layer 2.5 (filter),
    and Layer 3 (LLM judge) into the standard BaseDetector contract.
    """

    detector_name: str = "prompt_injection"

    def __init__(
        self,
        model_dir: str = "model_tmp",
        judge_backend: LLMBackend | None = None,
    ) -> None:
        self.classifier = get_classifier(model_dir=model_dir)
        self.intent_filter = get_filter()
        self.judge = LLMJudge(backend=judge_backend) if judge_backend else None

    def detect(
        self,
        prompt: str,
        response: str | None = None,
        context: list[str] | None = None,
        **kwargs,
    ) -> DetectorResult:
        """
        Analyze prompt for injection attempts.

        Returns standard DetectorResult:
          {
            "detector_name": "prompt_injection",
            "is_flagged": bool,
            "confidence": float,
            "sub_score": float,
            "evidence": dict
          }
        """
        # Step 1: Pre-process
        clean = preprocess(prompt)

        # Step 2: Sequence classification
        l2_res = self.classifier.predict(clean["clean_text"])
        injection_score = l2_res["injection_score"]

        # Step 2.5: Intent filter
        l25_res = self.intent_filter.decide(
            clean["clean_text"],
            l2_res["label"],
            injection_score,
        )

        decision = l25_res["decision"]
        judge_result = None

        # Step 3: LLM Judge if needed
        if decision == Routing.ROUTE_TO_JUDGE and self.judge:
            judge_result = self.judge.judge(
                clean["clean_text"],
                injection_score,
                l25_res["reason"],
            )
            final_verdict = judge_result["verdict"]
        elif decision == Routing.CONFIRMED_INJECTION:
            final_verdict = "INJECTION"
        else:
            final_verdict = "BENIGN"

        is_flagged = (final_verdict == "INJECTION")

        # Sub-score formulation: if flagged, use injection score (min 0.7); else use injection score
        sub_score = injection_score if is_flagged else round(injection_score * 0.5, 4)
        if is_flagged and sub_score < 0.7:
            sub_score = 0.75  # Ensure flagged items have a significant sub-score

        confidence = (
            judge_result["confidence"] if judge_result
            else l2_res["confidence"]
        )

        return DetectorResult(
            detector_name=self.detector_name,
            is_flagged=is_flagged,
            confidence=confidence,
            sub_score=sub_score,
            evidence={
                "preprocessing": clean,
                "classifier": l2_res,
                "intent_filter": l25_res,
                "judge": judge_result,
                "final_verdict": final_verdict,
            },
        )


_singleton: PromptInjectionDetector | None = None


def get_prompt_injection_detector(
    model_dir: str = "model_tmp",
    judge_backend: LLMBackend | None = None,
) -> PromptInjectionDetector:
    global _singleton
    if _singleton is None:
        _singleton = PromptInjectionDetector(
            model_dir=model_dir, judge_backend=judge_backend
        )
    return _singleton
