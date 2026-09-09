from detectors.base import BaseDetector, DetectorResult
from .config import SafetyThresholds, Profiles
from .lexicon import scan_lexicon
from .classifier import analyze_toxicity

class UnsafeOutputDetector(BaseDetector):
    def __init__(self, thresholds: SafetyThresholds = Profiles.MODERATE_ENTERPRISE):
        self.thresholds = thresholds

    @property
    def detector_name(self) -> str:
        return "Unsafe Output"

    def detect(self, prompt: str, response: str | None = None, context: str | None = None, conversation_history: list | None = None) -> DetectorResult:
        # We only care about what the LLM generates, not the prompt.
        if not response:
            return {
                "detector_name": self.detector_name,
                "is_flagged": False, 
                "sub_score": 0.0, 
                "confidence": 1.0, 
                "evidence": {"unsafe": False, "status": "No response provided"}
            }
        
        # 1. Lexicon Scan (Fast path)
        lex_scores = scan_lexicon(response)
        
        # 2. ML Classifier Scan (unitary/toxic-bert)
        ml_scores = analyze_toxicity(response)
        
        # Merge: Take the max of lexicon or ML for each category
        final_scores = {
            "hate": max(lex_scores["hate"], ml_scores["hate"]),
            "violence": max(lex_scores["violence"], ml_scores["violence"]),
            "sexual": max(lex_scores["sexual"], ml_scores["sexual"]),
            "self_harm": max(lex_scores["self_harm"], ml_scores["self_harm"]),
            "harassment": max(lex_scores["harassment"], ml_scores["harassment"])
        }
        
        # 3. Threshold check
        is_unsafe = False
        violated_categories = []
        
        if final_scores["hate"] >= self.thresholds.hate:
            is_unsafe = True; violated_categories.append("hate")
        if final_scores["violence"] >= self.thresholds.violence:
            is_unsafe = True; violated_categories.append("violence")
        if final_scores["sexual"] >= self.thresholds.sexual:
            is_unsafe = True; violated_categories.append("sexual")
        if final_scores["self_harm"] >= self.thresholds.self_harm:
            is_unsafe = True; violated_categories.append("self_harm")
        if final_scores["harassment"] >= self.thresholds.harassment:
            is_unsafe = True; violated_categories.append("harassment")
            
        max_severity = max(final_scores.values()) if final_scores.values() else 0.0
        
        evidence = {
            "unsafe": is_unsafe,
            "status": f"Unsafe content detected: {', '.join(violated_categories)}" if is_unsafe else "Content passed safety thresholds.",
            "categories": final_scores,
            "max_severity": float(max_severity),
            "violated_thresholds": violated_categories,
            "active_profile": {
                "hate_thresh": self.thresholds.hate,
                "violence_thresh": self.thresholds.violence,
                "sexual_thresh": self.thresholds.sexual,
                "self_harm_thresh": self.thresholds.self_harm,
                "harassment_thresh": self.thresholds.harassment
            }
        }
        
        return {
            "detector_name": self.detector_name,
            "is_flagged": is_unsafe,
            "sub_score": float(max_severity) if is_unsafe else 0.0,
            "confidence": 1.0, # Could blend based on distance from threshold, but 1.0 is fine for threshold breach
            "evidence": evidence
        }
