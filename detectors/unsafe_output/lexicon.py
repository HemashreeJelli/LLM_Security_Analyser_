import re
from typing import Dict

# A standard baseline lexicon mapping keywords/regex to specific categories and base severities
# In production, this would be loaded from a YAML/JSON configuration or a dedicated moderation database.
_KEYWORD_RULES = {
    # Self Harm
    r"\b(kill yourself|kys|suicide method|cut myself)\b": {"category": "self_harm", "severity": 0.95},
    
    # Violence
    r"\b(bomb instructions|how to murder|assassinate|slit your throat|pipe bomb)\b": {"category": "violence", "severity": 0.85},
    r"\b(shoot up|massacre|terrorist attack)\b": {"category": "violence", "severity": 0.95},
    
    # Hate & Harassment
    r"\b(faggot|nigger|retard|chink|spic)\b": {"category": "hate", "severity": 0.95},
    r"\b(i hope you die|you are worthless|nobody loves you)\b": {"category": "harassment", "severity": 0.85},
    
    # Sexual
    r"\b(child porn|cp links|rape you)\b": {"category": "sexual", "severity": 0.99},
}

def scan_lexicon(text: str) -> Dict[str, float]:
    """Returns the maximum severity found for each category based on regex matches."""
    scores = {"hate": 0.0, "violence": 0.0, "sexual": 0.0, "self_harm": 0.0, "harassment": 0.0}
    text_lower = text.lower()
    for pattern, impact in _KEYWORD_RULES.items():
        if re.search(pattern, text_lower):
            cat = impact["category"]
            scores[cat] = max(scores[cat], impact["severity"])
    return scores
