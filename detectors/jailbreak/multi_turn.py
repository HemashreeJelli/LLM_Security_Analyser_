"""
Layer 3: Multi-turn Escalation Tracking
=======================================
Detects "crescendo" attacks where the user gradually pushes boundaries
across multiple conversational turns.

Design:
Analyzes the `conversation_history` to track if the user repeatedly
makes requests after being refused by the assistant, indicating persistence
typical of manual jailbreak probing.
"""

from typing import NamedTuple

class EscalationMatch(NamedTuple):
    is_escalating: bool
    confidence: float
    escalation_reason: str | None

def check_multi_turn_escalation(history: list[dict[str, str]] | None) -> EscalationMatch:
    """
    Checks the conversation history for jailbreak probing behavior.
    
    Args:
        history: List of message dicts, e.g., [{"role": "user", "content": "..."}, ...]
    """
    if not history or len(history) < 3:
        # Not enough turns to establish a pattern
        return EscalationMatch(False, 0.0, None)

    refusal_keywords = ["i cannot", "i can't", "i am unable", "against my safety guidelines", "i'm sorry, but"]
    
    refusal_count = 0
    user_push_count = 0
    
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "").lower()
        
        if role == "assistant":
            if any(kw in content for kw in refusal_keywords):
                refusal_count += 1
        elif role == "user":
            if refusal_count > 0:
                # User continued prompting after a refusal
                user_push_count += 1

    if refusal_count >= 2 and user_push_count >= 2:
        # High likelihood of probing/escalation
        return EscalationMatch(
            is_escalating=True, 
            confidence=0.85, 
            escalation_reason=f"Detected {refusal_count} assistant refusals followed by {user_push_count} persistent user follow-ups."
        )
    elif refusal_count == 1 and user_push_count >= 1:
        # Moderate likelihood
        return EscalationMatch(
            is_escalating=True, 
            confidence=0.60, 
            escalation_reason="Detected persistent user follow-up after an assistant refusal."
        )

    return EscalationMatch(False, 0.0, None)
