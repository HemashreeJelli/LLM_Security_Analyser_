from dataclasses import dataclass

@dataclass
class SafetyThresholds:
    hate: float
    violence: float
    sexual: float
    self_harm: float
    harassment: float

class Profiles:
    # Highly restrictive (e.g., educational bots, children's applications)
    STRICT_EDUCATION = SafetyThresholds(hate=0.1, violence=0.1, sexual=0.1, self_harm=0.1, harassment=0.1)
    
    # Standard enterprise use (balances safety with allowing some edgy discussions if needed)
    MODERATE_ENTERPRISE = SafetyThresholds(hate=0.3, violence=0.3, sexual=0.5, self_harm=0.2, harassment=0.4)
    
    # Highly permissive (e.g., security research tools parsing malicious datasets)
    PERMISSIVE_RESEARCH = SafetyThresholds(hate=0.9, violence=0.9, sexual=0.9, self_harm=0.9, harassment=0.9)
