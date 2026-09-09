from transformers import pipeline

_pipeline = None

def get_classifier():
    global _pipeline
    if _pipeline is None:
        # We use toxic-bert. top_k=None forces the pipeline to return scores for ALL labels.
        _pipeline = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)
    return _pipeline

def analyze_toxicity(text: str) -> dict[str, float]:
    classifier = get_classifier()
    
    # BERT max sequence length is 512 tokens. We'll loosely truncate by characters to avoid crashing
    # on massive responses. 2000 chars is roughly 400-500 words.
    truncated_text = text[:2000]
    
    results = classifier(truncated_text)[0]
    
    # unitary/toxic-bert labels: toxic, severe_toxic, obscene, threat, insult, identity_hate
    raw_scores = {res["label"]: res["score"] for res in results}
    
    # Map toxic-bert labels to our PRD target categories
    mapped_scores = {
        "hate": raw_scores.get("identity_hate", 0.0),
        "violence": raw_scores.get("threat", 0.0),
        "sexual": raw_scores.get("obscene", 0.0), # Obscene is the closest proxy to sexual explicitly
        "self_harm": 0.0, # toxic-bert lacks this; we rely entirely on the lexicon layer
        "harassment": max(raw_scores.get("insult", 0.0), raw_scores.get("toxic", 0.0))
    }
    
    return mapped_scores
