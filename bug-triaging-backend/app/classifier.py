"""
classifier.py
Severity classification for NEW incoming reports, via API.

This is the exact same model + label descriptions + priority formula as
priority_scoring_v2.py — nothing reinvented, just wrapped so it's callable
over HTTP instead of run as a batch script.

The zero-shot-classification pipeline is loaded once at import time
(module-level singleton) since loading facebook/bart-large-mnli takes
real time — you don't want to pay that cost on every request.
"""
from functools import lru_cache

# Identical to priority_scoring_v2.py / zero_shot_classify_v2.py
LABEL_DESCRIPTIONS = {
    "a critical bug causing system crash, data loss, or complete failure": "Critical",
    "a high severity bug causing major functionality to break or work incorrectly": "High",
    "a medium severity bug causing a minor feature to misbehave but the system still works": "Medium",
    "a low severity bug that is a small cosmetic or trivial issue": "Low",
}
CANDIDATE_LABELS = list(LABEL_DESCRIPTIONS.keys())

# Identical to priority_scoring.py / priority_scoring_v2.py
SEVERITY_SCORE = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
DUPLICATE_PENALTY = 0.5

_classifier = None


def get_classifier():
    """
    Lazy singleton — the model only loads on first real request, not at
    server startup, so `uvicorn` boots instantly and Postman/docs are
    usable immediately. First classification call will be slow (~model load
    time); every call after that is fast.
    """
    global _classifier
    if _classifier is None:
        from transformers import pipeline
        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
    return _classifier


def classify_severity(text: str) -> tuple[str, float]:
    """
    Returns (predicted_severity, confidence) for a single bug report,
    using the exact same truncation (first 1000 chars) as your scripts.
    """
    text_for_model = (text or "")[:1000]

    if not text_for_model.strip():
        # Same fallback as priority_scoring.py for empty text
        return "Medium", 0.0

    clf = get_classifier()
    result = clf(text_for_model, CANDIDATE_LABELS)
    predicted_description = result["labels"][0]
    predicted_label = LABEL_DESCRIPTIONS[predicted_description]
    confidence = round(float(result["scores"][0]), 3)
    return predicted_label, confidence


def calculate_priority(severity: str, is_duplicate: bool) -> float:
    """Identical formula to priority_scoring.py's calculate_priority()."""
    base_score = SEVERITY_SCORE.get(severity, 2)
    if is_duplicate:
        return base_score * DUPLICATE_PENALTY
    return float(base_score)
