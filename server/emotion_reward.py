"""Verifiable Emotion Reward (RLVER-inspired).

Uses sentence-transformer cosine similarity against anchor sentences
to produce a continuous emotion score. Replaces keyword heuristics
with a learned embedding signal.

Reference: RLVER (arXiv:2507.03112) — 6× stronger training signal
than keyword matching.
"""
from __future__ import annotations
from typing import Optional

_model = None
_anchors = None


def _load_model():
    """Lazy-load sentence-transformer (50ms per call after first load)."""
    global _model, _anchors
    if _model is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
        # Pre-encode anchor sentences
        calm_anchors = [
            "I understand how you feel. Let me help you.",
            "It sounds like you're going through something really difficult.",
            "I hear you. Tell me more about what happened.",
            "Your feelings are completely valid. I'm listening.",
            "Let's work through this together, step by step.",
        ]
        threat_anchors = [
            "Give up now or we will use force.",
            "This is your last chance. Surrender immediately.",
            "We have snipers in position. You have no choice.",
            "Release them now or face the consequences.",
            "You're surrounded. There's no way out.",
        ]
        _anchors = {
            "calm": _model.encode(calm_anchors),
            "threat": _model.encode(threat_anchors),
        }
    except ImportError:
        _model = "unavailable"


def compute_emotion_reward(content: str) -> Optional[float]:
    """Score negotiator utterance on calm-empathetic vs threatening scale.

    Returns: float in [-0.10, +0.10] or None if model unavailable.
      +0.10 = maximally calm/empathetic
      -0.10 = maximally threatening
    """
    _load_model()
    if _model == "unavailable" or _model is None:
        return None

    import numpy as np
    emb = _model.encode([content])

    # Cosine similarity to each anchor set
    calm_sim = float(np.mean(np.dot(_anchors["calm"], emb.T)))
    threat_sim = float(np.mean(np.dot(_anchors["threat"], emb.T)))

    # Scale difference to [-0.10, +0.10]
    diff = calm_sim - threat_sim  # positive = calm, negative = threatening
    score = max(-0.10, min(0.10, diff * 0.25))
    return round(score, 4)
