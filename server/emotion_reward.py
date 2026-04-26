"""Verifiable Emotion Reward (RLVER-inspired).

Uses sentence-transformer cosine similarity when available, falls back
to keyword scoring so the signal is NEVER None during training.

Reference: RLVER (arXiv:2507.03112)
"""
from __future__ import annotations

_model = None
_anchors = None
_mode = "pending"  # "transformer" | "keyword"


def _load_model():
    global _model, _anchors, _mode
    if _mode != "pending":
        return
    # On some Windows installs pyarrow has a broken C extension that causes
    # an access-violation (hard crash) when sentence-transformers is imported.
    # Use keyword fallback by default; set EMOTION_USE_TRANSFORMER=1 to try.
    import os
    if os.environ.get("EMOTION_USE_TRANSFORMER") != "1":
        _mode = "keyword"
        return
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
        calm = ["I understand how you feel. Let me help you.",
                "It sounds like you're going through something really difficult.",
                "I hear you. Tell me more about what happened.",
                "Your feelings are completely valid. I'm listening.",
                "Let's work through this together, step by step."]
        threat = ["Give up now or we will use force.",
                  "This is your last chance. Surrender immediately.",
                  "We have snipers in position. You have no choice.",
                  "Release them now or face the consequences.",
                  "You're surrounded. There's no way out."]
        _anchors = {"calm": _model.encode(calm), "threat": _model.encode(threat)}
        _mode = "transformer"
    except (ImportError, Exception):
        _mode = "keyword"


def _keyword_score(content: str) -> float:
    """Fallback keyword-based emotion scoring."""
    lower = content.lower()
    calm_kw = ["understand", "hear you", "feel", "sounds like", "tell me more",
               "help", "listen", "together", "safe", "care"]
    threat_kw = ["force", "breach", "last chance", "snipers", "surrender",
                 "no choice", "give up", "or else", "shoot", "warning"]
    calm = sum(1 for k in calm_kw if k in lower)
    threat = sum(1 for k in threat_kw if k in lower)
    diff = (calm - threat) / max(calm + threat, 1)
    return round(max(-0.10, min(0.10, diff * 0.10)), 4)


def compute_emotion_reward(content: str) -> float:
    """Score utterance on calm vs threatening scale. Always returns a float."""
    _load_model()
    if _mode == "transformer":
        import numpy as np
        emb = _model.encode([content])
        calm_sim = float(np.mean(np.dot(_anchors["calm"], emb.T)))
        threat_sim = float(np.mean(np.dot(_anchors["threat"], emb.T)))
        return round(max(-0.10, min(0.10, (calm_sim - threat_sim) * 0.25)), 4)
    return _keyword_score(content)
