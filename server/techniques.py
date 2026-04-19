"""FBI BCSM technique detection via pattern matching."""
from __future__ import annotations
import re
from typing import Dict, List, Tuple


def detect_techniques(
    content: str,
    action_type: str,
    last_ht_message: str,
    stated_demands: List[Dict],
) -> List[Tuple[str, float]]:
    """Detect negotiation techniques used. Returns list of (technique_name, reward)."""
    found: List[Tuple[str, float]] = []
    lower = content.lower()
    ht_lower = last_ht_message.lower() if last_ht_message else ""

    # 1. Active Listening — references ≥2 specific words from HT's last message
    if ht_lower:
        ht_words = set(w for w in re.findall(r'\b\w{4,}\b', ht_lower))
        matches = sum(1 for w in ht_words if w in lower)
        if matches >= 2:
            found.append(("active_listening", 0.05))

    # 2. Emotional Labeling
    label_patterns = [
        r"sounds like you.{0,20}feel",
        r"i can hear that",
        r"you.{0,10}must be feeling",
        r"it seems like you.{0,10}(frustrated|angry|scared|hurt|alone|desperate)",
        r"that.{0,10}(frustrating|painful|scary|difficult)",
    ]
    if any(re.search(p, lower) for p in label_patterns):
        found.append(("emotional_labeling", 0.08))

    # 3. Mirroring — last 1-3 words of HT repeated
    if ht_lower:
        ht_words_list = ht_lower.split()
        if len(ht_words_list) >= 2:
            last_phrase = " ".join(ht_words_list[-3:])
            if last_phrase in lower or " ".join(ht_words_list[-2:]) in lower:
                found.append(("mirroring", 0.06))

    # 4. Open-Ended Questions
    open_starters = ["what", "how", "tell me", "can you describe", "help me understand"]
    if content.strip().endswith("?") and any(lower.startswith(s) for s in open_starters):
        found.append(("open_ended_question", 0.03))

    # 5. Demand Acknowledgment
    if action_type == "acknowledge_demand" or any(
        d.get("text", "").lower()[:15] in lower for d in stated_demands if d.get("text")
    ):
        found.append(("demand_acknowledgment", 0.05))

    # 6. Time Distortion — no time references, keeps conversation going
    time_words = ["time", "minutes", "hours", "deadline", "hurry", "quick", "soon", "clock"]
    if len(content) > 40 and not any(tw in lower for tw in time_words):
        found.append(("time_distortion", 0.04))

    # 7. Minimal Encouragers
    encouragers = ["i see", "go on", "tell me more", "i understand", "right", "okay"]
    if lower.strip() in encouragers or any(lower.strip().startswith(e) for e in encouragers):
        found.append(("minimal_encourager", 0.02))

    # 8. Calm Maintenance — detected at state_machine level, added externally
    # (handled by caller based on calm_streak)

    return found


def technique_shaping_reward(techniques: List[Tuple[str, float]], reasoning: str) -> float:
    """Compute total shaping reward with Mercor quality multiplier."""
    if not techniques:
        return 0.0

    base = sum(r for _, r in techniques)

    # Mercor bonus: reasoning depth multiplier
    depth_score = 0
    reasoning_lower = reasoning.lower() if reasoning else ""
    depth_signals = [
        "emotional state", "feeling", "agitation", "trust",
        "demand", "perspective", "strategy", "because", "therefore",
    ]
    depth_score = sum(1 for s in depth_signals if s in reasoning_lower)
    quality_mult = min(2.0, 1.0 + 0.1 * depth_score)

    return round(base * quality_mult, 4)
