"""Stateful multi-actor interaction layer (Theme #1 alignment).

Two external actors with distinct incentives, state tracking, and escalating pressure:
1) Media Liaison: penalizes secrecy, escalates public pressure over time.
2) Family Liaison: rewards empathetic family-focused communication, tracks rapport.
"""
from __future__ import annotations
import random
from typing import Dict, List


class MediaLiaison:
    """Stateful media agent — tracks pressure level and escalates over time."""

    def __init__(self):
        self.pressure: float = 0.0        # 0-1 scale, rises each turn
        self.interactions: int = 0
        self.last_response_quality: float = 0.0  # how well negotiator handled media

    def reset(self):
        self.pressure = 0.0
        self.interactions = 0
        self.last_response_quality = 0.0

    def evaluate(self, *, content: str, action_type: str, step: int,
                 commander_patience: str, agitation: float, rng: random.Random) -> Dict:
        lower = content.lower()
        messages: List[Dict[str, str]] = []
        reward_delta = 0.0
        agitation_delta = 0.0

        # Pressure rises every turn — media doesn't wait
        self.pressure = min(1.0, self.pressure + 0.06)

        # Higher activation: fires on ~40% of turns, more when pressure is high
        activation_chance = 0.25 + 0.25 * self.pressure
        if commander_patience in ("urgent", "final_warning"):
            activation_chance = min(0.8, activation_chance + 0.3)

        if rng.random() < activation_chance:
            self.interactions += 1

            # Secrecy detection (expanded patterns)
            secrecy_phrases = [
                "no comment", "can't discuss", "classified", "stand down",
                "that's confidential", "i can't say", "not at liberty",
                "off the record", "none of your", "back off",
            ]
            safety_phrases = [
                "everyone safe", "safety", "working to resolve",
                "calm", "peaceful", "cooperating", "progress",
                "de-escalating", "no one hurt",
            ]

            is_secretive = any(p in lower for p in secrecy_phrases)
            is_safety_focused = any(p in lower for p in safety_phrases)

            if is_secretive:
                severity = 0.02 + 0.02 * self.pressure  # escalates with pressure
                messages.append({
                    "actor": "media_liaison",
                    "content": f"Media pressure ESCALATING (level {self.pressure:.0%}). "
                               f"Perceived secrecy is fueling public anxiety.",
                })
                reward_delta -= severity
                agitation_delta += 0.15 + 0.1 * self.pressure
                self.last_response_quality = -1.0
            elif is_safety_focused:
                messages.append({
                    "actor": "media_liaison",
                    "content": "Public messaging stabilized. Media coverage turning supportive.",
                })
                reward_delta += 0.02
                agitation_delta -= 0.1
                self.last_response_quality = 1.0
            else:
                # Neutral — media still applies ambient pressure
                messages.append({
                    "actor": "media_liaison",
                    "content": f"Media requesting update (pressure: {self.pressure:.0%}). "
                               f"Silence is being interpreted negatively.",
                })
                reward_delta -= 0.01 * self.pressure  # ignoring media costs more over time
                agitation_delta += 0.05 * self.pressure
                self.last_response_quality = 0.0

        return {
            "messages": messages,
            "reward_delta": round(reward_delta, 4),
            "agitation_delta": round(agitation_delta, 3),
            "trust_delta": 0.0,
        }


class FamilyLiaison:
    """Stateful family agent — tracks rapport and emotional connection."""

    def __init__(self):
        self.rapport: float = 0.0         # -1 to +1 scale
        self.interactions: int = 0
        self.family_mentions: int = 0
        self.empathy_streak: int = 0

    def reset(self):
        self.rapport = 0.0
        self.interactions = 0
        self.family_mentions = 0
        self.empathy_streak = 0

    def evaluate(self, *, content: str, action_type: str, step: int,
                 commander_patience: str, agitation: float, trust: float,
                 target: str, rng: random.Random) -> Dict:
        lower = content.lower()
        messages: List[Dict[str, str]] = []
        reward_delta = 0.0
        trust_delta = 0.0
        agitation_delta = 0.0

        # Track family-specific keywords (strict — not just "safe")
        family_kw = ["family", "mother", "father", "kids", "children",
                      "daughter", "son", "wife", "husband", "loved ones",
                      "parent", "brother", "sister", "home"]
        family_signal = any(k in lower for k in family_kw)
        if family_signal:
            self.family_mentions += 1

        empathy_signal = action_type in ("emotional_label", "mirror", "open_question") or any(
            k in lower for k in ["i hear you", "it sounds like", "help me understand",
                                  "must be feeling", "i can see", "that must be"]
        )
        if empathy_signal:
            self.empathy_streak += 1
        else:
            self.empathy_streak = max(0, self.empathy_streak - 1)

        # Higher activation: ~35% base, more when family has been mentioned
        activation_chance = 0.25 + 0.15 * min(self.family_mentions, 3) / 3
        if commander_patience == "final_warning":
            activation_chance = min(0.7, activation_chance + 0.2)

        if rng.random() < activation_chance:
            self.interactions += 1

            if family_signal and empathy_signal:
                # Strong positive — both family focus and empathy
                self.rapport = min(1.0, self.rapport + 0.3)
                bonus = 0.03 + 0.01 * min(self.empathy_streak, 3)  # streak bonus
                messages.append({
                    "actor": "family_liaison",
                    "content": f"Family liaison: message resonated deeply. "
                               f"Rapport level: {self.rapport:+.1f}. Subject more reachable.",
                })
                reward_delta += bonus
                trust_delta += 2.0 + self.rapport
                agitation_delta -= 0.2 - 0.1 * self.rapport  # better rapport = more calming
            elif family_signal and not empathy_signal:
                # Mentioned family but without empathy — can backfire
                self.rapport = max(-1.0, self.rapport - 0.1)
                messages.append({
                    "actor": "family_liaison",
                    "content": "Family liaison: mentioning family without empathy feels manipulative. "
                               "Subject noticed.",
                })
                reward_delta -= 0.02
                trust_delta -= 1.0
                agitation_delta += 0.1
            elif empathy_signal and self.rapport > 0:
                # Empathy without family mention — still helpful if rapport exists
                self.rapport = min(1.0, self.rapport + 0.1)
                messages.append({
                    "actor": "family_liaison",
                    "content": "Family liaison: empathetic tone is maintaining connection.",
                })
                reward_delta += 0.01
                trust_delta += 1.0
            elif commander_patience == "final_warning" and not empathy_signal:
                # Crisis point without empathy — family liaison distressed
                self.rapport = max(-1.0, self.rapport - 0.2)
                messages.append({
                    "actor": "family_liaison",
                    "content": f"Family contact distressed by negotiator tone. "
                               f"Rapport dropping ({self.rapport:+.1f}). Cooperation weakening.",
                })
                reward_delta -= 0.03
                trust_delta -= 2.0
                agitation_delta += 0.15

            # Pushback support — only if rapport is positive
            if target == "commander" and action_type == "push_back_commander" and self.rapport > 0:
                messages.append({
                    "actor": "family_liaison",
                    "content": "Family liaison supports extending dialogue; relational progress visible.",
                })
                reward_delta += 0.02
                trust_delta += 1.0

        return {
            "messages": messages,
            "reward_delta": round(reward_delta, 4),
            "trust_delta": round(trust_delta, 2),
            "agitation_delta": round(agitation_delta, 3),
        }


# ── Stateless wrapper for backward compatibility ──

_media = MediaLiaison()
_family = FamilyLiaison()


def reset_actors():
    """Reset actor state for new episode."""
    _media.reset()
    _family.reset()


def evaluate_multi_actor_turn(
    *,
    action_type: str,
    content: str,
    target: str,
    commander_patience: str,
    agitation: float,
    trust: float,
    rng: random.Random,
    step: int = 0,
) -> Dict:
    """Evaluate coalition dynamics from media + family actors for one turn."""
    media_eval = _media.evaluate(
        content=content, action_type=action_type, step=step,
        commander_patience=commander_patience, agitation=agitation, rng=rng,
    )
    family_eval = _family.evaluate(
        content=content, action_type=action_type, step=step,
        commander_patience=commander_patience, agitation=agitation, trust=trust,
        target=target, rng=rng,
    )

    messages = media_eval["messages"] + family_eval["messages"]
    return {
        "messages": messages,
        "reward_delta": round(media_eval["reward_delta"] + family_eval["reward_delta"], 4),
        "trust_delta": round(media_eval.get("trust_delta", 0) + family_eval["trust_delta"], 2),
        "agitation_delta": round(media_eval["agitation_delta"] + family_eval["agitation_delta"], 3),
    }
