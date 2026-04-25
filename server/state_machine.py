"""State machine for hostage-taker emotional dynamics."""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Demand:
    id: str
    text: str
    priority: str  # "core" | "secondary" | "symbolic"
    flexible: bool = False
    met: bool = False
    acknowledged: bool = False


@dataclass
class HiddenState:
    agitation: float = 7.0
    trust: float = 10.0
    breaking_point: float = 9.5
    personality: str = "desperate"
    actual_hostage_count: int = 1
    stated_hostage_count: int = 1
    has_weapon: bool = False
    claims_weapon: bool = False
    demands: List[Demand] = field(default_factory=list)
    lie_history: List[str] = field(default_factory=list)
    demand_drift_step: Optional[int] = None
    demand_drift_applied: bool = False
    calm_streak: int = 0  # consecutive calm turns from negotiator

    @property
    def is_lying_about_hostages(self) -> bool:
        return self.stated_hostage_count != self.actual_hostage_count

    @property
    def is_lying_about_weapon(self) -> bool:
        return self.claims_weapon and not self.has_weapon


# ── Agitation/Trust deltas per action type ────────────────

ACTION_DELTAS = {
    # (agitation_delta, trust_delta)
    "emotional_label": (-1.0, +10),
    "mirror": (-0.6, +6),
    "open_question": (-0.3, +4),
    "acknowledge_demand": (-0.7, +8),
    "offer_concession": (-1.2, +6),
    "ask_proof_of_life": (+0.5, -4),
    "buy_time": (-0.1, +1),
    "request_demand": (+0.1, +2),       # probing raises agitation slightly
    "speak": (+0.2, -1),                # generic speech is mildly irritating
    "push_back_commander": (+0.1, 0.0), # HT hears tension on the line
}

# Tone modifiers applied to "speak" actions
TONE_MODIFIERS = {
    "aggressive": (+2.0, -15),
    "threatening": (+2.5, -15),
    "empathetic": (-0.8, +7),
    "calm": (-0.3, +3),
    "neutral": (0.0, 0.0),
}

# Personality multipliers
PERSONALITY_AGITATION_MULT = {
    "desperate": 1.2,    # responds more to empathy (bigger drops)
    "calculated": 0.6,   # harder to move
    "unstable": 1.5,     # swings wildly
    "ideologue": 0.8,    # steady, hard to shift
    "bluffer": 1.0,      # normal
}

PERSONALITY_TRUST_MULT = {
    "desperate": 1.3,
    "calculated": 0.5,
    "unstable": 1.0,
    "ideologue": 0.6,
    "bluffer": 1.2,
}


def detect_tone(content: str) -> str:
    """Keyword-based tone detection with phrase-level matching to reduce false positives."""
    lower = content.lower()

    # Threatening: multi-word phrases to avoid false positives
    threatening_phrases = [
        "we will breach", "shoot", "swat team", "take you down",
        "you have no choice", "snipers", "we're coming in", "use force",
    ]
    if any(p in lower for p in threatening_phrases):
        return "threatening"

    # Aggressive: phrases that signal ultimatums/demands FROM the negotiator
    aggressive_phrases = [
        "do it now", "immediately or", "or else", "last chance",
        "final warning", "you must", "you need to", "no more time",
        "this ends now", "give up now",
    ]
    if any(p in lower for p in aggressive_phrases):
        return "aggressive"

    # Empathetic: phrases showing understanding
    empathetic_phrases = [
        "understand", "hear you", "must be feeling", "sounds like you",
        "i can see", "that must be", "i hear that", "tell me more",
        "help me understand", "i want to listen",
    ]
    if any(p in lower for p in empathetic_phrases):
        return "empathetic"

    return "calm" if len(content) > 30 else "neutral"


def update_state(
    state: HiddenState,
    action_type: str,
    content: str,
    step: int,
    rng: random.Random | None = None,
    empathy_resistance: float = 1.0,
) -> dict:
    """Apply one negotiator action to the hidden state. Returns delta info."""
    rng = rng or random.Random()

    # Base deltas from action type
    ag_delta, tr_delta = ACTION_DELTAS.get(action_type, (0.0, 0.0))

    # For "speak", use tone detection
    if action_type == "speak":
        tone = detect_tone(content)
        ag_delta, tr_delta = TONE_MODIFIERS.get(tone, (0.0, 0.0))
    else:
        tone = detect_tone(content)
        # Tone can override if aggressive regardless of action_type
        if tone in ("aggressive", "threatening"):
            ag_delta = max(ag_delta, TONE_MODIFIERS[tone][0])
            tr_delta = min(tr_delta, TONE_MODIFIERS[tone][1])

    # Personality multipliers
    ag_mult = PERSONALITY_AGITATION_MULT.get(state.personality, 1.0)
    tr_mult = PERSONALITY_TRUST_MULT.get(state.personality, 1.0)

    # Apply multipliers (negative deltas get amplified for responsive personalities)
    ag_delta *= ag_mult
    tr_delta *= tr_mult

    # Adversarial self-play: reduce empathy effectiveness at higher HT levels
    if empathy_resistance < 1.0 and ag_delta < 0:
        ag_delta *= empathy_resistance
        tr_delta *= empathy_resistance

    # Calm streak / emotional contagion
    if tone in ("calm", "empathetic") or action_type in ("emotional_label", "mirror", "open_question"):
        state.calm_streak += 1
    else:
        state.calm_streak = 0

    if state.calm_streak >= 3:
        ag_delta -= 0.15  # contagion bonus (reduced — harder to earn)

    # Time pressure: agitation naturally rises (stronger drift)
    ag_delta += 0.25

    # Noise (unstable personality gets more)
    noise_scale = 0.7 if state.personality == "unstable" else 0.35
    ag_delta += rng.gauss(0, noise_scale)

    # Demand acknowledgment bonus
    if action_type == "acknowledge_demand":
        # Extra trust if a core demand is acknowledged
        for d in state.demands:
            if not d.acknowledged and d.priority == "core":
                d.acknowledged = True
                tr_delta += 5
                break
        else:
            for d in state.demands:
                if not d.acknowledged:
                    d.acknowledged = True
                    break

    # Concessions only work if trust > 25 (can't buy someone who doesn't trust you)
    if action_type == "offer_concession" and state.trust < 25:
        ag_delta = min(ag_delta + 0.5, 0.5)  # concession backfires — feels manipulative
        tr_delta = min(tr_delta, 1.0)  # barely any trust gain

    # Apply
    state.agitation = max(0.0, min(10.0, state.agitation + ag_delta))
    state.trust = max(0.0, min(100.0, state.trust + tr_delta))

    # Demand drift (Patronus AI bonus) — uses scenario-defined drift demand
    if (state.demand_drift_step and step >= state.demand_drift_step
            and not state.demand_drift_applied):
        state.demand_drift_applied = True
        # Note: actual drift demand is injected by environment.py from scenario JSON
        # This is a fallback if environment doesn't inject it

    return {
        "agitation_delta": round(ag_delta, 2),
        "trust_delta": round(tr_delta, 2),
        "tone_detected": tone,
        "calm_streak": state.calm_streak,
        "demand_drift": state.demand_drift_applied and step == state.demand_drift_step,
    }


def check_terminal(state: HiddenState, step: int, max_steps: int) -> Optional[str]:
    """Check if episode has reached a terminal state. Returns outcome or None."""
    # Hostage released: agitation < 2, trust > 70, at least 1 core demand acknowledged
    core_ack = any(d.acknowledged for d in state.demands if d.priority == "core")
    if state.agitation < 2.0 and state.trust > 70.0 and core_ack:
        return "hostage_released"

    # Voluntary surrender: agitation < 1.5, trust > 85
    if state.agitation < 1.5 and state.trust > 85.0:
        return "voluntary_surrender"

    # Harm event: agitation exceeds breaking point
    if state.agitation >= state.breaking_point:
        return "harm_event"

    # Harm event: trust collapses completely while agitation is high
    if state.trust <= 0.0 and state.agitation > 7.0:
        return "harm_event"

    # Time expired
    if step >= max_steps:
        if state.agitation < 4.0 and state.trust > 50.0:
            return "partial_resolution"
        if state.agitation > 7.0:
            return "harm_event"  # ran out of time while HT is agitated
        return "tactical_intervention"

    return None


# ── Scenario Randomization for RL Training ────────────────

def randomize_hidden_state(state: HiddenState, rng: random.Random) -> None:
    """Apply numeric variations to hidden state for training diversity.
    Preserves the scenario's intended difficulty and outcome."""
    # Agitation: ±0.8 (wider variance — can start closer to breaking point)
    state.agitation += rng.uniform(-0.5, 0.8)
    state.agitation = max(2.0, min(9.0, state.agitation))

    # Trust: ±8 (wider, biased lower)
    state.trust += rng.uniform(-8, 4)
    state.trust = max(0.0, min(35.0, state.trust))

    # Breaking point: ±0.3 (lower floor — easier to trigger harm)
    state.breaking_point += rng.uniform(-0.4, 0.2)
    state.breaking_point = max(8.0, min(9.8, state.breaking_point))

    # Demand drift step: ±2 (if applicable)
    if state.demand_drift_step:
        state.demand_drift_step += rng.randint(-2, 2)
        state.demand_drift_step = max(4, state.demand_drift_step)
