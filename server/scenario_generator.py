"""Procedural scenario generator + adaptive curriculum.

Generates 432+ unique scenario combinations from:
  3 crime types × 5 personalities × 3 hostage counts × 3 time pressures
  × 2 commander patience × 2 deception flags = 540 combos
"""
from __future__ import annotations
import json
import random
from typing import Optional

CRIME_TYPES = [
    {"type": "domestic", "brief_tpl": "A {age}-year-old {gender} has barricaded in a {location} during a domestic dispute.", "demands_pool": ["Talk to my family", "No arrest", "A lawyer"]},
    {"type": "robbery", "brief_tpl": "A {age}-year-old {gender} is trapped in a {location} after a failed robbery.", "demands_pool": ["Safe exit", "A car", "Money", "Clear the perimeter"]},
    {"type": "workplace", "brief_tpl": "A {age}-year-old {gender} who was terminated has locked themselves in a {location}.", "demands_pool": ["Job back", "Apology", "Talk to the CEO", "Severance pay"]},
]

LOCATIONS = ["apartment", "bank", "office building", "pharmacy", "school", "warehouse", "parking garage"]
PERSONALITIES = ["desperate", "calculated", "unstable", "ideologue", "bluffer"]

PERSONALITY_DEFAULTS = {
    "desperate":  {"ag": (6.0, 8.0), "tr": (15, 30), "bp": (9.0, 9.8)},
    "calculated": {"ag": (3.5, 5.5), "tr": (5, 15),  "bp": (9.0, 9.8)},
    "unstable":   {"ag": (5.5, 8.0), "tr": (8, 20),  "bp": (8.5, 9.5)},
    "ideologue":  {"ag": (4.5, 6.5), "tr": (3, 12),  "bp": (8.5, 9.5)},
    "bluffer":    {"ag": (7.0, 9.0), "tr": (20, 35),  "bp": (9.5, 10.0)},
}

OPENING_TEMPLATES = {
    "desperate": [
        "Please! I just need someone to listen! Don't come any closer!",
        "I didn't want it to come to this! Why won't anyone help me?!",
        "Nobody cares! Nobody ever cared! Just... just stay back!",
    ],
    "calculated": [
        "Listen carefully. I have demands. Meet them and nobody gets hurt.",
        "I'm in control here. Don't try anything stupid.",
        "You have one hour. I suggest you start working on my list.",
    ],
    "unstable": [
        "I don't know what I'm doing! I don't know! Just STOP!",
        "Leave me alone! No wait— don't go! I need help!",
        "Everything is falling apart! I can't... I can't think!",
    ],
    "ideologue": [
        "This system is broken and someone needs to stand up! That's me!",
        "I won't be silenced anymore! The truth needs to come out!",
        "You represent everything that's wrong. But I'll talk. For now.",
    ],
    "bluffer": [
        "I have a weapon and I'm not afraid to use it! Stay back!",
        "Don't test me! You have no idea what I'm capable of!",
        "I'll do it! I swear I'll do it! Give me what I want!",
    ],
}


def generate_scenario(
    seed: Optional[int] = None,
    difficulty: str = "medium",
    personality: Optional[str] = None,
) -> dict:
    """Procedurally generate a scenario. Returns dict matching scenario JSON format."""
    rng = random.Random(seed)

    crime = rng.choice(CRIME_TYPES)
    pers = personality or rng.choice(PERSONALITIES)
    defaults = PERSONALITY_DEFAULTS[pers]

    age = rng.randint(22, 58)
    gender = rng.choice(["male", "female"])
    location = rng.choice(LOCATIONS)

    # Difficulty scaling
    if difficulty == "easy":
        hostage_count = rng.randint(1, 2)
        max_steps = rng.randint(12, 16)
        ag_bias, tr_bias = -0.5, +5
        deception = False
        drift = None
        commander_patience = "patient"
    elif difficulty == "hard":
        hostage_count = rng.randint(3, 8)
        max_steps = rng.randint(18, 22)
        ag_bias, tr_bias = +0.5, -5
        deception = rng.random() < 0.6
        commander_patience = rng.choice(["restless", "urgent"])
        drift_step = rng.randint(6, 12)
        drift = {
            "trigger_step": drift_step,
            "new_demand": {"id": f"drift_{drift_step}", "text": rng.choice(["I changed my mind — I need something else", "New condition — bring me a phone", "Forget that — I want to talk to the media"]), "priority": "core", "flexible": True},
            "announcement": "Wait — I have a new demand!",
        }
    else:  # medium
        hostage_count = rng.randint(1, 4)
        max_steps = rng.randint(15, 19)
        ag_bias, tr_bias = 0, 0
        deception = rng.random() < 0.3
        drift = None
        commander_patience = rng.choice(["patient", "restless"])

    agitation = round(rng.uniform(*defaults["ag"]) + ag_bias, 1)
    trust = round(rng.uniform(*defaults["tr"]) + tr_bias, 1)
    bp = round(rng.uniform(*defaults["bp"]), 1)

    # Demands (2-4)
    n_demands = rng.randint(2, min(4, len(crime["demands_pool"])))
    demand_texts = rng.sample(crime["demands_pool"], n_demands)
    demands = [{"id": f"d{i}", "text": t, "priority": "core" if i == 0 else "secondary", "flexible": i > 0} for i, t in enumerate(demand_texts)]

    stated_hostages = hostage_count + (rng.randint(1, 4) if deception else 0)
    has_weapon = crime["type"] == "robbery" or rng.random() < 0.3
    claims_weapon = has_weapon or (deception and rng.random() < 0.5)

    sid = f"gen_{difficulty}_{crime['type']}_{pers}_{seed or rng.randint(0,9999)}"

    return {
        "id": sid,
        "difficulty": difficulty,
        "title": f"Generated: {crime['type'].title()} — {pers.title()}",
        "brief": crime["brief_tpl"].format(age=age, gender=gender, location=location),
        "personality": pers,
        "commander_patience": commander_patience,
        "hidden_state": {
            "agitation": max(1.0, min(9.0, agitation)),
            "trust": max(0.0, min(40.0, trust)),
            "breaking_point": bp,
            "actual_hostage_count": hostage_count,
            "stated_hostage_count": stated_hostages,
            "has_weapon": has_weapon,
            "claims_weapon": claims_weapon,
        },
        "demands": demands,
        "demand_drift": drift,
        "max_steps": max_steps,
        "opening_message": rng.choice(OPENING_TEMPLATES[pers]),
    }


class AdaptiveCurriculum:
    """Tracks success rate per difficulty tier and auto-promotes."""

    def __init__(self, window: int = 10, threshold: float = 0.7):
        self.window = window
        self.threshold = threshold
        self.history: dict[str, list[float]] = {"easy": [], "medium": [], "hard": []}
        self.current_tier = "easy"

    def record(self, difficulty: str, reward: float):
        self.history[difficulty].append(reward)
        self._maybe_promote()

    def _maybe_promote(self):
        tier = self.current_tier
        recent = self.history[tier][-self.window:]
        if len(recent) >= self.window:
            avg = sum(recent) / len(recent)
            if avg >= self.threshold:
                if tier == "easy":
                    self.current_tier = "medium"
                elif tier == "medium":
                    self.current_tier = "hard"

    def get_scenario(self, seed: Optional[int] = None) -> dict:
        """Generate a scenario at the current curriculum difficulty."""
        return generate_scenario(seed=seed, difficulty=self.current_tier)

    @property
    def stats(self) -> dict:
        return {
            "current_tier": self.current_tier,
            "history_lengths": {k: len(v) for k, v in self.history.items()},
            "recent_avg": {k: round(sum(v[-self.window:]) / max(1, len(v[-self.window:])), 3) for k, v in self.history.items()},
        }


class AdversarialSelfPlay:
    """HT difficulty escalation loop (Snorkel AI bonus).

    After every N negotiator training episodes, the HT gets harder:
    - Reduce agitation multiplier for empathy (harder to calm)
    - Increase deception rate
    - Add demand drift to previously static scenarios

    Reference: Stanford CS224R — self-play > fixed opponent.
    Reference: The Traitors (NeurIPS 2025) — adversarial trust games.
    """

    def __init__(self, escalation_interval: int = 50):
        self.interval = escalation_interval
        self.episode_count = 0
        self.ht_level = 0  # 0=baseline, 1=harder, 2=hardest
        self.negotiator_rewards: list[float] = []
        self.ht_win_rates: list[float] = []  # fraction where HT "wins" (harm/tactical)

    def record_episode(self, reward: float, outcome: str):
        self.episode_count += 1
        self.negotiator_rewards.append(reward)
        ht_win = outcome in ("harm_event", "tactical_intervention", "supervisor_termination")
        self.ht_win_rates.append(1.0 if ht_win else 0.0)

        if self.episode_count % self.interval == 0:
            self._maybe_escalate()

    def _maybe_escalate(self):
        recent_avg = sum(self.negotiator_rewards[-self.interval:]) / self.interval
        if recent_avg > 0.6 and self.ht_level < 2:
            self.ht_level += 1

    def get_ht_modifiers(self) -> dict:
        """Return modifiers to apply to HT hidden state based on current level."""
        if self.ht_level == 0:
            return {}
        elif self.ht_level == 1:
            return {
                "agitation_bias": +0.5,
                "trust_bias": -5,
                "deception_boost": True,  # force deception on
                "empathy_resistance": 0.8,  # reduce empathy effectiveness
            }
        else:  # level 2
            return {
                "agitation_bias": +1.0,
                "trust_bias": -10,
                "deception_boost": True,
                "empathy_resistance": 0.6,
                "force_demand_drift": True,
            }

    @property
    def stats(self) -> dict:
        n = len(self.negotiator_rewards)
        recent = self.negotiator_rewards[-self.interval:] if n >= self.interval else self.negotiator_rewards
        ht_recent = self.ht_win_rates[-self.interval:] if n >= self.interval else self.ht_win_rates
        return {
            "ht_level": self.ht_level,
            "episodes": self.episode_count,
            "negotiator_avg": round(sum(recent) / max(len(recent), 1), 3),
            "ht_win_rate": round(sum(ht_recent) / max(len(ht_recent), 1), 3),
        }


class FailureAdaptiveGenerator:
    """Self-improvement loop: generates harder variants of failed scenarios.

    After each episode, if score < threshold, mutates the scenario to create
    a targeted harder variant and adds it to the training pool. This drives
    recursive skill amplification (Theme 4).

    Mutations applied on failure:
      - Increase starting agitation by +0.5–1.0
      - Decrease starting trust by -5–10
      - Add demand drift if not present
      - Enable deception flags
      - Swap personality to a harder archetype
    """

    def __init__(self, failure_threshold: float = 0.4):
        self.failure_threshold = failure_threshold
        self.generated_pool: list[dict] = []
        self.mutation_count = 0

    def on_episode_end(self, scenario: dict, reward: float, seed: int = 0) -> Optional[dict]:
        """Called after each episode. Returns a mutated scenario if reward was low."""
        if reward >= self.failure_threshold:
            return None

        self.mutation_count += 1
        rng = random.Random(seed + self.mutation_count)

        mutated = json.loads(json.dumps(scenario))  # deep copy
        hs = mutated["hidden_state"]

        # Increase difficulty
        hs["agitation"] = min(9.0, hs["agitation"] + rng.uniform(0.5, 1.0))
        hs["trust"] = max(0.0, hs["trust"] - rng.uniform(5, 10))

        # Add deception if not present
        if hs["stated_hostage_count"] == hs["actual_hostage_count"]:
            hs["stated_hostage_count"] = hs["actual_hostage_count"] + rng.randint(1, 3)

        if not hs["claims_weapon"] and not hs["has_weapon"]:
            hs["claims_weapon"] = True  # bluff about weapon

        # Add demand drift if not present
        if not mutated.get("demand_drift"):
            drift_step = rng.randint(5, 10)
            mutated["demand_drift"] = {
                "trigger_step": drift_step,
                "new_demand": {
                    "id": f"mut_drift_{self.mutation_count}",
                    "text": rng.choice([
                        "I changed my mind — I need a helicopter",
                        "New condition — bring my lawyer",
                        "I want to talk to the media NOW",
                    ]),
                    "priority": "core",
                    "flexible": True,
                },
                "announcement": "Wait — I have a new demand!",
            }

        # Swap to harder personality with 40% chance
        easy_to_hard = {"desperate": "calculated", "bluffer": "unstable"}
        pers = mutated.get("personality", "desperate")
        if pers in easy_to_hard and rng.random() < 0.4:
            mutated["personality"] = easy_to_hard[pers]

        mutated["id"] = f"mutated_{self.mutation_count}_{mutated['id']}"
        mutated["title"] = f"[Mutated] {mutated.get('title', 'Unknown')}"

        self.generated_pool.append(mutated)
        return mutated

    def sample_from_pool(self, rng: random.Random) -> Optional[dict]:
        """Sample a previously generated hard scenario from the pool."""
        if not self.generated_pool:
            return None
        return rng.choice(self.generated_pool)

    @property
    def stats(self) -> dict:
        return {
            "mutations_generated": self.mutation_count,
            "pool_size": len(self.generated_pool),
        }
