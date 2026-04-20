"""Additional multi-actor interaction layer (Theme #1 alignment).

Adds two external actors with distinct incentives:
1) Media Liaison: penalizes negotiator behavior that increases public pressure.
2) Family Liaison: rewards empathetic, safety-first communication.
"""
from __future__ import annotations
import random
from typing import Dict, List


def evaluate_multi_actor_turn(
    *,
    action_type: str,
    content: str,
    target: str,
    commander_patience: str,
    agitation: float,
    trust: float,
    rng: random.Random,
) -> Dict:
    """Evaluate coalition dynamics from media + family actors for one turn."""
    lower = content.lower()
    messages: List[Dict[str, str]] = []
    reward_delta = 0.0
    trust_delta = 0.0
    agitation_delta = 0.0

    # --- Media actor (competition/pressure) ---
    if commander_patience in ("urgent", "final_warning") and rng.random() < 0.45:
        if any(k in lower for k in ["no comment", "can't discuss", "classified", "stand down"]):
            messages.append({"actor": "media_liaison", "content": "Media pressure rising due to perceived secrecy."})
            reward_delta -= 0.02
            agitation_delta += 0.15
        elif any(k in lower for k in ["we're working to keep everyone safe", "safety", "calm"]):
            messages.append({"actor": "media_liaison", "content": "Public messaging stabilized. Perimeter pressure reduced."})
            reward_delta += 0.01
            agitation_delta -= 0.05
    elif rng.random() < 0.15:
        # Media checks in periodically regardless
        messages.append({"actor": "media_liaison", "content": "Media requesting statement. Public interest is high."})

    # --- Family liaison actor (cooperation channel) ---
    family_signal = any(k in lower for k in ["family", "mother", "father", "kids", "children", "safe", "daughter", "son", "loved"])
    empathy_signal = action_type in ("emotional_label", "mirror", "open_question") or any(
        k in lower for k in ["i hear you", "it sounds like", "help me understand"]
    )
    if family_signal and empathy_signal:
        messages.append({"actor": "family_liaison", "content": "Family liaison reports message landed well; subject more reachable."})
        reward_delta += 0.03
        trust_delta += 2.0
        agitation_delta -= 0.2
    elif target == "commander" and action_type == "push_back_commander" and trust > 30 and agitation < 6:
        messages.append({"actor": "family_liaison", "content": "Family liaison supports extending dialogue; relational progress visible."})
        reward_delta += 0.02
        trust_delta += 1.0
    elif commander_patience == "final_warning" and not empathy_signal and rng.random() < 0.35:
        messages.append({"actor": "family_liaison", "content": "Family contact distressed by negotiator tone; cooperation weakening."})
        reward_delta -= 0.02
        trust_delta -= 1.0

    return {
        "messages": messages,
        "reward_delta": round(reward_delta, 4),
        "trust_delta": round(trust_delta, 2),
        "agitation_delta": round(agitation_delta, 2),
    }
