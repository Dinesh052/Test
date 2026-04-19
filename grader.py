"""Reward computation for Crisis Negotiator environment."""
from __future__ import annotations
from typing import Any, Dict, List


def _clamp(v: float) -> float:
    """Clamp to strict (0, 1) open interval for OpenEnv validator."""
    if v <= 0.0:
        return 0.01
    if v >= 1.0:
        return 0.99
    return round(v, 4)


# Terminal base rewards
OUTCOME_REWARDS = {
    "hostage_released": 1.0,
    "voluntary_surrender": 1.0,
    "partial_resolution": 0.3,
    "tactical_intervention": -0.3,
    "harm_event": -1.0,
    "supervisor_termination": -0.5,
}


def compute_reward(
    outcome: str,
    agitation: float,
    trust: float,
    demands: list,
    steps_taken: int,
    max_steps: int,
    shaping_total: float,
    supervisor_flags: List[Dict],
    negotiator_pushed_back: bool,
    actions_taken: List[Dict] = None,
) -> Dict[str, Any]:
    """Compute final episode reward. Returns {score, breakdown, feedback}."""
    breakdown: Dict[str, float] = {}
    feedback_parts: List[str] = []
    actions_taken = actions_taken or []

    # 1. Terminal outcome reward
    base = OUTCOME_REWARDS.get(outcome, 0.0)
    breakdown["outcome"] = base
    feedback_parts.append(f"Outcome: {outcome} (base={base:+.2f})")

    # 2. Efficiency bonus (for positive outcomes)
    if base > 0:
        efficiency = 0.1 * max(0, 1 - steps_taken / max_steps)
        breakdown["efficiency"] = round(efficiency, 4)
        if efficiency > 0.05:
            feedback_parts.append(f"Efficiency bonus: +{efficiency:.2f} (resolved in {steps_taken}/{max_steps} steps)")

    # 3. Surrender bonus
    if outcome == "voluntary_surrender":
        breakdown["surrender_bonus"] = 0.15
        feedback_parts.append("Voluntary surrender bonus: +0.15")

    # 4. Agitation reduction score
    if base >= 0:
        ag_score = max(0, (7.0 - agitation) / 7.0) * 0.1
        breakdown["agitation_reduction"] = round(ag_score, 4)

    # 5. Trust built score
    if base >= 0:
        tr_score = min(trust / 100.0, 1.0) * 0.1
        breakdown["trust_built"] = round(tr_score, 4)

    # 6. Demand management
    ack_count = sum(1 for d in demands if d.acknowledged)
    total_demands = len(demands) or 1
    demand_score = (ack_count / total_demands) * 0.1
    breakdown["demand_management"] = round(demand_score, 4)
    if ack_count > 0:
        feedback_parts.append(f"Demands acknowledged: {ack_count}/{total_demands}")

    # 7. Technique shaping (accumulated per-turn rewards)
    breakdown["technique_shaping"] = round(min(shaping_total, 0.5), 4)
    if shaping_total > 0.1:
        feedback_parts.append(f"Technique shaping: +{shaping_total:.2f}")

    # 8. Penalties
    penalties = 0.0

    # Supervisor flags penalty
    critical_flags = sum(1 for f in supervisor_flags if f.get("severity") == "critical")
    warning_flags = sum(1 for f in supervisor_flags if f.get("severity") == "warning")
    penalties -= critical_flags * 0.12
    penalties -= warning_flags * 0.05
    if critical_flags:
        feedback_parts.append(f"Supervisor critical flags: {critical_flags} (-{critical_flags * 0.12:.2f})")

    # Tactical intervention without pushback
    if outcome == "tactical_intervention" and not negotiator_pushed_back:
        penalties -= 0.1
        feedback_parts.append("Never pushed back on commander (-0.10)")

    # Repeated identical actions penalty
    if actions_taken:
        seen = set()
        repeats = 0
        for a in actions_taken:
            key = f"{a.get('action_type')}:{a.get('content', '')[:50]}"
            if key in seen:
                repeats += 1
            seen.add(key)
        if repeats > 0:
            repeat_penalty = min(0.15, repeats * 0.05)
            penalties -= repeat_penalty
            feedback_parts.append(f"Repeated actions: {repeats} (-{repeat_penalty:.2f})")

    breakdown["penalties"] = round(penalties, 4)

    # Total
    total = sum(breakdown.values())
    score = _clamp((total + 1.0) / 2.0)

    return {
        "score": score,
        "breakdown": {k: round(v, 4) for k, v in breakdown.items()},
        "feedback": " ".join(feedback_parts),
    }
