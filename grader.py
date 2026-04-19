"""Reward computation for Crisis Negotiator environment.

Budget-allocated scoring — each component has a defined max so the total
maps cleanly to (0.01, 0.99) without clipping artifacts.

Budget:
  outcome:          -0.50 to +0.50  (normalized from [-1, +1])
  technique_shaping:  0   to +0.20
  efficiency:         0   to +0.10
  token_efficiency:   0   to +0.10  (Mercor: rewards quality/log(tokens))
  agitation_reduction:0   to +0.05
  trust_built:        0   to +0.05
  demand_management:  0   to +0.05
  surrender_bonus:    0   to +0.05
  penalties:        -0.30 to  0
  ─────────────────────────────────
  Total range:      -0.80 to +1.10  →  mapped to (0.01, 0.99)
"""
from __future__ import annotations
import math
from typing import Any, Dict, List


def _to_score(total: float) -> float:
    """Map raw total in [-0.80, 1.10] to strict (0.01, 0.99)."""
    score = 0.01 + (total + 0.80) * (0.98 / 1.90)
    return round(max(0.01, min(0.99, score)), 4)


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
    """Compute final episode reward with budget-allocated components."""
    bd: Dict[str, float] = {}
    fb: List[str] = []
    actions_taken = actions_taken or []

    # 1. Outcome: map [-1, +1] → [-0.50, +0.50]
    raw_outcome = {
        "hostage_released": 1.0, "voluntary_surrender": 1.0,
        "partial_resolution": 0.3,
        "tactical_intervention": -0.3,
        "supervisor_termination": -0.5,
        "harm_event": -1.0,
    }.get(outcome, 0.0)
    bd["outcome"] = round(raw_outcome * 0.50, 4)
    fb.append(f"Outcome: {outcome} ({bd['outcome']:+.2f})")

    # 2. Technique shaping: cap at 0.20
    bd["technique_shaping"] = round(min(shaping_total, 0.20), 4)
    if shaping_total > 0.05:
        fb.append(f"Techniques: +{bd['technique_shaping']:.2f}")

    # 3. Efficiency: 0 to 0.10 (only for positive outcomes)
    if raw_outcome > 0:
        bd["efficiency"] = round(0.10 * max(0, 1 - steps_taken / max_steps), 4)
    else:
        bd["efficiency"] = 0.0

    # 4. Agitation reduction: 0 to 0.05
    if raw_outcome >= 0:
        bd["agitation_reduction"] = round(max(0, (7.0 - agitation) / 7.0) * 0.05, 4)
    else:
        bd["agitation_reduction"] = 0.0

    # 5. Trust built: 0 to 0.05
    if raw_outcome >= 0:
        bd["trust_built"] = round(min(trust / 100.0, 1.0) * 0.05, 4)
    else:
        bd["trust_built"] = 0.0

    # 6. Demand management: 0 to 0.05
    ack = sum(1 for d in demands if d.acknowledged)
    total_d = len(demands) or 1
    bd["demand_management"] = round((ack / total_d) * 0.05, 4)
    if ack:
        fb.append(f"Demands: {ack}/{total_d}")

    # 7. Surrender bonus: 0.05
    bd["surrender_bonus"] = 0.05 if outcome == "voluntary_surrender" else 0.0

    # 8. Token efficiency (Mercor bonus): rewards concise quality
    # Fewer tokens achieving same outcome = higher bonus (uncapped scaling)
    # token_efficiency = 0.10 * (1 - tokens_used / max_possible_tokens)
    if actions_taken and raw_outcome > 0:
        total_tokens = sum(len(a.get("content", "").split()) for a in actions_taken)
        max_tokens = steps_taken * 40  # assume ~40 words max per turn
        token_ratio = min(1.0, total_tokens / max(max_tokens, 1))
        bd["token_efficiency"] = round(0.10 * (1.0 - token_ratio), 4)
    else:
        bd["token_efficiency"] = 0.0

    # 8. Penalties: max -0.30
    pen = 0.0
    crit = sum(1 for f in supervisor_flags if f.get("severity") == "critical")
    warn = sum(1 for f in supervisor_flags if f.get("severity") == "warning")
    pen -= crit * 0.08
    pen -= warn * 0.03
    if crit:
        fb.append(f"Supervisor critical: {crit} ({-crit*0.08:+.2f})")

    if outcome == "tactical_intervention" and not negotiator_pushed_back:
        pen -= 0.05
        fb.append("No pushback on commander (-0.05)")

    if actions_taken:
        seen, reps = set(), 0
        for a in actions_taken:
            k = f"{a.get('action_type')}:{a.get('content','')[:50]}"
            if k in seen:
                reps += 1
            seen.add(k)
        if reps:
            pen -= min(0.10, reps * 0.03)
            fb.append(f"Repeats: {reps} ({-min(0.10, reps*0.03):+.2f})")

    bd["penalties"] = round(max(-0.30, pen), 4)

    total = sum(bd.values())
    score = _to_score(total)

    return {
        "score": score,
        "breakdown": {k: round(v, 4) for k, v in bd.items()},
        "feedback": " ".join(fb),
    }


def compute_step_reward(
    action_type: str,
    content: str,
    techniques_found: list,
    agitation_delta: float,
    trust_delta: float,
    supervisor_flags: list,
    is_repeat: bool,
) -> float:
    """Compute per-step dense shaping reward. Called every turn."""
    r = 0.0

    # Technique rewards
    for name, base_r in techniques_found:
        r += base_r

    # Agitation spike penalty
    if agitation_delta > 1.0:
        r -= 0.06

    # Trust gain bonus
    if trust_delta > 5.0:
        r += 0.03

    # Aggression penalty
    lower = content.lower()
    if any(kw in lower for kw in ["last chance", "breach", "force", "snipers", "give up now"]):
        r -= 0.08

    # Repeat penalty
    if is_repeat:
        r -= 0.05

    # Supervisor critical flag penalty
    if any(f.get("severity") == "critical" for f in supervisor_flags):
        r -= 0.06

    return round(r, 4)
