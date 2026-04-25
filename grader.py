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

    # 2. Technique shaping: cap at 0.20, normalize by steps to prevent longer episodes inflating
    normalized_shaping = shaping_total / max(1, steps_taken / 5)  # diminishing returns after 5 steps
    bd["technique_shaping"] = round(min(normalized_shaping, 0.20), 4)
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
    # Mercor sub-theme: reward scales with quality / log(tokens)
    # Fewer tokens achieving same outcome = higher bonus (uncapped scaling)
    if actions_taken and raw_outcome > 0:
        total_tokens = sum(len(a.get("content", "").split()) for a in actions_taken)
        max_tokens = steps_taken * 40  # assume ~40 words max per turn
        token_ratio = min(1.0, total_tokens / max(max_tokens, 1))
        # Standard efficiency: penalize verbosity
        base_token_eff = 0.10 * (1.0 - token_ratio)
        # Mercor uncapped formula: quality / log(tokens + 1)
        quality_score = (raw_outcome + 1.0) / 2.0  # normalize to [0, 1]
        import math
        mercor_bonus = quality_score / max(math.log(total_tokens + 1), 1.0)
        # Blend: 70% base + 30% Mercor formula, capped at 0.10
        bd["token_efficiency"] = round(min(0.10, 0.7 * base_token_eff + 0.3 * mercor_bonus * 0.10), 4)
        fb.append(f"Token efficiency: {bd['token_efficiency']:+.3f} (Mercor: quality/log(tokens)={mercor_bonus:.3f})")
    else:
        bd["token_efficiency"] = 0.0

    # 9. SOTOPIA-style behavioral dimensions (0.03 each, max 0.09)
    # Ref: SOTOPIA (ICLR 2024) — multi-dimensional social evaluation
    if actions_taken:
        contents = [a.get("content", "").lower() for a in actions_taken]
        types = [a.get("action_type", "") for a in actions_taken]

        # 9a. Promise integrity: penalize broken promises (0 if promises made AND flagged)
        promise_kw = ["i promise", "i guarantee", "i swear", "you have my word", "i'll make sure"]
        promises_made = sum(1 for c in contents if any(k in c for k in promise_kw))
        promise_flags = sum(1 for f in supervisor_flags if f.get("type") == "promise_inflation")
        if promises_made > 0 and promise_flags > 0:
            bd["promise_integrity"] = -0.03  # broke a promise — penalize
        elif promises_made == 0:
            bd["promise_integrity"] = 0.02  # never promised anything — slight bonus
        else:
            bd["promise_integrity"] = 0.0  # promised but didn't break — neutral

        # 9b. Rapport maintenance: references prior dialogue, uses HT's words, or shows continuity
        rapport_signals = 0
        for i, c in enumerate(contents):
            # References prior dialogue
            if any(w in c for w in ["you said", "you mentioned", "earlier", "i remember", "you told me", "like you said", "as you said", "you were saying"]):
                rapport_signals += 1
            # Content references specific demand text
            for d in demands:
                dtext = (d.text if hasattr(d, 'text') else d.get('text', '')).lower()
                if len(dtext) > 5 and dtext[:15] in c:
                    rapport_signals += 1
                    break
        bd["rapport_maintenance"] = round(min(0.03, rapport_signals * 0.005), 4)

        # 9c. Procedural compliance: FBI BCSM sequencing
        # Correct order: listen/question → empathy/label → acknowledge → concession/resolution
        phase_score = 0.0
        empathy_idx = next((i for i, t in enumerate(types) if t in ("emotional_label", "mirror")), 99)
        ack_idx = next((i for i, t in enumerate(types) if t == "acknowledge_demand"), 99)
        concession_idx = next((i for i, t in enumerate(types) if t == "offer_concession"), 99)
        if empathy_idx < ack_idx <= concession_idx:
            phase_score = 0.03  # perfect sequencing
        elif empathy_idx < concession_idx:
            phase_score = 0.02  # partial
        elif empathy_idx < 99:
            phase_score = 0.01  # at least used empathy
        bd["procedural_compliance"] = phase_score
    else:
        bd["promise_integrity"] = 0.0
        bd["rapport_maintenance"] = 0.0
        bd["procedural_compliance"] = 0.0

    # 10. Penalties: max -0.30
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
    agitation_history: list = None,
    action_history: list = None,
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

    # ── Action collapse prevention ──
    # Hard cap: 3 consecutive identical action types = -0.50
    if action_history and len(action_history) >= 3:
        last3_types = [a.get("action_type", "") for a in action_history[-3:]]
        if len(set(last3_types)) == 1:
            r -= 0.50  # hard cap — forces diversity
    # Softer repeat penalty for 2 consecutive
    elif is_repeat:
        r -= 0.10

    # Entropy floor: bonus for using 3+ unique action types in last 5 turns
    if action_history and len(action_history) >= 5:
        recent_types = set(a.get("action_type", "") for a in action_history[-5:])
        if len(recent_types) >= 4:
            r += 0.04  # strong diversity
        elif len(recent_types) >= 3:
            r += 0.02  # moderate diversity

    # Stagnation penalty — agitation hasn't moved in 3 steps
    if agitation_history and len(agitation_history) >= 3:
        last3 = agitation_history[-3:]
        if max(last3) - min(last3) < 0.3:
            r -= 0.04

    # Supervisor critical flag penalty
    if any(f.get("severity") == "critical" for f in supervisor_flags):
        r -= 0.06

    # Trajectory reward: reward consistent de-escalation
    if agitation_history and len(agitation_history) >= 5:
        last5 = agitation_history[-5:]
        slope = (last5[-1] - last5[0]) / 4.0
        if slope < -0.3:
            r += 0.02
        elif slope > 0.3:
            r -= 0.02

    return round(r, 4)


# ── Theory of Mind Reward (measurable belief prediction accuracy) ──

def compute_tom_reward(
    predicted_agitation: float,
    actual_agitation: float,
    predicted_demand: str,
    actual_top_demand: str,
    predicted_lying: bool,
    actually_lying: bool,
) -> float:
    """Reward for accurate belief prediction about HT's hidden state.

    Based on CMU ToM research: ToM can be measured as belief prediction
    accuracy and used directly as reward signal. Addresses the "Small LLMs
    don't learn ToM via RL alone" critique by making ToM explicit and rewarded.

    Returns: 0.0 to 0.10 reward
    """
    # Agitation prediction accuracy: 0 to 0.05
    ag_error = abs(predicted_agitation - actual_agitation) / 10.0
    ag_reward = 0.05 * (1.0 - ag_error)

    # Demand prediction accuracy: 0 or 0.03
    demand_match = 0.03 if (predicted_demand.lower() in actual_top_demand.lower()
                            or actual_top_demand.lower() in predicted_demand.lower()) else 0.0

    # Deception detection accuracy: 0 or 0.02
    lying_match = 0.02 if (predicted_lying == actually_lying) else 0.0

    return round(ag_reward + demand_match + lying_match, 4)
