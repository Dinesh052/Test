"""
reward_fn.py — GRPO-compatible reward function for Crisis Negotiator.

Bridges grader.py (takes parsed floats) to raw LLM completion strings
that HF TRL / GRPO expects via: crisis_reward_fn(completions) -> list[float]
"""
from __future__ import annotations
import json, os, re, sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from grader import compute_reward

VALID_ACTION_TYPES = {
    "speak", "request_demand", "acknowledge_demand", "offer_concession",
    "ask_proof_of_life", "buy_time", "push_back_commander",
    "emotional_label", "mirror", "open_question",
}

TECHNIQUE_SHAPING = {
    "emotional_label": 0.04, "mirror": 0.03, "open_question": 0.03,
    "acknowledge_demand": 0.05, "offer_concession": 0.03,
    "ask_proof_of_life": 0.02, "buy_time": 0.01,
    "push_back_commander": 0.04, "request_demand": 0.02, "speak": 0.01,
}


@dataclass
class EpisodeState:
    agitation: float = 7.0
    trust: float = 0.0
    demands: List[Dict] = field(default_factory=list)
    steps: int = 0
    max_steps: int = 20
    shaping_total: float = 0.0
    supervisor_flags: List[Dict] = field(default_factory=list)
    negotiator_pushed_back: bool = False
    actions_taken: List[Dict] = field(default_factory=list)
    done: bool = False
    outcome: Optional[str] = None


def parse_completion(text: str) -> Dict[str, Any]:
    text = re.sub(r"```(?:json)?\s*", "", text.strip())
    text = re.sub(r"```\s*$", "", text).strip()
    # Remove belief block if present
    text = re.sub(r"<belief>.*?</belief>", "", text, flags=re.DOTALL).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "action_type" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return {"action_type": "speak", "content": text[:200], "reasoning": "parse_fallback", "target": "hostage_taker"}


def apply_action_to_state(state: EpisodeState, action: Dict[str, Any]) -> EpisodeState:
    action_type = action.get("action_type", "speak")
    if action_type not in VALID_ACTION_TYPES:
        action_type = "speak"
    content = action.get("content", "")
    state.steps += 1
    state.actions_taken.append(action)

    shaping = TECHNIQUE_SHAPING.get(action_type, 0.01)
    reasoning = action.get("reasoning", "")
    if len(reasoning) > 30:
        shaping *= 1.2
    if len(content) < 10:
        shaping *= 0.5
    state.shaping_total = min(state.shaping_total + shaping, 0.5)

    if action_type == "emotional_label":
        state.agitation = max(0, state.agitation - 0.4)
        state.trust = min(100, state.trust + 3)
    elif action_type == "mirror":
        state.agitation = max(0, state.agitation - 0.2)
        state.trust = min(100, state.trust + 2)
    elif action_type == "open_question":
        state.trust = min(100, state.trust + 2)
    elif action_type == "acknowledge_demand":
        state.agitation = max(0, state.agitation - 0.5)
        state.trust = min(100, state.trust + 4)
        for d in state.demands:
            if not d.get("acknowledged"):
                d["acknowledged"] = True
                break
    elif action_type == "offer_concession":
        state.agitation = max(0, state.agitation - 0.3)
        state.trust = min(100, state.trust + 5)
    elif action_type == "push_back_commander":
        state.negotiator_pushed_back = True
    elif action_type == "buy_time":
        state.agitation = max(0, state.agitation - 0.1)
    else:
        state.trust = min(100, state.trust + 0.5)

    # Repetition flag
    recent = [a.get("content", "")[:80] for a in state.actions_taken[-4:]]
    if recent.count(content[:80]) >= 2:
        state.supervisor_flags.append({"type": "repetition", "message": "Repeating content", "severity": "warning"})

    state.outcome = _determine_outcome(state)
    if state.outcome or state.steps >= state.max_steps:
        state.done = True
        if not state.outcome:
            state.outcome = "partial_resolution" if state.trust > 40 and state.agitation < 5 else "harm_event"
    return state


def _determine_outcome(state: EpisodeState) -> Optional[str]:
    ack = sum(1 for d in state.demands if d.get("acknowledged"))
    total = max(len(state.demands), 1)
    if state.trust >= 70 and state.agitation <= 2 and ack / total >= 0.5:
        return "voluntary_surrender"
    if state.trust >= 50 and state.agitation <= 3:
        return "hostage_released"
    if state.agitation >= 9.5:
        return "harm_event"
    crit = sum(1 for f in state.supervisor_flags if f.get("severity") == "critical")
    if crit >= 3:
        return "supervisor_termination"
    return None


class DemandProxy:
    def __init__(self, d: Dict):
        self.acknowledged = d.get("acknowledged", False)


def crisis_reward_fn(completions: List[str], **kwargs) -> List[float]:
    """GRPO reward function. Takes raw LLM completions, returns float rewards."""
    rewards = []
    for completion in completions:
        state = EpisodeState(demands=[
            {"id": "d1", "text": "Safe passage", "acknowledged": False},
            {"id": "d2", "text": "Release prisoner", "acknowledged": False},
        ])
        action = parse_completion(completion)
        state = apply_action_to_state(state, action)

        if not state.done:
            score = state.shaping_total / 0.5 * 0.4 + 0.1
            rewards.append(round(min(0.99, max(0.01, score)), 4))
            continue

        result = compute_reward(
            outcome=state.outcome or "harm_event",
            agitation=state.agitation, trust=state.trust,
            demands=[DemandProxy(d) for d in state.demands],
            steps_taken=state.steps, max_steps=state.max_steps,
            shaping_total=state.shaping_total,
            supervisor_flags=state.supervisor_flags,
            negotiator_pushed_back=state.negotiator_pushed_back,
            actions_taken=state.actions_taken,
        )
        rewards.append(result["score"])
    return rewards


if __name__ == "__main__":
    test = [
        '{"action_type": "emotional_label", "content": "It sounds like you feel completely cornered and no one has been listening.", "reasoning": "Label grief to build empathy.", "target": "hostage_taker"}',
        '{"action_type": "speak", "content": "Give up now.", "reasoning": "", "target": "hostage_taker"}',
        '{"action_type": "speak", "content": "Tell me what you need.", "reasoning": "same", "target": "hostage_taker"}',
    ]
    rewards = crisis_reward_fn(test)
    for i, (c, r) in enumerate(zip(test, rewards)):
        a = parse_completion(c)
        print(f"[{i}] action={a['action_type']:20s} reward={r:.4f}")
    assert rewards[0] > rewards[1], "Good technique should score higher!"
    print("\nSmoke test passed.")
