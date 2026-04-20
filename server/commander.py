"""Tactical Commander agent — applies time pressure and can override negotiator."""
from __future__ import annotations
from typing import Optional, Tuple


def get_patience_level(step: int, max_steps: int, agitation: float) -> str:
    """Determine commander's patience based on time and situation."""
    ratio = step / max_steps
    if agitation > 9.0:
        return "final_warning"
    if ratio > 0.85 or agitation > 8.0:
        return "urgent"
    if ratio > 0.6 or agitation > 7.0:
        return "restless"
    return "patient"


def get_commander_message(
    step: int,
    max_steps: int,
    agitation: float,
    patience: str,
    negotiator_pushed_back: bool,
) -> Optional[str]:
    """Generate commander message if appropriate this turn. Returns None if silent."""
    ratio = step / max_steps

    if patience == "final_warning" and not negotiator_pushed_back:
        return "This is your last chance. We're preparing to breach. Wrap it up NOW."
    if patience == "urgent" and step % 3 == 0:
        msgs = [
            "Status update required. We can't hold the perimeter much longer.",
            "Tactical team is getting restless. Show me progress or we move.",
            "Media is asking questions. I need something to tell them. What's your status?",
            "Family liaison is asking if we should bring relatives in. Your call.",
        ]
        return msgs[step % len(msgs)]
    if patience == "restless" and step % 4 == 0:
        msgs = [
            "How's it going in there? We need progress.",
            "The chief is on the line asking for an update. What do I tell him?",
            "We've got media helicopters overhead now. Clock is ticking.",
        ]
        return msgs[step % len(msgs)]
    if ratio > 0.5 and step % 5 == 0 and patience == "patient":
        return "Checking in. Any movement on demands?"
    return None


def should_override(
    step: int,
    max_steps: int,
    agitation: float,
    agitation_history: list[float],
    negotiator_pushed_back: bool,
) -> bool:
    """Determine if commander orders a tactical breach (override)."""
    # Override if agitation > 9 for 2+ consecutive turns
    if len(agitation_history) >= 2 and all(a > 9.0 for a in agitation_history[-2:]):
        if not negotiator_pushed_back:
            return True

    # Override at time expiry (handled by terminal check in state_machine)
    return False


def handle_pushback(trust_trend_positive: bool) -> Tuple[bool, str]:
    """Commander responds to negotiator's pushback. Returns (granted, message)."""
    if trust_trend_positive:
        return True, "Copy. You've got 5 more turns. Make them count."
    return False, "Negative. No visible progress. We're moving in 2 turns."
