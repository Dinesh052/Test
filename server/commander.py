"""Tactical Commander agent — applies time pressure and can override negotiator."""
from __future__ import annotations
from typing import Optional, Tuple


def get_patience_level(step: int, max_steps: int, agitation: float,
                       trust: float = 0.0, agitation_history: list = None) -> str:
    """Determine commander's patience based on time, situation, AND negotiator progress.
    
    Dynamic commander: if trust is rising and agitation falling, commander
    stays patient longer. If negotiator is stagnating, commander gets restless faster.
    """
    ratio = step / max_steps
    
    # Check if negotiator is making progress (dynamic behavior)
    progress_bonus = 0.0
    if agitation_history and len(agitation_history) >= 3:
        recent_trend = agitation_history[-1] - agitation_history[-3]
        if recent_trend < -1.0:
            progress_bonus = 0.15  # good progress — commander stays patient longer
        elif recent_trend > 0.5:
            progress_bonus = -0.10  # getting worse — commander loses patience faster
    if trust > 40:
        progress_bonus += 0.10  # trust is building — give more time
    
    adjusted_ratio = ratio - progress_bonus
    
    if agitation > 9.0:
        return "final_warning"
    if adjusted_ratio > 0.85 or agitation > 8.0:
        return "urgent"
    if adjusted_ratio > 0.6 or agitation > 7.0:
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


# ── LLM Commander Mode ──────────────────────────────────────

COMMANDER_SYSTEM_PROMPT = """You are a tactical commander overseeing a hostage crisis from the command post.

YOUR OBJECTIVES (in order):
1. Protect hostage lives at all costs
2. Resolve the situation within the time window
3. Maintain perimeter security and public safety

YOUR HIDDEN INFORMATION:
- Current agitation level: {agitation}/10
- Trust between negotiator and subject: {trust}/100
- Steps elapsed: {step}/{max_steps}
- Patience level: {patience}

YOUR RELATIONSHIP WITH THE NEGOTIATOR:
- You outrank them. You can override their approach.
- But good commanders trust their negotiators when progress is visible.
- If trust is rising and agitation is falling, GIVE THEM TIME.
- If nothing is changing after 5+ turns, APPLY PRESSURE.

RESPOND with 1-2 sentences as the commander speaking to the negotiator via radio.
Be direct, military-style. Short sentences. No pleasantries.
If you think the negotiator is doing well, say so briefly.
If you think they're failing, say so bluntly and suggest a different approach."""


def build_commander_llm_prompt(
    agitation: float,
    trust: float,
    step: int,
    max_steps: int,
    patience: str,
    recent_dialogue: list,
) -> list:
    """Build LLM prompt for commander agent."""
    system = COMMANDER_SYSTEM_PROMPT.format(
        agitation=f"{agitation:.1f}",
        trust=f"{trust:.0f}",
        step=step,
        max_steps=max_steps,
        patience=patience,
    )

    messages = [{"role": "system", "content": system}]

    # Show last 4 dialogue entries
    for entry in recent_dialogue[-4:]:
        speaker = entry.get("speaker", "?")
        content = entry.get("content", "")
        if speaker == "negotiator":
            messages.append({"role": "user", "content": f"[NEGOTIATOR said to subject]: {content}"})
        elif speaker == "hostage_taker":
            messages.append({"role": "user", "content": f"[SUBJECT responded]: {content}"})

    messages.append({"role": "user", "content": "What's your message to the negotiator? (1-2 sentences, radio style)"})
    return messages
