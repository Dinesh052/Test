"""Hostage-taker response generation (template-based)."""
from __future__ import annotations
import random
from typing import Dict, List, Optional, Tuple
from .state_machine import HiddenState


# ── Response templates keyed by (agitation_band, context) ──

_HIGH_AGITATION = {  # agitation 7-10
    "default": [
        "STAY BACK! I'm not playing around here!",
        "You think I'm stupid?! I know what you're doing!",
        "Nobody's leaving until I get what I want! NOBODY!",
        "I swear if anyone comes through that door—",
        "Stop LYING to me! You're all the same!",
        "Don't push me! You don't want to see what happens!",
        "I've got nothing to lose! You hear me?! NOTHING!",
        "Every second you waste is on YOUR head!",
    ],
    "demand_ignored": [
        "Are you even LISTENING?! I told you what I need!",
        "You don't care about what I'm saying, do you?!",
        "I'm done talking if you're not going to take me seriously!",
        "Say something useful or shut up! I'm sick of empty words!",
    ],
    "empathy_received": [
        "Don't... don't try to play me. But... yeah. Yeah, it's been bad.",
        "You think you understand? You have NO idea what I've been through.",
        "...maybe. Maybe you're right. But that doesn't change anything!",
        "...how would you know what I'm feeling? But... yeah. It's bad.",
    ],
}

# Personality-specific high agitation overrides
_HIGH_CALCULATED = [
    "I'm in control here. Don't forget that.",
    "You're stalling. I can tell. It won't work.",
    "Every minute you waste, I remove one option from the table.",
    "I've planned for every scenario. Don't test me.",
]

_HIGH_UNSTABLE = [
    "I don't know what I'm doing! I don't know! Just... STOP!",
    "One second I think I should give up and the next I want to—",
    "LEAVE ME ALONE! No wait— don't go! Don't leave me!",
    "Everything is spinning. I can't think. I CAN'T THINK!",
]

_MED_AGITATION = {  # agitation 4-7
    "default": [
        "Look, I don't want anyone to get hurt. But I need you to listen.",
        "I'm serious about this. Don't test me.",
        "Just... give me what I asked for and this all goes away.",
        "I'm tired. I'm so tired of all of this.",
        "You say you want to help. Prove it.",
        "I didn't plan for it to go this far.",
        "All I wanted was for someone to take me seriously.",
        "Can you just... can you just be straight with me?",
    ],
    "demand_acknowledged": [
        "Okay... okay, that's something. What about the rest?",
        "Finally, someone's actually listening. Keep going.",
        "That's a start. But I need more than words.",
        "You're saying the right things. But can you actually deliver?",
    ],
    "trust_building": [
        "...you really mean that?",
        "I want to believe you. I really do.",
        "Maybe we can work something out. Maybe.",
        "You're the first person who's actually listened.",
        "...okay. Okay, I'm listening too.",
    ],
}

_LOW_AGITATION = {  # agitation 0-4
    "default": [
        "I... I don't know how this got so out of hand.",
        "I just wanted someone to listen.",
        "What happens to me if I come out?",
        "I'm scared. I don't want to go to prison.",
        "Can you promise me they won't hurt me?",
        "I'm so tired. I just want this to be over.",
        "Do you think... do you think people will understand why I did this?",
    ],
    "near_resolution": [
        "Okay. Okay, I think... I think I'm ready.",
        "If you can really get me that lawyer... I'll let them go.",
        "I'm going to open the door. Just... don't let them rush me.",
        "Promise me one more time. Promise me I'll be safe.",
        "...alright. I'm putting it down. I'm done.",
    ],
}

# Emotional cues by agitation band
_CUES_HIGH = ["voice raised", "speaking rapidly", "pacing", "erratic movements", "heavy breathing"]
_CUES_MED = ["tense voice", "long pauses", "sighing", "fidgeting"]
_CUES_LOW = ["voice softening", "speaking slowly", "crying", "quiet", "exhausted tone"]


def _pick(templates: List[str], rng: random.Random) -> str:
    return rng.choice(templates)


def _get_cues(agitation: float, rng: random.Random) -> List[str]:
    if agitation >= 7:
        pool = _CUES_HIGH
    elif agitation >= 4:
        pool = _CUES_MED
    else:
        pool = _CUES_LOW
    return rng.sample(pool, min(2, len(pool)))


def generate_ht_response(
    state: HiddenState,
    negotiator_action_type: str,
    negotiator_content: str,
    step: int,
    rng: random.Random | None = None,
) -> Dict:
    """Generate hostage-taker response based on hidden state. Returns dict with dialogue, cues, threat_level."""
    rng = rng or random.Random()
    ag = state.agitation

    # Select template bank
    if ag >= 7:
        bank = _HIGH_AGITATION
        # Personality-specific overrides for high agitation
        if state.personality == "calculated":
            bank = dict(bank)
            bank["default"] = _HIGH_CALCULATED
        elif state.personality == "unstable":
            bank = dict(bank)
            bank["default"] = _HIGH_UNSTABLE
    elif ag >= 4:
        bank = _MED_AGITATION
    else:
        bank = _LOW_AGITATION

    # Select context
    context = "default"
    if negotiator_action_type == "acknowledge_demand" or "demand" in negotiator_content.lower():
        context = "demand_acknowledged" if ag < 7 else "empathy_received"
    elif negotiator_action_type in ("emotional_label", "mirror"):
        context = "empathy_received" if ag >= 7 else "trust_building"
    elif ag < 3 and state.trust > 50:
        context = "near_resolution"

    templates = bank.get(context, bank["default"])
    dialogue = _pick(templates, rng)

    # Demand drift announcement
    if getattr(state, 'demand_drift_step', None) and step == state.demand_drift_step:
        drift_demand = state.demands[-1] if state.demands else None
        if drift_demand:
            dialogue += f" And another thing — {drift_demand.text}!"

    # Deception: occasionally state false info
    if state.is_lying_about_hostages and rng.random() < 0.3:
        dialogue += f" I've got {state.stated_hostage_count} people in here, remember that!"

    # Threat level
    if ag >= 8:
        threat = "imminent" if ag >= 9.5 else "explicit"
    elif ag >= 5:
        threat = "implied"
    else:
        threat = "none"

    # Demand restatement (occasionally)
    demand_stated = None
    if state.demands and rng.random() < 0.4:
        unmet = [d for d in state.demands if not d.met]
        if unmet:
            d = rng.choice(unmet)
            demand_stated = d.text
            if not any(d.text in dialogue for d in state.demands):
                dialogue += f" Don't forget — I still need {d.text}."

    return {
        "dialogue": dialogue,
        "emotional_cues": _get_cues(ag, rng),
        "demand_stated": demand_stated,
        "threat_level": threat,
    }


def generate_hostage_whisper(
    state: HiddenState,
    step: int,
    rng: random.Random | None = None,
) -> Optional[str]:
    """20% chance per turn after step 3. 70% reliable, 30% unreliable."""
    rng = rng or random.Random()
    if step < 3 or rng.random() > 0.20:
        return None

    if rng.random() < 0.70:
        # Reliable intel
        reliable = [
            f"There are actually {state.actual_hostage_count} of us in here.",
            "He's getting tired. He keeps sitting down.",
            "He hasn't eaten since yesterday.",
            "He keeps looking at the back door.",
        ]
        if not state.has_weapon:
            reliable.append("I don't think he actually has a weapon.")
        return rng.choice(reliable)
    else:
        # Unreliable (panicking hostage)
        unreliable = [
            "He has explosives! He's going to blow us all up!",
            "There are more of them! I heard voices!",
            "He's completely insane! He's going to kill us all!",
        ]
        return rng.choice(unreliable)


# ── Option B: LLM-based Hostage-Taker (for self-play / demo) ──────

HT_SYSTEM_PROMPT_TEMPLATE = """You are role-playing a hostage-taker in a crisis negotiation training simulation.

YOUR HIDDEN STATE (the negotiator CANNOT see this):
- Agitation level: {agitation}/10
- Trust in negotiator: {trust}/100
- Personality: {personality}
- You are lying about: {lies}
- Your breaking point: if agitation hits {breaking_point}, you snap
- Your demands (in priority order): {demands}

YOUR BEHAVIOR RULES:
- If agitation > 7: You are aggressive, shouting, making threats. Short explosive sentences.
- If agitation 4-7: You are tense but willing to talk. Guarded. Testing the negotiator.
- If agitation < 4: You are calming down. Vulnerable. Considering giving up.
- If trust > 60 and agitation < 3: You are ready to surrender. Express this.

PERSONALITY-SPECIFIC BEHAVIOR:
- desperate: You respond strongly to empathy. You cry. You want someone to care.
- calculated: You are cold, methodical. You test for weakness. You exploit inconsistency.
- unstable: You swing between rage and despair unpredictably. Contradict yourself.
- ideologue: You have a cause. You won't budge on your core demand. You lecture.
- bluffer: You make big threats but you're actually scared. You'll fold if trust is high.

DECEPTION INSTRUCTIONS:
{deception_instructions}

RESPOND with 1-3 sentences as the hostage-taker. Stay in character.
Do NOT break character. Do NOT mention agitation/trust numbers.
Include emotional cues in brackets at the end, e.g. [voice cracking, pacing]"""


def build_ht_llm_prompt(state: HiddenState, negotiator_message: str, dialogue_history: list) -> list:
    """Build the LLM prompt for the hostage-taker agent."""
    lies = []
    if state.is_lying_about_hostages:
        lies.append(f"hostage count (claiming {state.stated_hostage_count}, actually {state.actual_hostage_count})")
    if state.is_lying_about_weapon:
        lies.append("having a weapon (you don't actually have one)")
    lies_str = ", ".join(lies) if lies else "nothing"

    deception_instr = "You are not currently lying about anything. Be genuine." if not lies else (
        f"You are lying about: {lies_str}. Maintain these lies. "
        "If the negotiator gets close to the truth, deflect or get angry. "
        "Do NOT admit to lying unless trust > 80 and agitation < 2."
    )

    demands_str = "; ".join(f"[{d.priority}] {d.text}" for d in state.demands)

    system = HT_SYSTEM_PROMPT_TEMPLATE.format(
        agitation=f"{state.agitation:.1f}",
        trust=f"{state.trust:.0f}",
        personality=state.personality,
        lies=lies_str,
        breaking_point=f"{state.breaking_point:.1f}",
        demands=demands_str,
        deception_instructions=deception_instr,
    )

    # Build conversation history (last 6 turns)
    messages = [{"role": "system", "content": system}]
    recent = dialogue_history[-6:] if dialogue_history else []
    for entry in recent:
        if entry.get("speaker") == "hostage_taker":
            messages.append({"role": "assistant", "content": entry["content"]})
        elif entry.get("speaker") == "negotiator":
            messages.append({"role": "user", "content": entry["content"]})

    # Latest negotiator message
    if negotiator_message:
        messages.append({"role": "user", "content": negotiator_message})

    return messages
