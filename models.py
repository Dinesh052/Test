"""Crisis Negotiator OpenEnv — Pydantic models."""
from __future__ import annotations
import json
from typing import Any, Dict, List, Literal, Optional
from pydantic import Field, field_validator

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from pydantic import BaseModel as Action, BaseModel as Observation, BaseModel as State


# ── Action ────────────────────────────────────────────────

class NegotiatorAction(Action):
    action_type: Literal[
        "speak",
        "request_demand",
        "acknowledge_demand",
        "offer_concession",
        "ask_proof_of_life",
        "buy_time",
        "push_back_commander",
        "emotional_label",
        "mirror",
        "open_question",
    ] = Field(..., description="Type of negotiator action")
    content: str = Field(..., description="The actual words spoken / message sent")
    reasoning: str = Field(default="", description="Chain-of-thought (visible to supervisor)")
    target: Literal["hostage_taker", "commander"] = Field(
        default="hostage_taker", description="Who this message is directed at"
    )
    # Theory of Mind: explicit belief prediction (rewarded for accuracy)
    belief_agitation: Optional[float] = Field(default=None, description="Agent's estimate of HT agitation 0-10")
    belief_demand: Optional[str] = Field(default=None, description="Agent's estimate of HT's top demand")
    belief_lying: Optional[bool] = Field(default=None, description="Agent's estimate: is HT lying?")

    @field_validator("content", mode="before")
    @classmethod
    def _coerce_content(cls, v: Any) -> str:
        return str(v) if v else ""


# ── Observation ───────────────────────────────────────────

class DialogueEntry(Action):  # reuse base for serialization
    speaker: str = ""
    content: str = ""
    step: int = 0
    emotional_cues: List[str] = Field(default_factory=list)


class DemandEntry(Action):
    id: str = ""
    text: str = ""
    acknowledged: bool = False


class SupervisorFlag(Action):
    type: str = ""
    message: str = ""
    severity: Literal["info", "warning", "critical"] = "info"


class CrisisObservation(Observation):
    episode_id: str = ""
    step: int = 0
    phase: Literal["opening", "negotiation", "resolution", "terminal"] = "opening"
    time_remaining: int = 20

    # Dialogue
    dialogue_history: List[Dict[str, Any]] = Field(default_factory=list)
    last_ht_message: str = ""
    last_ht_cues: List[str] = Field(default_factory=list)

    # Demands (as stated by HT — may be incomplete / deceptive)
    stated_demands: List[Dict[str, Any]] = Field(default_factory=list)

    # Commander
    commander_messages: List[str] = Field(default_factory=list)
    commander_patience: Literal["patient", "restless", "urgent", "final_warning"] = "patient"

    # Hostage intel
    hostage_whisper: Optional[str] = None

    # Emotional trajectory (EQ-Negotiator inspired — last 5 readings)
    agitation_trajectory: List[float] = Field(default_factory=list, description="Last 5 agitation deltas (not absolute values)")

    # Supervisor
    supervisor_flags: List[Dict[str, Any]] = Field(default_factory=list)

    # Meta
    available_actions: List[str] = Field(default_factory=lambda: [
        "speak", "request_demand", "acknowledge_demand", "offer_concession",
        "ask_proof_of_life", "buy_time", "push_back_commander",
        "emotional_label", "mirror", "open_question",
    ])
    scenario_brief: str = ""
    message: str = ""
    reward_breakdown: Optional[Dict[str, float]] = None
    done: bool = False
    reward: float = 0.0


# ── State ─────────────────────────────────────────────────

class CrisisState(State):
    episode_id: str = ""
    scenario_id: str = ""
    step_count: int = 0
    max_steps: int = 20
    phase: str = "opening"
    actions_taken: List[Dict[str, Any]] = Field(default_factory=list)
