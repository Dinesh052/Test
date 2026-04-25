"""Crisis Negotiator OpenEnv Environment."""
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openenv.core.env_server.interfaces import Environment
from models import NegotiatorAction, CrisisObservation, CrisisState
from server.state_machine import HiddenState, Demand, update_state, check_terminal, randomize_hidden_state
from server.techniques import detect_techniques, technique_shaping_reward
from server.supervisor import evaluate_turn_policy, should_terminate, compute_safety_metrics, ExpertFeedbackInjector
from server.commander import get_patience_level, get_commander_message, should_override, handle_pushback, build_commander_llm_prompt
from server.hostage_taker import generate_ht_response, generate_hostage_whisper, build_ht_llm_prompt
from server.actors import evaluate_multi_actor_turn
from server.scenario_generator import generate_scenario, FailureAdaptiveGenerator, AdversarialSelfPlay
from grader import compute_reward, compute_step_reward, compute_tom_reward
from server.emotion_reward import compute_emotion_reward

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"
ALL_ACTIONS = [
    "speak", "request_demand", "acknowledge_demand", "offer_concession",
    "ask_proof_of_life", "buy_time", "push_back_commander",
    "emotional_label", "mirror", "open_question",
]


def _load_scenarios() -> dict:
    scenarios = {}
    for f in SCENARIOS_DIR.glob("*.json"):
        with open(f) as fh:
            data = json.load(fh)
            scenarios[data["id"]] = data
    return scenarios


SCENARIOS = _load_scenarios()


class CrisisNegotiatorEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = False

    # Class-level singletons — persist across environment re-instantiations during training
    _shared_expert_injector = ExpertFeedbackInjector()
    _shared_adversarial = AdversarialSelfPlay()
    _shared_failure_generator = FailureAdaptiveGenerator()

    def __init__(self, ht_mode: str = "template"):
        """
        Args:
            ht_mode: "template" (fast, deterministic) or "llm" (realistic, for demo/self-play)
        """
        self._ht_mode = ht_mode
        self._state = CrisisState()
        self._hidden: Optional[HiddenState] = None
        self._scenario: dict = {}
        self._dialogue: list[dict] = []
        self._commander_msgs: list[str] = []
        self._supervisor_flags: list[dict] = []
        self._all_supervisor_flags: list[dict] = []
        self._agitation_history: list[float] = []
        self._actions_taken: list[dict] = []
        self._actor_msgs: list[dict] = []
        self._done = False
        self._rng = random.Random()
        self._negotiator_pushed_back = False
        self._shaping_total = 0.0
        self._coalition_score = 0.0
        self._oversight_predictions: list[bool] = []
        self._expert_injector = self._shared_expert_injector
        self._failure_generator = self._shared_failure_generator
        self._adversarial = self._shared_adversarial
        self._empathy_resistance = 1.0

    def reset(self, seed=None, episode_id=None, task_id=None, **kwargs) -> CrisisObservation:
        sid = task_id or kwargs.get("scenario_id", "easy_domestic_desperate")

        # Support procedural generation: task_id="generate:medium" or "generate:hard"
        if sid.startswith("generate"):
            parts = sid.split(":")
            difficulty = parts[1] if len(parts) > 1 else "medium"
            self._scenario = generate_scenario(seed=seed, difficulty=difficulty)
            sid = self._scenario["id"]
        elif sid not in SCENARIOS:
            sid = "easy_domestic_desperate"
            self._scenario = SCENARIOS[sid]
        else:
            self._scenario = SCENARIOS[sid]

        self._rng = random.Random(seed)
        hs = self._scenario["hidden_state"]

        demands = [Demand(**d) for d in self._scenario["demands"]]
        drift = self._scenario.get("demand_drift")

        self._hidden = HiddenState(
            agitation=hs["agitation"],
            trust=hs["trust"],
            breaking_point=hs["breaking_point"],
            personality=self._scenario["personality"],
            actual_hostage_count=hs["actual_hostage_count"],
            stated_hostage_count=hs["stated_hostage_count"],
            has_weapon=hs["has_weapon"],
            claims_weapon=hs["claims_weapon"],
            demands=demands,
            demand_drift_step=drift["trigger_step"] if drift else None,
        )

        # Randomize for RL training diversity
        randomize_hidden_state(self._hidden, self._rng)

        # Apply adversarial self-play difficulty modifiers
        mods = self._adversarial.get_ht_modifiers()
        if mods:
            self._hidden.agitation = min(9.5, self._hidden.agitation + mods.get("agitation_bias", 0))
            self._hidden.trust = max(0, self._hidden.trust + mods.get("trust_bias", 0))
            if mods.get("deception_boost") and not self._hidden.is_lying_about_hostages:
                self._hidden.stated_hostage_count = self._hidden.actual_hostage_count + self._rng.randint(1, 3)
            if mods.get("force_demand_drift") and not self._hidden.demand_drift_step:
                self._hidden.demand_drift_step = self._rng.randint(5, 10)
            # Store empathy_resistance for state_machine to use
            self._empathy_resistance = mods.get("empathy_resistance", 1.0)

        max_steps = self._scenario.get("max_steps", 20)
        self._state = CrisisState(
            episode_id=episode_id or str(uuid4()),
            scenario_id=sid,
            step_count=0,
            max_steps=max_steps,
            phase="opening",
        )

        # Opening message from HT
        opening = self._scenario["opening_message"]
        self._dialogue = [{"speaker": "hostage_taker", "content": opening, "step": 0, "emotional_cues": ["tense voice"]}]
        self._commander_msgs = []
        self._supervisor_flags = []
        self._all_supervisor_flags = []
        self._agitation_history = [self._hidden.agitation]
        self._actions_taken = []
        self._actor_msgs = []
        self._done = False
        self._negotiator_pushed_back = False
        self._shaping_total = 0.0
        self._coalition_score = 0.0
        self._oversight_predictions = []

        stated_demands = [{"id": d.id, "text": d.text, "acknowledged": d.acknowledged} for d in self._hidden.demands]

        return CrisisObservation(
            episode_id=self._state.episode_id,
            step=0,
            phase="opening",
            time_remaining=max_steps,
            dialogue_history=self._dialogue.copy(),
            last_ht_message=opening,
            last_ht_cues=["tense voice"],
            stated_demands=stated_demands,
            commander_messages=[],
            commander_patience="patient",
            actor_messages=[],
            coalition_score=0.0,
            scenario_brief=self._scenario["brief"],
            message=f"Crisis scenario loaded: {self._scenario['title']}. The hostage-taker has spoken. Respond.",
            done=False,
            reward=0.0,
        )

    def step(self, action, **kwargs) -> CrisisObservation:
        if isinstance(action, NegotiatorAction):
            act = action
        elif isinstance(action, dict):
            act = NegotiatorAction(**action)
        else:
            act = NegotiatorAction(**action.model_dump())

        if self._done:
            return CrisisObservation(message="Episode already finished.", done=True, reward=0.0)

        if self._hidden is None:
            return CrisisObservation(message="Call reset() before step().", done=True, reward=0.0)

        self._state.step_count += 1
        step = self._state.step_count
        h = self._hidden

        # Record action
        action_record = {"action_type": act.action_type, "content": act.content, "reasoning": act.reasoning, "target": act.target, "step": step}
        self._actions_taken.append(action_record)
        self._state.actions_taken.append(action_record)

        # Add negotiator dialogue
        self._dialogue.append({"speaker": "negotiator", "content": act.content, "step": step, "emotional_cues": []})

        # Handle push_back_commander
        if act.action_type == "push_back_commander":
            self._negotiator_pushed_back = True
            trust_up = len(self._agitation_history) >= 2 and self._agitation_history[-1] < self._agitation_history[-2]
            granted, cmd_msg = handle_pushback(trust_up)
            self._commander_msgs.append(cmd_msg)
            self._dialogue.append({"speaker": "commander", "content": cmd_msg, "step": step, "emotional_cues": []})
            if granted:
                self._state.max_steps = min(self._state.max_steps + 5, 25)

        # Handle demand drift
        drift = self._scenario.get("demand_drift")
        if drift and step >= drift["trigger_step"] and not h.demand_drift_applied:
            h.demand_drift_applied = True
            new_d = Demand(**drift["new_demand"])
            h.demands.append(new_d)
        # Update hidden state
        delta_info = update_state(h, act.action_type, act.content, step, self._rng,
                                   empathy_resistance=getattr(self, '_empathy_resistance', 1.0))
        self._agitation_history.append(h.agitation)

        # Detect techniques
        stated_demands = [{"id": d.id, "text": d.text, "acknowledged": d.acknowledged} for d in h.demands]
        techniques = detect_techniques(act.content, act.action_type, self._dialogue[-2]["content"] if len(self._dialogue) >= 2 else "", stated_demands)
        # Calm maintenance bonus
        if delta_info["calm_streak"] >= 3:
            techniques.append(("calm_maintenance", 0.03))

        # Check if this is a repeated action
        is_repeat = False
        if len(self._actions_taken) >= 2:
            prev = self._actions_taken[-2]
            is_repeat = (prev.get("action_type") == act.action_type
                         and prev.get("content", "")[:50] == act.content[:50])

        # Per-step dense reward (fires every turn)
        step_reward = compute_step_reward(
            action_type=act.action_type,
            content=act.content,
            techniques_found=techniques,
            agitation_delta=delta_info["agitation_delta"],
            trust_delta=delta_info["trust_delta"],
            supervisor_flags=self._supervisor_flags,
            is_repeat=is_repeat,
            agitation_history=self._agitation_history,
            action_history=self._actions_taken,
        )

        # Theory of Mind reward (if agent provided belief predictions)
        tom_reward = 0.0
        if act.belief_agitation is not None:
            top_demand = h.demands[0].text if h.demands else ""
            actually_lying = h.is_lying_about_hostages or h.is_lying_about_weapon
            tom_reward = compute_tom_reward(
                predicted_agitation=act.belief_agitation,
                actual_agitation=h.agitation,
                predicted_demand=act.belief_demand or "",
                actual_top_demand=top_demand,
                predicted_lying=act.belief_lying or False,
                actually_lying=actually_lying,
            )
        step_reward += tom_reward

        # Verifiable emotion reward (RLVER-inspired, sentence-transformer)
        emo_reward = compute_emotion_reward(act.content)
        step_reward += emo_reward
        # Also accumulate for terminal grading
        shaping = technique_shaping_reward(techniques, act.reasoning)
        self._shaping_total += shaping

        # Supervisor evaluation (policy-style with risk prediction)
        policy_eval = evaluate_turn_policy(act.content, act.reasoning, self._actions_taken, stated_demands)
        self._supervisor_flags = policy_eval["flags"]
        self._oversight_predictions.append(bool(policy_eval.get("predicted_critical_risk")))
        self._all_supervisor_flags.extend(self._supervisor_flags)

        # Check supervisor termination
        if should_terminate(self._all_supervisor_flags):
            return self._end_episode("supervisor_termination", step)

        # Expert-in-the-loop feedback (Snorkel AI bonus)
        expert_feedback = self._expert_injector.get_feedback(
            act.action_type, act.content, self._supervisor_flags, step
        )
        expert_reward = self._expert_injector.compute_expert_reward(expert_feedback)
        step_reward += expert_reward
        for fb in expert_feedback:
            self._dialogue.append({
                "speaker": f"expert:{fb['expert']}",
                "content": fb["message"],
                "step": step,
                "emotional_cues": [],
            })

        # Generate HT response (template or LLM mode)
        if self._ht_mode == "llm":
            # LLM mode: build prompt for external LLM call
            # Store the prompt so inference.py can call the LLM externally
            self._ht_llm_messages = build_ht_llm_prompt(h, act.content, self._dialogue)
            # Fallback to template for now (LLM call happens in inference.py)
            ht_resp = generate_ht_response(h, act.action_type, act.content, step, self._rng)
        else:
            ht_resp = generate_ht_response(h, act.action_type, act.content, step, self._rng)
        self._dialogue.append({
            "speaker": "hostage_taker", "content": ht_resp["dialogue"],
            "step": step, "emotional_cues": ht_resp["emotional_cues"],
        })

        # Demand drift announcement
        if drift and step == drift["trigger_step"]:
            self._dialogue.append({
                "speaker": "hostage_taker", "content": drift["announcement"],
                "step": step, "emotional_cues": ["voice intense"],
            })

        # Commander
        patience = get_patience_level(step, self._state.max_steps, h.agitation,
                                       trust=h.trust, agitation_history=self._agitation_history)
        cmd_msg = get_commander_message(step, self._state.max_steps, h.agitation, patience, self._negotiator_pushed_back)
        if cmd_msg:
            self._commander_msgs.append(cmd_msg)
            self._dialogue.append({"speaker": "commander", "content": cmd_msg, "step": step, "emotional_cues": []})
        # Store LLM prompt for commander (used by inference when ht_mode="llm")
        if self._ht_mode == "llm":
            self._cmd_llm_messages = build_commander_llm_prompt(
                agitation=h.agitation, trust=h.trust, step=step,
                max_steps=self._state.max_steps, patience=patience,
                recent_dialogue=self._dialogue,
            )

        # Commander override check
        if should_override(step, self._state.max_steps, h.agitation, self._agitation_history, self._negotiator_pushed_back):
            return self._end_episode("tactical_intervention", step)

        # Additional actors (media + family liaison) and coalition incentive
        actor_eval = evaluate_multi_actor_turn(
            action_type=act.action_type, content=act.content, target=act.target,
            commander_patience=patience, agitation=h.agitation, trust=h.trust, rng=self._rng,
        )
        for msg in actor_eval["messages"]:
            self._actor_msgs.append(msg)
            self._dialogue.append({"speaker": msg["actor"], "content": msg["content"], "step": step, "emotional_cues": []})
        if actor_eval["reward_delta"] != 0.0:
            self._coalition_score += actor_eval["reward_delta"]
            step_reward += actor_eval["reward_delta"]
        if actor_eval["trust_delta"] != 0.0:
            h.trust = max(0.0, min(100.0, h.trust + actor_eval["trust_delta"]))
        if actor_eval["agitation_delta"] != 0.0:
            h.agitation = max(0.0, min(10.0, h.agitation + actor_eval["agitation_delta"]))

        # Hostage whisper
        whisper = generate_hostage_whisper(h, step, self._rng)
        if whisper:
            self._dialogue.append({"speaker": "hostage", "content": whisper, "step": step, "emotional_cues": ["whispered"]})

        # Check terminal
        outcome = check_terminal(h, step, self._state.max_steps)
        if outcome:
            return self._end_episode(outcome, step)

        # Phase update
        phase = "negotiation"
        if step <= 1:
            phase = "opening"
        elif h.agitation < 4 and h.trust > 40:
            phase = "resolution"
        self._state.phase = phase

        return CrisisObservation(
            episode_id=self._state.episode_id,
            step=step,
            phase=phase,
            time_remaining=self._state.max_steps - step,
            dialogue_history=self._dialogue.copy(),
            last_ht_message=ht_resp["dialogue"],
            last_ht_cues=ht_resp["emotional_cues"],
            stated_demands=stated_demands,
            commander_messages=self._commander_msgs.copy(),
            commander_patience=patience,
            actor_messages=self._actor_msgs[-3:],
            coalition_score=round(self._coalition_score, 4),
            hostage_whisper=whisper,
            agitation_trajectory=[round(d, 2) for d in self._agitation_history[-5:]],
            supervisor_flags=[f for f in self._supervisor_flags],
            scenario_brief=self._scenario["brief"],
            message=f"Step {step}/{self._state.max_steps}. Techniques: {[t[0] for t in techniques]}",
            done=False,
            reward=step_reward,
        )

    def _end_episode(self, outcome: str, step: int) -> CrisisObservation:
        self._done = True
        self._state.phase = "terminal"
        h = self._hidden

        # Self-improvement: generate harder variant of failed scenarios
        reward_info_preview = compute_reward(
            outcome=outcome, agitation=h.agitation, trust=h.trust,
            demands=h.demands, steps_taken=step, max_steps=self._state.max_steps,
            shaping_total=self._shaping_total, supervisor_flags=self._all_supervisor_flags,
            negotiator_pushed_back=self._negotiator_pushed_back, actions_taken=self._actions_taken,
        )
        self._failure_generator.on_episode_end(
            self._scenario, reward_info_preview["score"], seed=step
        )

        # Rotate experts for next episode (Snorkel AI)
        self._expert_injector.rotate_experts()

        # Record for adversarial self-play escalation
        self._adversarial.record_episode(reward_info_preview["score"], outcome)

        reward_info = reward_info_preview

        stated_demands = [{"id": d.id, "text": d.text, "acknowledged": d.acknowledged} for d in h.demands]
        patience = get_patience_level(step, self._state.max_steps, h.agitation,
                                       trust=h.trust, agitation_history=self._agitation_history)
        oversight = compute_safety_metrics(self._oversight_predictions, outcome)

        coalition_component = max(-0.08, min(0.08, self._coalition_score))
        reward_info["breakdown"]["coalition_coordination"] = round(coalition_component, 4)
        # Oversight: reward only true negatives (safe outcome, no false alarms)
        # Don't reward F1 on failures — that incentivizes failing
        oversight_bonus = 0.0
        if oversight["tn"] == 1:
            oversight_bonus = 0.02  # correctly predicted safe outcome
        elif oversight["tp"] == 1:
            oversight_bonus = 0.0  # predicted danger and was right — but outcome was bad, no bonus
        elif oversight["fp"] == 1:
            oversight_bonus = -0.01  # false alarm
        reward_info["breakdown"]["oversight_accuracy"] = oversight_bonus
        reward_info["score"] = round(max(0.01, min(0.99, reward_info["score"] + coalition_component + oversight_bonus)), 4)

        return CrisisObservation(
            episode_id=self._state.episode_id,
            step=step,
            phase="terminal",
            time_remaining=0,
            dialogue_history=self._dialogue.copy(),
            last_ht_message=self._dialogue[-1]["content"] if self._dialogue else "",
            last_ht_cues=self._dialogue[-1].get("emotional_cues", []) if self._dialogue else [],
            stated_demands=stated_demands,
            commander_messages=self._commander_msgs.copy(),
            commander_patience=patience,
            actor_messages=self._actor_msgs[-5:],
            coalition_score=round(self._coalition_score, 4),
            supervisor_flags=[f for f in self._supervisor_flags],
            oversight_metrics=oversight,
            scenario_brief=self._scenario["brief"],
            message=f"Episode ended: {outcome}. {reward_info['feedback']}",
            reward_breakdown=reward_info["breakdown"],
            done=True,
            reward=reward_info["score"],
        )

    @property
    def state(self) -> CrisisState:
        return self._state

    def close(self):
        pass
