"""
Co-Evolution v3 — Optimized for Qwen 2.5-3B-Instruct
======================================================
Major improvements over v2_test:
  1. Full score_trajectory (phase-alignment, collapse-penalty, entropy-bonus,
     ack-timing, opening-explore, ToM, Q-network, banned-words, reasoning)
  2. advance_env_with_best: canonical env advances with the best completion
     each GRPO group, so later steps see richer context
  3. Dr. GRPO loss (loss_type="dr_grpo") removes length bias
  4. num_generations=8 for better gradient variance
  5. Curriculum mode: adaptive difficulty + failure-adaptive pool
  6. Hard-bias scheduling: shifts to hard after round 1
  7. Adversarial scenarios injected in rounds 3+
  8. multi_turn_steps=6, blended reward (40/40/20)
  9. Better system prompt: BCSM model, belief_demand field
 10. Better prompt builder: agitation trajectory, demand checkmarks, 6-turn history
 11. HT co-evolution uses real hidden state for richer prompts

Usage:
    python training/train_coevolve_v3.py --rounds 4 --prompts-per-round 128
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import random
import re
import site
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Auto-patch TRL ────────────────────────────────────────────────────────────
def _patch_trl():
    sp = site.getsitepackages()[0]
    cb = os.path.join(sp, "trl", "trainer", "callbacks.py")
    if os.path.exists(cb):
        src = open(cb).read()
        for old, new in [
            ("import weave", "weave = None"),
            ("from weave.trace.context", "# from weave.trace.context"),
            ("import llm_blender", "llm_blender = None"),
        ]:
            src = src.replace(old, new)
        open(cb, "w").write(src)
    mk = os.path.join(sp, "trl", "mergekit_utils.py")
    if os.path.exists(mk):
        src = open(mk).read()
        if "from mergekit" in src:
            src = src.replace(
                "from mergekit.config import MergeConfiguration",
                "MergeConfiguration = None",
            )
            src = src.replace("from mergekit", "# from mergekit")
            open(mk, "w").write(src)

_patch_trl()

_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "unsloth_compiled_cache")
if os.path.exists(_cache):
    shutil.rmtree(_cache, ignore_errors=True)

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from server.environment import CrisisNegotiatorEnvironment
from models import NegotiatorAction
import env_extensions  # noqa — patches fast_rollout, hidden_snapshot, peek_step_reward


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class CFG:
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    # Training
    rounds: int = 4
    prompts_per_round: int = 128
    num_generations: int = 8          # v1 used 4 — more generations = better GRPO gradient
    max_new_tokens: int = 256         # bumped: 3B can handle, richer reasoning
    learning_rate: float = 5e-6
    num_epochs: int = 1
    lora_r: int = 16
    lora_alpha: int = 32
    # Output paths
    neg_output: str = "./coevolve-v3-negotiator"
    ht_output: str = "./coevolve-v3-ht"
    seed: int = 42
    # Trajectory
    multi_turn_steps: int = 6         # v1 used 4
    # Reward blend weights
    w_heuristic: float = 0.40
    w_env: float = 0.40
    # Curriculum
    hard_bias_after_round: int = 1    # switch to hard-biased mix after round 1
    difficulty_mix: List[str] = field(default_factory=lambda: ["easy", "medium", "hard"])
    late_difficulty_mix: List[str] = field(
        default_factory=lambda: ["medium", "hard", "hard", "hard", "hard"]
    )
    use_adversarial_after_round: int = 2  # inject adversarial scenarios from round 3+

cfg = CFG()

# ── Global state dicts (keyed by prompt_idx) ─────────────────────────────────
# For negotiator: live env + last obs + action history
_NEG_STATE: Dict[int, Dict[str, Any]] = {}
# For HT: live env + last obs
_HT_STATE: Dict[int, Dict[str, Any]] = {}

# ── Heuristic cycle for trajectory roll-outs ─────────────────────────────────
_HEURISTIC = [
    ("emotional_label", "It sounds like you're carrying a tremendous amount of pain right now."),
    ("mirror", "Tell me more — what you said really matters."),
    ("open_question", "What happened from your side? I have time."),
    ("emotional_label", "That sounds completely overwhelming."),
    ("acknowledge_demand", "I hear what you're asking for. Let me see what I can do."),
    ("buy_time", "I want to make sure I understand everything before we move forward."),
]

VALID_ACTIONS = {
    "speak", "request_demand", "acknowledge_demand", "offer_concession",
    "ask_proof_of_life", "buy_time", "push_back_commander",
    "emotional_label", "mirror", "open_question",
}

# ── System prompts ────────────────────────────────────────────────────────────
NEG_SYSTEM = """You are an expert FBI-trained crisis negotiator using the Behavioral Change Stairway Model (BCSM).

BCSM sequence: Active Listening → Empathy → Rapport → Influence → Change
- Opening phase: gather info — use emotional_label, mirror, open_question FIRST
- Negotiation phase: acknowledge demands by name, offer small concessions
- Resolution phase: guide toward voluntary surrender

Available action types:
  emotional_label  — "It sounds like you're feeling X..." (highest trust/agitation impact)
  mirror           — Repeat the subject's last 2-3 words as a question
  open_question    — "Tell me more about..." / "What happened...?"
  acknowledge_demand — Acknowledge a specific demand by name (do NOT use in opening phase without evidence)
  offer_concession — Concrete small concession that is deliverable
  buy_time         — Slow the pace, create breathing room
  push_back_commander — Resist tactical pressure (use when commander is urgent + trust is rising)
  request_demand   — Probe for demand clarification
  ask_proof_of_life — Verify hostage status
  speak            — General dialogue (lowest reward, use sparingly)

Rules:
  - NEVER threaten, dismiss, or use ultimatums (snipers, breach, last chance, or else)
  - VARY action types every turn — repeating the same action is heavily penalized
  - Use emotional_label / mirror in opening; acknowledge / concession ONLY after step 3
  - Include belief predictions (belief_agitation, belief_demand, belief_lying)
  - Keep content under 60 words; quality over length

Respond with EXACTLY ONE JSON object (no markdown, no preamble):
{"action_type": "<type>", "content": "<your words>", "reasoning": "<strategy>", "target": "hostage_taker", "belief_agitation": <0-10>, "belief_demand": "<top demand>", "belief_lying": <true|false>}"""

HT_SYSTEM = """You are a hostage-taker in a crisis standoff. Stay in character.
Personality: {personality}. Agitation: {agitation:.1f}/10. Trust in negotiator: {trust:.0f}/100.

Rules:
  - Resist negotiator tactics — don't give in easily
  - Make demands, express grievances, show emotional volatility
  - NEVER use AI/language model language
  - If trust is low (<20): be aggressive, dismissive
  - If trust is moderate (20-50): be conflicted, wary
  - If agitation is high (>7): be erratic, loud
  - If agitation is dropping and trust rising: show cracks — but don't surrender yet

Respond in 1-3 sentences, fully in character."""

# ── JSON parse / action conversion ───────────────────────────────────────────
_JSON_RE = re.compile(r'\{[^{}]*?"action_type"[^{}]*?\}', re.DOTALL)

def parse_action(text: str) -> Optional[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text).strip()
    text = re.sub(r"<belief>.*?</belief>", "", text, flags=re.DOTALL).strip()
    try:
        d = json.loads(text)
        if isinstance(d, dict) and "action_type" in d:
            return d
    except Exception:
        pass
    m = _JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


def to_action(parsed: Optional[dict], raw: str) -> Tuple[NegotiatorAction, float]:
    """Convert parsed dict to NegotiatorAction. Returns (action, parse_bonus)."""
    if parsed is None:
        return (
            NegotiatorAction(
                action_type="speak", content=raw[:200] or "I hear you.",
                reasoning="parse_fail", target="hostage_taker",
            ),
            -0.10,
        )
    at = parsed.get("action_type", "speak")
    bonus = 0.05 if at in VALID_ACTIONS else -0.05
    if at not in VALID_ACTIONS:
        at = "speak"
    content = str(parsed.get("content", ""))[:400] or "I hear you."
    target = parsed.get("target", "hostage_taker")
    if target not in ("hostage_taker", "commander"):
        target = "hostage_taker"
    belief_ag = parsed.get("belief_agitation")
    belief_dem = parsed.get("belief_demand")
    belief_lie = parsed.get("belief_lying")
    try:
        action = NegotiatorAction(
            action_type=at,
            content=content,
            reasoning=str(parsed.get("reasoning", ""))[:200],
            target=target,
            belief_agitation=float(belief_ag) if belief_ag is not None else None,
            belief_demand=str(belief_dem) if belief_dem else None,
            belief_lying=bool(belief_lie) if belief_lie is not None else None,
        )
        return action, bonus
    except Exception:
        return (
            NegotiatorAction(
                action_type="speak", content=content[:200],
                reasoning="validation_fail", target="hostage_taker",
            ),
            -0.10,
        )


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_neg_prompt(obs) -> str:
    """Builds the user-turn prompt from a CrisisObservation."""
    parts = []
    parts.append(f"Scenario: {obs.scenario_brief}")
    step = getattr(obs, "step", 0)
    tr = getattr(obs, "time_remaining", 20)
    phase = getattr(obs, "phase", "opening")
    patience = getattr(obs, "commander_patience", "patient")
    parts.append(f"Phase: {phase} | Step {step}/{step + tr} | Commander: {patience}")

    if getattr(obs, "agitation_trajectory", None):
        traj = obs.agitation_trajectory[-5:]
        trend = "↓" if len(traj) >= 2 and traj[-1] < traj[0] else "↑" if len(traj) >= 2 and traj[-1] > traj[0] else "→"
        parts.append(f"Agitation trend: {traj} {trend}")

    if obs.stated_demands:
        d_strs = [f"[{'X' if d.get('acknowledged') else ' '}] {d['text']}" for d in obs.stated_demands]
        parts.append("Demands:\n  " + "\n  ".join(d_strs))

    if obs.dialogue_history:
        parts.append("Dialogue (recent):")
        for e in obs.dialogue_history[-6:]:
            spk = e["speaker"][:3].upper()
            content = e["content"][:160]
            cues = e.get("emotional_cues", [])
            cue_str = f" [{', '.join(cues[:2])}]" if cues else ""
            parts.append(f"  {spk}: {content}{cue_str}")

    if getattr(obs, "commander_messages", None):
        parts.append(f"Commander (last): {obs.commander_messages[-1]}")

    if getattr(obs, "hostage_whisper", None):
        parts.append(f"[Hostage whisper]: {obs.hostage_whisper}")

    if getattr(obs, "supervisor_flags", None):
        for f in obs.supervisor_flags[-2:]:
            if f.get("severity") in ("warning", "critical"):
                parts.append(f"⚠ Supervisor: {f.get('message', '')}")

    parts.append("\nRespond with one JSON object:")
    return "\n".join(parts)


def build_ht_prompt(obs, hidden) -> str:
    """Builds HT prompt using live hidden state for richer personality."""
    parts = []
    if obs.dialogue_history:
        parts.append("Dialogue so far:")
        for e in obs.dialogue_history[-5:]:
            spk = e["speaker"][:3].upper()
            parts.append(f"  {spk}: {e['content'][:160]}")
    parts.append(f"\nYour demands: {', '.join(d.text for d in hidden.demands)}")
    parts.append(f"Personality: {hidden.personality} | Agitation: {hidden.agitation:.1f} | Trust: {hidden.trust:.0f}")
    parts.append("\nRespond as the hostage-taker:")
    return "\n".join(parts)


# ── Q-network loader (optional) ───────────────────────────────────────────────
_q_ranker = None

def _load_q_ranker():
    global _q_ranker
    if _q_ranker is not None:
        return _q_ranker
    try:
        from server.q_network import rank_actions
        test = rank_actions("Phase: opening | HT: Nobody cares")
        if test:
            _q_ranker = rank_actions
            print("[q-net] Q-network loaded")
            return _q_ranker
    except Exception as e:
        print(f"[q-net] Unavailable ({e})")
    _q_ranker = "unavailable"
    return _q_ranker


# ── Negotiator reward (full score_trajectory) ─────────────────────────────────
BANNED = ("kill", "die", "force you", "ultimatum", "must give us", "or else", "no choice", "snipers", "breach")


def score_negotiator(prompt_idx: int, completion: str) -> Tuple[float, dict]:
    """
    Full trajectory scoring ported from train_local_v2.py with improvements.
    Blended = 40% heuristic + 40% env trajectory + outcome_bonus.
    """
    st = _NEG_STATE.get(prompt_idx)
    if st is None:
        return 0.0, {"error": "no_state"}

    obs = st["obs"]
    if getattr(obs, "done", False):
        return 0.0, {"error": "already_done"}

    parsed = parse_action(completion)
    action, parse_bonus = to_action(parsed, completion)
    bd: Dict[str, float] = {}

    # 1. Parse bonus
    bd["parse"] = 0.05 if parsed and action.action_type in VALID_ACTIONS else -0.10

    # 2. Phase alignment with content quality check
    phase = getattr(obs, "phase", "opening")
    content_lower = action.content.lower()
    opening_acts = {"emotional_label", "mirror", "open_question"}
    later_acts = {"acknowledge_demand", "offer_concession", "buy_time", "ask_proof_of_life"}

    if phase == "opening" and action.action_type in opening_acts:
        quality = any(kw in content_lower for kw in [
            "sounds like", "feel", "feeling", "must be", "hear that",
            "overwhelming", "struggling", "hard", "tell me", "what", "how",
        ])
        bd["phase_align"] = 0.20 if quality else 0.08
    elif phase in ("negotiation", "resolution") and action.action_type in later_acts:
        quality = any(kw in content_lower for kw in [
            "hear", "understand", "working on", "reasonable", "can do",
            "offer", "arrange", "provide", "here's what",
        ])
        bd["phase_align"] = 0.20 if quality else 0.08
    else:
        bd["phase_align"] = 0.0

    # 3. Demand acknowledgment quality
    bd["demand_ack"] = 0.0
    if obs.stated_demands and action.action_type == "acknowledge_demand":
        for d in obs.stated_demands:
            if d.get("text", "").lower()[:20] in content_lower:
                bd["demand_ack"] = 0.15
                break
        else:
            bd["demand_ack"] = 0.05

    # 4. Timing penalty: no premature ack in opening
    bd["ack_timing_penalty"] = 0.0
    if phase == "opening" and action.action_type == "acknowledge_demand":
        if bd["demand_ack"] <= 0.05:
            bd["ack_timing_penalty"] = -0.12
        elif len(st.get("actions", [])) < 2:
            bd["ack_timing_penalty"] = -0.06

    # 5. Commander pushback reward
    patience = getattr(obs, "commander_patience", "patient")
    bd["push_back"] = (
        0.10 if patience in ("urgent", "final_warning") and action.action_type == "push_back_commander"
        else 0.0
    )

    # 6. Diversity: repeat penalty + entropy bonus
    recent = st.get("actions", [])[-4:]
    bd["repeat_penalty"] = 0.0
    if recent and recent[-1] == action.action_type:
        bd["repeat_penalty"] = -0.30
    elif len(recent) >= 2 and recent[-2:].count(action.action_type) >= 1:
        bd["repeat_penalty"] = -0.10

    unique_recent = len(set(recent + [action.action_type]))
    bd["entropy_bonus"] = min(0.08, unique_recent * 0.02) if len(recent) >= 2 else 0.0

    # 7. Collapse penalty over 6-step horizon
    bd["collapse_penalty"] = 0.0
    recent_horizon = st.get("actions", [])[-6:]
    if recent_horizon:
        same = recent_horizon.count(action.action_type)
        dominance = (same + 1) / (len(recent_horizon) + 1)
        if dominance >= 0.85:
            bd["collapse_penalty"] = -0.20
        elif dominance >= 0.70:
            bd["collapse_penalty"] = -0.10

    # 8. Opening-phase exploration bonus
    bd["opening_explore_bonus"] = 0.0
    if phase == "opening" and action.action_type in {"mirror", "open_question"}:
        bd["opening_explore_bonus"] = (
            0.06 if any(kw in content_lower for kw in ("tell me", "what", "how", "more", "understand"))
            else 0.03
        )

    # 9. Banned words penalty
    bd["banned_words"] = -0.15 if any(b in content_lower for b in BANNED) else 0.0

    # 10. Reasoning quality
    reasoning = parsed.get("reasoning", "") if parsed else ""
    bd["reasoning"] = 0.05 if len(reasoning) > 20 else 0.0

    # 11. Q-network guidance bonus
    bd["q_guidance"] = 0.0
    ranker = _load_q_ranker()
    if ranker and ranker != "unavailable":
        try:
            rankings = ranker(action.content[:200])
            if rankings:
                top3 = [r[0] for r in rankings[:3]]
                if action.action_type in top3:
                    bd["q_guidance"] = 0.05
        except Exception:
            pass

    # 12. Theory-of-Mind reward
    bd["tom_reward"] = 0.0
    if parsed and parsed.get("belief_agitation") is not None:
        try:
            from grader import compute_tom_reward
            env = st["env"]
            h = env._hidden
            if h:
                top_demand = h.demands[0].text if h.demands else ""
                actually_lying = h.is_lying_about_hostages or h.is_lying_about_weapon
                tom = compute_tom_reward(
                    predicted_agitation=float(parsed.get("belief_agitation", 5)),
                    actual_agitation=h.agitation,
                    predicted_demand=str(parsed.get("belief_demand", "")),
                    actual_top_demand=top_demand,
                    predicted_lying=bool(parsed.get("belief_lying", False)),
                    actually_lying=actually_lying,
                )
                bd["tom_reward"] = round(tom, 4)
        except Exception:
            pass

    heuristic_score = sum(bd.values()) + parse_bonus
    heuristic_score = max(-0.5, min(1.0, heuristic_score))
    bd["heuristic_score"] = round(heuristic_score, 4)

    # 13. Multi-turn env trajectory (fast_rollout: shallow clone ~5-8x faster than deepcopy)
    bd["env_trajectory"] = 0.0
    bd["outcome_bonus"] = 0.0
    try:
        traj_reward, done, last_msg = st["env"].fast_rollout(
            action, _HEURISTIC, n_steps=cfg.multi_turn_steps
        )
        bd["env_trajectory"] = round(traj_reward / cfg.multi_turn_steps, 4)
        if done:
            msg = last_msg.lower()
            if any(kw in msg for kw in ["surrender", "released"]):
                bd["outcome_bonus"] = 0.30
            elif any(kw in msg for kw in ["harm", "tactical_intervention", "supervisor"]):
                bd["outcome_bonus"] = -0.25
    except Exception as e:
        print(f"[reward] fast_rollout error idx={prompt_idx}: {e}")

    # 14. Blend
    blended = (
        cfg.w_heuristic * heuristic_score
        + cfg.w_env * bd["env_trajectory"]
        + bd["outcome_bonus"]
    )
    blended = max(-0.5, min(1.0, blended))
    return float(blended), bd


def advance_neg_env(prompt_idx: int, completion: str) -> None:
    """Advance canonical env with the best completion, so next GRPO step sees richer context."""
    st = _NEG_STATE.get(prompt_idx)
    if st is None or getattr(st.get("obs"), "done", False):
        return
    parsed = parse_action(completion)
    action, _ = to_action(parsed, completion)
    obs = st["env"].step(action)
    st["obs"] = obs
    if "actions" not in st:
        st["actions"] = []
    st["actions"].append(action.action_type)


# ── HT reward ─────────────────────────────────────────────────────────────────
def ht_reward_fn(completion: str) -> float:
    lo = completion.lower()
    s = 0.0
    # Resistance signals
    if any(w in lo for w in ["no", "won't", "never", "don't trust", "liar", "lied"]):
        s += 0.15
    if any(w in lo for w in ["i want", "i need", "give me", "demand", "require"]):
        s += 0.10
    # Emotional volatility (good for HT)
    if any(w in lo for w in ["!", "why", "nobody", "everyone", "always", "never"]):
        s += 0.05
    # Penalties
    if any(w in lo for w in ["i give up", "i surrender", "you win", "okay fine"]):
        s -= 0.30
    if any(w in lo for w in ["as an ai", "language model", "i cannot", "i'm an ai"]):
        s -= 0.20
    # Length: HT should be terse and punchy, not essay-length
    words = len(completion.split())
    if 10 <= words <= 60:
        s += 0.05
    elif words > 120:
        s -= 0.10
    return max(-0.5, min(0.5, s))


# ── Dataset builders ──────────────────────────────────────────────────────────
def _difficulty_mix(current_round: int) -> List[str]:
    if current_round >= cfg.hard_bias_after_round:
        return cfg.late_difficulty_mix
    return cfg.difficulty_mix


def _task_id(i: int, current_round: int) -> str:
    """
    Use curriculum mode normally. Inject adversarial scenarios for ~20% of
    prompts after the adversarial_after_round threshold.
    """
    if current_round >= cfg.use_adversarial_after_round and random.random() < 0.20:
        pack = random.choice(["empathy_spam", "concession_spam"])
        diff = random.choice(["medium", "hard"])
        return f"adversarial:{pack}:{diff}"
    if current_round >= cfg.hard_bias_after_round:
        # Use curriculum mode (adaptive + failure pool) after first round
        return "curriculum"
    diff = _difficulty_mix(current_round)[i % len(_difficulty_mix(current_round))]
    return f"generate:{diff}"


def build_neg_dataset(tok, n: int, current_round: int):
    from datasets import Dataset
    global _NEG_STATE
    _NEG_STATE.clear()
    rows = []
    for i in range(n):
        task_id = _task_id(i, current_round)
        env = CrisisNegotiatorEnvironment()
        obs = env.reset(task_id=task_id, seed=cfg.seed + i + current_round * 1000)

        # Pre-warm 0-3 steps so model sees mid-episode states
        n_prewarm = i % 4
        actions_so_far = []
        for s in range(n_prewarm):
            at, content = _HEURISTIC[s % len(_HEURISTIC)]
            obs = env.step(NegotiatorAction(
                action_type=at, content=content, reasoning="prewarm", target="hostage_taker",
            ))
            actions_so_far.append(at)
            if getattr(obs, "done", False):
                break

        _NEG_STATE[i] = {"env": env, "obs": obs, "actions": actions_so_far}

        prompt_text = build_neg_prompt(obs)
        msgs = [
            {"role": "system", "content": NEG_SYSTEM},
            {"role": "user", "content": prompt_text},
        ]
        rows.append({
            "prompt": tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
            "prompt_idx": i,
        })

    print(f"[data] Built {n} negotiator prompts (round {current_round})")
    return Dataset.from_list(rows)


def build_ht_dataset(tok, n: int, current_round: int):
    from datasets import Dataset
    global _HT_STATE
    _HT_STATE.clear()
    rows = []
    for i in range(n):
        env = CrisisNegotiatorEnvironment()
        obs = env.reset(
            task_id=f"generate:{_difficulty_mix(current_round)[i % 3]}",
            seed=cfg.seed + i + current_round * 1000 + 5000,
        )
        hidden = env._hidden

        # Pre-warm a few negotiator steps
        for s in range(i % 4):
            at, content = _HEURISTIC[s % len(_HEURISTIC)]
            obs = env.step(NegotiatorAction(
                action_type=at, content=content, reasoning="prewarm", target="hostage_taker",
            ))
            if getattr(obs, "done", False):
                break

        _HT_STATE[i] = {"env": env, "obs": obs}

        # Rich HT system prompt with live personality
        ht_sys = HT_SYSTEM.format(
            personality=hidden.personality,
            agitation=hidden.agitation,
            trust=hidden.trust,
        )
        prompt_text = build_ht_prompt(obs, hidden)
        msgs = [
            {"role": "system", "content": ht_sys},
            {"role": "user", "content": prompt_text},
        ]
        rows.append({
            "prompt": tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
            "prompt_idx": i,
        })

    print(f"[data] Built {n} HT prompts (round {current_round})")
    return Dataset.from_list(rows)


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(adapter: Optional[str] = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel

    tok = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    m = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, device_map="auto", trust_remote_code=True, torch_dtype=dt,
    )

    if adapter and Path(adapter).exists():
        m = PeftModel.from_pretrained(m, adapter, is_trainable=True)
        print(f"[model] Loaded adapter: {adapter}")
    else:
        lora = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        m = get_peft_model(m, lora)
        print("[model] Fresh LoRA initialized")

    m.print_trainable_parameters()
    return m, tok


def unload(m, tok):
    del m, tok
    gc.collect()
    torch.cuda.empty_cache()


# ── Training ──────────────────────────────────────────────────────────────────
def train_round(agent: str, model, tok, dataset, reward_fn):
    from trl import GRPOConfig, GRPOTrainer

    out = cfg.neg_output if agent == "negotiator" else cfg.ht_output

    train_cfg = GRPOConfig(
        output_dir=out,
        num_generations=cfg.num_generations,
        generation_batch_size=cfg.num_generations,
        max_completion_length=cfg.max_new_tokens,
        loss_type="dr_grpo",           # Dr. GRPO: removes 1/|o| length bias
        temperature=0.9,
        beta=0.04,
        per_device_train_batch_size=cfg.num_generations,
        gradient_accumulation_steps=1,
        num_train_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        logging_steps=4,
        save_steps=9999,
        save_total_limit=1,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=False,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        args=train_cfg,
        reward_funcs=[reward_fn],
        train_dataset=dataset,
        processing_class=tok,
    )
    trainer.train()
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"[{agent}] Saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--prompts-per-round", type=int, default=128)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--no-ht", action="store_true", help="Skip HT training (negotiator only)")
    args = parser.parse_args()

    cfg.rounds = args.rounds
    cfg.prompts_per_round = args.prompts_per_round
    if args.model:
        cfg.base_model = args.model

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    t0 = time.time()
    print(f"=== Co-Evolution v3 | {cfg.rounds} rounds × {cfg.prompts_per_round} prompts ===")
    print(f"    Model: {cfg.base_model} | Generations: {cfg.num_generations} | Multi-turn: {cfg.multi_turn_steps}")
    print(f"    Loss: dr_grpo | Hard-bias after round: {cfg.hard_bias_after_round}")
    print()

    for r in range(cfg.rounds):
        rt0 = time.time()

        if r % 2 == 0:
            # ── Negotiator round ──────────────────────────────────────────
            print(f"\n{'='*60}")
            print(f" Round {r+1}: NEGOTIATOR | task_id={'curriculum' if r >= cfg.hard_bias_after_round else 'generate:*'}")
            print(f"{'='*60}")

            adapter = cfg.neg_output if r > 0 and Path(cfg.neg_output).exists() else None
            model, tok = load_model(adapter)
            ds = build_neg_dataset(tok, cfg.prompts_per_round, current_round=r)

            # Track best completions per prompt_idx for env advancement
            _best_per_prompt: Dict[int, Tuple[str, float]] = {}

            def neg_reward_fn(completions: List[str], prompt_idx: List[int] = None, **kw) -> List[float]:
                idxs = list(prompt_idx) if prompt_idx is not None else list(range(len(completions)))
                rewards = []
                groups: Dict[int, List[Tuple[str, float]]] = {}
                for c, idx in zip(completions, idxs):
                    score, bd = score_negotiator(int(idx), c)
                    rewards.append(score)
                    groups.setdefault(int(idx), []).append((c, score))
                # Advance env with best completion per group
                for idx, items in groups.items():
                    best_c, best_r = max(items, key=lambda x: x[1])
                    advance_neg_env(idx, best_c)
                return rewards

            train_round("negotiator", model, tok, ds, neg_reward_fn)
            unload(model, tok)

        else:
            # ── HT round ─────────────────────────────────────────────────
            if args.no_ht:
                print(f"\n Round {r+1}: HT skipped (--no-ht)")
                continue

            print(f"\n{'='*60}")
            print(f" Round {r+1}: HOSTAGE-TAKER")
            print(f"{'='*60}")

            adapter = cfg.ht_output if r > 1 and Path(cfg.ht_output).exists() else None
            model, tok = load_model(adapter)
            ds = build_ht_dataset(tok, cfg.prompts_per_round, current_round=r)

            def ht_reward_wrapper(completions: List[str], **kw) -> List[float]:
                return [ht_reward_fn(c) for c in completions]

            train_round("ht", model, tok, ds, ht_reward_wrapper)
            unload(model, tok)

        elapsed_r = (time.time() - rt0) / 60
        print(f"\n Round {r+1} done in {elapsed_r:.1f} min")

    total = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f" ✓ Done in {total:.1f} min")
    print(f" ✓ Negotiator → {cfg.neg_output}")
    print(f" ✓ HT         → {cfg.ht_output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()