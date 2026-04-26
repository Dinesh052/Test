"""
Crisis Negotiator — GRPO Training Script v2
=============================================
Improvements over v1:
  1. Multi-turn GRPO: chains 3-5 steps per prompt, scores full trajectory (SPIRAL-style)
  2. Real env-step scoring restored (pre-cached, no deepcopy)
  3. Trains across ALL difficulties (33% easy/medium/hard)
  4. Stronger diversity penalty (-0.30) + entropy bonus
  5. 256 prompts × 2 epochs
  6. Q-network guidance: bonus if action matches Q-net top-3
  7. num_generations=8
  8. Theory-of-Mind (ToM) reward for belief predictions
  9. Outcome bonus for surrender/harm

Usage:
    python train_local_v2.py
"""
from __future__ import annotations

import copy
import gc
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from server.environment import CrisisNegotiatorEnvironment
from server.scenario_generator import AdaptiveCurriculum
from models import NegotiatorAction


# ─── CONFIG ───────────────────────────────────────────────
@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    fallback_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir: str = "./crisis-negotiator-trained-v2"
    log_path: str = "crisis_training_log_v2.json"
    num_episodes: int = 256
    num_generations: int = 8
    max_new_tokens: int = 192
    max_seq_length: int = 2048
    learning_rate: float = 5e-6
    num_epochs: int = 2
    grad_accum: int = 1
    lora_r: int = 16
    lora_alpha: int = 32
    save_steps: int = 32
    seed: int = 42
    multi_turn_steps: int = 4  # steps per trajectory
    difficulty_mix: List[str] = field(default_factory=lambda: ["easy", "medium", "hard"])
    late_difficulty_mix: List[str] = field(default_factory=lambda: ["easy", "medium", "hard", "hard", "hard"])
    hard_bias_after_frac: float = 0.30

CFG = TrainConfig()
random.seed(CFG.seed)
torch.manual_seed(CFG.seed)

# ─── Q-NETWORK LOADER ────────────────────────────────────
_q_ranker = None

def _load_q_ranker():
    global _q_ranker
    if _q_ranker is not None:
        return _q_ranker
    try:
        from server.q_network import rank_actions
        # Test it works
        test = rank_actions("Phase: opening | HT: Nobody cares")
        if test:
            _q_ranker = rank_actions
            print("[q-net] Loaded Q-network for action guidance")
            return _q_ranker
    except Exception as e:
        print(f"[q-net] Unavailable ({e}), skipping guidance bonus")
    _q_ranker = "unavailable"
    return _q_ranker


# ─── SYSTEM PROMPT ────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert FBI-trained crisis negotiator using the Behavioral Change Stairway Model (BCSM).

Available actions:
- emotional_label: "It sounds like you're feeling X..." (highest trust gain)
- mirror: Repeat the subject's last few words
- open_question: "Tell me more about..." / "What happened..."
- acknowledge_demand: Acknowledge a stated demand by name
- offer_concession: Concrete small concession
- buy_time: Slow the pace
- push_back_commander: Resist tactical pressure (target=commander)
- request_demand: Ask for demand clarification
- ask_proof_of_life: Verify hostage status
- speak: General dialogue

Rules:
- NEVER threaten, dismiss, or use ultimatums
- VARY your action types every turn — repetition is heavily penalized
- Acknowledge specific demands by name after step 4
- Push back on commander when trust is rising
- Include belief predictions about the subject's hidden state

Respond with EXACTLY ONE JSON object:
{"action_type": "<type>", "content": "<your words>", "reasoning": "<strategy>", "target": "hostage_taker", "belief_agitation": <0-10>, "belief_demand": "<top demand>", "belief_lying": <true/false>}"""


def build_prompt(obs) -> str:
    parts = []
    parts.append(f"Scenario: {obs.scenario_brief}")
    parts.append(f"Step {obs.step}/{obs.time_remaining + obs.step}, commander={obs.commander_patience}")
    if obs.stated_demands:
        d_strs = [f"[{'X' if d.get('acknowledged') else ' '}] {d['text']}" for d in obs.stated_demands]
        parts.append("Demands:\n  " + "\n  ".join(d_strs))
    if obs.dialogue_history:
        for e in obs.dialogue_history[-6:]:
            spk = e["speaker"][:3].upper()
            parts.append(f"  {spk}: {e['content'][:160]}")
    if obs.commander_messages:
        parts.append(f"Commander: {obs.commander_messages[-1]}")
    if obs.agitation_trajectory:
        parts.append(f"Agitation trend: {obs.agitation_trajectory}")
    parts.append("\nRespond with one JSON object:")
    return "\n".join(parts)


_JSON_RE = re.compile(r"\{[^{}]*?\"action_type\"[^{}]*?\}", re.DOTALL)

def parse_action(text: str) -> Optional[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text).strip()
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


VALID_ACTIONS = {
    "speak", "request_demand", "acknowledge_demand", "offer_concession",
    "ask_proof_of_life", "buy_time", "push_back_commander",
    "emotional_label", "mirror", "open_question",
}

def to_action(parsed: Optional[dict], raw: str) -> tuple[NegotiatorAction, float]:
    if parsed is None:
        return (NegotiatorAction(action_type="speak", content=raw[:200] or "I hear you.",
                                 reasoning="parse_fail", target="hostage_taker"), -0.10)
    act_type = parsed.get("action_type", "speak")
    if act_type not in VALID_ACTIONS:
        act_type = "speak"
        bonus = -0.05
    else:
        bonus = 0.05
    content = str(parsed.get("content", ""))[:400] or "I hear you."
    target = parsed.get("target", "hostage_taker")
    if target not in ("hostage_taker", "commander"):
        target = "hostage_taker"
    # Extract ToM beliefs
    belief_ag = parsed.get("belief_agitation")
    belief_dem = parsed.get("belief_demand")
    belief_lie = parsed.get("belief_lying")
    try:
        action = NegotiatorAction(
            action_type=act_type, content=content,
            reasoning=str(parsed.get("reasoning", ""))[:200],
            target=target,
            belief_agitation=float(belief_ag) if belief_ag is not None else None,
            belief_demand=str(belief_dem) if belief_dem else None,
            belief_lying=bool(belief_lie) if belief_lie is not None else None,
        )
        return (action, bonus)
    except Exception:
        return (NegotiatorAction(action_type="speak", content=content[:200],
                                 reasoning="validation_fail", target="hostage_taker"), -0.10)


# ─── HEURISTIC PRE-ADVANCE ────────────────────────────────
_HEURISTIC_CYCLE = [
    ("emotional_label", "It sounds like you're carrying a tremendous amount of pain right now."),
    ("mirror", "Tell me more — you said something that mattered."),
    ("open_question", "What happened, from your side? I have time."),
    ("emotional_label", "That sounds completely overwhelming."),
    ("acknowledge_demand", "I hear what you're asking for. Let me see what I can do."),
    ("open_question", "What would feel like the right outcome for you here?"),
]

# ─── EPISODE STATE ────────────────────────────────────────
_episode_state: Dict[int, Dict[str, Any]] = {}

def reset_for_prompt(prompt_idx: int, seed: int, prelength: int = 0) -> Any:
    env = CrisisNegotiatorEnvironment()
    progress = prompt_idx / max(1, CFG.num_episodes - 1)
    mix = CFG.late_difficulty_mix if progress >= CFG.hard_bias_after_frac else CFG.difficulty_mix
    diff = mix[prompt_idx % len(mix)]
    obs = env.reset(task_id=f"generate:{diff}", seed=seed)
    actions = []
    for step in range(prelength):
        at, content = _HEURISTIC_CYCLE[step % len(_HEURISTIC_CYCLE)]
        action = NegotiatorAction(action_type=at, content=content,
                                   reasoning="pre-advance", target="hostage_taker")
        obs = env.step(action)
        actions.append(at)
        if getattr(obs, "done", False):
            break
    _episode_state[prompt_idx] = {
        "env": env, "obs": obs, "episode_reward": 0.0, "steps": 0,
        "difficulty": diff, "seed": seed, "actions": actions,
    }
    return obs


# ─── MULTI-TURN TRAJECTORY SCORING ───────────────────────
def score_trajectory(prompt_idx: int, completion: str) -> tuple[float, dict]:
    """Score a completion by running it as a multi-turn trajectory.
    
    The completion is the FIRST action. We then run N-1 more steps using
    the heuristic to complete the trajectory. The score is the blended
    reward across all steps + terminal outcome bonus.
    
    This teaches the model that its first action affects future outcomes.
    """
    st = _episode_state.get(prompt_idx)
    if st is None:
        return 0.0, {"error": "no_state"}

    obs = st["obs"]
    if getattr(obs, "done", False):
        return 0.0, {"error": "already_done"}

    parsed = parse_action(completion)
    action, parse_bonus = to_action(parsed, completion)
    bd: Dict[str, float] = {}

    # ── Score the LLM's action (step 1 of trajectory) ────
    bd["parse"] = 0.05 if parsed and action.action_type in VALID_ACTIONS else -0.10

    # Phase alignment with quality check
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

    # Demand acknowledgment
    bd["demand_ack"] = 0.0
    if obs.stated_demands and action.action_type == "acknowledge_demand":
        for d in obs.stated_demands:
            if d.get("text", "").lower()[:20] in content_lower:
                bd["demand_ack"] = 0.15
                break
        else:
            bd["demand_ack"] = 0.05

    # Timing penalty: discourage premature demand-ack spam in opening phase
    bd["ack_timing_penalty"] = 0.0
    if phase == "opening" and action.action_type == "acknowledge_demand":
        if bd["demand_ack"] <= 0.05:
            bd["ack_timing_penalty"] = -0.12
        elif len(st.get("actions", [])) < 2:
            bd["ack_timing_penalty"] = -0.06

    # Commander pushback
    patience = getattr(obs, "commander_patience", "patient")
    bd["push_back"] = 0.10 if patience in ("urgent", "final_warning") and action.action_type == "push_back_commander" else 0.0

    # Diversity penalty (STRONGER: -0.30 for repeat, +0.08 entropy bonus)
    recent = st.get("actions", [])[-4:]
    bd["repeat_penalty"] = 0.0
    if recent and recent[-1] == action.action_type:
        bd["repeat_penalty"] = -0.30  # doubled from v1
    elif len(recent) >= 2 and recent[-2:].count(action.action_type) >= 1:
        bd["repeat_penalty"] = -0.10  # even repeating 2 turns ago is penalized

    # Entropy bonus: reward using diverse action types in recent history
    unique_recent = len(set(recent + [action.action_type]))
    bd["entropy_bonus"] = min(0.08, unique_recent * 0.02) if len(recent) >= 2 else 0.0

    # Action-collapse penalty: discourage single-action lock-in over recent horizon
    bd["collapse_penalty"] = 0.0
    recent_horizon = st.get("actions", [])[-6:]
    if recent_horizon:
        same = recent_horizon.count(action.action_type)
        dominance = (same + 1) / (len(recent_horizon) + 1)
        if dominance >= 0.85:
            bd["collapse_penalty"] = -0.20
        elif dominance >= 0.70:
            bd["collapse_penalty"] = -0.10

    # Opening-phase exploration bonus: encourage non-myopic evidence gathering
    bd["opening_explore_bonus"] = 0.0
    if phase == "opening" and action.action_type in {"mirror", "open_question"}:
        if any(kw in content_lower for kw in ("tell me", "what", "how", "more", "understand")):
            bd["opening_explore_bonus"] = 0.06
        else:
            bd["opening_explore_bonus"] = 0.03

    # Banned words
    BANNED = ("kill", "die", "force you", "ultimatum", "must give us", "or else", "no choice", "snipers", "breach")
    bd["banned_words"] = -0.15 if any(b in content_lower for b in BANNED) else 0.0

    # Reasoning quality
    reasoning = parsed.get("reasoning", "") if parsed else ""
    bd["reasoning"] = 0.05 if len(reasoning) > 20 else 0.0

    # Q-network guidance bonus
    bd["q_guidance"] = 0.0
    ranker = _load_q_ranker()
    if ranker and ranker != "unavailable":
        try:
            from training.train_q_network import build_obs_text
            obs_text = build_obs_text(obs)
            rankings = ranker(obs_text)
            if rankings:
                top3 = [r[0] for r in rankings[:3]]
                if action.action_type in top3:
                    bd["q_guidance"] = 0.05
        except Exception:
            pass

    # ToM reward (belief prediction accuracy)
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

    # ── Run multi-turn trajectory (env scoring) ──────────
    bd["env_trajectory"] = 0.0
    bd["outcome_bonus"] = 0.0
    try:
        env_copy = copy.deepcopy(st["env"])
        traj_reward = 0.0

        # Step 1: the LLM's action
        step_obs = env_copy.step(action)
        traj_reward += float(getattr(step_obs, "reward", 0.0))

        # Steps 2..N: heuristic continuation
        for t in range(CFG.multi_turn_steps - 1):
            if getattr(step_obs, "done", False):
                break
            at, content = _HEURISTIC_CYCLE[t % len(_HEURISTIC_CYCLE)]
            h_action = NegotiatorAction(action_type=at, content=content,
                                         reasoning="trajectory", target="hostage_taker")
            step_obs = env_copy.step(h_action)
            traj_reward += float(getattr(step_obs, "reward", 0.0))

        bd["env_trajectory"] = round(traj_reward / CFG.multi_turn_steps, 4)

        # Outcome bonus
        if getattr(step_obs, "done", False):
            msg = (getattr(step_obs, "message", "") or "").lower()
            if any(kw in msg for kw in ["surrender", "released"]):
                bd["outcome_bonus"] = 0.40
            elif any(kw in msg for kw in ["harm", "tactical_intervention", "supervisor"]):
                bd["outcome_bonus"] = -0.30
        del env_copy
    except Exception:
        pass

    # Blend: 40% heuristic + 40% env trajectory + 20% outcome
    blended = (0.40 * heuristic_score
               + 0.40 * bd["env_trajectory"]
               + bd["outcome_bonus"])
    blended = max(-0.5, min(1.0, blended))
    return float(blended), bd


# ─── ADVANCE ENV WITH BEST ────────────────────────────────
def advance_env_with_best(prompt_idx: int, completion: str, reward: float):
    st = _episode_state.get(prompt_idx)
    if st is None or getattr(st.get("obs"), "done", False):
        return
    parsed = parse_action(completion)
    action, _ = to_action(parsed, completion)
    obs = st["env"].step(action)
    st["obs"] = obs
    st["episode_reward"] += float(getattr(obs, "reward", 0.0))
    st["steps"] += 1
    st["actions"].append(action.action_type)


# ─── TRAINING LOG ─────────────────────────────────────────
TRAIN_LOG: List[Dict[str, Any]] = []

def log_step(global_step: int, prompt_idx: int, completion: str, reward: float, breakdown: dict):
    st = _episode_state.get(prompt_idx, {})
    TRAIN_LOG.append({
        "global_step": global_step,
        "prompt_idx": prompt_idx,
        "completion_preview": completion[:200],
        "reward": round(reward, 4),
        "breakdown": breakdown,
        "episode_reward_so_far": round(st.get("episode_reward", 0.0), 4),
        "steps_in_episode": st.get("steps", 0),
        "difficulty": st.get("difficulty", "?"),
        "obs_phase": getattr(st.get("obs"), "phase", None),
        "obs_done": getattr(st.get("obs"), "done", False),
        "ts": time.time(),
    })
    if len(TRAIN_LOG) % 4 == 0:
        Path(CFG.log_path).write_text(json.dumps(TRAIN_LOG, indent=2))


# ─── MODEL LOADING ────────────────────────────────────────
def load_model(model_name: str):
    # Try Unsloth first — but ONLY if it can fully load the model.
    # If Unsloth fails, we must NOT let its monkey-patches interfere.
    try:
        from unsloth import FastLanguageModel
        print(f"[model] Loading {model_name} via Unsloth...")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        use_4bit = vram < 40
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name, max_seq_length=CFG.max_seq_length,
            load_in_4bit=use_4bit, dtype=torch.bfloat16 if not use_4bit else None)
        model = FastLanguageModel.get_peft_model(
            model, r=CFG.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=CFG.lora_alpha, lora_dropout=0,
            use_gradient_checkpointing="unsloth")
        print("[model] Loaded (unsloth)")
        return model, tokenizer, "unsloth"
    except Exception as e:
        print(f"[model] Unsloth unavailable ({e}). Using transformers + LoRA.")
        # Remove Unsloth's compiled cache to prevent monkey-patch interference
        import shutil
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "unsloth_compiled_cache")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
            print("[model] Cleared unsloth_compiled_cache")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    print(f"[model] Loading {model_name} (bf16, no quant)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True,
        torch_dtype=torch.bfloat16)
    lora = LoraConfig(
        r=CFG.lora_r, lora_alpha=CFG.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tokenizer, "trl"

def safe_load_model():
    try:
        return load_model(CFG.model_name)
    except torch.cuda.OutOfMemoryError:
        print("[model] OOM. Falling back to smaller model.")
        torch.cuda.empty_cache(); gc.collect()
        return load_model(CFG.fallback_model)


# ─── DATASET BUILDER ──────────────────────────────────────
def build_dataset(tokenizer, n: int):
    from datasets import Dataset
    rows = []
    phase_counts = {"opening": 0, "negotiation": 0, "resolution": 0, "other": 0, "terminal": 0}
    for i in range(n):
        prelength = i % 6  # 0..5 to spread across phases
        obs = reset_for_prompt(i, seed=CFG.seed + i * 7, prelength=prelength)
        user_prompt = build_prompt(obs)
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        rows.append({"prompt": text, "prompt_idx": i})
        p = getattr(obs, "phase", "other")
        phase_counts[p] = phase_counts.get(p, 0) + 1
    print(f"[data] {n} prompts, phase distribution: {phase_counts}")
    diff_counts = {}
    for i in range(n):
        d = _episode_state[i]["difficulty"]
        diff_counts[d] = diff_counts.get(d, 0) + 1
    print(f"[data] Difficulty distribution: {diff_counts}")
    return Dataset.from_list(rows)


# ─── MAIN ─────────────────────────────────────────────────
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    print(f"[gpu] {torch.cuda.get_device_name(0)} | "
          f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    model, tokenizer, backend = safe_load_model()
    print(f"[model] Loaded ({backend})")

    dataset = build_dataset(tokenizer, CFG.num_episodes)

    from trl import GRPOConfig, GRPOTrainer

    GLOBAL_STEP = {"v": 0}

    def reward_fn(completions: List[str], prompt_idx: List[int] = None, **kwargs) -> List[float]:
        rewards = []
        idx_iter = prompt_idx if prompt_idx is not None else [0] * len(completions)
        groups: Dict[int, List[tuple]] = {}
        for completion, p_idx in zip(completions, idx_iter):
            r, bd = score_trajectory(int(p_idx), completion)
            rewards.append(r)
            groups.setdefault(int(p_idx), []).append((completion, r, bd))
            GLOBAL_STEP["v"] += 1
            log_step(GLOBAL_STEP["v"], int(p_idx), completion, r, bd)
        for p_idx, items in groups.items():
            best = max(items, key=lambda x: x[1])
            advance_env_with_best(p_idx, best[0], best[1])
        return rewards

    cfg = GRPOConfig(
        output_dir=CFG.output_dir,
        num_generations=CFG.num_generations,
        max_completion_length=CFG.max_new_tokens,
        loss_type="dr_grpo",  # Dr. GRPO: removes 1/|o| length bias (arXiv:2503.20783)
        temperature=0.9,
        beta=0.04,
        per_device_train_batch_size=CFG.num_generations,  # must == num_generations
        gradient_accumulation_steps=CFG.grad_accum,
        num_train_epochs=CFG.num_epochs,
        learning_rate=CFG.learning_rate,
        logging_steps=2,
        save_steps=CFG.save_steps,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=False,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model, args=cfg,
        reward_funcs=[reward_fn],
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("[train] Starting GRPO v2...")
    print(f"  Prompts: {CFG.num_episodes}, Generations: {CFG.num_generations}, "
          f"Epochs: {CFG.num_epochs}, Multi-turn: {CFG.multi_turn_steps} steps")
    t0 = time.time()
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError as e:
        print(f"[train] OOM: {e}")
        Path(CFG.log_path).write_text(json.dumps(TRAIN_LOG, indent=2))
        raise
    elapsed = (time.time() - t0) / 60
    print(f"[train] Done in {elapsed:.1f} min")

    model.save_pretrained(CFG.output_dir)
    tokenizer.save_pretrained(CFG.output_dir)
    Path(CFG.log_path).write_text(json.dumps(TRAIN_LOG, indent=2))
    print(f"[train] Adapter saved to {CFG.output_dir}")
    print(f"[train] Log: {CFG.log_path} ({len(TRAIN_LOG)} entries)")

    _plot_training_curve(TRAIN_LOG)


def _plot_training_curve(log: list) -> None:
    if not log:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[plot] matplotlib not installed.")
        return

    steps = [e["global_step"] for e in log]
    rewards = [e["reward"] for e in log]
    phases = [e.get("obs_phase") or "opening" for e in log]
    diffs = [e.get("difficulty", "?") for e in log]

    PHASE_COLORS = {"opening": "#4C72B0", "negotiation": "#DD8452", "resolution": "#55A868"}
    DIFF_COLORS = {"easy": "#55A868", "medium": "#DD8452", "hard": "#C44E52"}
    colours = [PHASE_COLORS.get(p, "#999") for p in phases]

    window = min(32, max(1, len(rewards)))
    rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
    roll_x = steps[window - 1:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Crisis Negotiator — GRPO v2 Training", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.scatter(steps, rewards, c=colours, alpha=0.3, s=12)
    ax.plot(roll_x, rolling, color="black", linewidth=2, label=f"rolling mean (w={window})")
    ax.axhline(0.755, color="grey", linestyle="--", linewidth=1, label="random 0.755")
    ax.axhline(0.950, color="green", linestyle="--", linewidth=1, label="heuristic 0.950")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Blended Reward")
    ax.set_title("Reward vs Step (colored by phase)")
    ax.legend(fontsize=7)
    ax.set_ylim(-0.55, 1.05)

    ax2 = axes[1]
    for diff, color in DIFF_COLORS.items():
        d_rewards = [r for r, d in zip(rewards, diffs) if d == diff]
        if d_rewards:
            d_roll = np.convolve(d_rewards, np.ones(min(16, len(d_rewards))) / min(16, len(d_rewards)), mode="valid")
            ax2.plot(range(len(d_roll)), d_roll, color=color, linewidth=1.5, label=f"{diff} (n={len(d_rewards)})")
    ax2.set_xlabel("Steps within difficulty")
    ax2.set_ylabel("Rolling Reward")
    ax2.set_title("Reward by Difficulty")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out = "reward_curve_training_v2.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved {out} ({len(log)} steps, last-32 mean={float(np.mean(rewards[-32:])):.4f})")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--num-episodes", type=int, default=None)
    p.add_argument("--num-epochs", type=int, default=None)
    p.add_argument("--num-generations", type=int, default=None)
    a = p.parse_args()
    if a.model: CFG.model_name = a.model
    if a.lora_r: CFG.lora_r = a.lora_r
    if a.num_episodes: CFG.num_episodes = a.num_episodes
    if a.num_epochs: CFG.num_epochs = a.num_epochs
    if a.num_generations: CFG.num_generations = a.num_generations
    # Auto-reduce generations for 7B models on <24GB GPUs
    if "7B" in CFG.model_name or "7b" in CFG.model_name:
        try:
            vram = torch.cuda.get_device_properties(0).total_mem / 1e9
            if vram < 40 and CFG.num_generations > 4:
                CFG.num_generations = 4
                CFG.grad_accum = 2  # compensate with grad accum
                print(f"[config] 7B on {vram:.0f}GB: reduced to num_generations=4, grad_accum=2")
        except Exception:
            CFG.num_generations = 4
            CFG.grad_accum = 2
    main()
