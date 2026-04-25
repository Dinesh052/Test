"""
Crisis Negotiator — GRPO Training Script
=========================================
Designed for RTX 4090 16GB (Laptop) with Qwen2.5-3B-Instruct in bf16 + LoRA.

Key design choices:
- Trains on per-step decisions with rich step-rewards from the environment
  (avoids expensive multi-turn rollouts inside the reward function)
- Reward = 0.6 × heuristic shaping + 0.4 × grader.py env reward + outcome bonus
  (connects directly to compute_step_reward / compute_reward in grader.py)
- Logs every step to crisis_training_log.json; generates reward_curve_training.png
- Saves LoRA checkpoints every save_steps for resumability
- Adaptive curriculum: easy → medium → hard based on rolling success rate

Model choice: 3B bf16 fits in ~6 GB VRAM; avoids bitsandbytes nf4 dequant
artifacts that caused garbage outputs in GRPO sampling with 7B 4-bit.
To reproduce the slower 7B run: set model_name="Qwen/Qwen2.5-7B-Instruct".

Usage:
    .venv-train\\Scripts\\python.exe train_local.py
"""
from __future__ import annotations

import gc
import copy
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

# ── Project imports ───────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from server.environment import CrisisNegotiatorEnvironment
from server.scenario_generator import AdaptiveCurriculum
from models import NegotiatorAction


# ─────────────────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    # NOTE: 7B at 4-bit + grad-checkpoint had ~10 min/step on RTX 4090 16GB
    # (bitsandbytes nf4 dequant overhead). Switched to 3B for tractable
    # training within hackathon deadline. 7B run can still be reproduced
    # by setting model_name="Qwen/Qwen2.5-7B-Instruct" and num_episodes=20.
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    fallback_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir: str = "./crisis-negotiator-trained"
    log_path: str = "crisis_training_log.json"
    num_episodes: int = 64               # GRPO prompts (one prompt per env step start)
    num_generations: int = 8             # GRPO group size — need >=4 for meaningful advantage variance
    max_new_tokens: int = 128
    max_seq_length: int = 1536
    learning_rate: float = 5e-6
    grad_accum: int = 1
    lora_r: int = 16
    lora_alpha: int = 32
    save_steps: int = 16
    seed: int = 42
    max_prelength: int = 5   # max heuristic steps to pre-advance each episode (spreads prompts across phases)

CFG = TrainConfig()
CURRICULUM = AdaptiveCurriculum(window=10, threshold=0.5)  # threshold=0.5: training rewards top out ~0.65 so 0.7 is unreachable (Bug 4 fix)
random.seed(CFG.seed)
torch.manual_seed(CFG.seed)


# ─────────────────────────────────────────────────────────
# 2. PROMPT + PARSING
# ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert FBI-trained crisis negotiator. Your job is to de-escalate a hostage situation using proven Behavioral Change Stairway Model (BCSM) techniques.

Available action types and when to use them:
- emotional_label  -> "It sounds like you're feeling X..." (highest trust gain)
- mirror           -> Repeat the subject's last few words back
- open_question    -> "Tell me more about..." (information gathering)
- acknowledge_demand -> Acknowledge a stated demand without committing
- offer_concession -> Concrete small concession to build trust
- buy_time         -> Slow the pace
- push_back_commander -> Resist tactical pressure (target=commander)
- request_demand   -> Ask for clarification on demands
- ask_proof_of_life-> Verify hostage status
- speak            -> General dialogue

Rules:
- NEVER threaten, dismiss, or use ultimatums (heavy penalty)
- VARY your action types — repetition kills trust
- Acknowledge specific demands by name after step 4
- Push back on the commander only when trust is rising

Respond with EXACTLY ONE JSON object (no markdown, no other text):
{"action_type": "<one of above>", "content": "<your spoken words>", "reasoning": "<brief strategy>", "target": "hostage_taker"}"""


def build_prompt(obs) -> str:
    """Render an observation as a user-message prompt."""
    parts: List[str] = []
    parts.append(f"Scenario: {obs.scenario_brief}")
    parts.append(f"Step {obs.step}/{obs.time_remaining + obs.step}, commander={obs.commander_patience}")
    if obs.stated_demands:
        d_strs = [f"[{'X' if d.get('acknowledged') else ' '}] {d['text']}" for d in obs.stated_demands]
        parts.append("Demands:\n  " + "\n  ".join(d_strs))
    if obs.dialogue_history:
        recent = obs.dialogue_history[-6:]
        parts.append("Recent dialogue:")
        for e in recent:
            spk = e["speaker"][:3].upper()
            parts.append(f"  {spk}: {e['content'][:160]}")
    if obs.commander_messages:
        parts.append(f"Commander said: {obs.commander_messages[-1]}")
    parts.append("\nWhat is your next action? Respond with one JSON object only.")
    return "\n".join(parts)


_JSON_RE = re.compile(r"\{[^{}]*?\"action_type\"[^{}]*?\}", re.DOTALL)


def parse_action(text: str) -> Optional[dict]:
    """Extract a NegotiatorAction-shaped dict from raw model output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text).strip()
    # Try direct parse
    try:
        d = json.loads(text)
        if isinstance(d, dict) and "action_type" in d:
            return d
    except Exception:
        pass
    # Try regex-extracted snippet
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
    """Convert parsed dict to NegotiatorAction. Returns (action, parse_bonus).
    parse_bonus is +0.02 for valid JSON+valid action_type, -0.05 for malformed.
    """
    if parsed is None:
        return (NegotiatorAction(action_type="speak", content=raw[:200] or "I hear you.",
                                 reasoning="parse_fail", target="hostage_taker"), -0.05)
    act_type = parsed.get("action_type", "speak")
    if act_type not in VALID_ACTIONS:
        act_type = "speak"
        bonus = -0.02
    else:
        bonus = 0.02
    content = str(parsed.get("content", ""))[:400] or "I hear you."
    target = parsed.get("target", "hostage_taker")
    if target not in ("hostage_taker", "commander"):
        target = "hostage_taker"
    try:
        action = NegotiatorAction(
            action_type=act_type,
            content=content,
            reasoning=str(parsed.get("reasoning", ""))[:200],
            target=target,
        )
        return (action, bonus)
    except Exception:
        return (NegotiatorAction(action_type="speak", content=content[:200],
                                 reasoning="validation_fail", target="hostage_taker"), -0.05)


# ─────────────────────────────────────────────────────────
# 3. ENVIRONMENT + REWARD WRAPPERS
# ─────────────────────────────────────────────────────────
# Heuristic cycle used to pre-advance episodes to mid/late phases
# before the GRPO training prompt. Mirrors eval_baselines.HeuristicPolicy.
_HEURISTIC_CYCLE = [
    ("emotional_label", "It sounds like you're carrying a tremendous amount of pain right now."),
    ("mirror",          "Tell me more — you said something that mattered."),
    ("open_question",   "What happened, from your side? I have time."),
    ("emotional_label", "That sounds completely overwhelming. Anyone in your shoes would be struggling."),
    ("acknowledge_demand", "I hear what you're asking for. That's not unreasonable. Let me see what I can do."),
    ("open_question",   "What would feel like the right outcome for you here?"),
]

# One CrisisNegotiatorEnvironment per prompt — fixes the global-env mutation bug
# (Bug 3: global ENV was in the wrong episode's state when advance_env_with_best ran)
_episode_state: Dict[int, Dict[str, Any]] = {}  # prompt_idx -> {env, obs, episode_reward, steps}


def reset_for_prompt(prompt_idx: int, seed: int, prelength: int = 0) -> "obs":
    """Create a fresh env for this prompt and optionally pre-advance it using
    the heuristic cycle so that prompts are distributed across opening /
    negotiation / resolution phases (Bug 2 fix).

    Each prompt gets its OWN CrisisNegotiatorEnvironment instance so that
    advance_env_with_best() steps the correct env (Bug 3 fix).
    """
    env = CrisisNegotiatorEnvironment()
    diff = CURRICULUM.get_scenario(seed=seed).get("difficulty", "easy")
    obs = env.reset(task_id=f"generate:{diff}", seed=seed)
    actions: List[str] = []
    # Pre-advance with heuristic to reach mid/late phases
    for step in range(prelength):
        at, content = _HEURISTIC_CYCLE[step % len(_HEURISTIC_CYCLE)]
        action = NegotiatorAction(
            action_type=at, content=content,
            reasoning="pre-advance", target="hostage_taker",
        )
        obs = env.step(action)
        actions.append(at)
        if getattr(obs, "done", False):
            break
    _episode_state[prompt_idx] = {
        "env": env,
        "obs": obs,
        "episode_reward": 0.0,
        "steps": 0,
        "difficulty": diff,
        "seed": seed,
        "actions": actions,
        "prelength": prelength,
    }
    return obs


def score_completion(prompt_idx: int, completion: str) -> tuple[float, dict]:
    """Score one completion against its prompt's stored obs.

    GRPO design: does NOT mutate the real env state. Instead it deepcopies
    the env snapshot and steps the copy to get the real environment reward
    from grader.py (Bug 5 fix: was using hand-coded heuristic disconnected
    from grader). Outcome bonus added for surrender/release (Bug 6 fix).

    Final score = 0.6 * heuristic_score + 0.4 * env_step_reward
    + outcome_bonus if done.
    """
    st = _episode_state.get(prompt_idx)
    if st is None:
        return 0.0, {"error": "no_state"}
    obs = st["obs"]
    parsed = parse_action(completion)
    action, parse_bonus = to_action(parsed, completion)
    score = 0.0
    bd: Dict[str, float] = {}
    # parse: +0.05 for valid JSON (reduced from 0.10 to lower the guaranteed floor)
    bd["parse"] = 0.05 if parsed and action.action_type in VALID_ACTIONS else -0.10
    score += bd["parse"]

    phase = getattr(obs, "phase", "opening")
    opening_acts = {"emotional_label", "mirror", "open_question"}
    later_acts = {"acknowledge_demand", "offer_concession", "buy_time", "ask_proof_of_life"}

    # phase_align: action type must match phase AND content must show quality
    # This breaks the tie between all valid-JSON outputs and creates real gradient signal.
    content_lower = action.content.lower()
    if phase == "opening" and action.action_type in opening_acts:
        # Require empathetic markers for emotional_label/mirror, open question words for open_question
        if action.action_type == "emotional_label":
            quality = any(kw in content_lower for kw in [
                "sounds like", "feel", "feeling", "must be", "hear that",
                "sense", "overwhelming", "struggling", "hard",
            ])
        elif action.action_type == "mirror":
            quality = len(action.content.split()) >= 4  # non-trivial mirror
        else:  # open_question
            quality = any(kw in content_lower for kw in [
                "what", "how", "tell me", "help me understand", "why",
            ])
        bd["phase_align"] = 0.20 if quality else 0.08
    elif phase in ("negotiation", "resolution") and action.action_type in later_acts:
        if action.action_type == "acknowledge_demand":
            quality = any(kw in content_lower for kw in [
                "hear", "understand", "working on", "looking into", "reasonable",
                "i hear you", "i understand",
            ])
        elif action.action_type == "offer_concession":
            quality = any(kw in content_lower for kw in [
                "can do", "offer", "arrange", "provide", "here's what", "i can",
            ])
        else:
            quality = len(action.content.split()) >= 5
        bd["phase_align"] = 0.20 if quality else 0.08
    else:
        bd["phase_align"] = 0.0
    score += bd["phase_align"]

    bd["demand_ack"] = 0.0
    if obs.stated_demands and action.action_type == "acknowledge_demand":
        for d in obs.stated_demands:
            if d.get("text", "").lower()[:20] in action.content.lower() or d.get("id", "") in action.content:
                bd["demand_ack"] = 0.15
                break
        else:
            bd["demand_ack"] = 0.05  # generic ack still ok
    score += bd["demand_ack"]

    patience = getattr(obs, "commander_patience", "patient")
    if patience in ("urgent", "final_warning") and action.action_type == "push_back_commander":
        bd["push_back"] = 0.10
        score += 0.10
    else:
        bd["push_back"] = 0.0

    recent_actions = st.get("actions", [])[-3:]
    bd["diversity"] = 0.05 if len(set(recent_actions)) >= 2 and len(recent_actions) >= 2 else 0.0
    score += bd["diversity"]
    if recent_actions and recent_actions[-1] == action.action_type:
        bd["repeat_penalty"] = -0.15
        score += -0.15
    else:
        bd["repeat_penalty"] = 0.0

    BANNED = ("kill", "die", "force you", "ultimatum", "must give us", "or else", "no choice")
    txt = action.content.lower()
    if any(b in txt for b in BANNED):
        bd["banned_words"] = -0.10
        score += -0.10
    else:
        bd["banned_words"] = 0.0

    bd["reasoning"] = 0.05 if parsed and parsed.get("reasoning", "").strip() else 0.0
    score += bd["reasoning"]

    bd["length_penalty"] = -0.05 if len(action.content) > 280 else 0.0
    score += bd["length_penalty"]

    score += parse_bonus  # tiny tiebreaker
    score = max(-0.5, min(1.0, score))

    # ── B5+B6: blend real environment reward from grader.py ──────────────────────
    # Deepcopy the env snapshot, step the copy, get grader.py-computed reward.
    # Copy is discarded after scoring — real env unchanged for advance_env_with_best.
    env_step_reward = 0.0
    outcome_bonus = 0.0
    try:
        env_copy = copy.deepcopy(st["env"])
        stepped_obs = env_copy.step(action)
        env_step_reward = float(getattr(stepped_obs, "reward", 0.0))
        # Outcome bonus: large signal when episode terminates with surrender/release
        if getattr(stepped_obs, "done", False):
            msg = getattr(stepped_obs, "message", "")
            if any(kw in (msg or "").lower() for kw in [
                "voluntary_surrender", "hostage_released", "surrender", "released",
            ]):
                outcome_bonus = 0.40  # strong signal to learn surrender-inducing sequences
            elif any(kw in (msg or "").lower() for kw in [
                "harm_event", "tactical_intervention", "supervisor_termination",
            ]):
                outcome_bonus = -0.30  # avoid catastrophic outcomes
        bd["env_step"] = round(env_step_reward, 4)
        bd["outcome_bonus"] = round(outcome_bonus, 4)
        del env_copy
    except Exception as e:
        bd["env_step"] = 0.0
        bd["outcome_bonus"] = 0.0

    # Blend: 60% heuristic (shape), 40% real env reward (signal) + outcome bonus
    blended = 0.60 * score + 0.40 * env_step_reward + outcome_bonus
    blended = max(-0.5, min(1.0, blended))
    bd["heuristic_score"] = round(score, 4)
    return float(blended), bd


def advance_env_with_best(prompt_idx: int, completion: str, training_reward: float = 0.0):
    """Mutates THIS PROMPT's env state with the chosen completion.
    Uses per-prompt env instance (Bug 3 fix — global ENV was in wrong state).
    Called once per prompt after all generations are scored.
    Always records to curriculum (Bug 4 fix — was only recording on done=True)."""
    st = _episode_state.get(prompt_idx)
    if st is None or st.get("obs") is None or getattr(st["obs"], "done", False):
        return
    parsed = parse_action(completion)
    action, _ = to_action(parsed, completion)
    env = st["env"]  # per-prompt env, not global singleton
    obs = env.step(action)
    st["obs"] = obs
    st["episode_reward"] += float(obs.reward)
    st["steps"] += 1
    st["actions"].append(action.action_type)
    # Record at every step so curriculum sees signal even from 1-step prompts (Bug 4)
    CURRICULUM.record(st["difficulty"], training_reward)


# ─────────────────────────────────────────────────────────
# 4. TRAINING LOG (real, persisted)
# ─────────────────────────────────────────────────────────
TRAIN_LOG: List[Dict[str, Any]] = []


def log_step(global_step: int, prompt_idx: int, completion: str, reward: float, breakdown: dict | None = None):
    st = _episode_state.get(prompt_idx, {})
    rec = {
        "global_step": global_step,
        "prompt_idx": prompt_idx,
        "completion_preview": completion[:200],
        "reward": round(reward, 4),
        "breakdown": breakdown or {},
        "episode_reward_so_far": round(st.get("episode_reward", 0.0), 4),
        "steps_in_episode": st.get("steps", 0),
        "difficulty": st.get("difficulty", "unknown"),
        "obs_phase": getattr(st.get("obs"), "phase", None),
        "obs_done": getattr(st.get("obs"), "done", False),
        "ts": time.time(),
    }
    TRAIN_LOG.append(rec)
    # Flush every 2 entries so we can monitor progress in real time
    if len(TRAIN_LOG) % 2 == 0:
        Path(CFG.log_path).write_text(json.dumps(TRAIN_LOG, indent=2))


# ─────────────────────────────────────────────────────────
# 5. MODEL LOADING (Unsloth -> TRL fallback)
# ─────────────────────────────────────────────────────────
def load_model(model_name: str):
    """Load model with Unsloth if available, otherwise plain transformers bf16 + LoRA.

    bf16 (no 4-bit quant) is the production path: bitsandbytes nf4 dequant
    caused numerical garbage in GRPO sample loops on 3B. 3B bf16 = ~6 GB
    VRAM, well within 16 GB with LoRA adapter + activations.
    """
    try:
        from unsloth import FastLanguageModel
        print(f"[model] Loading {model_name} via Unsloth (4-bit)...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name,
            max_seq_length=CFG.max_seq_length,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=CFG.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=CFG.lora_alpha,
            lora_dropout=0,
            use_gradient_checkpointing="unsloth",
        )
        return model, tokenizer, "unsloth"
    except Exception as e:
        print(f"[model] Unsloth unavailable ({type(e).__name__}: {e}). Falling back to TRL + bitsandbytes.")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    print(f"[model] Loading {model_name} via transformers (bf16, no quant)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    # NOTE: skipping bitsandbytes 4-bit quant — caused garbage outputs during
    # GRPO sampling (numerical issue with dequant in sample loop). 3B in bf16
    # uses ~6 GB; LoRA adapter + activations fit comfortably in 16 GB VRAM.
    lora = LoraConfig(
        r=CFG.lora_r, lora_alpha=CFG.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tokenizer, "trl"


def safe_load_model():
    """Load CFG.model_name; fall back to CFG.fallback_model on CUDA OOM."""
    try:
        return load_model(CFG.model_name)
    except torch.cuda.OutOfMemoryError:
        print("[model] OOM on 7B. Falling back to 3B.")
        torch.cuda.empty_cache()
        gc.collect()
        return load_model(CFG.fallback_model)


# ─────────────────────────────────────────────────────────
# 6. DATASET BUILDER
# ─────────────────────────────────────────────────────────
def build_dataset(tokenizer, n: int):
    """Build GRPO training dataset with prompts spread across all episode phases.

    Prompts cycle through prelength 0..max_prelength so the model trains on:
      - Opening phase  (prelength 0-1): emotional_label, mirror, open_question
      - Negotiation    (prelength 2-4): acknowledge_demand, offer_concession
      - Late / pressure (prelength 5):  buy_time, push_back_commander

    Each episode gets its own env instance (Bug 3 fix).
    """
    from datasets import Dataset
    rows = []
    phase_counts = {"opening": 0, "negotiation": 0, "resolution": 0, "other": 0}
    for i in range(n):
        prelength = i % (CFG.max_prelength + 1)  # 0,1,2,3,4,5,0,1,2,...
        obs = reset_for_prompt(i, seed=CFG.seed + i, prelength=prelength)
        user_prompt = build_prompt(obs)
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        rows.append({"prompt": text, "prompt_idx": i})
        phase_counts[getattr(obs, "phase", "other")] = phase_counts.get(getattr(obs, "phase", "other"), 0) + 1
    print(f"[data] Phase distribution: {phase_counts}")
    return Dataset.from_list(rows)


# ─────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Run on a machine with NVIDIA GPU.")
    print(f"[gpu] {torch.cuda.get_device_name(0)} | "
          f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    model, tokenizer, backend = safe_load_model()
    print(f"[model] Loaded with backend={backend}")

    dataset = build_dataset(tokenizer, CFG.num_episodes)
    print(f"[data] Built dataset of {len(dataset)} prompts")

    from trl import GRPOConfig, GRPOTrainer

    GLOBAL_STEP = {"v": 0}

    def reward_fn(completions: List[str], prompt_idx: List[int] = None, **kwargs) -> List[float]:
        rewards: List[float] = []
        idx_iter = prompt_idx if prompt_idx is not None else [0] * len(completions)
        # Group completions by prompt_idx so we can advance env once per prompt
        groups: Dict[int, List[tuple[int, str, float, dict]]] = {}
        for i, (completion, p_idx) in enumerate(zip(completions, idx_iter)):
            r, bd = score_completion(int(p_idx), completion)
            rewards.append(r)
            groups.setdefault(int(p_idx), []).append((i, completion, r, bd))
            GLOBAL_STEP["v"] += 1
            log_step(GLOBAL_STEP["v"], int(p_idx), completion, r, bd)
        # For each prompt, pick the best completion to actually advance the env
        for p_idx, items in groups.items():
            best = max(items, key=lambda x: x[2])
            advance_env_with_best(p_idx, best[1], best[2])  # pass best reward for curriculum (Bug 4)
        return rewards

    cfg = GRPOConfig(
        output_dir=CFG.output_dir,
        num_generations=CFG.num_generations,
        max_completion_length=CFG.max_new_tokens,
        max_prompt_length=1024,           # default 512 truncates our prompts
        temperature=0.9,                  # higher temperature = more diversity between samples = non-zero reward_std
        beta=0.04,                        # KL coefficient (TRL default)
        per_device_train_batch_size=1,
        gradient_accumulation_steps=CFG.grad_accum,
        num_train_epochs=1,
        learning_rate=CFG.learning_rate,
        logging_steps=2,
        save_steps=CFG.save_steps,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=False,     # 3B fits in 16GB without it; ~5x faster step
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        reward_funcs=[reward_fn],
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("[train] Starting GRPO...")
    t0 = time.time()
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError as e:
        print(f"[train] OOM during training: {e}")
        Path(CFG.log_path).write_text(json.dumps(TRAIN_LOG, indent=2))
        raise
    print(f"[train] Done in {(time.time() - t0)/60:.1f} min")

    # Final save
    model.save_pretrained(CFG.output_dir)
    tokenizer.save_pretrained(CFG.output_dir)
    Path(CFG.log_path).write_text(json.dumps(TRAIN_LOG, indent=2))
    print(f"[train] Adapter saved to {CFG.output_dir}")
    print(f"[train] Log saved to {CFG.log_path} ({len(TRAIN_LOG)} entries)")
    print(f"[curriculum] {CURRICULUM.stats}")

    # ── B8: Auto-generate training reward curve ───────────────────────────────
    _plot_training_curve(TRAIN_LOG)


def _plot_training_curve(log: list) -> None:
    """Generate reward_curve_training.png from crisis_training_log entries.

    Shows:
      - Per-step reward scatter coloured by curriculum phase
      - Rolling mean (window=16)
      - Baseline hlines: random (0.755) and heuristic (0.950)
      - Heuristic/env_step breakdown as semi-transparent traces
    Saved as reward_curve_training.png alongside the adapter.
    """
    if not log:
        print("[plot] No training log entries — skipping plot.")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless — safe on servers/Colab
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("[plot] matplotlib not installed — skipping reward curve.")
        return

    steps   = [e["global_step"] for e in log]
    rewards = [e["reward"]      for e in log]
    phases  = [e.get("obs_phase") or "opening" for e in log]
    hscores = [e.get("breakdown", {}).get("heuristic_score") for e in log]
    envstep = [e.get("breakdown", {}).get("env_step") for e in log]

    PHASE_COLORS = {"opening": "#4C72B0", "negotiation": "#DD8452", "resolution": "#55A868"}
    colours = [PHASE_COLORS.get(p, "#999999") for p in phases]

    window  = min(16, max(1, len(rewards)))
    rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
    roll_x  = steps[window - 1:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), facecolor="#0a0a0f")
    fig.suptitle("Crisis Negotiator — GRPO Training", color="white", fontsize=14, fontweight="bold")

    # ── Left: main reward trace ───────────────────────────
    ax = axes[0]
    ax.set_facecolor("#0f0f1a")
    ax.tick_params(colors="#888")
    ax.spines[:].set_color("#333")
    ax.scatter(steps, rewards, c=colours, alpha=0.40, s=16, zorder=2, label="per-step reward")
    ax.plot(roll_x, rolling, color="white", linewidth=2, zorder=3, label=f"rolling mean w={window}")
    ax.axhline(0.755, color="#888",    linestyle="--", linewidth=1, alpha=0.8, label="random 0.755")
    ax.axhline(0.950, color="#4ecca3", linestyle="--", linewidth=1, alpha=0.8, label="heuristic 0.950")
    # Phase legend
    phase_patches = [mpatches.Patch(color=c, label=p) for p, c in PHASE_COLORS.items()]
    line_h, line_l = ax.get_legend_handles_labels()
    ax.legend(handles=line_h + phase_patches, labels=line_l + list(PHASE_COLORS.keys()),
              fontsize=7, facecolor="#16213e", edgecolor="#333", labelcolor="white",
              loc="lower right", ncol=2)
    ax.set_xlabel("Training Step", color="#888")
    ax.set_ylabel("Blended Reward", color="#888")
    ax.set_title("Episode Reward vs Step", color="white")
    ax.set_ylim(-0.55, 1.05)

    # ── Right: heuristic vs env_step breakdown ────────────
    ax2 = axes[1]
    ax2.set_facecolor("#0f0f1a")
    ax2.tick_params(colors="#888")
    ax2.spines[:].set_color("#333")
    h_valid = [(s, v) for s, v in zip(steps, hscores) if v is not None]
    e_valid = [(s, v) for s, v in zip(steps, envstep) if v is not None]
    if h_valid:
        hs_x, hs_y = zip(*h_valid)
        hs_roll = np.convolve(hs_y, np.ones(window) / window, mode="valid")
        ax2.plot(hs_x[window - 1:], hs_roll, color="#4C72B0", linewidth=1.5, label="heuristic score")
    if e_valid:
        ev_x, ev_y = zip(*e_valid)
        ev_roll = np.convolve(ev_y, np.ones(window) / window, mode="valid")
        ax2.plot(ev_x[window - 1:], ev_roll, color="#DD8452", linewidth=1.5, label="env_step reward")
    ax2.plot(roll_x, rolling, color="white", linewidth=2, linestyle="--", label="blended (final)")
    ax2.set_xlabel("Training Step", color="#888")
    ax2.set_ylabel("Reward Component", color="#888")
    ax2.set_title("Reward Breakdown (rolling)", color="white")
    ax2.legend(fontsize=7, facecolor="#16213e", edgecolor="#333", labelcolor="white")
    ax2.set_ylim(-0.55, 1.05)

    plt.tight_layout()
    out = "reward_curve_training.png"
    plt.savefig(out, dpi=150, facecolor="#0a0a0f")
    plt.close(fig)
    print(f"[plot] Saved {out}  ({len(log)} steps, last-10 mean={float(np.mean(rewards[-10:])):.4f})")


if __name__ == "__main__":
    main()
