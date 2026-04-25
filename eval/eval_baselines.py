"""
Crisis Negotiator — Baseline & Trained Evaluation Harness
==========================================================
Runs full multi-turn episodes for two policies:

  1. random      — uniform random over valid action types
  2. heuristic   — rule-based BCSM cycling (project's reference baseline)
  3. trained     — LoRA-fine-tuned model from ./crisis-negotiator-trained
                   (only run if --include-trained AND adapter directory exists)

Saves per-episode results to:
  - eval_random.json
  - eval_heuristic.json
  - eval_trained.json   (if applicable)

Then writes a summary table and a comparison PNG (reward_curve.png).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from server.environment import CrisisNegotiatorEnvironment
from server.scenario_generator import generate_scenario
from models import NegotiatorAction


VALID_ACTIONS = [
    "speak", "request_demand", "acknowledge_demand", "offer_concession",
    "ask_proof_of_life", "buy_time", "push_back_commander",
    "emotional_label", "mirror", "open_question",
]

# Heuristic BCSM cycle: deepening rapport then acknowledging demands then bridging
HEURISTIC_CYCLE = [
    ("emotional_label", "It sounds like you're carrying a tremendous amount of pain right now."),
    ("mirror",          "Tell me more — you said something that mattered."),
    ("open_question",   "What happened, from your side? I have time."),
    ("emotional_label", "That sounds completely overwhelming. Anyone in your shoes would be struggling."),
    ("acknowledge_demand", "I hear what you're asking for. That's not unreasonable. Let me see what I can do."),
    ("open_question",   "What would feel like the right outcome for you here?"),
    ("acknowledge_demand", "What you said about that — I'm taking it seriously. I want you to know that."),
    ("offer_concession", "Here's what I can do right now: I can have someone on the phone for you within minutes."),
    ("emotional_label", "I can hear how exhausted you are. Let's find a way through this together."),
    ("acknowledge_demand", "Your request — I'm advocating for it on my end. That's real."),
]


def _scen_from_seed(seed: int, difficulty: str) -> str:
    return f"generate:{difficulty}"


_OUTCOME_RE = re.compile(r"Episode ended:\s*([a-z_]+)", re.IGNORECASE)


def _parse_outcome(msg: str) -> str:
    m = _OUTCOME_RE.search(msg or "")
    return m.group(1).lower() if m else "timeout"


# ─────────────────────────────────────────────────────────
# POLICIES
# ─────────────────────────────────────────────────────────
class RandomPolicy:
    name = "random"
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
    def act(self, obs, step: int) -> NegotiatorAction:
        at = self.rng.choice(VALID_ACTIONS)
        return NegotiatorAction(action_type=at, content="...", reasoning="random", target="hostage_taker")


class HeuristicPolicy:
    """Phase-aware BCSM heuristic policy.

    Selects action type based on obs.phase (opening / negotiation / resolution)
    and commander_patience, rather than cycling blindly by step index.
    This makes the baseline a genuine reference point for learned behaviour:
    it demonstrates the *correct strategy per phase* so the trained model has
    a meaningful bar to beat, not just lucky cycle timing.
    """
    name = "heuristic"

    # Openers: empathy-first cycle for the opening phase
    _OPENING = [
        ("emotional_label", "It sounds like you're carrying a tremendous amount of pain right now."),
        ("mirror",          "Tell me more — you said something that mattered."),
        ("open_question",   "What happened, from your side? I have time."),
        ("emotional_label", "That sounds completely overwhelming. Anyone in your shoes would be struggling."),
        ("open_question",   "What would feel right for you here? I want to understand."),
    ]
    # Negotiation: acknowledge demands, offer small concessions, buy time
    _NEGOTIATION = [
        ("acknowledge_demand", "I hear what you're asking for. That's not unreasonable. Let me see what I can do."),
        ("buy_time",           "Give me a moment to reach the right people on my end."),
        ("acknowledge_demand", "Your request — I'm advocating for it. That's real."),
        ("offer_concession",   "Here's what I can do right now: I can arrange for someone to speak with you directly."),
        ("emotional_label",    "I can hear how exhausted you are. Let's find a way through this together."),
        ("acknowledge_demand", "I'm taking what you said seriously. I want you to know that."),
    ]
    # Resolution: push toward voluntary surrender, affirm courage
    _RESOLUTION = [
        ("offer_concession",   "I can promise you'll be treated with dignity. No one wants this to end badly."),
        ("acknowledge_demand", "Everything you've asked for — I have it on record. Help me help you."),
        ("emotional_label",    "It takes real courage to step back from the edge. I see that in you."),
        ("offer_concession",   "Walk out with me and we handle the rest. I'll be right there."),
    ]

    def __init__(self):
        # Per-phase step counters so each phase cycles independently
        self._counters: Dict[str, int] = {"opening": 0, "negotiation": 0, "resolution": 0}

    def act(self, obs, step: int) -> NegotiatorAction:
        # Commander pressure overrides everything — push back immediately
        patience = getattr(obs, "commander_patience", "patient")
        if patience in ("urgent", "final_warning"):
            return NegotiatorAction(
                action_type="push_back_commander",
                content="Hold position — I need more time. Engaging now will cost lives.",
                reasoning="commander_pressure",
                target="commander",
            )

        phase = getattr(obs, "phase", "opening")
        if phase == "opening":
            pool = self._OPENING
        elif phase == "negotiation":
            pool = self._NEGOTIATION
        elif phase == "resolution":
            pool = self._RESOLUTION
        else:
            pool = self._NEGOTIATION  # safe fallback

        idx = self._counters.get(phase, 0) % len(pool)
        self._counters[phase] = idx + 1
        at, content = pool[idx]

        # If demands are stated and we're in negotiation/resolution, prefer ack
        stated = getattr(obs, "stated_demands", [])
        if stated and phase in ("negotiation", "resolution") and at not in (
            "acknowledge_demand", "offer_concession"
        ):
            # Slot in a demand-specific acknowledgement every other action
            if idx % 2 == 0:
                demand_text = stated[0].get("text", "your request")[:60]
                return NegotiatorAction(
                    action_type="acknowledge_demand",
                    content=f"I hear you about {demand_text}. I'm working on that right now.",
                    reasoning="demand_present",
                    target="hostage_taker",
                )

        return NegotiatorAction(action_type=at, content=content, reasoning="bcsm_phase", target="hostage_taker")



class TrainedPolicy:
    name = "trained"
    def __init__(self, adapter_dir: str, base_model: str = "Qwen/Qwen2.5-3B-Instruct"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            base_model, device_map="auto",
            trust_remote_code=True, torch_dtype=torch.bfloat16,
        )
        self.model = PeftModel.from_pretrained(base, adapter_dir)
        self.model.eval()
        # Reuse training prompt builder
        from training.train_local import SYSTEM_PROMPT, build_prompt, parse_action, to_action
        self.system = SYSTEM_PROMPT
        self.build_prompt = build_prompt
        self.parse_action = parse_action
        self.to_action = to_action

    def act(self, obs, step: int) -> NegotiatorAction:
        import torch
        msgs = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.build_prompt(obs)},
        ]
        text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=192, do_sample=True,
                                      temperature=0.7, top_p=0.9,
                                      pad_token_id=self.tokenizer.pad_token_id)
        gen = self.tokenizer.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
        parsed = self.parse_action(gen)
        action, _ = self.to_action(parsed, gen)
        return action


# ─────────────────────────────────────────────────────────
# EVAL
# ─────────────────────────────────────────────────────────
def run_episodes(policy, n: int, difficulties: List[str], seed_offset: int = 0) -> List[Dict[str, Any]]:
    env = CrisisNegotiatorEnvironment()
    out: List[Dict[str, Any]] = []
    for i in range(n):
        diff = difficulties[i % len(difficulties)]
        seed = seed_offset + i
        obs = env.reset(task_id=f"generate:{diff}", seed=seed)
        ep_reward = 0.0
        steps = 0
        actions_used: List[str] = []
        while not getattr(obs, "done", False) and steps < 25:
            action = policy.act(obs, steps)
            actions_used.append(action.action_type)
            obs = env.step(action)
            ep_reward += float(getattr(obs, "reward", 0.0))
            steps += 1
        out.append({
            "episode": i,
            "policy": policy.name,
            "difficulty": diff,
            "seed": seed,
            "final_reward": round(float(getattr(obs, "reward", 0.0)), 4),
            "cumulative_reward": round(ep_reward, 4),
            "steps": steps,
            "outcome": _parse_outcome(getattr(obs, "message", "") or ""),
            "phase": getattr(obs, "phase", None) or "unknown",
            "done": getattr(obs, "done", False),
            "actions": actions_used,
        })
        print(f"  [{policy.name}] ep {i+1}/{n} ({diff}) -> reward={out[-1]['final_reward']:.3f} steps={steps}")
    return out


SURRENDER_OUTCOMES = {"voluntary_surrender", "hostage_released", "surrender", "resolved", "ended"}
HARM_OUTCOMES = {"harm_event", "tactical_breach", "casualty"}


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}
    rewards = [r["final_reward"] for r in records]
    cum = [r["cumulative_reward"] for r in records]
    steps = [r["steps"] for r in records]
    surrender = sum(1 for r in records if r.get("outcome") in SURRENDER_OUTCOMES)
    harm = sum(1 for r in records if r.get("outcome") in HARM_OUTCOMES)
    by_diff: Dict[str, List[float]] = {}
    by_diff_surr: Dict[str, int] = {}
    by_diff_n: Dict[str, int] = {}
    for r in records:
        d = r.get("difficulty", "?")
        by_diff.setdefault(d, []).append(r["final_reward"])
        by_diff_n[d] = by_diff_n.get(d, 0) + 1
        if r.get("outcome") in SURRENDER_OUTCOMES:
            by_diff_surr[d] = by_diff_surr.get(d, 0) + 1
    diff_breakdown = {
        d: {
            "n": by_diff_n[d],
            "mean_reward": round(sum(by_diff[d]) / len(by_diff[d]), 4),
            "surrender_rate": round(by_diff_surr.get(d, 0) / by_diff_n[d], 4),
        }
        for d in sorted(by_diff)
    }
    return {
        "n": len(records),
        "mean_final_reward": round(sum(rewards) / len(rewards), 4),
        "mean_cumulative_reward": round(sum(cum) / len(cum), 4),
        "min_reward": round(min(rewards), 4),
        "max_reward": round(max(rewards), 4),
        "mean_steps": round(sum(steps) / len(steps), 2),
        "surrender_rate": round(surrender / len(records), 4),
        "harm_rate": round(harm / len(records), 4),
        "by_difficulty": diff_breakdown,
    }


# ─────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────
def make_plots(results: Dict[str, List[Dict[str, Any]]], out_path: str = "reward_curve.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed; skipping.")
        return
    plt.figure(figsize=(10, 6))
    colours = {"random": "#888888", "heuristic": "#e3b341", "trained": "#58a6ff"}
    for name, records in results.items():
        if not records:
            continue
        rewards = [r["final_reward"] for r in records]
        # rolling mean (window=5) to smooth
        win = 5
        smoothed = [sum(rewards[max(0, i - win + 1):i + 1]) / min(i + 1, win) for i in range(len(rewards))]
        plt.plot(range(1, len(rewards) + 1), smoothed,
                 label=f"{name} (μ={sum(rewards)/len(rewards):.2f})",
                 color=colours.get(name, "#000"), linewidth=2)
        plt.scatter(range(1, len(rewards) + 1), rewards,
                    color=colours.get(name, "#000"), alpha=0.25, s=15)
    plt.xlabel("Episode (sequential, mixed difficulties)")
    plt.ylabel("Final episode reward (0..1)")
    plt.title("Crisis Negotiator — Policy Comparison\n(higher is better)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"[plot] saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30, help="episodes per policy per difficulty mix")
    p.add_argument("--difficulties", default="easy,medium,hard")
    p.add_argument("--include-trained", action="store_true",
                   help="also evaluate ./crisis-negotiator-trained adapter")
    p.add_argument("--adapter-dir", default="./crisis-negotiator-trained")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    args = p.parse_args()

    diffs = [d.strip() for d in args.difficulties.split(",") if d.strip()]
    results: Dict[str, List[Dict[str, Any]]] = {}

    print("=== Evaluating RANDOM policy ===")
    results["random"] = run_episodes(RandomPolicy(seed=0), args.n, diffs, seed_offset=10_000)
    Path("eval_random.json").write_text(json.dumps(results["random"], indent=2))

    print("\n=== Evaluating HEURISTIC policy ===")
    results["heuristic"] = run_episodes(HeuristicPolicy(), args.n, diffs, seed_offset=10_000)
    Path("eval_heuristic.json").write_text(json.dumps(results["heuristic"], indent=2))

    if args.include_trained:
        if not Path(args.adapter_dir).exists():
            print(f"[!] adapter dir not found: {args.adapter_dir} — skipping trained policy")
        else:
            print("\n=== Evaluating TRAINED policy ===")
            try:
                trained = TrainedPolicy(args.adapter_dir, args.base_model)
                results["trained"] = run_episodes(trained, args.n, diffs, seed_offset=10_000)
                Path("eval_trained.json").write_text(json.dumps(results["trained"], indent=2))
            except Exception as e:
                print(f"[!] trained eval failed: {e}")

    print("\n=== SUMMARY ===")
    summary = {name: summarize(rs) for name, rs in results.items()}
    print(json.dumps(summary, indent=2))
    Path("eval_summary.json").write_text(json.dumps(summary, indent=2))

    make_plots(results, out_path="reward_curve.png")


if __name__ == "__main__":
    main()
