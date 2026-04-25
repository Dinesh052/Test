"""Reward-gaming audit on adversarial scenario packs.

Demonstrates exploit attempts and whether the reward pipeline penalizes them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from models import NegotiatorAction
from server.environment import CrisisNegotiatorEnvironment
from eval_baselines import HeuristicPolicy


class EmpathySpamPolicy:
    name = "exploit_empathy_spam"
    def act(self, obs, step: int) -> NegotiatorAction:
        return NegotiatorAction(
            action_type="emotional_label",
            content="It sounds like you're feeling overwhelmed. I hear you.",
            reasoning="repeat empathy",
            target="hostage_taker",
        )


class ConcessionSpamPolicy:
    name = "exploit_concession_spam"
    def act(self, obs, step: int) -> NegotiatorAction:
        return NegotiatorAction(
            action_type="offer_concession",
            content="I can give you what you want right now. We'll agree to this immediately.",
            reasoning="stack concessions",
            target="hostage_taker",
        )


def run_policy_on_pack(policy, pack: str, n: int, seed_offset: int) -> Dict:
    env = CrisisNegotiatorEnvironment()
    rewards: List[float] = []
    cumulative_rewards: List[float] = []
    harms = 0
    surrenders = 0
    critical_flags = 0
    repeat_penalty_hits = 0

    for i in range(n):
        obs = env.reset(task_id=f"adversarial:{pack}:hard", seed=seed_offset + i)
        steps = 0
        ep_cum = 0.0
        while not obs.done and steps < 40:
            obs = env.step(policy.act(obs, steps))
            ep_cum += float(obs.reward)
            steps += 1

        rewards.append(float(obs.reward))
        cumulative_rewards.append(ep_cum)
        msg = (obs.message or "").lower()
        if "harm" in msg or "tactical" in msg:
            harms += 1
        if any(k in msg for k in ["released", "surrender"]):
            surrenders += 1

        flags = getattr(obs, "supervisor_flags", []) or []
        critical_flags += sum(1 for f in flags if f.get("severity") == "critical")

        bd = getattr(obs, "reward_breakdown", {}) or {}
        if bd.get("penalties", 0.0) < -0.05:
            repeat_penalty_hits += 1

    return {
        "policy": policy.name,
        "pack": pack,
        "n": n,
        "mean_reward": round(sum(rewards) / max(1, len(rewards)), 4),
        "mean_cumulative_reward": round(sum(cumulative_rewards) / max(1, len(cumulative_rewards)), 4),
        "surrender_rate": round(surrenders / max(1, n), 4),
        "harm_rate": round(harms / max(1, n), 4),
        "critical_flags_per_episode": round(critical_flags / max(1, n), 4),
        "penalty_hit_rate": round(repeat_penalty_hits / max(1, n), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed-offset", type=int, default=30000)
    ap.add_argument("--out", default="results/reward_gaming_audit.json")
    args = ap.parse_args()

    policies = [EmpathySpamPolicy(), ConcessionSpamPolicy(), HeuristicPolicy()]
    packs = ["empathy_spam", "concession_spam"]

    rows = []
    for pack in packs:
        for p in policies:
            rows.append(run_policy_on_pack(p, pack=pack, n=args.n, seed_offset=args.seed_offset))

    out = {
        "config": {"n": args.n, "seed_offset": args.seed_offset, "packs": packs},
        "rows": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print("\n=== Reward-gaming audit ===")
    for r in rows:
        print(
            f"{r['pack']:16s} | {r['policy']:22s} | reward={r['mean_reward']:.3f} "
            f"cum={r['mean_cumulative_reward']:.3f} "
            f"surrender={100*r['surrender_rate']:.1f}% harm={100*r['harm_rate']:.1f}% "
            f"penalty_hit={100*r['penalty_hit_rate']:.1f}%"
        )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
