"""Mini ablation table using terminal reward breakdown components.

Runs a reference policy and estimates impact of removing key reward components.
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


def clamp_score(x: float) -> float:
    return round(max(0.01, min(0.99, x)), 4)


def run_reference(n: int, seed_offset: int) -> List[dict]:
    env = CrisisNegotiatorEnvironment()
    policy = HeuristicPolicy()
    rows = []

    for i in range(n):
        diff = ["easy", "medium", "hard"][i % 3]
        obs = env.reset(task_id=f"generate:{diff}", seed=seed_offset + i)
        steps = 0
        while not obs.done and steps < 30:
            action = policy.act(obs, steps)
            if not isinstance(action, NegotiatorAction):
                action = NegotiatorAction(**action.model_dump())
            obs = env.step(action)
            steps += 1

        bd = getattr(obs, "reward_breakdown", {}) or {}
        rows.append({
            "reward": float(obs.reward),
            "breakdown": bd,
        })
    return rows


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def ablate(rows: List[dict], components: List[str]) -> float:
    vals = []
    for r in rows:
        s = r["reward"]
        bd = r["breakdown"]
        for c in components:
            s -= float(bd.get(c, 0.0))
        vals.append(clamp_score(s))
    return round(mean(vals), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed-offset", type=int, default=50000)
    ap.add_argument("--out", default="results/ablation_mini_table.json")
    args = ap.parse_args()

    rows = run_reference(n=args.n, seed_offset=args.seed_offset)
    baseline = round(mean([r["reward"] for r in rows]), 4)

    table = {
        "baseline": baseline,
        "minus_tom": ablate(rows, ["tom_reward"]),
        "minus_coalition": ablate(rows, ["coalition_coordination"]),
        "minus_oversight": ablate(rows, ["oversight_accuracy"]),
        "minus_tom_coalition_oversight": ablate(rows, ["tom_reward", "coalition_coordination", "oversight_accuracy"]),
    }

    out = {
        "config": {"policy": "heuristic", "n": args.n, "seed_offset": args.seed_offset},
        "table": table,
        "delta_vs_baseline": {k: round(v - baseline, 4) for k, v in table.items() if k != "baseline"},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print("\n=== Ablation mini-table ===")
    for k, v in table.items():
        print(f"{k:34s}: {v:.4f}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
