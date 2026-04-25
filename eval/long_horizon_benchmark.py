"""Long-horizon benchmark split (25–40 turn budget, delayed pivot)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from models import NegotiatorAction
from server.environment import CrisisNegotiatorEnvironment
from eval_baselines import RandomPolicy, HeuristicPolicy, TrainedPolicy


def run(policy, n: int, seed_offset: int = 40000) -> Dict[str, Any]:
    env = CrisisNegotiatorEnvironment()
    rewards, steps = [], []
    surrender, harm = 0, 0

    for i in range(n):
        obs = env.reset(task_id="generate:long", seed=seed_offset + i)
        ep_steps = 0
        while not obs.done and ep_steps < 45:
            action = policy.act(obs, ep_steps)
            if not isinstance(action, NegotiatorAction):
                action = NegotiatorAction(**action.model_dump())
            obs = env.step(action)
            ep_steps += 1

        rewards.append(float(obs.reward))
        steps.append(ep_steps)
        msg = (obs.message or "").lower()
        if any(k in msg for k in ["released", "surrender"]):
            surrender += 1
        if any(k in msg for k in ["harm", "tactical"]):
            harm += 1

    return {
        "n": n,
        "mean_reward": round(sum(rewards) / n, 4),
        "mean_steps": round(sum(steps) / n, 2),
        "surrender_rate": round(surrender / n, 4),
        "harm_rate": round(harm / n, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed-offset", type=int, default=40000)
    ap.add_argument("--include-trained", action="store_true")
    ap.add_argument("--adapter", default="./crisis-negotiator-trained-v2")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--out", default="results/long_horizon_benchmark.json")
    args = ap.parse_args()

    results: Dict[str, Dict[str, Any]] = {
        "random": run(RandomPolicy(seed=args.seed_offset), n=args.n, seed_offset=args.seed_offset),
        "heuristic": run(HeuristicPolicy(), n=args.n, seed_offset=args.seed_offset),
    }

    if args.include_trained:
        if not Path(args.adapter).exists():
            raise FileNotFoundError(f"Adapter not found: {args.adapter}")
        tp = TrainedPolicy(adapter_dir=args.adapter, base_model=args.base_model)
        results["trained"] = run(tp, n=args.n, seed_offset=args.seed_offset)

    out = {
        "config": {
            "n": args.n,
            "seed_offset": args.seed_offset,
            "task": "generate:long",
            "include_trained": args.include_trained,
        },
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print("\n=== Long-horizon benchmark ===")
    for k, v in results.items():
        print(
            f"{k:9s} reward={v['mean_reward']:.3f} steps={v['mean_steps']:.2f} "
            f"surrender={100*v['surrender_rate']:.1f}% harm={100*v['harm_rate']:.1f}%"
        )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
