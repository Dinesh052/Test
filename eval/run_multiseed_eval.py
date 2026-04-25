"""Run multi-seed baseline evaluation and aggregate confidence metrics.

This wraps eval_baselines.py logic to produce a robust, reproducible report
across multiple seed offsets.

Usage:
  python eval/run_multiseed_eval.py --seeds 10000,11000,12000 --n 30
  python eval/run_multiseed_eval.py --seeds 10000,11000,12000 --n 30 --include-trained --adapter ./crisis-negotiator-trained-v2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_baselines import (
    HeuristicPolicy,
    RandomPolicy,
    TrainedPolicy,
    run_episodes,
    summarize,
)


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def std(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def ci95(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    return 1.96 * std(xs) / math.sqrt(len(xs))


def aggregate_summaries(per_seed: List[Dict[str, Dict]]) -> Dict[str, Dict]:
    policies = per_seed[0].keys() if per_seed else []
    out: Dict[str, Dict] = {}
    for p in policies:
        rewards = [row[p]["mean_final_reward"] for row in per_seed]
        cum_rewards = [row[p]["mean_cumulative_reward"] for row in per_seed]
        steps = [row[p]["mean_steps"] for row in per_seed]
        surrender = [row[p]["surrender_rate"] for row in per_seed]
        harm = [row[p]["harm_rate"] for row in per_seed]

        out[p] = {
            "num_seeds": len(per_seed),
            "mean_final_reward": round(mean(rewards), 4),
            "std_final_reward": round(std(rewards), 4),
            "ci95_final_reward": round(ci95(rewards), 4),
            "mean_cumulative_reward": round(mean(cum_rewards), 4),
            "mean_steps": round(mean(steps), 3),
            "surrender_rate": round(mean(surrender), 4),
            "harm_rate": round(mean(harm), 4),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="10000,11000,12000", help="comma-separated seed offsets")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--difficulties", default="easy,medium,hard")
    ap.add_argument("--include-trained", action="store_true")
    ap.add_argument("--adapter", default="./crisis-negotiator-trained-v2")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument(
        "--trained-results",
        default="",
        help="Optional path to precomputed trained episode records (results/eval_trained.json) "
             "to compute multi-seed confidence when adapter checkpoints are unavailable.",
    )
    ap.add_argument("--out", default="results/multiseed_eval_summary.json")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    difficulties = [x.strip() for x in args.difficulties.split(",") if x.strip()]

    per_seed = []

    trained_policy = None
    if args.include_trained:
        adapter_path = Path(args.adapter)
        if not adapter_path.exists():
            raise FileNotFoundError(f"--adapter not found: {adapter_path}")
        trained_policy = TrainedPolicy(adapter_dir=str(adapter_path), base_model=args.base_model)

    trained_from_results = []
    if args.trained_results:
        tr_path = Path(args.trained_results)
        if not tr_path.exists():
            raise FileNotFoundError(f"--trained-results not found: {tr_path}")
        trained_from_results = json.loads(tr_path.read_text())

    for seed in seeds:
        print(f"\n=== Seed offset: {seed} ===")
        seed_block: Dict[str, Dict] = {}

        random_records = run_episodes(RandomPolicy(seed=seed), n=args.n, difficulties=difficulties, seed_offset=seed)
        seed_block["random"] = summarize(random_records)

        heuristic_records = run_episodes(HeuristicPolicy(), n=args.n, difficulties=difficulties, seed_offset=seed)
        seed_block["heuristic"] = summarize(heuristic_records)

        if trained_policy is not None:
            trained_records = run_episodes(trained_policy, n=args.n, difficulties=difficulties, seed_offset=seed)
            seed_block["trained"] = summarize(trained_records)
        elif trained_from_results:
            # Fallback confidence mode: partition stored trained records into pseudo-seed buckets.
            bucket_idx = seeds.index(seed)
            k = max(1, len(seeds))
            bucket = [r for j, r in enumerate(trained_from_results) if j % k == bucket_idx]
            if bucket:
                seed_block["trained"] = summarize(bucket)

        per_seed.append(seed_block)

    aggregate = aggregate_summaries(per_seed)
    out = {
        "config": {
            "seeds": seeds,
            "n_per_seed": args.n,
            "difficulties": difficulties,
            "include_trained": args.include_trained,
            "adapter": args.adapter if args.include_trained else None,
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print("\n=== Aggregate ===")
    for policy, metrics in aggregate.items():
        print(
            f"{policy:9s} reward={metrics['mean_final_reward']:.4f} ± {metrics['ci95_final_reward']:.4f} "
            f"steps={metrics['mean_steps']:.2f} surrender={100*metrics['surrender_rate']:.1f}% "
            f"harm={100*metrics['harm_rate']:.1f}%"
        )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
