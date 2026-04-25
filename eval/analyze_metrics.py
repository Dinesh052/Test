"""Analyze policy metrics and highlight model-tuning opportunities.

Usage:
    python eval/analyze_metrics.py \
      --summary results/eval_summary.json \
      --trained results/eval_trained.json \
      --heuristic results/eval_heuristic.json \
      --random results/eval_random.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Dict, List


def load_json(path: str):
    return json.loads(Path(path).read_text())


def action_stats(episodes: List[dict]) -> Dict[str, float]:
    c = collections.Counter()
    total_steps = 0
    for ep in episodes:
        acts = ep.get("actions", [])
        c.update(acts)
        total_steps += len(acts)

    if not total_steps:
        return {
            "total_steps": 0,
            "dominant_action": "none",
            "dominant_share": 0.0,
            "action_entropy": 0.0,
            "n_actions_used": 0,
        }

    probs = [v / total_steps for v in c.values() if v > 0]
    entropy = -sum(p * math.log(p + 1e-12, 2) for p in probs)
    dominant_action, dominant_count = c.most_common(1)[0]
    return {
        "total_steps": total_steps,
        "dominant_action": dominant_action,
        "dominant_share": round(dominant_count / total_steps, 4),
        "action_entropy": round(entropy, 4),
        "n_actions_used": len(c),
    }


def ci95(samples: List[float]) -> tuple[float, float]:
    if not samples:
        return (0.0, 0.0)
    n = len(samples)
    mean = sum(samples) / n
    if n == 1:
        return (mean, 0.0)
    var = sum((x - mean) ** 2 for x in samples) / (n - 1)
    se = math.sqrt(var / n)
    return (mean, 1.96 * se)


def episode_reward_ci(episodes: List[dict]) -> Dict[str, float]:
    rewards = [float(ep.get("final_reward", 0.0)) for ep in episodes]
    mean, half = ci95(rewards)
    return {
        "mean": round(mean, 4),
        "ci95_halfwidth": round(half, 4),
        "n": len(rewards),
    }


def print_summary_block(label: str, block: dict):
    print(f"\n[{label}]")
    print(
        f"  reward={block['mean_final_reward']:.4f} | cumulative={block['mean_cumulative_reward']:.4f}"
        f" | surrender={100*block['surrender_rate']:.1f}% | harm={100*block['harm_rate']:.1f}%"
        f" | steps={block['mean_steps']:.2f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/eval_summary.json")
    ap.add_argument("--trained", default="results/eval_trained.json")
    ap.add_argument("--heuristic", default="results/eval_heuristic.json")
    ap.add_argument("--random", default="results/eval_random.json")
    args = ap.parse_args()

    summary = load_json(args.summary)
    trained_eps = load_json(args.trained)
    heuristic_eps = load_json(args.heuristic)
    random_eps = load_json(args.random)

    print("=== POLICY SCORE SNAPSHOT ===")
    for key in ("random", "heuristic", "trained"):
        print_summary_block(key, summary[key])

    print("\n=== PER-POLICY REWARD CI (episode-level final reward) ===")
    for name, eps in (("random", random_eps), ("heuristic", heuristic_eps), ("trained", trained_eps)):
        ci = episode_reward_ci(eps)
        print(f"  {name:9s}: mean={ci['mean']:.4f} ± {ci['ci95_halfwidth']:.4f} (n={ci['n']})")

    print("\n=== ACTION DIVERSITY DIAGNOSTICS ===")
    stats = {}
    for name, eps in (("random", random_eps), ("heuristic", heuristic_eps), ("trained", trained_eps)):
        s = action_stats(eps)
        stats[name] = s
        print(
            f"  {name:9s}: dominant={s['dominant_action']} ({100*s['dominant_share']:.1f}%), "
            f"entropy={s['action_entropy']:.3f}, actions_used={s['n_actions_used']}"
        )

    # Focused gap analysis for trained policy
    if "trained" not in summary or "heuristic" not in summary:
        print("\n[warn] Trained or heuristic results missing from eval_summary.json — skipping gap analysis.")
        return
    tr = summary["trained"]
    he = summary["heuristic"]
    hard_gap = tr.get("by_difficulty", {}).get("hard", {}).get("mean_reward", 0) - he.get("by_difficulty", {}).get("hard", {}).get("mean_reward", 0)

    print("\n=== TUNING RECOMMENDATIONS ===")
    print("1) Keep trained > heuristic edge on hard tasks while reducing policy collapse.")
    print(
        f"   - Hard-tier reward gap (trained - heuristic): {hard_gap:+.4f}. "
        "Protect this by tuning penalties for repetitive action loops, not by removing high-value demand acknowledgment."
    )

    dom = stats["trained"]["dominant_share"]
    if dom > 0.65:
        print(
            f"2) Action collapse alert: trained dominant-action share is {100*dom:.1f}% "
            "(target <= 55%). Increase repeat/collapse penalties and add opening-phase exploration bonuses."
        )
    else:
        print("2) Action diversity is acceptable; keep regularization as-is.")

    print(
        "3) Add multi-seed robustness reporting: run at least 3 seeds and report mean±CI for "
        "reward, surrender rate, harm rate, and steps."
    )
    print(
        "4) Run curriculum with hard-biased sampling after midpoint (e.g., 20/30/50 easy/medium/hard) "
        "to increase separation on hard scenarios."
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
