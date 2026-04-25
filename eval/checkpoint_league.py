"""Checkpoint league evaluation for trained adapters.

Evaluates every checkpoint directory under a training output root and selects
best-by-hard-tier (not just latest), helping catch regressions.

Usage:
  python eval/checkpoint_league.py --root ./crisis-negotiator-trained-v2 --n 20
  python eval/checkpoint_league.py --root ./crisis-negotiator-trained-v2 --n 20 --difficulties hard
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from eval_baselines import TrainedPolicy, run_episodes, summarize

_CKPT_RE = re.compile(r"checkpoint-(\d+)$")


def checkpoint_step(path: Path) -> int:
    m = _CKPT_RE.search(path.name)
    return int(m.group(1)) if m else -1


def list_checkpoints(root: Path) -> List[Path]:
    cks = [p for p in root.iterdir() if p.is_dir() and _CKPT_RE.search(p.name)]
    return sorted(cks, key=checkpoint_step)


def evaluate_checkpoint(
    ckpt_dir: Path,
    base_model: str,
    n: int,
    difficulties: List[str],
    seed_offset: int,
) -> Dict[str, Any]:
    policy = TrainedPolicy(adapter_dir=str(ckpt_dir), base_model=base_model)
    records = run_episodes(policy, n=n, difficulties=difficulties, seed_offset=seed_offset)
    summ = summarize(records)
    hard_rewards = [r["final_reward"] for r in records if r.get("difficulty") == "hard"]
    hard_mean = sum(hard_rewards) / len(hard_rewards) if hard_rewards else float("-inf")
    return {
        "checkpoint": ckpt_dir.name,
        "checkpoint_step": checkpoint_step(ckpt_dir),
        "summary": summ,
        "hard_mean_reward": round(hard_mean, 4) if hard_rewards else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Directory containing checkpoint-* subdirs")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--difficulties", default="easy,medium,hard")
    ap.add_argument("--seed-offset", type=int, default=10000)
    ap.add_argument("--out", default="results/checkpoint_league.json")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Invalid --root: {root}")

    difficulties = [d.strip() for d in args.difficulties.split(",") if d.strip()]
    checkpoints = list_checkpoints(root)
    if not checkpoints:
        raise RuntimeError(f"No checkpoint-* directories found under {root}")

    print(f"Found {len(checkpoints)} checkpoints under {root}")
    league_rows = []

    for ckpt in checkpoints:
        print(f"\nEvaluating {ckpt.name} ...")
        row = evaluate_checkpoint(
            ckpt_dir=ckpt,
            base_model=args.base_model,
            n=args.n,
            difficulties=difficulties,
            seed_offset=args.seed_offset,
        )
        league_rows.append(row)
        s = row["summary"]
        print(
            f"  reward={s['mean_final_reward']:.4f} hard={row['hard_mean_reward']} "
            f"steps={s['mean_steps']:.2f} surrender={100*s['surrender_rate']:.1f}%"
        )

    # Best by hard-tier reward first, then overall reward, then checkpoint step.
    ranked = sorted(
        league_rows,
        key=lambda r: (
            r["hard_mean_reward"] if r["hard_mean_reward"] is not None else -1e9,
            r["summary"]["mean_final_reward"],
            r["checkpoint_step"],
        ),
        reverse=True,
    )

    best = ranked[0]
    out = {
        "config": {
            "root": str(root),
            "base_model": args.base_model,
            "n": args.n,
            "difficulties": difficulties,
            "seed_offset": args.seed_offset,
        },
        "best_checkpoint": best,
        "ranked": ranked,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print("\n=== BEST CHECKPOINT (by hard-tier) ===")
    print(
        f"{best['checkpoint']} | hard={best['hard_mean_reward']} "
        f"overall={best['summary']['mean_final_reward']:.4f}"
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
