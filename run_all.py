#!/usr/bin/env python3
"""
Crisis Negotiator — Master Training & Evaluation Pipeline
==========================================================
Runs ALL training and evaluation in sequence on a single GPU.
Designed for HuggingFace Spaces with GPU (L4/A10G/A100).

Total estimated time: ~2 hours on L4, ~1 hour on A100.
Total estimated cost: ~$2-4 on HF.

Usage:
    python run_all.py                    # run everything
    python run_all.py --skip-train       # only eval (if models already trained)
    python run_all.py --quick            # reduced episodes for testing
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def run(cmd: str, desc: str, cwd: str = None):
    """Run a command, print timing."""
    print(f"\n{'='*60}")
    print(f">>> {desc}")
    print(f">>> {cmd}")
    print(f"{'='*60}\n", flush=True)
    t0 = time.time()
    result = subprocess.run(
        cmd, shell=True, cwd=cwd or str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"\n>>> {desc}: {status} in {elapsed:.0f}s\n", flush=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true", help="Skip training, only run eval")
    parser.add_argument("--quick", action="store_true", help="Reduced episodes for quick test")
    parser.add_argument("--a100", action="store_true", help="A100 mode: 7B model, r=32, more data")
    args = parser.parse_args()

    if args.quick:
        n, prompts, rounds, q_episodes = 10, 32, 2, 200
        model_args = ""
    elif args.a100:
        n, prompts, rounds, q_episodes = 50, 64, 6, 800
        model_args = "--model Qwen/Qwen2.5-7B-Instruct --lora-r 32"
    else:
        n, prompts, rounds, q_episodes = 30, 64, 4, 500
        model_args = ""

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    plots_dir = ROOT / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Pre-flight: install missing TRL deps and clear Unsloth cache
    import shutil
    cache_dir = ROOT / "unsloth_compiled_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
        print("[preflight] Cleared unsloth_compiled_cache")
    subprocess.run("pip install -q mergekit llm-blender matplotlib 2>/dev/null || true", shell=True)

    pipeline_log = {"steps": [], "start_time": time.strftime("%Y-%m-%d %H:%M:%S")}
    t_start = time.time()

    # ── PHASE 1: Training ──────────────────────────────────
    if not args.skip_train:
        # 1a. Main GRPO training
        run(
            f"python training/train_local_v2.py --num-episodes {prompts} --num-epochs 1 {model_args}",
            f"STEP 1/8: Main GRPO Training {'(7B + r=32)' if args.a100 else '(3B + r=16)'}"
        )
        pipeline_log["steps"].append({"name": "grpo_train", "time": time.time() - t_start})

        # 1b. Adversarial Co-Evolution
        run(
            f"python training/train_coevolve_grpo.py --rounds {rounds} --prompts-per-round {prompts}",
            "STEP 2/8: Adversarial Co-Evolution Training"
        )
        pipeline_log["steps"].append({"name": "coevolution", "time": time.time() - t_start})

        # 1c. Q-Network Training
        run(
            f"python training/train_q_network.py --episodes {q_episodes}",
            "STEP 3/8: DialogXpert Q-Network Training"
        )
        pipeline_log["steps"].append({"name": "q_network", "time": time.time() - t_start})

    # ── PHASE 2: Evaluation ────────────────────────────────

    # 2a. Baselines + Trained model eval
    run(
        f"python eval/eval_baselines.py --n {n} --difficulties easy,medium,hard --include-trained",
        "STEP 4/8: Baseline & Trained Model Evaluation"
    )
    pipeline_log["steps"].append({"name": "eval_baselines", "time": time.time() - t_start})

    # 2b. Theory-of-Mind Probe / Belief Convergence
    run(
        f"python eval/plot_belief_convergence.py --n {n}",
        "STEP 5/8: Theory-of-Mind Probe & Belief Convergence"
    )
    pipeline_log["steps"].append({"name": "tom_probe", "time": time.time() - t_start})

    # 2c. Reward-Hacking Exploit Analysis
    run(
        f"python eval/eval_exploit.py --n {n}",
        "STEP 6/8: Reward-Hacking Failure Analysis"
    )
    pipeline_log["steps"].append({"name": "exploit_analysis", "time": time.time() - t_start})

    # 2d. Cross-Personality Generalization
    for personality in ["unstable", "calculated", "bluffer"]:
        run(
            f"python eval/eval_generalization.py --personality {personality} --n {n}",
            f"STEP 7/8: Generalization Test ({personality})"
        )
    pipeline_log["steps"].append({"name": "generalization", "time": time.time() - t_start})

    # 2e. Mechanistic Dialogue Dissection
    run(
        f"python eval/generate_dissection.py",
        "STEP 8/8: Mechanistic Dialogue Dissection"
    )
    pipeline_log["steps"].append({"name": "dissection", "time": time.time() - t_start})

    # ── PHASE 3: Summary ───────────────────────────────────
    total_time = time.time() - t_start
    pipeline_log["total_time_seconds"] = total_time
    pipeline_log["total_time_human"] = f"{total_time/60:.1f} minutes"

    with open(results_dir / "pipeline_log.json", "w") as f:
        json.dump(pipeline_log, f, indent=2)

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE — {total_time/60:.1f} minutes total")
    print(f"{'='*60}")
    print(f"\nResults saved to: {results_dir}/")
    print(f"Plots saved to:   {plots_dir}/")
    print(f"\nFiles generated:")
    for f in sorted(results_dir.glob("*.json")):
        print(f"  {f.name} ({f.stat().st_size/1024:.1f}KB)")
    for f in sorted(plots_dir.glob("*.png")):
        print(f"  {f.name} ({f.stat().st_size/1024:.1f}KB)")


if __name__ == "__main__":
    main()
