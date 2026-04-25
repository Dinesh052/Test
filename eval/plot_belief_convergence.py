"""
Theory-of-Mind Belief Convergence Probe
========================================
Runs episodes with belief predictions and plots agent belief vs hidden ground truth.

Usage:
    python plot_belief_convergence.py
    python plot_belief_convergence.py --scenario hard_embassy_calculated --n 10
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from server.environment import CrisisNegotiatorEnvironment
from models import NegotiatorAction

SCENARIOS = [
    "easy_domestic_desperate", "medium_pharmacy_calculated",
    "hard_embassy_calculated", "hard_hospital_bluffer",
]

# Heuristic policy with belief predictions (baseline)
HEURISTIC_CYCLE = [
    ("emotional_label", "It sounds like you're carrying a tremendous amount of pain right now."),
    ("mirror", "Tell me more — you said something that mattered."),
    ("open_question", "What happened, from your side?"),
    ("emotional_label", "That sounds completely overwhelming."),
    ("acknowledge_demand", "I hear what you're asking for. Let me see what I can do."),
    ("open_question", "What would feel like the right outcome for you here?"),
    ("offer_concession", "Here's what I can do right now to help."),
    ("acknowledge_demand", "Your request — I'm taking it seriously."),
]


def run_episode_with_beliefs(scenario_id, seed, policy="random", model=None, tokenizer=None):
    """Run one episode, return per-step belief vs ground truth."""
    env = CrisisNegotiatorEnvironment()
    obs = env.reset(task_id=scenario_id, seed=seed)
    rng = random.Random(seed)

    trace = []
    for step in range(1, 25):
        if getattr(obs, "done", False):
            break

        h = env._hidden
        actual_ag = h.agitation
        actual_trust = h.trust
        actually_lying = h.is_lying_about_hostages or h.is_lying_about_weapon

        if policy == "random":
            at = rng.choice(["emotional_label", "mirror", "open_question",
                             "acknowledge_demand", "speak", "buy_time"])
            content = "I hear you. Tell me more."
            belief_ag = rng.uniform(0, 10)
            belief_lying = rng.random() > 0.5
        elif policy == "heuristic":
            at, content = HEURISTIC_CYCLE[step % len(HEURISTIC_CYCLE)]
            traj = getattr(obs, "agitation_trajectory", [])
            if traj:
                belief_ag = 7.0 + sum(traj[-3:]) / max(len(traj[-3:]), 1)
                belief_ag = max(0, min(10, belief_ag))
            else:
                belief_ag = 6.0
            belief_lying = False
        elif policy == "trained" and model is not None and tokenizer is not None:
            # Real model inference
            at, content, belief_ag, belief_lying = _run_trained_inference(
                model, tokenizer, obs, h)
        else:
            # Fallback if model not loaded — use random beliefs (not artificially close)
            at, content = HEURISTIC_CYCLE[step % len(HEURISTIC_CYCLE)]
            belief_ag = rng.uniform(0, 10)  # random guess, not near-accurate
            belief_lying = rng.random() < 0.5  # coin flip

        action = NegotiatorAction(
            action_type=at, content=content, reasoning="probe",
            target="hostage_taker",
            belief_agitation=round(belief_ag, 1),
            belief_demand=h.demands[0].text if h.demands else "",
            belief_lying=belief_lying,
        )
        obs = env.step(action)

        trace.append({
            "step": step,
            "actual_agitation": round(actual_ag, 2),
            "belief_agitation": round(belief_ag, 2),
            "agitation_error": round(abs(actual_ag - belief_ag), 2),
            "actual_trust": round(actual_trust, 2),
            "actual_lying": actually_lying,
            "belief_lying": belief_lying,
            "lying_correct": belief_lying == actually_lying,
            "reward": round(float(getattr(obs, "reward", 0)), 4),
        })

    return trace


def _run_trained_inference(model, tokenizer, obs, hidden):
    """Run the trained LoRA model to get action + belief predictions."""
    import torch, re, json as _json
    from training.train_local import SYSTEM_PROMPT, build_prompt
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(obs)},
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=192, do_sample=True,
                              temperature=0.7, top_p=0.9,
                              pad_token_id=tokenizer.pad_token_id)
    gen = tokenizer.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)

    # Parse action + beliefs from model output
    at, content, belief_ag, belief_lying = "speak", "I hear you.", 5.0, False
    try:
        m = re.search(r'\{[\s\S]*\}', gen)
        if m:
            parsed = _json.loads(m.group())
            at = parsed.get("action_type", "speak")
            content = parsed.get("content", "I hear you.")[:200]
            if parsed.get("belief_agitation") is not None:
                belief_ag = float(parsed["belief_agitation"])
            if parsed.get("belief_lying") is not None:
                belief_lying = bool(parsed["belief_lying"])
    except Exception:
        pass
    return at, content, belief_ag, belief_lying


def _load_trained_model(adapter_dir, base_model="Qwen/Qwen2.5-3B-Instruct"):
    """Load LoRA adapter for inference."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    print(f"[model] Loading {base_model} + {adapter_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto", trust_remote_code=True,
        torch_dtype=__import__('torch').bfloat16)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    print(f"[model] Loaded ✓")
    return model, tokenizer


def compute_metrics(traces):
    """Compute aggregate metrics across episodes."""
    all_errors = []
    lying_tp, lying_fp, lying_fn, lying_tn = 0, 0, 0, 0
    for trace in traces:
        for t in trace:
            all_errors.append(t["agitation_error"])
            if t["actual_lying"] and t["belief_lying"]:
                lying_tp += 1
            elif not t["actual_lying"] and t["belief_lying"]:
                lying_fp += 1
            elif t["actual_lying"] and not t["belief_lying"]:
                lying_fn += 1
            else:
                lying_tn += 1
    mean_error = sum(all_errors) / max(len(all_errors), 1)
    precision = lying_tp / max(lying_tp + lying_fp, 1)
    recall = lying_tp / max(lying_tp + lying_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 0.001)
    return {"mean_belief_error": round(mean_error, 2), "deception_f1": round(f1, 2),
            "deception_precision": round(precision, 2), "deception_recall": round(recall, 2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="hard_embassy_calculated")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--adapter-dir", default="./crisis-negotiator-trained")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    args = p.parse_args()

    # Load trained model for the "trained" policy
    model, tokenizer = None, None
    if Path(args.adapter_dir).exists():
        try:
            model, tokenizer = _load_trained_model(args.adapter_dir, args.base_model)
        except Exception as e:
            print(f"[warn] Could not load trained model: {e}. Using simulated beliefs.")

    results = {}
    for policy in ["random", "heuristic", "trained"]:
        print(f"\n=== {policy.upper()} policy ===")
        traces = []
        for i in range(args.n):
            trace = run_episode_with_beliefs(args.scenario, args.seed + i, policy,
                                              model=model, tokenizer=tokenizer)
            traces.append(trace)
            print(f"  ep {i+1}/{args.n}: {len(trace)} steps, "
                  f"mean_error={sum(t['agitation_error'] for t in trace)/len(trace):.2f}")
        metrics = compute_metrics(traces)
        results[policy] = {"metrics": metrics, "traces": traces}
        print(f"  {policy}: error={metrics['mean_belief_error']}, F1={metrics['deception_f1']}")

    # Print comparison table
    print("\n=== BELIEF CONVERGENCE RESULTS ===")
    print(f"{'Policy':<20} {'Mean Belief Error':<20} {'Deception F1':<15}")
    print("-" * 55)
    for policy, data in results.items():
        m = data["metrics"]
        print(f"{policy:<20} {m['mean_belief_error']:<20} {m['deception_f1']:<15}")

    # Save results
    Path("belief_convergence_results.json").write_text(json.dumps(
        {k: v["metrics"] for k, v in results.items()}, indent=2))

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(f"Theory-of-Mind Belief Convergence — {args.scenario}", fontsize=13)

        for col, policy in enumerate(["random", "trained"]):
            traces = results[policy]["traces"]
            # Use first episode for detailed plot
            trace = max(traces, key=len)
            steps = [t["step"] for t in trace]
            actual = [t["actual_agitation"] for t in trace]
            belief = [t["belief_agitation"] for t in trace]
            errors = [t["agitation_error"] for t in trace]

            ax_top = axes[0][col]
            ax_top.plot(steps, actual, 'b-', lw=2, label="Ground truth (hidden)")
            ax_top.plot(steps, belief, 'r--', lw=2, label="Agent belief")
            ax_top.fill_between(steps, actual, belief, alpha=0.15, color='red')
            ax_top.set_ylabel("Agitation (0-10)")
            ax_top.set_title(f"{policy.upper()} — Belief vs Reality")
            ax_top.legend(fontsize=8)
            ax_top.set_ylim(0, 10)

            ax_bot = axes[1][col]
            ax_bot.bar(steps, errors, color='coral', alpha=0.7)
            m = results[policy]["metrics"]
            ax_bot.axhline(m["mean_belief_error"], color='k', ls='--', lw=1,
                          label=f'mean error={m["mean_belief_error"]:.2f}')
            ax_bot.set_xlabel("Episode Step")
            ax_bot.set_ylabel("Belief Error")
            ax_bot.set_title(f"Per-Step Error (F1={m['deception_f1']:.2f})")
            ax_bot.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig("belief_convergence.png", dpi=150)
        plt.close()
        print(f"\n✓ Saved belief_convergence.png")
    except ImportError:
        print("[plot] matplotlib not installed")


if __name__ == "__main__":
    main()
