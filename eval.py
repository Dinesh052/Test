"""Deterministic evaluation harness.

Runs fixed scenarios with fixed seeds, outputs JSON metrics + plots.
Use to compare checkpoints apples-to-apples.

Usage:
  python eval.py                    # run all scenarios
  python eval.py --difficulty easy  # run only easy
  python eval.py --plot             # generate reward_curve.png
"""
import sys, json, argparse
sys.path.insert(0, '.')

from server.environment import CrisisNegotiatorEnvironment
from models import NegotiatorAction

EVAL_SCENARIOS = [
    ("easy_domestic_desperate", 42), ("easy_bank_surrender", 42), ("easy_workplace_grievance", 42),
    ("medium_custody_ideologue", 42), ("medium_pharmacy_calculated", 42),
    ("medium_bridge_unstable", 42), ("medium_protest_drift", 42),
    ("hard_embassy_calculated", 42), ("hard_hospital_bluffer", 42),
    ("hard_school_unstable_drift", 42), ("hard_compound_ideologue", 42),
]

# Fixed good-negotiator policy for baseline eval
POLICY = [
    ("emotional_label", "It sounds like you're feeling overwhelmed right now."),
    ("open_question", "Tell me more about what brought you here."),
    ("acknowledge_demand", "I hear what you need. That's understandable."),
    ("mirror", "You just want someone to listen. I'm listening."),
    ("offer_concession", "Let me see what I can arrange for you."),
    ("emotional_label", "I can hear how exhausted you are."),
    ("acknowledge_demand", "Your other request — I'm working on it."),
    ("mirror", "You want this to be over. So do I."),
    ("offer_concession", "I've arranged what you asked. Let's talk next steps."),
    ("emotional_label", "You sound ready. Nobody will hurt you."),
]


def run_eval(difficulty_filter=None):
    results = []
    for scenario_id, seed in EVAL_SCENARIOS:
        if difficulty_filter and not scenario_id.startswith(difficulty_filter):
            continue

        env = CrisisNegotiatorEnvironment()
        obs = env.reset(task_id=scenario_id, seed=seed)

        steps = 0
        for i, (atype, content) in enumerate(POLICY * 3):
            if obs.done:
                break
            action = NegotiatorAction(
                action_type=atype, content=content, reasoning="eval",
                target="hostage_taker", belief_agitation=5.0,
                belief_demand="", belief_lying=False,
            )
            obs = env.step(action)
            steps = i + 1

        outcome = "timeout"
        if obs.done and obs.message:
            parts = obs.message.split(":")
            if len(parts) > 1:
                outcome = parts[1].split(".")[0].strip()

        results.append({
            "scenario": scenario_id,
            "seed": seed,
            "reward": obs.reward,
            "steps": steps,
            "outcome": outcome,
            "done": obs.done,
            "breakdown": obs.reward_breakdown,
        })

    return results


def print_results(results):
    print(f"\n{'='*70}")
    print(f"{'Scenario':<35} {'Reward':>7} {'Steps':>6} {'Outcome':<20}")
    print(f"{'='*70}")
    for r in results:
        icon = "✅" if r["reward"] > 0.5 else "⚠️" if r["reward"] > 0.2 else "❌"
        print(f"{icon} {r['scenario']:<33} {r['reward']:>7.3f} {r['steps']:>5} {r['outcome']:<20}")

    rewards = [r["reward"] for r in results]
    successes = sum(1 for r in results if r["reward"] > 0.5)
    print(f"{'='*70}")
    print(f"Avg Reward: {sum(rewards)/len(rewards):.3f}")
    print(f"Success Rate: {successes}/{len(results)} ({100*successes/len(results):.0f}%)")
    print(f"Avg Steps: {sum(r['steps'] for r in results)/len(results):.1f}")


def save_results(results, path="eval_results.json"):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {path}")


def plot_results(results, path="reward_curve.png"):
    try:
        import matplotlib.pyplot as plt
        rewards = [r["reward"] for r in results]
        labels = [r["scenario"].replace("_", "\n") for r in results]
        colors = ["#4ecca3" if r > 0.5 else "#f0a500" if r > 0.2 else "#e94560" for r in rewards]

        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.bar(range(len(rewards)), rewards, color=colors, width=0.7)
        ax.axhline(y=0.5, color="#4ecca3", linestyle="--", alpha=0.5, label="Success threshold")
        ax.set_xticks(range(len(rewards)))
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax.set_ylabel("Reward")
        ax.set_title("Crisis Negotiator — Evaluation Results")
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        print(f"Plot saved to {path}")
    except ImportError:
        print("matplotlib not installed, skipping plot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    results = run_eval(args.difficulty)
    print_results(results)
    save_results(results)
    if args.plot:
        plot_results(results)
