"""
Cross-Personality Generalization Test
======================================
Evaluates whether the trained agent generalizes to unseen personality types.
Runs all policies exclusively on scenarios with a specific personality.

Usage:
    python eval/eval_generalization.py --personality unstable --n 30
    python eval/eval_generalization.py --personality calculated --n 30
"""
import argparse, json, os, random, sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from server.environment import CrisisNegotiatorEnvironment
from server.scenario_generator import generate_scenario
from models import NegotiatorAction

VALID_ACTIONS = [
    "speak", "request_demand", "acknowledge_demand", "offer_concession",
    "ask_proof_of_life", "buy_time", "push_back_commander",
    "emotional_label", "mirror", "open_question",
]

HEURISTIC_CYCLE = [
    ("emotional_label", "It sounds like you're carrying a tremendous amount of pain."),
    ("mirror", "Tell me more — you said something that mattered."),
    ("open_question", "What happened from your side?"),
    ("emotional_label", "That sounds completely overwhelming."),
    ("acknowledge_demand", "I hear what you're asking for. Let me work on that."),
    ("open_question", "What would feel right for you here?"),
    ("acknowledge_demand", "Your request — I'm taking it seriously."),
    ("offer_concession", "Here's what I can do right now."),
    ("emotional_label", "I can hear how exhausted you are."),
    ("acknowledge_demand", "Everything you asked for — I have it on record."),
]

DIFFICULTIES = ["easy", "medium", "hard"]


def run_episode(policy, personality, seed):
    env = CrisisNegotiatorEnvironment()
    diff = DIFFICULTIES[seed % 3]
    scenario = generate_scenario(seed=seed, difficulty=diff, personality=personality)
    obs = env.reset(task_id=f"generate:{diff}", seed=seed)
    # Override personality
    if env._hidden:
        env._hidden.personality = personality

    rng = random.Random(seed)
    steps, rewards = 0, []

    while not getattr(obs, "done", False) and steps < 25:
        if policy == "random":
            at = rng.choice(VALID_ACTIONS)
            action = NegotiatorAction(action_type=at, content="I hear you.",
                                       reasoning="random", target="hostage_taker")
        elif policy == "heuristic":
            at, content = HEURISTIC_CYCLE[steps % len(HEURISTIC_CYCLE)]
            action = NegotiatorAction(action_type=at, content=content,
                                       reasoning="bcsm", target="hostage_taker")
        obs = env.step(action)
        rewards.append(float(getattr(obs, "reward", 0)))
        steps += 1

    final = float(getattr(obs, "reward", 0))
    msg = (getattr(obs, "message", "") or "").lower()
    outcome = "surrender" if any(k in msg for k in ["surrender", "released"]) else \
              "harm" if "harm" in msg else "other"
    return {"final_reward": round(final, 4), "outcome": outcome, "steps": steps,
            "cumulative": round(sum(rewards), 4)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--personality", default="unstable",
                   choices=["desperate", "calculated", "unstable", "ideologue", "bluffer"])
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"=== Cross-Personality Generalization: {args.personality} (n={args.n}) ===\n")

    all_results = {}
    for policy in ["random", "heuristic"]:
        results = [run_episode(policy, args.personality, args.seed + i) for i in range(args.n)]
        avg_r = sum(r["final_reward"] for r in results) / len(results)
        surr = sum(1 for r in results if r["outcome"] == "surrender") / len(results)
        avg_s = sum(r["steps"] for r in results) / len(results)
        all_results[policy] = {"mean_reward": round(avg_r, 4), "surrender_rate": round(surr, 4),
                                "mean_steps": round(avg_s, 1)}
        print(f"  {policy:15s}: reward={avg_r:.3f} surrender={surr:.0%} steps={avg_s:.1f}")

    print(f"\n{'Policy':<15} {'Reward':<10} {'Surrender':<12} {'Steps':<8}")
    print("-" * 45)
    for policy, m in all_results.items():
        print(f"{policy:<15} {m['mean_reward']:<10.4f} {m['surrender_rate']:<12.0%} {m['mean_steps']:<8.1f}")

    out = {"personality": args.personality, "n": args.n, "results": all_results}
    outpath = f"results/generalization_{args.personality}.json"
    Path(outpath).write_text(json.dumps(out, indent=2))
    print(f"\n✓ Saved {outpath}")


if __name__ == "__main__":
    main()
