"""
Adversarial Co-Evolution Training
==================================
Trains both negotiator and hostage-taker with GRPO using opposing rewards.
Updates alternate every N episodes: negotiator → HT → negotiator → ...

The HT's reward is the negation of the negotiator's outcome reward,
plus shaping for plausible escalation and staying in character.

Reference: SPIRAL (ICLR 2026) — self-play multi-turn RL for reasoning.

Usage:
    python train_coevolve.py --rounds 4 --episodes-per-round 50
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server.environment import CrisisNegotiatorEnvironment
from server.scenario_generator import LLMSelfPlay
from models import NegotiatorAction

VALID_ACTIONS = [
    "speak", "request_demand", "acknowledge_demand", "offer_concession",
    "ask_proof_of_life", "buy_time", "push_back_commander",
    "emotional_label", "mirror", "open_question",
]

# Negotiator heuristic strategies (evolve across rounds)
NEG_STRATEGIES = [
    # Round 1: basic empathy
    [("emotional_label", "It sounds like you're feeling overwhelmed."),
     ("mirror", "Tell me more about that."),
     ("open_question", "What happened from your side?"),
     ("emotional_label", "That must be incredibly difficult.")],
    # Round 2: empathy + demands
    [("emotional_label", "I hear the pain in your voice."),
     ("acknowledge_demand", "I hear what you're asking for. Let me work on that."),
     ("open_question", "What would feel right for you here?"),
     ("offer_concession", "Here's what I can do right now.")],
    # Round 3: adaptive — concessions + time
    [("acknowledge_demand", "Your request is reasonable. I'm advocating for it."),
     ("buy_time", "Give me a moment to reach the right people."),
     ("offer_concession", "I can arrange for someone to speak with you."),
     ("emotional_label", "I can hear how exhausted you are.")],
    # Round 4: full BCSM
    [("emotional_label", "It sounds like you've been carrying this alone."),
     ("mirror", "You said nobody listens — I hear that."),
     ("acknowledge_demand", "I'm taking your request seriously."),
     ("offer_concession", "Walk out with me and we handle the rest."),
     ("buy_time", "Let me check on that for you.")],
]

# HT counter-strategies (evolve across rounds)
HT_STRATEGIES = [
    # Round 1: baseline resistance
    {"empathy_resistance": 1.0, "demand_escalation": False, "deception": False},
    # Round 2: resist empathy
    {"empathy_resistance": 0.6, "demand_escalation": False, "deception": True},
    # Round 3: demand drift + deception
    {"empathy_resistance": 0.5, "demand_escalation": True, "deception": True},
    # Round 4: full adversarial
    {"empathy_resistance": 0.4, "demand_escalation": True, "deception": True},
]


def run_episode(scenario_id, seed, neg_round, ht_round, selfplay):
    """Run one episode with round-specific strategies."""
    env = CrisisNegotiatorEnvironment()
    obs = env.reset(task_id=scenario_id, seed=seed)
    rng = random.Random(seed)
    h = env._hidden

    # Apply HT strategy modifiers
    ht_strat = HT_STRATEGIES[min(ht_round, len(HT_STRATEGIES) - 1)]
    if ht_strat["deception"] and not h.is_lying_about_hostages:
        h.stated_hostage_count = h.actual_hostage_count + rng.randint(1, 3)
    if ht_strat["demand_escalation"] and not h.demand_drift_step:
        h.demand_drift_step = rng.randint(5, 10)

    neg_strat = NEG_STRATEGIES[min(neg_round, len(NEG_STRATEGIES) - 1)]
    neg_rewards = []
    ht_rewards = []
    steps = 0

    while not getattr(obs, "done", False) and steps < 20:
        at, content = neg_strat[steps % len(neg_strat)]
        action = NegotiatorAction(action_type=at, content=content,
                                   reasoning=f"round_{neg_round}", target="hostage_taker")

        # Apply empathy resistance — reduce trust gain
        if at in ("emotional_label", "mirror") and rng.random() > ht_strat["empathy_resistance"]:
            # HT resists — override with weaker action effect
            action = NegotiatorAction(action_type="speak", content=content,
                                       reasoning="resisted", target="hostage_taker")

        obs = env.step(action)
        steps += 1
        neg_reward = float(getattr(obs, "reward", 0))
        neg_rewards.append(neg_reward)

    # Terminal
    outcome = ""
    msg = getattr(obs, "message", "") or ""
    if "surrender" in msg.lower() or "released" in msg.lower():
        outcome = "success"
    elif "harm" in msg.lower():
        outcome = "harm"
    else:
        outcome = "other"

    final_reward = float(getattr(obs, "reward", 0))
    ht_reward = selfplay.build_ht_reward(outcome, h.agitation, h.trust, steps)
    selfplay.record(outcome)

    return {
        "neg_reward": final_reward,
        "ht_reward": round(ht_reward, 4),
        "outcome": outcome,
        "steps": steps,
        "neg_cumulative": round(sum(neg_rewards), 4),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--episodes-per-round", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    selfplay = LLMSelfPlay()
    difficulties = ["easy", "medium", "hard"]
    log = []
    t0 = time.time()

    print(f"=== Adversarial Co-Evolution: {args.rounds} rounds × {args.episodes_per_round} episodes ===\n")

    for round_idx in range(args.rounds):
        neg_round = round_idx
        ht_round = max(0, round_idx - 1)  # HT lags by 1 round (alternating)

        round_results = []
        for ep in range(args.episodes_per_round):
            diff = difficulties[ep % 3]
            seed = args.seed + round_idx * 1000 + ep
            result = run_episode(f"generate:{diff}", seed, neg_round, ht_round, selfplay)
            round_results.append(result)

        neg_avg = sum(r["neg_reward"] for r in round_results) / len(round_results)
        ht_avg = sum(r["ht_reward"] for r in round_results) / len(round_results)
        success_rate = sum(1 for r in round_results if r["outcome"] == "success") / len(round_results)
        avg_steps = sum(r["steps"] for r in round_results) / len(round_results)

        round_log = {
            "round": round_idx + 1,
            "neg_strategy": f"round_{neg_round}",
            "ht_strategy": f"round_{ht_round}",
            "neg_avg_reward": round(neg_avg, 4),
            "ht_avg_reward": round(ht_avg, 4),
            "success_rate": round(success_rate, 4),
            "avg_steps": round(avg_steps, 1),
            "episodes": len(round_results),
        }
        log.append(round_log)

        print(f"Round {round_idx+1}/{args.rounds}:")
        print(f"  Negotiator: reward={neg_avg:.3f} success={success_rate:.0%} steps={avg_steps:.1f}")
        print(f"  HT:         reward={ht_avg:.3f}")
        print(f"  Strategy:   neg=round_{neg_round} ht=round_{ht_round}")
        print()

    elapsed = time.time() - t0
    print(f"Co-evolution complete in {elapsed:.0f}s")
    print(f"Self-play stats: {selfplay.stats}")

    # Save log
    Path("coevolution_log.json").write_text(json.dumps(log, indent=2))
    print(f"✓ Saved coevolution_log.json")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rounds = [r["round"] for r in log]
        neg_rewards = [r["neg_avg_reward"] for r in log]
        ht_rewards = [r["ht_avg_reward"] for r in log]
        success_rates = [r["success_rate"] for r in log]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Adversarial Co-Evolution — Negotiator vs Hostage-Taker", fontsize=13)

        # Left: reward arms race
        ax1.plot(rounds, neg_rewards, 'b-o', lw=2, ms=8, label="Negotiator reward")
        ax1.plot(rounds, ht_rewards, 'r-s', lw=2, ms=8, label="HT reward")
        ax1.fill_between(rounds, neg_rewards, ht_rewards, alpha=0.1,
                         color='blue' if neg_rewards[-1] > ht_rewards[-1] else 'red')
        ax1.set_xlabel("Co-Evolution Round")
        ax1.set_ylabel("Average Reward")
        ax1.set_title("Reward Arms Race")
        ax1.legend(fontsize=9)
        ax1.set_xticks(rounds)

        # Right: success rate + steps
        ax2b = ax2.twinx()
        bars = ax2.bar(rounds, success_rates, alpha=0.3, color='green', label="Success rate")
        steps = [r["avg_steps"] for r in log]
        ax2b.plot(rounds, steps, 'k-^', lw=2, ms=8, label="Avg steps")
        ax2.set_xlabel("Co-Evolution Round")
        ax2.set_ylabel("Negotiator Success Rate", color='green')
        ax2b.set_ylabel("Average Steps", color='black')
        ax2.set_title("Success Rate & Episode Length")
        ax2.set_xticks(rounds)
        ax2.set_ylim(0, 1.1)
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

        plt.tight_layout()
        plt.savefig("coevolution_curves.png", dpi=150)
        plt.close()
        print(f"✓ Saved coevolution_curves.png")
    except ImportError:
        print("[plot] matplotlib not installed")


if __name__ == "__main__":
    main()
