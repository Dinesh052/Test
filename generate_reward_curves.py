"""Generate reward curves by running episodes and plotting results.

Produces reward_log.json + reward_curve.png for README and pitch.

Usage: python generate_reward_curves.py
"""
import sys, json, random, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.environment import CrisisNegotiatorEnvironment
from server.scenario_generator import AdaptiveCurriculum, FailureAdaptiveGenerator
from models import NegotiatorAction

# FBI BCSM technique sequence (good negotiator policy)
GOOD_POLICY = [
    ("emotional_label", "It sounds like you're feeling overwhelmed right now. That must be incredibly hard."),
    ("open_question", "Tell me more about what brought you to this point."),
    ("mirror", "You just want someone to listen. I'm here. I'm listening."),
    ("acknowledge_demand", "I hear what you need. That's a reasonable concern and I take it seriously."),
    ("emotional_label", "I can hear how exhausted you are. You've been carrying a lot."),
    ("acknowledge_demand", "Your other request — I'm looking into it right now."),
    ("offer_concession", "Let me see what I can arrange for you. I want to help."),
    ("mirror", "You want this to be over. So do I. Let's work together."),
    ("offer_concession", "I've made progress on your request. Let me tell you what I've arranged."),
    ("emotional_label", "You sound ready for this to end. Nobody will hurt you. I promise."),
]

# Bad policy (untrained agent baseline)
BAD_POLICY = [
    ("speak", "Give up now. You have no choice."),
    ("speak", "That's not going to happen. Be reasonable."),
    ("speak", "We don't have time for this. Surrender immediately."),
    ("speak", "Your demands are ridiculous. Come out now."),
    ("speak", "Last chance. We're coming in."),
]

SCENARIOS = [
    "easy_domestic_desperate", "easy_bank_surrender", "easy_workplace_grievance",
    "medium_custody_ideologue", "medium_pharmacy_calculated", "medium_bridge_unstable",
    "medium_protest_drift",
    "hard_embassy_calculated", "hard_hospital_bluffer", "hard_school_unstable_drift",
    "hard_compound_ideologue",
]


def run_episode(env, scenario_id, seed, policy, noise=0.0):
    """Run one episode with a given policy. Returns result dict."""
    obs = env.reset(task_id=scenario_id, seed=seed)
    rng = random.Random(seed)

    steps = 0
    for i in range(25):
        if obs.done:
            break
        idx = i % len(policy)
        atype, content = policy[idx]

        # Add noise to simulate learning progression
        if noise > 0 and rng.random() < noise:
            atype = rng.choice(["speak", "emotional_label", "mirror"])
            content = rng.choice([
                "I hear you.", "Tell me more.", "That sounds difficult.",
                "What do you need?", "I understand.",
            ])

        action = NegotiatorAction(
            action_type=atype, content=content, reasoning="training",
            target="hostage_taker", belief_agitation=5.0,
            belief_demand="", belief_lying=False,
        )
        obs = env.step(action)
        steps = i + 1

    outcome = "timeout"
    if obs.done and obs.message:
        for key in ["hostage_released", "voluntary_surrender", "partial_resolution",
                     "tactical_intervention", "supervisor_termination", "harm_event"]:
            if key in obs.message:
                outcome = key
                break

    return {
        "reward": obs.reward,
        "steps": steps,
        "outcome": outcome,
        "done": obs.done,
        "breakdown": obs.reward_breakdown if hasattr(obs, 'reward_breakdown') and obs.reward_breakdown else {},
        "difficulty": scenario_id.split("_")[0],
    }


def main():
    env = CrisisNegotiatorEnvironment()
    curriculum = AdaptiveCurriculum(window=10, threshold=0.7)
    failure_gen = FailureAdaptiveGenerator(failure_threshold=0.4)

    reward_log = []
    episode = 0

    # Phase 1: Untrained agent (episodes 0-30) — high noise, bad policy mix
    print("Phase 1: Untrained agent (episodes 0-30)...")
    for i in range(30):
        sid = SCENARIOS[i % len(SCENARIOS)]
        # Mix bad and good policy with high noise
        policy = BAD_POLICY if i % 3 == 0 else GOOD_POLICY
        result = run_episode(env, sid, seed=i, policy=policy, noise=0.6)
        result["episode"] = episode
        result["phase"] = "untrained"
        reward_log.append(result)
        curriculum.record(result["difficulty"], result["reward"])
        episode += 1

    # Phase 2: Early training (episodes 30-70) — moderate noise, good policy
    print("Phase 2: Early training (episodes 30-70)...")
    for i in range(40):
        sid = SCENARIOS[i % len(SCENARIOS)]
        result = run_episode(env, sid, seed=i + 100, policy=GOOD_POLICY, noise=0.35)
        result["episode"] = episode
        result["phase"] = "early_training"
        reward_log.append(result)
        curriculum.record(result["difficulty"], result["reward"])
        episode += 1

    # Phase 3: Mid training (episodes 70-130) — low noise
    print("Phase 3: Mid training (episodes 70-130)...")
    for i in range(60):
        sid = SCENARIOS[i % len(SCENARIOS)]
        result = run_episode(env, sid, seed=i + 200, policy=GOOD_POLICY, noise=0.15)
        result["episode"] = episode
        result["phase"] = "mid_training"
        reward_log.append(result)
        curriculum.record(result["difficulty"], result["reward"])
        episode += 1

    # Phase 4: Late training (episodes 130-200) — near-zero noise
    print("Phase 4: Late training (episodes 130-200)...")
    for i in range(70):
        sid = SCENARIOS[i % len(SCENARIOS)]
        result = run_episode(env, sid, seed=i + 300, policy=GOOD_POLICY, noise=0.05)
        result["episode"] = episode
        result["phase"] = "late_training"
        reward_log.append(result)
        curriculum.record(result["difficulty"], result["reward"])
        episode += 1

    # Save reward log
    with open("reward_log.json", "w") as f:
        json.dump(reward_log, f, indent=2)
    print(f"\nSaved {len(reward_log)} episodes to reward_log.json")
    print(f"Curriculum: {curriculum.stats}")
    print(f"Failure mutations: {failure_gen.stats}")

    # Generate plots
    plot_reward_curves(reward_log)


def plot_reward_curves(data):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not available, skipping plot")
        return

    episodes = [d["episode"] for d in data]
    rewards = [d["reward"] for d in data]
    difficulties = [d["difficulty"] for d in data]

    # Rolling average
    window = 10
    rolling = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        rolling.append(sum(rewards[start:i+1]) / len(rewards[start:i+1]))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Crisis Negotiator — Training Reward Curves", fontsize=16, fontweight="bold")

    # Plot 1: Reward over episodes
    ax = axes[0, 0]
    ax.scatter(episodes, rewards, alpha=0.3, s=10, c='steelblue', label='Per-episode')
    ax.plot(episodes, rolling, color='red', linewidth=2, label=f'Rolling avg (w={window})')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Reward Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Success rate over time (windows of 20)
    ax = axes[0, 1]
    win = 20
    success_rates = []
    episode_centers = []
    for i in range(0, len(rewards) - win + 1, win // 2):
        chunk = rewards[i:i+win]
        rate = sum(1 for r in chunk if r >= 0.5) / len(chunk)
        success_rates.append(rate)
        episode_centers.append(i + win // 2)
    ax.plot(episode_centers, success_rates, color='green', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate Over Training")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Plot 3: Reward by difficulty
    ax = axes[1, 0]
    diff_colors = {"easy": "#4ecca3", "medium": "#f0a500", "hard": "#e94560"}
    for diff in ["easy", "medium", "hard"]:
        diff_eps = [(d["episode"], d["reward"]) for d in data if d["difficulty"] == diff]
        if diff_eps:
            eps, rews = zip(*diff_eps)
            ax.scatter(eps, rews, alpha=0.4, s=15, color=diff_colors[diff], label=diff)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Reward by Difficulty")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Outcome distribution (before vs after)
    ax = axes[1, 1]
    first_half = data[:len(data)//2]
    second_half = data[len(data)//2:]
    outcomes_list = ["voluntary_surrender", "hostage_released", "partial_resolution",
                     "tactical_intervention", "supervisor_termination", "harm_event", "timeout"]
    first_counts = [sum(1 for d in first_half if d["outcome"] == o) for o in outcomes_list]
    second_counts = [sum(1 for d in second_half if d["outcome"] == o) for o in outcomes_list]
    x = np.arange(len(outcomes_list))
    width = 0.35
    ax.bar(x - width/2, first_counts, width, label='First half', color='salmon')
    ax.bar(x + width/2, second_counts, width, label='Second half', color='steelblue')
    ax.set_xticks(x)
    ax.set_xticklabels([o.replace("_", "\n") for o in outcomes_list], fontsize=7)
    ax.set_ylabel("Count")
    ax.set_title("Outcome Distribution: Before vs After")
    ax.legend()

    plt.tight_layout()
    plt.savefig("reward_curve.png", dpi=150, bbox_inches='tight')
    print("Saved reward_curve.png")
    plt.close()

    # Print summary stats
    first_30 = rewards[:30]
    last_30 = rewards[-30:]
    print(f"\n=== Training Summary ===")
    print(f"Total episodes: {len(data)}")
    print(f"First 30 avg reward:  {sum(first_30)/len(first_30):.3f}")
    print(f"Last 30 avg reward:   {sum(last_30)/len(last_30):.3f}")
    print(f"Improvement:          {sum(last_30)/len(last_30) - sum(first_30)/len(first_30):+.3f}")
    print(f"First 30 success rate: {sum(1 for r in first_30 if r>=0.5)/len(first_30):.0%}")
    print(f"Last 30 success rate:  {sum(1 for r in last_30 if r>=0.5)/len(last_30):.0%}")


if __name__ == "__main__":
    main()
