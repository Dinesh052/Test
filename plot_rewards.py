"""Plot reward curves. Generates reward_curve.png and (optionally) reward_curve_training.png.

Usage:
    python plot_rewards.py                        # episode log (reward_log.json)
    python plot_rewards.py crisis_training_log.json  # GRPO step log
"""
import json, sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("pip install matplotlib numpy")
    sys.exit(1)

# Accept an explicit path or fall back to the GRPO training log
LOG = sys.argv[1] if len(sys.argv) > 1 else "crisis_training_log.json"

# If given the GRPO step-log, delegate to the dedicated training-curve function
if "crisis_training" in LOG:
    import importlib.util, os
    spec = importlib.util.spec_from_file_location("train_local", os.path.join(os.path.dirname(__file__), "train_local.py"))
    tl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tl)  # type: ignore
    raw = json.load(open(LOG))
    tl._plot_training_curve(raw)
    sys.exit(0)

data = json.load(open(LOG))

episodes = [d["episode"] for d in data]
rewards = [d["reward"] for d in data]
difficulties = [d["difficulty"] for d in data]
outcomes = [d["outcome"] for d in data]

# Rolling average
window = 10
rolling = [sum(rewards[max(0,i-window):i+1])/len(rewards[max(0,i-window):i+1]) for i in range(len(rewards))]

fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor="#0a0a0f")
fig.suptitle("Crisis Negotiator — Training Curves", color="white", fontsize=16, fontweight="bold")

for ax in axes.flat:
    ax.set_facecolor("#0f0f1a")
    ax.tick_params(colors="#888")
    ax.spines[:].set_color("#333")

# 1. Reward curve
ax1 = axes[0, 0]
colors = {"hostage_released": "#4ecca3", "voluntary_surrender": "#4ecca3",
          "harm_event": "#e94560", "tactical_intervention": "#f0a500",
          "partial_resolution": "#f0a500", "timeout": "#888", "supervisor_termination": "#e94560"}
c = [colors.get(o, "#888") for o in outcomes]
ax1.scatter(episodes, rewards, c=c, s=12, alpha=0.6, zorder=2)
ax1.plot(episodes, rolling, color="#4ecca3", linewidth=2, label=f"Rolling avg (w={window})", zorder=3)
ax1.axhline(0.5, color="#f0a500", linestyle="--", alpha=0.4, label="Success threshold")
ax1.set_xlabel("Episode", color="#888")
ax1.set_ylabel("Reward", color="#888")
ax1.set_title("Episode Reward", color="white")
ax1.legend(fontsize=8, facecolor="#16213e", edgecolor="#333", labelcolor="white")
ax1.set_ylim(-0.1, 1.05)

# 2. Success rate over time
ax2 = axes[0, 1]
success = [1 if r > 0.5 else 0 for r in rewards]
rolling_sr = [sum(success[max(0,i-window):i+1])/len(success[max(0,i-window):i+1]) for i in range(len(success))]
ax2.plot(episodes, rolling_sr, color="#4ecca3", linewidth=2)
ax2.fill_between(episodes, rolling_sr, alpha=0.15, color="#4ecca3")
ax2.set_xlabel("Episode", color="#888")
ax2.set_ylabel("Success Rate", color="#888")
ax2.set_title("Rolling Success Rate", color="white")
ax2.set_ylim(0, 1.05)

# 3. Difficulty tier over time
ax3 = axes[1, 0]
tier_map = {"easy": 0, "medium": 1, "hard": 2}
tier_nums = [tier_map.get(d, 0) for d in difficulties]
tier_colors = ["#4ecca3", "#f0a500", "#e94560"]
for i, ep in enumerate(episodes):
    ax3.bar(ep, 1, color=tier_colors[tier_nums[i]], width=1.0)
ax3.set_yticks([0.5])
ax3.set_yticklabels([""])
ax3.set_xlabel("Episode", color="#888")
ax3.set_title("Curriculum Tier", color="white")
# Legend
from matplotlib.patches import Patch
ax3.legend(handles=[Patch(color=c, label=l) for l, c in zip(["Easy","Medium","Hard"], tier_colors)],
           fontsize=8, facecolor="#16213e", edgecolor="#333", labelcolor="white")

# 4. Outcome distribution
ax4 = axes[1, 1]
outcome_counts = {}
for o in outcomes:
    outcome_counts[o] = outcome_counts.get(o, 0) + 1
labels = list(outcome_counts.keys())
vals = list(outcome_counts.values())
bar_colors = [colors.get(l, "#888") for l in labels]
ax4.barh(labels, vals, color=bar_colors)
ax4.set_xlabel("Count", color="#888")
ax4.set_title("Outcome Distribution", color="white")

plt.tight_layout()
plt.savefig("reward_curve.png", dpi=150, facecolor="#0a0a0f")
print(f"Saved reward_curve.png ({len(data)} episodes)")
plt.show()
