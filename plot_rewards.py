"""Generate reward_curve.png from real eval rollouts.

Reads eval_random.json, eval_heuristic.json, eval_trained.json
(produced by eval_baselines.py) and renders a 4-panel policy comparison.

Usage:
    python plot_rewards.py
"""
import json
import os
import sys
from collections import Counter

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("pip install matplotlib")
    sys.exit(1)

SERIES = [
    ("random",    "eval_random.json",    "#888888"),
    ("heuristic", "eval_heuristic.json", "#f0a500"),
    ("trained",   "eval_trained.json",   "#4ecca3"),
]

OUTCOME_COLORS = {
    "hostage_released":       "#4ecca3",
    "voluntary_surrender":    "#4ecca3",
    "partial_resolution":     "#f0a500",
    "timeout":                "#888888",
    "tactical_intervention":  "#e94560",
    "harm_event":             "#e94560",
    "supervisor_termination": "#a855f7",
    "error":                  "#555555",
}


def load(path):
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception as e:
        print(f"[warn] {path}: {e}")
        return None


def rolling(xs, w=8):
    return [sum(xs[max(0, i - w):i + 1]) / len(xs[max(0, i - w):i + 1])
            for i in range(len(xs))]


def summarize(name, log, color):
    if not log:
        return None
    rewards = [r["final_reward"] for r in log]
    outcomes = Counter(r.get("outcome", "?") for r in log)
    steps = [r.get("steps", 0) for r in log]
    mean = sum(rewards) / len(rewards)
    SURRENDER = {"hostage_released", "voluntary_surrender"}
    succ = sum(1 for r in log if r.get("outcome") in SURRENDER) / len(log)
    mean_steps = sum(steps) / len(steps) if steps else 0
    return {
        "name": name, "n": len(log), "mean": mean, "success": succ,
        "outcomes": outcomes, "rewards": rewards,
        "steps": steps, "mean_steps": mean_steps, "color": color,
    }


loaded = [(name, load(path), color) for name, path, color in SERIES]
summaries = [summarize(n, l, c) for n, l, c in loaded if l is not None]

if not summaries:
    print("No eval logs found. Run eval_baselines.py first.")
    sys.exit(1)

fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor="#0a0a0f")
fig.suptitle("Crisis Negotiator — Policy Comparison (real eval rollouts)",
             color="white", fontsize=16, fontweight="bold")

for ax in axes.flat:
    ax.set_facecolor("#0f0f1a")
    ax.tick_params(colors="#888")
    ax.spines[:].set_color("#333")

names = [s["name"] for s in summaries]
bar_colors = [s["color"] for s in summaries]

# ── Panel 1: per-episode final reward with rolling average ──────────────────
ax1 = axes[0, 0]
for s in summaries:
    xs = list(range(len(s["rewards"])))
    ax1.scatter(xs, s["rewards"], c=s["color"], s=14, alpha=0.4)
    ax1.plot(xs, rolling(s["rewards"], 8), color=s["color"], linewidth=2,
             label=f"{s['name']} (n={s['n']}, avg={s['mean']:.3f})")
ax1.axhline(0.5, color="#f0a500", linestyle="--", alpha=0.4, label="Success threshold")
ax1.set_xlabel("Episode", color="#888")
ax1.set_ylabel("Reward", color="#888")
ax1.set_title("Final Episode Reward (rolling avg window=8)", color="white")
ax1.legend(fontsize=8, facecolor="#16213e", edgecolor="#333", labelcolor="white")
ax1.set_ylim(-0.05, 1.05)

# ── Panel 2: success rate bars ───────────────────────────────────────────────
ax2 = axes[0, 1]
succs = [s["success"] for s in summaries]
bars = ax2.bar(names, succs, color=bar_colors)
for i, (bar, v) in enumerate(zip(bars, succs)):
    ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
             f"{v:.0%}", ha="center", color="white", fontsize=11, fontweight="bold")
ax2.set_ylabel("Surrender rate (hostage_released / voluntary_surrender)", color="#888")
ax2.set_title("Success Rate by Policy", color="white")
ax2.set_ylim(0, 1.2)

# ── Panel 3: mean steps to resolution ───────────────────────────────────────
ax3 = axes[1, 0]
mean_steps = [s["mean_steps"] for s in summaries]
bars3 = ax3.bar(names, mean_steps, color=bar_colors)
for bar, v in zip(bars3, mean_steps):
    ax3.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
             f"{v:.1f}", ha="center", color="white", fontsize=11, fontweight="bold")
ax3.set_ylabel("Mean steps to resolution", color="#888")
ax3.set_title("Step Efficiency (fewer = better)", color="white")
ax3.set_ylim(0, max(mean_steps) * 1.2 + 1)

# ── Panel 4: stacked outcome distribution ───────────────────────────────────
ax4 = axes[1, 1]
all_outcomes = sorted({o for s in summaries for o in s["outcomes"]})
bottoms = [0.0] * len(summaries)
for outcome in all_outcomes:
    pct = [s["outcomes"].get(outcome, 0) / s["n"] for s in summaries]
    ax4.bar(names, pct, bottom=bottoms,
            color=OUTCOME_COLORS.get(outcome, "#666666"), label=outcome)
    bottoms = [a + b for a, b in zip(bottoms, pct)]
ax4.set_ylabel("Share of episodes", color="#888")
ax4.set_title("Outcome Distribution", color="white")
ax4.set_ylim(0, 1.05)
ax4.legend(fontsize=7, facecolor="#16213e", edgecolor="#333",
           labelcolor="white", ncol=2, loc="lower right")

plt.tight_layout()
plt.savefig("reward_curve.png", dpi=150, facecolor="#0a0a0f")
total = sum(s["n"] for s in summaries)
print(f"Saved reward_curve.png — {total} episodes across {len(summaries)} policies")
for s in summaries:
    print(f"  {s['name']:12s}: n={s['n']:3d}  mean={s['mean']:.3f}  "
          f"succ={s['success']:.0%}  steps={s['mean_steps']:.1f}")
