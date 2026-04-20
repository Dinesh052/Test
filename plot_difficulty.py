"""Plot scenario difficulty gradient from batch_run outputs.
Usage: python plot_difficulty.py
"""
import json, glob, os, sys

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("pip install matplotlib"); sys.exit(1)

# Load from whichever runs dir has data
run_dir = "runs_llm" if os.path.exists("runs_llm") else "runs"
order = [
    "easy_domestic_desperate", "easy_bank_surrender", "easy_workplace_grievance",
    "medium_custody_ideologue", "medium_pharmacy_calculated", "medium_bridge_unstable",
    "medium_protest_drift",
    "hard_embassy_calculated", "hard_hospital_bluffer", "hard_school_unstable_drift",
    "hard_compound_ideologue",
]

avgs, labels, colors = [], [], []
color_map = {"easy": "#4ecca3", "medium": "#f0a500", "hard": "#e94560"}

for sid in order:
    f = os.path.join(run_dir, f"{sid}.json")
    if not os.path.exists(f):
        continue
    data = json.load(open(f))
    rewards = [ep["terminal_reward"] if "terminal_reward" in ep else ep.get("reward", 0) for ep in data]
    avgs.append(sum(rewards) / len(rewards))
    labels.append(sid.replace("_", "\n"))
    tier = sid.split("_")[0]
    colors.append(color_map.get(tier, "#888"))

fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0a0a0f")
ax.set_facecolor("#0f0f1a")
bars = ax.bar(range(len(avgs)), avgs, color=colors, width=0.7, edgecolor="#333")
ax.axhline(0.5, color="#4ecca3", linestyle="--", alpha=0.4, label="Success threshold")
ax.set_xticks(range(len(avgs)))
ax.set_xticklabels(labels, fontsize=7, color="#888")
ax.set_ylabel("Avg Reward", color="#888")
ax.set_title("Scenario Difficulty Gradient", color="white", fontsize=14)
ax.tick_params(colors="#888")
ax.spines[:].set_color("#333")
ax.legend(fontsize=9, facecolor="#16213e", edgecolor="#333", labelcolor="white")
ax.set_ylim(0, 1.05)

for i, v in enumerate(avgs):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color="white")

plt.tight_layout()
plt.savefig("difficulty_gradient.png", dpi=150, facecolor="#0a0a0f")
print(f"Saved difficulty_gradient.png ({len(avgs)} scenarios from {run_dir}/)")
