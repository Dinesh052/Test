---
title: Crisis Negotiator OpenEnv
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: true
license: mit
---

# 🚨 Crisis Negotiator — Multi-Agent De-escalation RL Environment

> **Train AI agents to negotiate hostage crises using FBI techniques, Theory-of-Mind reasoning, and adversarial self-play.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗-HuggingFace_Spaces-yellow)](https://huggingface.co/spaces/Dinesh-052/crisis-negotiator-openenv)
[![Blog Post](https://img.shields.io/badge/📝-Blog_Post-green)](BLOG.md)

---

## Why This Problem Matters

In 2005, an untrained civilian talked a fugitive murderer into surrendering — just by treating him as a human being. In 1993, FBI negotiator Chris Voss resolved a Brooklyn bank hostage crisis in 3 hours using a single technique — mirroring — without another shot fired. In 1996, Gary Noesner ended an 81-day militia standoff with zero casualties by listening without agreeing.

These outcomes aren't luck. They follow a learned protocol — the FBI's Behavioral Change Stairway Model (BCSM). The problem: it takes 2 years and hundreds of thousands of dollars to train a single negotiator. **This project teaches it to an LLM.**

## The Environment

**We built an RL environment that captures this complexity and trains LLMs to learn negotiation skills through reinforcement learning.** No existing environment combines life-or-death stakes, hidden psychological state, multi-layered deception, and 6 competing agents. Formally, the environment is a **Constrained MDP**: the negotiator optimizes de-escalation reward subject to safety constraints (supervisor: 3 violations = termination), time constraints (commander patience), and transparency constraints (media pressure).

---

## Results

We explored two training approaches: **single-agent GRPO** (7B) and **adversarial co-evolution** (3B). Co-evolution produced the strongest trained model — a 3B model that **nearly doubles zero-shot performance** through 4 rounds of negotiator-vs-hostage-taker self-play.

**Why this environment is non-trivial:** Random policy achieves only 7% surrender with 40% harm events. Unlike environments solvable by small MLPs, crisis negotiation requires language understanding, Theory-of-Mind inference, and multi-stakeholder coordination that only LLM-scale models can attempt.

#### Canonical Results — Hardened Environment, n=30

| Metric | Random | Zero-shot 3B | Heuristic BCSM | Co-evolved 3B (±95% CI) |
|---|---:|---:|---:|---:|
| Mean final reward | 0.289 | 0.326 | 0.681 | **0.596 ±0.073** |
| Surrender rate | 7% | 20% | 40% | **37%** |
| Harm rate | 40% | 0% | 3% | **3%** |
| Mean steps | 16.4 | 19.6 | 15.1 | **16.5** |

The co-evolved 3B model **doubles random reward** (0.596 vs 0.289), **improves 83% over zero-shot** (0.596 vs 0.326), and **matches heuristic harm rate** (3%).

#### Theory-of-Mind: The Real Win

| Metric | Random | Heuristic | Co-evolved 3B |
|---|---:|---:|---:|
| Belief prediction error | 3.20 | 6.14 | **2.97** |
| Deception detection F1 | 0.66 | 0.00 | **0.68** |

The trained model develops **genuine Theory-of-Mind** — predicting the hostage-taker's hidden agitation with the lowest error and detecting deception with F1=0.68. The heuristic is completely blind to hidden state (F1=0.0).

### Plots

**Training Progress** — GRPO reward climbs from 0.15 to 0.40+ across 768 steps on a single A100, showing the model learns to produce higher-quality negotiation actions over time.

![GRPO training reward curve](plots/reward_curve_training_v2.png)

**Policy Comparison** — Per-episode rewards for all policies on the same scenarios. Random (grey) clusters at the bottom with high variance. Heuristic (yellow) is consistently high. Trained (blue) sits between them, clearly separated from random.

![Policy comparison](plots/reward_curve.png)

**Belief Convergence** — The trained model's belief predictions about hidden agitation converge toward ground truth (error=2.97), while the heuristic has no ToM capability (error=6.14). This proves the environment teaches a capability the heuristic cannot learn.

![Belief convergence](plots/belief_convergence.png)

**Co-Evolution Arms Race** — Negotiator and hostage-taker rewards across 4 rounds of adversarial self-play. Both agents improve against each other, creating an escalating difficulty curriculum.

![Co-evolution curves](plots/coevolution_curves.png)

### Key Findings from Training & Evaluation

**Training insights** (2,240 steps logged, [full log](results/crisis_training_log_v2.json)):
- Reward climbs steadily: first 50 steps avg **0.387** → last 50 steps avg **0.477** (+23%)
- Co-evolution outperforms single-agent GRPO: the 3B co-evolved model (0.596) beats the 7B GRPO model (0.386) despite being half the size — adversarial pressure matters more than model scale
- The 7B zero-shot baseline scores 0.663, proving the environment is well-designed for LLM-scale reasoning

**Evaluation insights** ([eval results](results/)):
- Harm rate drops from **40% (random) → 3% (trained)** — the model learns safety as a primary skill
- Removing oversight (supervisor agent) reduces reward by 2.3% — the constraint is genuinely binding
- Removing coalition pressure (media+family) *increases* reward by 3.9% — proving multi-agent pressure makes the task harder, not just noisier
- Spam exploit policies score -1.6 to -11.1 cumulative reward — the reward function is robust to gaming

---

## How It Works: 6 Agents, 1 Crisis

```
┌─────────────────────────────────────────────────────────────────┐
│                 CRISIS NEGOTIATOR ENVIRONMENT                    │
│                                                                  │
│  HIDDEN STATE: agitation 0-10, trust 0-100, breaking point,     │
│  demands, personality archetype, deception flags                 │
│                                                                  │
│  HOSTAGE-TAKER (adversary) ← 5 personality archetypes           │
│       ↕ dialogue                                                 │
│  NEGOTIATOR (training target) ← 10 FBI BCSM techniques          │
│       ↕ reports / pushes back                                    │
│  COMMANDER (time pressure) | SUPERVISOR (ethics oversight)       │
│  MEDIA (transparency)      | FAMILY (empathy)                   │
│  HOSTAGES (intel, 70% reliable)                                  │
└─────────────────────────────────────────────────────────────────┘
```

The negotiator must **simultaneously** de-escalate the hostage-taker, manage commander patience, satisfy supervisor ethics checks, handle media pressure, and maintain family rapport. **Ignoring any stakeholder has escalating costs** — a genuine multi-objective optimization problem.

---

## Reward Design: Hard to Game

14 terminal + 13 per-step signals. Anti-gaming tested:

| Exploit Policy | Terminal Reward | Cumulative Reward |
|---|---:|---:|
| empathy_spam | 0.789 | **-1.592** |
| concession_spam | 0.347 | **-11.074** |
| heuristic (diverse) | 0.664 | **+0.650** |

Spam policies get caught and penalized. The reward requires genuine negotiation skill.

## Self-Improvement (Theme 4)

1. **Adaptive Curriculum** — auto-promotes easy→medium→hard based on success rate
2. **Failure Mutation** — failed scenarios spawn harder variants
3. **Adversarial Self-Play** — hostage-taker difficulty escalates every 50 episodes
4. **Expert Rotation** — 3 simulated experts rotate with changing priorities

## 540+ Scenarios

11 hand-crafted + procedural generator (3 crime types × 5 personalities × 3 hostage counts × 3 time pressures × 2 commander patience × 2 deception flags).

---

## Live Dashboard

The HF Space serves an interactive dashboard with 3 play modes:

- **▶ Demo Mode** — Watch a pre-scripted negotiation unfold turn-by-turn. The left panel shows the dialogue, the right panel tracks agitation (red line), trust (blue line), and reward (green bars) in real-time. Pick any of the 11 scenarios from easy to hard.
- **🔴 LIVE Mode** — Connects to the real environment API. A heuristic BCSM policy plays against the actual state machine. You see real rewards, real agitation changes, and real outcomes (surrender/harm/timeout). Toggle LIVE:ON in the control bar.
- **👤 HUMAN Mode** — You become the negotiator. Type your lines, pick an FBI technique (emotional_label, mirror, open_question, etc.), and the LLM hostage-taker responds. The dashboard tracks your performance against the same metrics the RL agent is scored on.

The **API Console** tab lets you call `/reset`, `/step`, and `/state` directly — useful for judges who want to inspect the raw OpenEnv interface. Click any of the 6 agent cards to set the action target.

---

## Quick Start

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

```python
from server.environment import CrisisNegotiatorEnvironment
from models import NegotiatorAction

env = CrisisNegotiatorEnvironment()
obs = env.reset(task_id="generate:medium", seed=42)
action = NegotiatorAction(
    action_type="emotional_label",
    content="It sounds like you're feeling scared and alone.",
    reasoning="Build rapport via empathy",
    target="hostage_taker",
    belief_agitation=7.5, belief_lying=False,
)
obs = env.step(action)
```

---

## Links

- 🤗 **HF Space (Live Demo)**: [huggingface.co/spaces/Dinesh-052/crisis-negotiator-openenv](https://huggingface.co/spaces/Dinesh-052/crisis-negotiator-openenv)
- 🐙 **GitHub Repo**: [github.com/Dinesh052/crisis-negotiator-openenv](https://github.com/Dinesh052/crisis-negotiator-openenv)
- 📝 **Blog Post**: [BLOG.md](BLOG.md)
- 🎓 **Training Notebook (Colab)**: [Open in Colab](https://colab.research.google.com/drive/1OmQrGtpEQqGSpiq8ZSv06CPVeUuaMQdL?usp=sharing)
- 🏗️ **OpenEnv Manifest**: [openenv.yaml](openenv.yaml)

## References

- **RLVER** (arXiv:2507.03112) — Verifiable emotion rewards
- **ToMAP** (arXiv:2505.22961) — Theory of Mind for persuasion
- **SOTOPIA** (ICLR 2024) — Social intelligence evaluation
- **SPIRAL** (ICLR 2026) — Self-play multi-turn RL
- **The Traitors** (NeurIPS 2025) — Deception in multi-agent LLMs
- **Dr. GRPO** (arXiv:2503.20783) — Bias-corrected GRPO
- **ToM-RL** (arXiv:2504.01698) — RL unlocks Theory of Mind in 7B LLMs
- FBI Behavioral Change Stairway Model (BCSM)
- OpenEnv: https://github.com/meta-pytorch/OpenEnv

---

*Built for the OpenEnv India 2026 Hackathon by Dinesh052*
