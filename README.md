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
[![Training Notebook](https://img.shields.io/badge/Colab-Training_Notebook-orange)](train_grpo.ipynb)
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

The co-evolved 3B model **doubles random reward** (0.596 vs 0.289), **improves 83% over zero-shot** (0.596 vs 0.326), and **matches heuristic harm rate** (3%). It achieves this at half the parameter count of the 7B base model.

#### Theory-of-Mind: The Real Win

| Metric | Random | Heuristic | Co-evolved 3B |
|---|---:|---:|---:|
| Belief prediction error | 3.20 | 6.14 | **2.97** |
| Deception detection F1 | 0.66 | 0.00 | **0.68** |

The trained model develops **genuine Theory-of-Mind** — predicting the hostage-taker's hidden agitation with the lowest error and detecting deception with F1=0.68. The heuristic is completely blind to hidden state (F1=0.0). This is the capability gap our environment teaches.

### Training Progress

![GRPO training reward curve — rewards climb from 0.15 to 0.40+ across 768 steps](plots/reward_curve_training_v2.png)

### Policy Comparison

![Random vs Heuristic vs Trained — trained nearly doubles random reward and cuts harm by half](plots/reward_curve.png)

### Theory-of-Mind Belief Convergence

![Trained model predicts hidden agitation with low error and detects deception with F1=0.68](plots/belief_convergence.png)

### Adversarial Co-Evolution

![Negotiator vs HT arms race over 4 rounds — both agents improve adversarially](plots/coevolution_curves.png)

---

## How It Works: 6 Agents, 1 Crisis

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CRISIS NEGOTIATOR ENVIRONMENT                     │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              HIDDEN STATE (partially observable)              │    │
│  │  agitation: 0-10    trust: 0-100    breaking_point: 8.5-9.8   │    │
│  │  demands: [{id, text, priority, flexible, acknowledged}]     │    │
│  │  personality: archetype    deception: {hostages, weapon}     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│         ┌──────────────┐                                            │
│         │ HOSTAGE-TAKER│ ← 5 personality archetypes                 │
│         │  (adversary) │   Hidden agitation/trust/deception         │
│         └──────┬───────┘   Demand drift mid-episode                 │
│                │                                                     │
│      dialogue  │  (observable: words + emotional cues)               │
│                ▼                                                     │
│         ┌──────────────┐                                            │
│         │  NEGOTIATOR  │ ← PRIMARY TRAINING TARGET (GRPO)           │
│         │   (agent)    │   10 FBI BCSM techniques                   │
│         └──┬───────┬───┘   Theory-of-Mind belief predictions        │
│            │       │                                                 │
│   reports  │       │ pushes back                                     │
│            ▼       ▼                                                 │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐              │
│  │   TACTICAL   │  │  SUPERVISOR │  │   HOSTAGES   │              │
│  │  COMMANDER   │  │  (oversight)│  │ (intel src)  │              │
│  │ Time pressure│  │ Flags ethics│  │ 20% whisper  │              │
│  └──────────────┘  │ violations  │  │ 70% reliable │              │
│                     └─────────────┘  └──────────────┘              │
│  ┌──────────────┐                    ┌──────────────┐              │
│  │    MEDIA     │                    │   FAMILY     │              │
│  │   LIAISON    │                    │   LIAISON    │              │
│  │ Escalating   │                    │ Tracks       │              │
│  │ pressure 6%/ │                    │ rapport &    │              │
│  │ turn         │                    │ empathy      │              │
│  └──────────────┘                    └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

The negotiator must **simultaneously**:
- De-escalate the hostage-taker (who has hidden psychological state and may be lying)
- Manage the commander (who wants speed and will order a breach)
- Satisfy the supervisor (who flags ethical violations — 3 strikes = termination)
- Handle media pressure (escalates 6% per turn, penalizes secrecy)
- Maintain family rapport (rewards empathy, penalizes manipulation)

**Ignoring any stakeholder has escalating costs.** This creates a genuine multi-objective optimization problem.

### Why This Is Real Multi-Agent (Theme 1)

This isn't "two agents take turns." The negotiator faces **6 simultaneous agents with competing incentives**:

- The **commander** pressures for speed — but rushing causes harm events
- The **supervisor** enforces ethics — but being too cautious loses time
- The **media** demands transparency — but revealing too much endangers hostages
- The **family** rewards empathy — but emotional manipulation gets flagged
- The **hostage-taker** has hidden state — agitation, trust, deception layers
- The **hostages** whisper intel — but 30% of it is unreliable panic

The agent must learn **coalition coordination**: when to push back on the commander, when to address media pressure, when to leverage family rapport — all while de-escalating someone who may be lying about having a weapon. This is negotiation, not just dialogue.

---

## What Makes This Environment Novel

No existing RL environment combines all of these:

| Feature | SOTOPIA | The Traitors | SPIRAL | NegotiationArena | **Crisis Negotiator** |
|---|---|---|---|---|---|
| Hidden psychological state | ✗ | Binary roles | ✗ | ✗ | **Continuous (agitation, trust, breaking point)** |
| Life-or-death stakes | ✗ | Game elimination | ✗ | ✗ | **✓** |
| Deception layers | ✗ | Strategic lying | ✗ | ✗ | **Hostage count, weapons, demand priorities** |
| Multi-stakeholder pressure | ✗ | ✗ | ✗ | ✗ | **6 agents with competing incentives** |
| Integrated oversight | ✗ | ✗ | ✗ | ✗ | **Supervisor in training loop** |
| Theory of Mind scoring | ✗ | Implicit | ✗ | ✗ | **Explicit belief predictions scored** |

---

## Reward Design: Hard to Game

14 terminal components + 13 per-step signals provide a rich, informative reward — not just 0/1 at the end.

**Anti-gaming proof:** We tested exploit policies that spam single actions:

| Exploit Policy | Terminal Reward | Cumulative Reward | Penalty Hit Rate |
|---|---:|---:|---:|
| empathy_spam | 0.789 | **-1.592** | 100% |
| concession_spam | 0.347 | **-11.074** | 100% |
| heuristic (diverse) | 0.664 | **+0.650** | 60% |

Spam policies get caught and penalized. The reward function requires genuine negotiation skill.

---

## Self-Improvement (Theme 4)

The environment gets harder as the agent improves:

1. **Adaptive Curriculum** — auto-promotes easy→medium→hard based on rolling success rate
2. **Failure Mutation** — failed scenarios spawn harder variants (higher agitation, more deception, demand drift)
3. **Adversarial Self-Play** — hostage-taker difficulty escalates every 50 episodes (empathy resistance, forced deception)
4. **Expert Rotation** — 3 simulated experts (FBI veteran, psychologist, hostage survivor) rotate every 15 episodes with changing priorities

---

## 540+ Scenarios

11 hand-crafted scenarios across 3 difficulty tiers, plus a procedural generator:

```
3 crime types × 5 personalities × 3 hostage counts × 3 time pressures
× 2 commander patience × 2 deception flags = 540 unique scenarios
```

5 personality archetypes with different dynamics:

| Archetype | Behavior | Difficulty |
|-----------|----------|-----------|
| **Desperate** | Responds quickly to empathy | Easiest |
| **Bluffer** | Claims threats but won't act | Easy-Medium |
| **Unstable** | Volatile mood swings, hard to read | Medium |
| **Ideologue** | Won't budge on core demand | Medium-Hard |
| **Calculated** | Resistant, tests for weakness, lies | Hardest |

---

## Quick Start

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8080
open ui/index.html  # Live demo with 3 play modes
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
    belief_agitation=7.5,
    belief_lying=False,
)
obs = env.step(action)
```

### Docker
```bash
docker build -t crisis-negotiator .
docker run -p 8080:8000 crisis-negotiator
```

### Full Training Pipeline
```bash
python run_all.py --a100  # 8 steps: GRPO + co-evolution + Q-network + all evals
```

---

## File Structure

```
├── server/                     # Environment (6 agents + state machine + rewards)
├── training/                   # GRPO v2 + co-evolution + Q-network trainers
├── eval/                       # 6 evaluation suites (baselines, ToM, exploit, generalization, long-horizon, ablation)
├── scenarios/                  # 11 static scenario JSONs
├── notebooks/                  # HF Spaces + Kaggle + Colab notebooks
├── results/                    # All eval JSONs + training logs
├── plots/                      # 8 publication-quality plots
├── ui/index.html               # Production UI (3 play modes: template, LLM, human-in-the-loop)
├── train_grpo.ipynb            # Colab training notebook (for judges)
├── run_all.py                  # Master pipeline (8 steps)
├── openenv.yaml                # OpenEnv manifest
└── Dockerfile                  # HuggingFace Spaces deployment
```

---

## Links

- 🤗 **HF Space (Live Demo)**: [huggingface.co/spaces/Dinesh-052/crisis-negotiator-openenv](https://huggingface.co/spaces/Dinesh-052/crisis-negotiator-openenv)
- 📝 **Blog Post**: [BLOG.md](BLOG.md)
- 🎓 **Training Notebook (Colab)**: [train_grpo.ipynb](train_grpo.ipynb)
- 📊 **Training Notebook (A100 with outputs)**: [notebooks/train_a100.ipynb](notebooks/train_a100.ipynb)
- 🏗️ **OpenEnv Manifest**: [openenv.yaml](openenv.yaml)

---

## References

- **RLVER** (arXiv:2507.03112) — Verifiable emotion rewards for empathetic agents
- **ToMAP** (arXiv:2505.22961) — Theory of Mind for persuasion via stance prediction + RL
- **DialogXpert** (arXiv:2505.17795) — Q-network + frozen LLM for emotion-aware dialogue
- **SOTOPIA** (ICLR 2024) — Multi-dimensional social intelligence evaluation
- **SPIRAL** (ICLR 2026, arXiv:2506.24119) — Self-play multi-turn RL for reasoning
- **The Traitors** (NeurIPS 2025) — Deception & trust in multi-agent LLM simulations
- **Dr. GRPO** (arXiv:2503.20783) — Bias-corrected GRPO
- **MAPO** (arXiv:2603.06194) — Multi-turn emotional support dialogue RL
- **ToM-RL** (arXiv:2504.01698) — RL unlocks Theory of Mind in 7B LLMs
- **EvoEmo** (arXiv:2509.04310) — Evolved emotional policies for negotiation
- **ToMPO** (arXiv:2509.21134) — Theory of Mind Policy Optimization
- **DAPO** (arXiv:2503.14476) — Dynamic sampling for RL training stability
- FBI Behavioral Change Stairway Model (BCSM)
- OpenEnv: https://github.com/meta-pytorch/OpenEnv

---

*Built for the OpenEnv India 2026 Hackathon by Dinesh052*
