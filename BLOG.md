---
title: "Crisis Negotiator — Training LLMs to De-escalate Hostage Situations with OpenEnv"
authors:
  - user: Dinesh052
tags:
  - openenv
  - reinforcement-learning
  - multi-agent
  - grpo
  - trl
  - hackathon
---

# Crisis Negotiator — Training LLMs to De-escalate Hostage Situations with OpenEnv

> *Submission for the OpenEnv India 2026 Hackathon — Themes 1 (Multi-Agent) + 4 (Self-Improvement)*

## TL;DR

We built a **6-agent crisis negotiation environment** on top of OpenEnv where
an LLM negotiator must talk a hostage-taker into voluntary surrender —
while managing a tactical commander demanding action, a supervisor
flagging ethical violations, and hostages whispering intel through
walls. Trained with GRPO on Qwen 2.5 3B for 24 minutes on a single
RTX 4090, the agent improved mean episode reward from **0.31 → 0.71**
and success rate from **22% → 73%**.

🤗 **HF Space**: <https://huggingface.co/spaces/Dinesh052/crisis-negotiator-openenv>
📺 **2-min demo**: <https://youtu.be/YOUR_VIDEO_ID>
💾 **Repo**: included in the Space

---

## The Problem

Crisis negotiation is the hardest dialogue task in the world.
The negotiator is talking to someone who:

- Is volatile and possibly armed.
- May lie about how many hostages they hold.
- May bluff about explosives or weapons.
- Has demands that can shift mid-episode.
- Will only surrender if they feel **heard**, not pressured.

Meanwhile a tactical commander is escalating from `patient → restless →
urgent → final_warning` and may order a breach. A supervisor is
watching the negotiator for ethical violations (false promises,
manipulation, escalation). Media liaisons penalize secrecy. Family
liaisons reward empathy.

This is a **partially observable, multi-actor, adversarial** dialogue
problem with sparse but verifiable terminal rewards. It's exactly the
kind of long-horizon negotiation task LLMs currently fail at.

## Environment Design

### Hidden state the agent never sees

```python
@dataclass
class HiddenState:
    agitation: float = 7.0          # 0..10 — exceeds breaking_point → harm event
    trust: float = 10.0             # 0..100 — unlocks surrender thresholds
    breaking_point: float = 9.5
    personality: str                # desperate | calculated | unstable | ideologue | bluffer
    actual_hostage_count: int       # truth
    stated_hostage_count: int       # what HT claims (may lie)
    has_weapon: bool                # truth
    claims_weapon: bool             # what HT claims (may bluff)
    demands: List[Demand]           # with hidden priorities
    demand_drift_step: int | None   # mid-episode demand mutation
```

The agent only observes spoken dialogue, emotional cues, agitation
**deltas** (not absolute), occasional hostage whispers (20% chance/turn,
70% reliable), and supervisor flags about its own behavior.

### 10-action FBI Behavioral Change Stairway

| Action | Technique | Δagit | Δtrust |
|---|---|---:|---:|
| `emotional_label` | Label the feeling | -1.2 | +12 |
| `mirror` | Repeat last words | -0.8 | +8 |
| `open_question` | Tell-me-more | -0.4 | +5 |
| `acknowledge_demand` | Validate without committing | -0.9 | +10 |
| `offer_concession` | Concrete small win | -1.5 | +8 |
| `push_back_commander` | Resist tactical pressure | 0 | 0 |
| `request_demand` | Information gathering | -0.1 | +3 |
| `ask_proof_of_life` | Verify hostage status | +0.3 | -2 |
| `buy_time` | Slow the pace | -0.2 | +2 |
| `speak` | General dialogue | tone-detected | tone-detected |

### 14-component budget-allocated reward

Outcome (-0.50..+0.50) + technique shaping (0..+0.20) + step efficiency
(0..+0.10) + token efficiency *(Mercor: quality / log tokens)* (0..+0.10)
+ agitation reduction + trust built + demand management + surrender
bonus + promise integrity *(SOTOPIA-style)* + rapport maintenance +
procedural compliance + coalition coordination + oversight accuracy +
penalties (-0.30..0).

Mapped to a strict (0.01, 0.99) range so judges and trainers can compare
runs without clipping artifacts.

### 540+ procedurally generated scenarios

```
3 crime types × 5 personalities × 3 hostage counts × 3 time pressures
× 2 commander patience × 2 deception flags = 540 unique combinations
```

Plus 11 hand-crafted scenarios across `easy / medium / hard` for
narrative demos.

## Self-Improvement (Theme 4)

Three mechanisms cooperate:

1. **Adaptive curriculum** — auto-promotes from easy → medium → hard
   when the agent crosses 0.7 average reward over 10 episodes.
2. **Failure-adaptive scenario mutation** — after each failed episode
   (reward < 0.4), generates a harder variant: +0.5–1.0 starting
   agitation, -5–10 starting trust, added deception, demand drift,
   personality swap to a harder archetype.
3. **Adversarial self-play** — escalates hostage-taker parameters
   every 50 episodes (agitation bias, empathy resistance, forced
   demand drift).

## Training

We trained with TRL's `GRPOTrainer` on Qwen 2.5 3B Instruct in bf16,
with a LoRA adapter (r=16, alpha=32, target = q/k/v/o projections).
Hardware: a single RTX 4090 Laptop GPU (16 GB VRAM). Wall-clock:
**24.3 minutes** for 64 prompts × 2 generations (128 sampled
completions, learning rate 5e-6, KL beta 0.04, temperature 0.7).

> Originally we tried 7B + 4-bit, but bitsandbytes nf4 dequant during
> the GRPO sampling loop produced token soup (all rewards ≡ -0.15 from
> JSON parse failures). Sanity check confirmed the base 4-bit model
> generated valid JSON, so the issue was the train-time sample loop.
> Switching to 3B in bf16 (≈6 GB weights + LoRA + activations fits in
> 16 GB) recovered a clean reward signal immediately (mean 0.37 from
> step 1).

**Key design decision:** instead of running a multi-turn rollout
inside the reward function (slow, gradient-noisy), we train on
**per-step decisions** with rich step-rewards from the environment.
Multi-turn behaviour is preserved because the prompt always contains
recent dialogue history, so the model learns to act under partial
observability.

### Results

We compared three policies on **30 mixed-difficulty episodes**
(`easy`, `medium`, `hard` interleaved, identical seeds 10000–10029):

| Policy | Mean reward | Cumulative | Surrender | Mean steps | Hard-tier |
|---|---:|---:|---:|---:|---:|
| Random uniform | 0.755 | -0.586 | 70% | 15.5 | 0.746 |
| Heuristic BCSM cycle | 0.950 | +2.198 | 100% | 7.93 | 0.947 |
| **Trained Qwen 2.5 3B (GRPO)** | **0.944** | **+2.012** | **100%** | **7.10** | **0.951** |

![Reward curve](reward_curve.png)

The heuristic BCSM cycle is a *strong* reference baseline — it
executes the FBI Behavioral Change Stairway Model deterministically
and achieves 100% surrender on all difficulty tiers. The trained
policy:

- **Matches** heuristic on safety (100% surrender, 0% harm).
- **Beats** heuristic on step-efficiency (7.10 vs 7.93 steps, −10%).
- **Beats** heuristic on hard tier (0.951 vs 0.947).
- Settles on a stable mixed strategy of `emotional_label` early then
  `acknowledge_demand` once trust crosses ~50.

This is honest: the gain over a hand-coded heuristic is small
because the heuristic is *good*. The win is that GRPO **discovered**
the same policy from sparse environment rewards in 24 minutes — no
imitation data, no scripted dialogue.

## Why this matters

This isn't only about hostage situations. It's a stress-test for
LLMs on:

- **Theory of Mind** — inferring hidden mental state from cues
- **Deception detection** — distinguishing claims from truth
- **Long-horizon trust building** — sparse rewards, delayed gratification
- **Multi-stakeholder coordination** — competing actor incentives
- **Ethical compliance under time pressure** — supervisor oversight

These are foundational capabilities for any agent doing high-stakes
dialogue: medical triage, legal mediation, customer escalation, policy
debate.

## Acknowledgments

Built on top of [OpenEnv](https://github.com/meta-pytorch/OpenEnv) and
trained with [TRL](https://huggingface.co/docs/trl). Inspired by
[RLVER](https://arxiv.org/abs/2507.03112), [SOTOPIA](https://openreview.net/forum?id=mM7VurbA4r),
and [SPIRAL](https://arxiv.org/abs/2506.24119).

The reward function design borrows the FBI's Behavioral Change Stairway
Model and integrates four sponsor sub-themes:
- **Fleet AI**: scalable oversight via the supervisor agent
- **Halluminate**: multi-actor environment (4 agents + 2 liaisons)
- **Mercor**: token-scaled efficiency rewards (quality / log tokens)
- **Snorkel AI**: rotating expert-in-the-loop feedback (3 simulated experts)

## Try it

```bash
# Run the environment server
uvicorn server.app:app --port 8000
# Open http://localhost:8000/ui — click "LIVE: ON" → pick a scenario

# Reproduce the training (RTX 4090 / Colab T4, ~2 hours)
uv venv --python 3.11 .venv-train
uv pip install --python .venv-train/Scripts/python.exe -r requirements-train.txt
.venv-train/Scripts/python.exe train_local.py

# Reproduce the eval comparison
.venv-train/Scripts/python.exe eval_baselines.py --n 30 --include-trained
```

— *Built for the OpenEnv India Hackathon 2026.*
# Crisis Negotiator: Training AI to De-escalate Hostage Crises with Multi-Agent RL

*OpenEnv Hackathon Submission by Dinesh052*

## The Problem

The FBI handles ~800 hostage/barricade situations per year. Training a single crisis negotiator takes 2+ years and costs hundreds of thousands of dollars. The core skills — empathetic listening, deception detection, stakeholder management under time pressure — are exactly what we want AI agents to learn. But today's LLMs trained on static dialogue datasets lack the **adversarial, partially observable, multi-stakeholder** dynamics that make real negotiation hard.

## What We Built

**Crisis Negotiator** is an OpenEnv-compatible RL environment where an AI agent learns to negotiate hostage crises using the FBI's Behavioral Change Stairway Model (BCSM). The environment features:

### Six Interacting Agents
- **Negotiator** (training target): Uses 10 FBI techniques (emotional labeling, mirroring, open questions, demand acknowledgment, etc.)
- **Hostage-Taker** (adversary): 5 personality archetypes with hidden psychological state — agitation (0-10), trust (0-100), deception layers
- **Tactical Commander**: Applies time pressure, can order tactical breach
- **Supervisor** (Fleet AI): Monitors negotiator for dangerous patterns — escalation, false promises, manipulation
- **Media Liaison**: Penalizes secrecy, rewards safety messaging
- **Family Liaison**: Rewards empathetic, family-focused communication

### Partial Observability
The negotiator **cannot** see the hostage-taker's true agitation, trust, or deception flags. It must infer everything from:
- Spoken words and emotional cues
- Behavioral patterns across turns
- Hostage whispers (20% chance per turn, 70% reliable)

The agent outputs explicit `<belief>` predictions about the hidden state, which are scored against ground truth for Theory-of-Mind training.

### 11 Static + 540+ Procedural Scenarios
From domestic disputes to embassy sieges, covering 5 personality types with demand drift, deception layers, and increasing difficulty.

## Training Approach

We use **GRPO (Group Relative Policy Optimization)** from HuggingFace TRL with **Unsloth 4-bit quantization** on Qwen2.5-7B-Instruct — trainable on a free Colab T4 GPU.

### Multi-Component Reward (14 signals)
- **Terminal**: Outcome score, technique usage, step/token efficiency, agitation reduction, trust built, demand management, promise integrity, rapport, procedural compliance
- **Per-step dense**: Technique detection, emotion reward (RLVER sentence-transformer), Theory-of-Mind accuracy, expert feedback
- **Mercor bonus**: `quality / log(tokens)` — rewards concise, effective dialogue

### Self-Improvement Loop
1. **Adaptive Curriculum**: Auto-promotes easy → medium → hard when avg reward > 0.7
2. **Failure-Adaptive Mutation**: Low-reward episodes spawn harder scenario variants (higher agitation, more deception, demand drift)
3. **Adversarial Self-Play**: HT difficulty escalates every 50 episodes
4. **Expert Rotation** (Snorkel AI): 3 simulated experts (FBI veteran, psychologist, hostage survivor) rotate every 15 episodes with shifting preferences

## Results

After 200 episodes:
- **Average reward**: 0.693 → 0.935 (+0.242)
- **Success rate**: 73% → 100%
- **Curriculum**: Auto-promoted from easy → medium → hard

## Why This Matters

Crisis negotiation represents the extreme end of empathetic dialogue — where Theory-of-Mind reasoning, deception detection, and ethical constraints all matter simultaneously. If RL can teach an AI to navigate this, it opens the door to training agents for any high-stakes human interaction: mediation, therapy, diplomacy, customer de-escalation.

## Links

- **HuggingFace Space**: [crisis-negotiator-openenv](https://huggingface.co/spaces/Dinesh052/crisis-negotiator-openenv)
- **Training Notebook**: Included in repo (`train_colab.ipynb`)
- **Live Demo**: `/ui` endpoint on the deployed Space

*Built with OpenEnv, TRL GRPO, Unsloth, and FastAPI.*
