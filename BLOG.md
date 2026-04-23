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
