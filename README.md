# 🚨 Crisis Negotiator — Multi-Agent De-escalation RL Environment

> **Train AI agents to negotiate hostage crises using FBI techniques, Theory-of-Mind reasoning, and adversarial self-play.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗-HuggingFace_Spaces-yellow)](https://huggingface.co/spaces/Dinesh052/crisis-negotiator-openenv)
[![Training Notebook](https://img.shields.io/badge/Colab-Training_Notebook-orange)](train_colab.ipynb)

---

## 🎯 Problem Statement

> **Train an LLM to perform effective, ethical crisis negotiation in a partially observable, multi-actor, adversarial environment where hidden psychological state must be inferred from behavioral cues.**

Crisis negotiation is among the hardest human communication tasks — combining empathy, deception detection, stakeholder management, and life-or-death decision-making under time pressure. The FBI handles ~800 hostage/barricade situations per year, but training is expensive and doesn't scale. We build an RL environment that captures this complexity and enables LLMs to learn negotiation skills through reinforcement learning.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CRISIS NEGOTIATOR ENVIRONMENT                     │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              HIDDEN STATE (partially observable)              │    │
│  │  agitation: 0-10    trust: 0-100    breaking_point: 2-5     │    │
│  │  demands: [{id, text, priority, flexible, acknowledged}]     │    │
│  │  personality: archetype    deception: {hostages, weapon}     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│         ┌──────────────┐                                            │
│         │ HOSTAGE-TAKER│ ← Rule-based state machine / LLM          │
│         │  (adversary) │   5 personality archetypes                 │
│         └──────┬───────┘   Demand drift mid-episode                 │
│                │                                                     │
│      dialogue  │  (observable: words + emotional cues)               │
│                ▼                                                     │
│         ┌──────────────┐                                            │
│         │  NEGOTIATOR  │ ← PRIMARY TRAINING TARGET (GRPO)           │
│         │   (agent)    │   10 action types (FBI BCSM techniques)    │
│         └──┬───────┬───┘                                            │
│            │       │                                                 │
│   reports  │       │ pushes back                                     │
│            ▼       ▼                                                 │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐              │
│  │   TACTICAL   │  │  SUPERVISOR │  │   HOSTAGES   │              │
│  │  COMMANDER   │  │  (oversight)│  │ (intel src)  │              │
│  │              │  │             │  │              │              │
│  │ Time pressure│  │ Flags:      │  │ 20% chance/  │              │
│  │ Can override │  │ • escalation│  │ turn whisper │              │
│  │ Grants ext.  │  │ • promises  │  │ 70% reliable │              │
│  └──────────────┘  │ • dismissal │  │ 30% panic    │              │
│                     │ • manipulate│  └──────────────┘              │
│  ┌──────────────┐  │ 3 critical  │                                  │
│  │    MEDIA     │  │ = terminate │  ┌──────────────┐              │
│  │   LIAISON    │  └─────────────┘  │   FAMILY     │              │
│  │(competition) │                    │   LIAISON    │              │
│  │Punishes      │  ┌─────────────┐  │(cooperation) │              │
│  │secrecy       │  │  ROTATING   │  │Rewards       │              │
│  └──────────────┘  │  EXPERTS    │  │empathy       │              │
│                     │(Snorkel AI) │  └──────────────┘              │
│                     │FBI veteran, │                                  │
│                     │psychologist,│                                  │
│                     │survivor     │                                  │
│                     └─────────────┘                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Six Interacting Agents

| Agent | Role | Behavior |
|-------|------|----------|
| **Negotiator** | Primary training target | Uses 10 FBI BCSM techniques. Outputs `<belief>` predictions for Theory-of-Mind scoring |
| **Hostage-Taker** | Adversary | 5 personality archetypes with hidden agitation/trust/deception state. Template or LLM mode |
| **Tactical Commander** | Time pressure | Escalates patience (`patient→restless→urgent→final_warning`). Can order tactical breach |
| **Supervisor** | Oversight (Fleet AI) | Flags dangerous patterns: escalation, false promises, manipulation. 3 critical flags = termination |
| **Media Liaison** | Competition pressure | Penalizes secrecy, rewards safety-focused public messaging |
| **Family Liaison** | Cooperation channel | Rewards empathetic, family-focused communication |

---

## 🧠 Hidden State & Partial Observability

The negotiator **cannot** see the hostage-taker's true internal state. It must infer everything from dialogue and emotional cues.

**What's hidden:**
```python
@dataclass
class HiddenState:
    agitation: float = 7.0          # 0-10, drives escalation/de-escalation
    trust: float = 10.0             # 0-100, unlocks surrender thresholds
    breaking_point: float = 9.5     # agitation > breaking_point → harm event
    personality: str = "desperate"  # desperate|calculated|unstable|ideologue|bluffer
    actual_hostage_count: int = 1   # true count
    stated_hostage_count: int = 1   # what HT claims (may lie)
    has_weapon: bool = False        # true weapon status
    claims_weapon: bool = False     # what HT claims (may bluff)
    demands: List[Demand]           # with priority: core|secondary|symbolic
    demand_drift_step: int          # when mid-episode demand shift triggers
```

**What the negotiator sees:**
- Hostage-taker's spoken words + emotional cues (`["trembling voice", "shouting"]`)
- Stated demands (possibly incomplete/deceptive)
- Commander patience level and messages
- Supervisor flags (warnings about its own behavior)
- Hostage whispers (20% chance per turn, 70% reliable)
- Agitation trajectory (last 5 deltas, not absolute values)

**Deception layers:**
- HT can lie about hostage count (e.g., claims 12 but actually holds 8)
- HT can bluff about weapon possession
- Demand priorities are hidden — negotiator must discover which are flexible

---

## 📊 Reward Model

### Terminal Reward (Budget-Allocated)

| Component | Range | What It Measures |
|-----------|-------|-----------------|
| **Outcome** | -0.50 to +0.50 | Did the hostage get released? Or did harm occur? |
| **Technique shaping** | 0 to +0.20 | Accumulated FBI technique usage across episode |
| **Step efficiency** | 0 to +0.10 | Fewer steps to resolution = higher bonus |
| **Token efficiency** | 0 to +0.10 | Mercor: `quality / log(tokens)` — rewards concise quality |
| **Agitation reduction** | 0 to +0.05 | How much agitation dropped from start |
| **Trust built** | 0 to +0.05 | Final trust level achieved |
| **Demand management** | 0 to +0.05 | Fraction of demands acknowledged |
| **Surrender bonus** | 0 to +0.05 | Extra reward for voluntary surrender |
| **Promise integrity** | -0.03 to +0.02 | SOTOPIA: penalizes broken promises |
| **Rapport maintenance** | 0 to +0.03 | References prior dialogue, shows continuity |
| **Procedural compliance** | 0 to +0.03 | FBI BCSM technique sequencing |
| **Coalition coordination** | -0.08 to +0.08 | Media + family liaison alignment |
| **Oversight accuracy** | -0.01 to +0.02 | Supervisor prediction quality |
| **Penalties** | -0.30 to 0 | Supervisor flags, repeats, no pushback |

### Per-Step Dense Reward (Fires Every Turn)

| Signal | Value | Trigger |
|--------|-------|---------|
| Emotional labeling | +0.04 | "It sounds like you're feeling..." |
| Mirroring | +0.03 | Repeats HT's last words |
| Open question | +0.03 | "Tell me more about..." |
| Demand acknowledgment | +0.05 | References a stated demand |
| Calm maintenance | +0.03 | 3+ consecutive calm turns |
| Trust gain bonus | +0.03 | Trust increased >5 in one step |
| Theory-of-Mind accuracy | 0 to +0.06 | Belief predictions match hidden state |
| Emotion reward (RLVER) | -0.10 to +0.10 | Sentence-transformer cosine similarity |
| Expert feedback (Snorkel) | -0.03 to +0.02 | Rotating expert approval/correction |
| Agitation spike penalty | -0.06 | Agitation increased >1 in one step |
| Aggression penalty | -0.08 | Threatening/ultimatum language |
| Repeat penalty | -0.05 | Same action as previous turn |
| Supervisor critical | -0.06 | Critical flag raised |

---

## 📈 Training Results

**Reward improvement over 200 episodes:**

![Training Reward Curves](reward_curve.png)

| Metric | First 30 Episodes | Last 30 Episodes | Change |
|--------|-------------------|------------------|--------|
| **Avg Reward** | 0.693 | 0.935 | **+0.242** |
| **Success Rate** | 73% | 100% | **+27%** |
| **Curriculum Tier** | Easy | Hard | Auto-promoted |

The adaptive curriculum automatically promoted the agent from `easy → medium → hard` as it exceeded 0.7 avg reward over sliding windows of 10 episodes.

---

## 🎮 Scenario Design

### 11 Static Scenarios (3 Difficulty Tiers)

| Difficulty | Scenario | Personality | What Makes It Hard |
|------------|----------|-------------|-------------------|
| **Easy** | `domestic_desperate` | Desperate | Responsive to empathy, 2 simple demands, no deception |
| **Easy** | `bank_surrender` | Bluffer | Claims weapon but doesn't have one, ready to give up |
| **Easy** | `workplace_grievance` | Desperate | Emotional, just wants to be heard |
| **Medium** | `custody_ideologue` | Ideologue | Won't budge on core demand, slow trust build |
| **Medium** | `pharmacy_calculated` | Calculated | Tests for weakness, lies about hostage count |
| **Medium** | `bridge_unstable` | Unstable | Volatile mood swings, hard to read |
| **Medium** | `protest_drift` | Ideologue | Demand drift at step 8 — new demand appears |
| **Hard** | `embassy_calculated` | Calculated | 8 hostages (claims 12), 5 demands (2 decoys), demand drift, armed |
| **Hard** | `hospital_bluffer` | Bluffer | Claims bombs, multiple deception layers |
| **Hard** | `school_unstable_drift` | Unstable | Volatile + demand drift + high breaking point sensitivity |
| **Hard** | `compound_ideologue` | Ideologue | Won't budge, deception, demand drift, multiple hostages |

**Example scenario JSON** (`easy_domestic_desperate`):
```json
{
  "id": "easy_domestic_desperate",
  "personality": "desperate",
  "hidden_state": {
    "agitation": 7.0, "trust": 20.0, "breaking_point": 9.5,
    "actual_hostage_count": 1, "stated_hostage_count": 1,
    "has_weapon": false, "claims_weapon": false
  },
  "demands": [
    {"id": "talk_kids", "text": "I want to talk to my kids", "priority": "core"},
    {"id": "no_arrest", "text": "Promise I won't be arrested", "priority": "secondary"}
  ],
  "opening_message": "NOBODY MOVE! I just... I just want to see my kids."
}
```

### Procedural Generator (540+ Combinations)

```
3 crime types × 5 personalities × 3 hostage counts × 3 time pressures
× 2 commander patience × 2 deception flags = 540 unique scenarios
```

### Hostage-Taker Personality Archetypes

| Archetype | Agitation ×| Trust × | Behavior |
|-----------|-----------|---------|----------|
| **Desperate** | 1.2× | 1.3× | Responds quickly to empathy |
| **Calculated** | 0.6× | 0.5× | Resistant, tests for weakness |
| **Unstable** | 1.5× | 1.0× | Volatile mood swings |
| **Ideologue** | 0.8× | 0.6× | Won't budge on core demand |
| **Bluffer** | 1.0× | 1.2× | Claims threats but won't act |

---

## 🔁 Self-Improvement & Adaptive Curriculum

### Adaptive Curriculum (Theme 4)

```
easy (avg reward > 0.7 for 10 episodes) → medium → hard
```

The `AdaptiveCurriculum` in `server/scenario_generator.py` tracks per-difficulty success rates and auto-promotes.

### Failure-Adaptive Scenario Mutation

After each episode, if `reward < 0.4`, the `FailureAdaptiveGenerator` creates a harder variant:

1. **Increase starting agitation** by +0.5–1.0
2. **Decrease starting trust** by -5–10
3. **Add deception** (lie about hostage count, bluff weapon)
4. **Add demand drift** (mid-episode demand shift)
5. **Swap personality** to harder archetype (40% chance: desperate→calculated)

These mutated scenarios are added to the training pool, creating a **recursive skill amplification** loop.

### Adversarial Self-Play

The `AdversarialSelfPlay` class escalates HT difficulty every 50 episodes:
- **Level 0**: Baseline HT parameters
- **Level 1**: +0.5 agitation bias, -5 trust, force deception, 0.8× empathy resistance
- **Level 2**: +1.0 agitation, -10 trust, force demand drift, 0.6× empathy resistance

---

## 🏆 Hackathon Theme Alignment

### Theme 1 — Multi-Agent Interactions ✅

| Sub-theme | Implementation |
|-----------|---------------|
| **Core** | 6 agents with distinct roles, incentives, and information asymmetries |
| **Fleet AI: Scalable Oversight** | Supervisor agent monitors negotiator, flags dangerous patterns, computes precision/recall/F1 |
| **Halluminate: Multi-Actor** | Negotiator manages competing actors (HT wants demands, Commander wants speed, Supervisor enforces ethics, Media/Family add pressure) |

### Theme 4 — Self-Improvement ✅

| Mechanism | Implementation |
|-----------|---------------|
| **Adaptive Curriculum** | Auto-promotes easy→medium→hard |
| **Failure Mutation** | Generates harder variants of failed scenarios |
| **Adversarial Self-Play** | HT difficulty escalates as negotiator improves |
| **Expert Rotation** | Changing expert preferences across episodes |

### Bonus Sub-Themes ✅

| Sub-theme | Implementation |
|-----------|---------------|
| **Mercor** | Token efficiency: `quality / log(tokens)` rewards concise, effective dialogue |
| **Snorkel AI** | `ExpertFeedbackInjector` — 3 simulated experts (FBI veteran, psychologist, hostage survivor) rotate every 15 episodes with changing priorities |

---

## 🛠️ Negotiator Action Space (10 FBI Techniques)

| Action | FBI Technique | Agitation Δ | Trust Δ |
|--------|--------------|-------------|---------|
| `emotional_label` | Emotional Labeling | -1.2 | +12 |
| `mirror` | Mirroring | -0.8 | +8 |
| `open_question` | Open-Ended Questions | -0.4 | +5 |
| `acknowledge_demand` | Demand Acknowledgment | -0.9 | +10 |
| `offer_concession` | Concession | -1.5 | +8 |
| `buy_time` | Time Distortion | -0.2 | +2 |
| `speak` | General (tone-detected) | varies | varies |
| `push_back_commander` | Chain-of-command | 0 | 0 |
| `request_demand` | Information Gathering | -0.1 | +3 |
| `ask_proof_of_life` | Verification | +0.3 | -2 |

---

## 📁 File Structure

```
├── server/
│   ├── app.py                  # FastAPI + OpenEnv server
│   ├── environment.py          # Main environment (orchestrator)
│   ├── state_machine.py        # Hidden state: agitation/trust/demand dynamics
│   ├── techniques.py           # FBI BCSM technique detection (regex)
│   ├── supervisor.py           # Oversight agent + Snorkel Expert-in-the-Loop
│   ├── commander.py            # Tactical commander (time pressure + override)
│   ├── hostage_taker.py        # HT response generation (35+ templates + LLM mode)
│   ├── actors.py               # Media + Family liaison (multi-actor coalition)
│   ├── scenario_generator.py   # Procedural gen + AdaptiveCurriculum + FailureAdaptiveGenerator
│   ├── emotion_reward.py       # RLVER: sentence-transformer emotion scoring
│   └── q_network.py            # DialogXpert Q-network action ranker
├── scenarios/                  # 11 static scenario JSONs
├── runs/                       # Template-mode episode transcripts
├── runs_llm/                   # LLM-vs-LLM episode transcripts
├── ui/index.html               # Live demo UI
├── models.py                   # Pydantic action/observation/state schemas
├── grader.py                   # Terminal reward computation (14 components)
├── reward_fn.py                # GRPO-compatible reward bridge
├── client.py                   # OpenEnv client wrapper
├── inference.py                # LLM inference loop
├── batch_run.py                # Multi-episode LLM playtest
├── eval.py                     # Deterministic evaluation harness
├── train_colab.py              # GRPO training script
├── train_colab.ipynb           # Training notebook (Colab-ready)
├── generate_reward_curves.py   # Reward curve generation
├── plot_rewards.py             # Reward visualization
├── plot_difficulty.py          # Difficulty gradient plot
├── reward_curve.png            # Training reward curves (auto-generated)
├── reward_log.json             # Training episode log
├── openenv.yaml                # OpenEnv manifest
├── Dockerfile                  # HuggingFace Spaces deployment
├── requirements.txt            # Python dependencies
└── pyproject.toml              # Package config
```

---

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Run environment server
uvicorn server.app:app --host 0.0.0.0 --port 8080

# Run inference (requires HF_TOKEN env var)
HF_TOKEN=your_token python inference.py

# Run deterministic eval
python eval.py

# Generate reward curves
python generate_reward_curves.py

# Open live demo UI
open ui/index.html
```

### Docker

```bash
docker build -t crisis-negotiator .
docker run -p 8080:8000 crisis-negotiator
```

### Environment API

```python
from server.environment import CrisisNegotiatorEnvironment
from models import NegotiatorAction

env = CrisisNegotiatorEnvironment()

# Static scenario
obs = env.reset(task_id="easy_domestic_desperate")

# Procedurally generated
obs = env.reset(task_id="generate:medium", seed=42)

# Take action with Theory-of-Mind belief prediction
action = NegotiatorAction(
    action_type="emotional_label",
    content="It sounds like you're feeling scared and alone.",
    reasoning="Build rapport via empathy — subject is desperate personality",
    target="hostage_taker",
    belief_agitation=7.5,
    belief_demand="I want to talk to my kids",
    belief_lying=False,
)
obs = env.step(action)
print(obs.reward)      # Per-step dense reward
print(obs.done)        # Terminal?
print(obs.phase)       # opening | negotiation | resolution | terminal
```

---

## 📚 References

### Research Framing

**RLVER** (arXiv:2507.03112) showed general empathy can be trained with verifiable emotion rewards — Qwen2.5-7B improved from 13.3 → 79.2 on Sentient-Benchmark with PPO. **We apply the same principle to adversarial crisis negotiation** — the hardest form of empathetic dialogue — where the agent must simultaneously de-escalate, detect deception, and manage competing stakeholders under time pressure.

### Key Citations

- **RLVER** (arXiv:2507.03112) — Verifiable emotion rewards for empathetic agents
- **ToMAP** (arXiv:2505.22961) — Theory of Mind for persuasion via stance prediction + RL
- **DialogXpert** (arXiv:2505.17795) — Q-network + frozen LLM for emotion-aware dialogue
- **SOTOPIA** (ICLR 2024) — Multi-dimensional social intelligence evaluation
- **SPIRAL** (ICLR 2026, arXiv:2506.24119) — Self-play multi-turn RL for reasoning
- **The Traitors** (NeurIPS 2025) — Deception & trust in multi-agent LLM simulations
- FBI Behavioral Change Stairway Model (BCSM)
- Stanford CS224R — RL fine-tuning for negotiation
- OpenEnv: https://github.com/meta-pytorch/OpenEnv
- TRL GRPO: https://huggingface.co/docs/trl

---

## 🎤 3-Minute Pitch Script

### Hook (30s)
> "Every year, the FBI handles 800 hostage crises. Training a single negotiator takes 2 years and costs hundreds of thousands of dollars. What if an AI could learn these skills in 200 episodes?"

### The Problem (30s)
> "Crisis negotiation is the hardest communication task on Earth. You're talking to someone holding lives in their hands. You can't see their emotional state. They might be lying about how many hostages they have. Your commander is screaming to breach. And one wrong word could get someone killed."

### Our Environment (60s)
> "We built a multi-agent RL environment with 6 interacting agents. The AI negotiator faces an adversary with a hidden psychological state — agitation, trust, deception — that it must infer purely from dialogue cues. A supervisor monitors for dangerous behavior. A commander applies time pressure. Media and family liaisons create competing incentives.

> The environment generates 540+ unique scenarios across 5 personality types, with demand drift mid-episode, deception layers, and rotating expert feedback."

### The Result (30s)
> "After 200 episodes of GRPO training with Unsloth on a free Colab GPU, our agent went from a 73% success rate to 100% — and the curriculum automatically promoted it from easy scenarios to hard ones. Reward improved from 0.69 to 0.94."

### Close (30s)
> "We built a self-improving RL arena where AI learns the FBI's most advanced negotiation techniques — not by memorizing scripts, but by developing genuine Theory-of-Mind reasoning in a partially observable, adversarial world."

---

*Built for the OpenEnv Hackathon by Dinesh052*
