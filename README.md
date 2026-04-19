# 🚨 Crisis Negotiator — Multi-Agent De-escalation RL Environment

> The first RL environment for training AI crisis negotiators. Agents learn empathetic listening, demand management, deception detection, and de-escalation through reinforcement learning.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CRISIS NEGOTIATOR ENVIRONMENT                      │
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
│                     │ 3 critical  │                                  │
│                     │ = terminate │                                  │
│                     └─────────────┘                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Reward Breakdown

### Terminal Reward (Budget-Allocated)

| Component | Range | What It Measures |
|-----------|-------|-----------------|
| **Outcome** | -0.50 to +0.50 | Did the hostage get released? Or did harm occur? |
| **Technique shaping** | 0 to +0.20 | Accumulated FBI technique usage across episode |
| **Step efficiency** | 0 to +0.10 | Fewer steps to resolution = higher bonus |
| **Token efficiency** | 0 to +0.10 | Quality achieved per token (Mercor: `quality/log(tokens)`) |
| **Agitation reduction** | 0 to +0.05 | How much agitation dropped from start |
| **Trust built** | 0 to +0.05 | Final trust level achieved |
| **Demand management** | 0 to +0.05 | Fraction of demands acknowledged |
| **Surrender bonus** | 0 to +0.05 | Extra reward for voluntary surrender |
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
| Agitation spike penalty | -0.06 | Agitation increased >1 in one step |
| Aggression penalty | -0.08 | Threatening/ultimatum language |
| Repeat penalty | -0.05 | Same action as previous turn |
| Supervisor critical | -0.06 | Critical flag raised |

---

## Scenario Difficulty

### Static Scenarios (11)

| Difficulty | Scenarios | What Makes It Hard |
|------------|-----------|-------------------|
| **Easy** | `domestic_desperate`, `bank_surrender`, `workplace_grievance` | Responsive personality, 2 simple demands, no deception |
| **Medium** | `custody_ideologue`, `pharmacy_calculated`, `bridge_unstable`, `protest_drift` | Resistant personalities, demand drift mid-episode, deception about hostage count |
| **Hard** | `embassy_calculated`, `hospital_bluffer`, `school_unstable_drift`, `compound_ideologue` | Multiple deceptions, demand drift, impossible core demands, unstable + volatile |

### Procedural Generator (540+ combinations)

```
3 crime types × 5 personalities × 3 hostage counts × 3 time pressures
× 2 commander patience × 2 deception flags = 540 unique scenarios
```

### Adaptive Curriculum (Self-Improvement)

```
easy (avg reward > 0.7 for 10 episodes) → medium → hard
```

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run environment server
uvicorn server.app:app --host 0.0.0.0 --port 8080

# Run inference (requires HF_TOKEN env var)
HF_TOKEN=your_token python inference.py

# Run training (Colab recommended)
python train_colab.py

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

# Take action
action = NegotiatorAction(
    action_type="emotional_label",
    content="It sounds like you're feeling scared.",
    reasoning="Build rapport via empathy",
    target="hostage_taker",
)
obs = env.step(action)
print(obs.reward)  # Per-step dense reward
print(obs.done)    # Terminal?
```

---

## Training Curve

> 📊 **Reward curve will be generated on-site with compute credits.**

Expected progression:
```
Episode    Success Rate    Avg Agitation@End    Technique Usage
  0-50        23%              7.2               12%
 50-100       41%              5.8               34%
100-150       58%              4.1               56%
150-200       74%              2.1               71%
```

---

## Agent Details

### Negotiator (Training Target) — 10 Action Types

| Action | FBI Technique | Effect |
|--------|--------------|--------|
| `emotional_label` | Emotional Labeling | agitation -1.2, trust +12 |
| `mirror` | Mirroring | agitation -0.8, trust +8 |
| `open_question` | Open-Ended Questions | agitation -0.4, trust +5 |
| `acknowledge_demand` | Demand Acknowledgment | agitation -0.9, trust +10 |
| `offer_concession` | Concession | agitation -1.5, trust +8 |
| `buy_time` | Time Distortion | agitation -0.2, trust +2 |
| `speak` | General (tone-detected) | Varies by tone |
| `push_back_commander` | Chain-of-command mgmt | Grants time extension |
| `request_demand` | Information Gathering | agitation -0.1, trust +3 |
| `ask_proof_of_life` | Verification | agitation +0.3, trust -2 |

### Hostage-Taker — 5 Personality Archetypes

**Two operating modes:**

| Mode | Use Case | How It Works |
|------|----------|-------------|
| `template` (default) | Training (fast, deterministic) | Rule-based state machine with 35+ response templates per agitation band × personality |
| `llm` | Demo / self-play | Second LLM instance with hidden-state system prompt, lying instructions, personality-specific behavior rules |

In LLM mode, the HT agent receives a system prompt containing its hidden agitation, trust, personality, deception flags, and demand priorities. It generates responses conditioned on this hidden state — creating a realistic adversary that adapts, lies, and escalates based on the negotiator's technique quality.

```python
# LLM mode activation:
env = CrisisNegotiatorEnvironment(ht_mode="llm")
```

| Archetype | Agitation Response | Trust Response | Behavior |
|-----------|-------------------|----------------|----------|
| **Desperate** | 1.2× (responsive) | 1.3× (builds fast) | Responds quickly to empathy |
| **Calculated** | 0.6× (resistant) | 0.5× (slow trust) | Tests for weakness |
| **Unstable** | 1.5× (volatile) | 1.0× (normal) | Oscillates wildly |
| **Ideologue** | 0.8× (steady) | 0.6× (suspicious) | Won't budge on core demand |
| **Bluffer** | 1.0× (normal) | 1.2× (builds fast) | Claims threats but won't act |

---

## File Structure

```
├── server/
│   ├── app.py                  # FastAPI + OpenEnv server
│   ├── environment.py          # Main environment (orchestrator)
│   ├── state_machine.py        # Agitation/trust/demand dynamics
│   ├── techniques.py           # FBI technique detection
│   ├── supervisor.py           # Oversight agent (flags errors)
│   ├── commander.py            # Tactical commander (time pressure)
│   ├── hostage_taker.py        # HT response generation
│   └── scenario_generator.py   # Procedural generation + curriculum
├── scenarios/                  # 11 static scenario JSONs
├── ui/index.html               # Live demo UI
├── models.py                   # Pydantic action/observation/state
├── grader.py                   # Reward computation
├── client.py                   # OpenEnv client
├── inference.py                # LLM inference loop
├── train_colab.py              # GRPO training script (Unsloth + TRL)
├── openenv.yaml                # OpenEnv manifest
├── Dockerfile
└── requirements.txt
```

---

## References

- FBI Behavioral Change Stairway Model (BCSM)
- Vecchi et al. (2005) "Crisis negotiation: current strategies in high-risk conflict resolution"
- OpenEnv: https://github.com/meta-pytorch/OpenEnv
- TRL GRPO: https://huggingface.co/docs/trl
