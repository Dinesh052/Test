# 🚨 Crisis Negotiator — Multi-Agent Hostage De-escalation Arena

> **The first RL environment for training AI crisis negotiators.** Agents learn empathetic listening, demand management, deception detection, and de-escalation — skills that can't be taught from next-token prediction alone.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Theme](https://img.shields.io/badge/Theme-Multi--Agent%20%2B%20Self--Improvement-purple)]()
[![Bonus](https://img.shields.io/badge/Bonus-Halluminate%20%7C%20Mercor%20%7C%20Patronus%20%7C%20Snorkel-gold)]()

---

## Problem Statement

Every year, ~800 hostage/barricade situations occur in the US alone. The FBI trains negotiators using expensive live role-play that doesn't scale and can't be rewound. Current LLM training produces agents that can *talk about* negotiation but can't *do* it — they lack theory-of-mind reasoning, emotional state modeling, and long-horizon strategic planning under adversarial pressure.

**Crisis Negotiator** is a multi-agent OpenEnv environment where:
- A **Negotiator agent** must de-escalate a crisis by modeling the hidden emotional state of an adversary
- A **Hostage-Taker agent** has hidden beliefs, lies, changes demands, and responds to (or resists) persuasion techniques
- A **Tactical Commander** applies external time pressure and can override the negotiator
- A **Supervisor agent** monitors the negotiator's reasoning for dangerous patterns
- **Hostages** occasionally leak unreliable intelligence

The environment trains **theory of mind**, **deception detection**, **emotional contagion modeling**, and **long-horizon persuasion under time pressure** — genuinely unsolved capabilities in frontier LLMs.

---

