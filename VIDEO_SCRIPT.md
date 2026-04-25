# Crisis Negotiator — 2-Minute Video Script

**Total runtime: ~115 seconds. Read at conversational pace.**

---

## [0:00 – 0:10] HOOK — Cold open with a stat

> *On screen: red `🚨 CRISIS NEGOTIATOR` logo on dark background.*

> "Every year the FBI handles 800 hostage and barricade situations.
> Each negotiator costs hundreds of thousands of dollars to train.
> What if we could train an AI to do this — in 80 episodes?"

---

## [0:10 – 0:30] PROBLEM — Why this is hard

> *Cut to terminal showing `python eval_baselines.py --n 10` running.*

> "Crisis negotiation is the hardest dialogue task on Earth.
> The agent talks to someone who is volatile, may be lying about
> hostages, may be bluffing about weapons.
>
> Meanwhile a tactical commander is screaming to breach,
> a supervisor is watching for ethical violations,
> and the agent can't see what the hostage-taker is actually feeling."

---

## [0:30 – 0:55] ENVIRONMENT — Show what we built

> *Switch to UI at http://localhost:8000/ui — open `Embassy — Calculated Actor` (hard).*
> *Click `LIVE: ON` button.*
> *Press Play.*

> "We built an OpenEnv environment with **6 interacting agents**:
> hostage-taker, tactical commander, supervisor, media liaison,
> family liaison, and rotating expert advisors.
>
> The hostage-taker has a **hidden state** — agitation, trust,
> breaking-point — that the agent must infer from dialogue cues.
> Some scenarios include deception layers and demands that
> shift mid-episode.
>
> Watch the agitation gauge drop as the negotiator labels emotions.
> Watch trust climb when demands are acknowledged. The supervisor
> flags any dangerous patterns."

> *Highlight the `LIVE: ON` button — this is the real OpenEnv API.*

---

## [0:55 – 1:25] TRAINING — Show real curves

> *Cut to `reward_curve.png` opened full-screen.*

> "We trained Qwen 2.5 3B with GRPO from Hugging Face TRL on a
> single RTX 4090 Laptop GPU — 24 minutes, LoRA r=16, bf16.
>
> Use the **Final canonical table** (same as README + BLOG):
> random **0.7547**, heuristic **0.9476**, trained **0.9537** mean reward.
> Surrender/harm on this split: heuristic and trained are **100% / 0%**.
> Keep a one-line note that these are the "Final" numbers, and pilot
> hardened numbers are shown separately in docs as historical context.
>
> The reward function is **14-component**, budget-allocated:
> outcome, technique usage, demand acknowledgment, trust,
> agitation reduction, oversight accuracy, coalition alignment,
> token efficiency. No single signal can be gamed in isolation."

---

## [1:25 – 1:55] SELF-IMPROVEMENT — Theme 4 hook

> *Back to UI — show generating procedural scenarios.*
> *Click 🎲 Generate: HARD card.*

> "The environment is **self-improving**:
>
> An adaptive curriculum auto-promotes from easy → medium → hard
> when the agent crosses 0.7 average reward over 10 episodes.
>
> Failed episodes spawn **harder mutated variants** —
> increased agitation, added deception, demand drift —
> creating a recursive skill amplification loop.
>
> An adversarial self-play module escalates hostage-taker difficulty
> every 50 episodes, keeping the agent at the edge of its capability."

---

## [1:55 – 2:00] CLOSE — Call to action

> *Show README on Hugging Face Space page.*

> "Crisis Negotiator. Built on OpenEnv. Hosted on Hugging Face Spaces.
> Train your own agent. Negotiate with consequences."

> *Final frame: `🤗 huggingface.co/spaces/<your-handle>/crisis-negotiator-openenv`*

---

## Recording Tips

| Section | Key Visual | What to Stress |
|---|---|---|
| Hook | Logo on black | The 800/year stat — anchors the real-world impact |
| Problem | Terminal running eval | This isn't a toy — it's evaluating right now |
| Environment | UI with LIVE: ON | The button proves you're hitting real API |
| Training | reward_curve.png full-screen | Overlay 3 KPIs on-screen: surrender rate, harm rate, steps-to-resolution |
| Self-Improvement | Procedural card | 540+ scenarios — not just 10 hand-crafted |
| Close | HF Space URL | Make it ridiculously easy for judges to click |

## Software for Recording

- **Windows**: OBS Studio (free) + a quiet room.
- **Audio**: USB mic preferred. If using laptop mic, post-process with
  https://podcast.adobe.com/enhance (free, removes noise).
- **Cursor**: Enable "Highlight cursor" in OBS for visibility.
- **Screen res**: Record at 1920x1080 minimum.
- **Export**: MP4 (H.264) — YouTube and HF prefer this.
- **Length**: Stay under 2:00. Trim ruthlessly.

## Pitch Variants (if you have less time)

### 60-second version (cut Self-Improvement section)
Hook (10) → Problem (15) → Environment (20) → Training (15) → Close

### 30-second version (just for HF Space description)
> "Multi-agent crisis negotiation environment for OpenEnv. 6 agents,
> hidden state, 14-component reward, 540+ procedural scenarios.
> Trained Qwen 2.5 3B with GRPO in 24 minutes — 100% surrender, 7.1 steps, matches FBI baseline."
