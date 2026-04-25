# Model Tuning Notes (Metric-Driven)

Date: 2026-04-25

This note is based on:

```bash
python eval/analyze_metrics.py
```

## Key findings from current metrics

- Trained policy is currently strongest on aggregate reward (`0.9537`) and steps (`7.13`).
- Hard-tier reward still has room for robustness margin.
- The trained policy shows action-collapse:
  - dominant action = `acknowledge_demand`
  - dominant share = **82.2%**
  - action entropy = **0.907** (very low)

## Implemented training-time tweaks

In `training/train_local_v2.py` scoring:

1. Added **`ack_timing_penalty`**
   - Penalizes early `acknowledge_demand` usage in opening phase when no specific demand grounding is present.

2. Added **`collapse_penalty`**
   - Penalizes high single-action dominance over recent action horizon.

3. Added **`opening_explore_bonus`**
   - Rewards `mirror`/`open_question` with quality cues in opening phase to improve evidence gathering and policy diversity.

These changes are intended to preserve final outcome performance while reducing degenerate one-action policies.

## Suggested next run config

- Keep model/checkpoint unchanged for controlled A/B.
- Run 3 seeds with current script and compare vs pre-change checkpoint:
  - mean final reward
  - hard-tier reward
  - dominant-action share
  - action entropy
  - steps-to-resolution

Recommended command:

```bash
python eval/run_multiseed_eval.py --seeds 10000,11000,12000 --n 30
```

Also enabled in training:
- Late-stage hard-biased curriculum scheduling in `train_local_v2.py` (default after 50% progress).
- Adversarial scenario packs can now be requested via environment reset:
  - `adversarial:empathy_spam:hard`
  - `adversarial:concession_spam:hard`

Checkpoint league evaluation (best-by-hard-tier):

```bash
python eval/checkpoint_league.py --root ./crisis-negotiator-trained-v2 --n 20 --difficulties easy,medium,hard
```

## Success criteria for this tuning pass

- Dominant action share: **<= 55%** (from 82.2%)
- Action entropy: **>= 1.5**
- No regression on:
  - surrender rate (target: 100%)
  - harm rate (target: 0%)
  - hard-tier reward (avoid drop > 0.01)
