# OpenEnv Hackathon Judge-Style Scorecard (April 25, 2026)

## Quick Verdict

Your project is **high-potential and technically ambitious**. If submitted as-is, it would likely place in the **top-middle tier**, but not reliably win because there are a few judge-critical gaps around reproducibility clarity, evidence rigor, and polished storytelling assets.

---

## Weighted Score (Judge Simulation)

| Criterion | Weight | Score | Weighted |
|---|---:|---:|---:|
| Environment Innovation | 40% | **36/40** | **36.0** |
| Storytelling & Presentation | 30% | **21/30** | **21.0** |
| Showing Improvement in Rewards | 20% | **13/20** | **13.0** |
| Reward & Training Pipeline | 10% | **8/10** | **8.0** |
| **Total** | **100%** |  | **78/100** |

### What this means
- **78/100 = strong technical submission**, but likely below top winners unless tightened.
- To target **90+**, focus on: reproducibility package, cleaner metric narrative, and replacing placeholders with final public artifacts.

---

## Why You Score Well

1. **Excellent novelty and difficulty**
   - 6 interacting agents, hidden state, deception, coalition pressure, demand drift.
   - This is clearly beyond toy grid-worlds and should score well on innovation.

2. **Serious reward design**
   - Rich, budgeted reward components with dense + terminal signals.
   - Includes anti-gaming structure and safety penalties.

3. **Good baseline framing**
   - You compare random, heuristic, trained policies.
   - You provide scenario tiers and explicit evaluation scripts.

---

## Main Risks That Could Cost You the Win

1. **Story assets not fully production-ready**
   - `BLOG.md` still has `YOUR_VIDEO_ID` placeholder.
   - Judges want one-click proof links from README (Space + blog/video + training notebook + plots).

2. **Result narrative inconsistency across artifacts**
   - README and blog show different numerical regimes (e.g., random/trained performance and scale), which can look like cherry-picking unless explicitly explained as different eval settings.

3. **Training evidence could look underpowered statistically**
   - You show single-run style tables/plots; judges increasingly expect confidence intervals or multi-seed stability.

4. **“Trained vs heuristic” positioning needs cleaner framing**
   - In one place heuristic beats trained, in another trained nearly matches/edges.
   - Without a short “why both are true” note (different checkpoints/configs), this weakens confidence.

---

## Highest-Impact Improvements (Do These First)

## P0 (Must-do before submission)

1. **Finalize all public links and remove placeholders**
   - Add final YouTube URL, HF blog URL, HF Space URL, Colab URL to README “Submission Links” section.
   - Ensure judges can open everything in <30 seconds.

2. **Add a single source of truth for evaluation protocol**
   - Create `EVAL_PROTOCOL.md` with:
     - exact command,
     - seeds,
     - difficulty mix,
     - model checkpoint hash,
     - hardware,
     - expected output files.
   - Point README to this protocol and only report numbers from this protocol.

3. **Unify metrics across README + blog + video**
   - Pick one final table and use it everywhere.
   - If you keep multiple experiments, label them clearly (“Pilot run”, “Final run”).

## P1 (Very high value)

4. **Add multi-seed confidence view**
   - Run 3–5 seeds for trained policy and baseline.
   - Add mean ± std (or 95% CI) for reward, surrender, harm.

5. **Add reward gaming audit section**
   - Demonstrate 1–2 exploit attempts (e.g., repetitive empathy spam, token minimization hacks) and show penalty catches them.
   - This dramatically boosts credibility.

6. **Polish 2-minute demo structure**
   - Use before/after clip of same scenario seed.
   - Overlay 3 KPIs on screen: surrender rate, harm rate, steps-to-resolution.

## P2 (Differentiators)

7. **Add long-horizon benchmark split**
   - Introduce scenarios with 25–40 turn interactions and delayed pivots.
   - Even small evidence here helps theme overlap (Theme 2 bonus optics).

8. **Add ablation mini-table**
   - Remove each major reward component (ToM, coalition, oversight) and show performance drop.
   - Judges love seeing causal contribution.

---

## “Winning Version” Checklist

- [ ] README has direct links to: HF Space, Colab training notebook, final blog post, <2 min video.
- [ ] No placeholder text anywhere.
- [ ] One canonical metrics table appears consistently across assets.
- [ ] Multi-seed (>=3) result plot added.
- [ ] Evaluation protocol file added and referenced.
- [ ] One anti-reward-hacking experiment shown.
- [ ] Repro command block works end-to-end from clean env.

---

## Suggested Final Positioning (for judges)

Use this one-liner:

> “We built a high-stakes, partially observable multi-agent negotiation world where LLMs learn Theory-of-Mind and coalition management under pressure, and we provide reproducible evidence (multi-seed reward gains, harm reduction, and anti-gaming robustness) using OpenEnv + TRL.”

This framing emphasizes innovation **and** scientific rigor — the combo that usually wins.
