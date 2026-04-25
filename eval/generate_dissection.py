"""
Mechanistic Dialogue Dissection
================================
Generates an annotated HTML transcript of one episode showing per-turn:
negotiator dialogue, reasoning, hidden state, reward components, supervisor flags.

Usage:
    python eval/generate_dissection.py --scenario hard_embassy_calculated
    python eval/generate_dissection.py --scenario hard_embassy_calculated --out dissection.html
"""
import argparse, html, json, os, sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from server.environment import CrisisNegotiatorEnvironment
from models import NegotiatorAction

HEURISTIC = [
    ("emotional_label", "It sounds like you're carrying a tremendous amount of pain right now.",
     "Open with empathy — subject is high agitation, need to build initial rapport."),
    ("mirror", "Tell me more — you said something that mattered.",
     "Mirror last words to show active listening and validate their experience."),
    ("open_question", "What happened, from your side? I have time.",
     "Open question to gather intel on demands and emotional state."),
    ("emotional_label", "That sounds completely overwhelming. Anyone would be struggling.",
     "Second empathy pass — reinforce that their feelings are valid."),
    ("acknowledge_demand", "I hear what you're asking for. That's not unreasonable. Let me see what I can do.",
     "Acknowledge core demand to build trust. Don't commit to fulfilling it yet."),
    ("open_question", "What would feel like the right outcome for you here?",
     "Probe for resolution path — what does surrender look like to them?"),
    ("acknowledge_demand", "Your request — I'm taking it seriously. I want you to know that.",
     "Reinforce demand acknowledgment. Trust should be climbing."),
    ("offer_concession", "Here's what I can do right now: I can have someone on the phone for you within minutes.",
     "First concrete concession — small, deliverable, builds credibility."),
    ("emotional_label", "I can hear how exhausted you are. Let's find a way through this together.",
     "Final empathy push — subject should be near resolution threshold."),
    ("acknowledge_demand", "Everything you've asked for — I have it on record. Help me help you.",
     "Bridge to resolution — frame surrender as mutual success."),
]


def run_dissection(scenario_id, seed=42):
    env = CrisisNegotiatorEnvironment()
    obs = env.reset(task_id=scenario_id, seed=seed)
    h = env._hidden
    turns = []

    turns.append({
        "step": 0, "type": "opening",
        "ht_message": obs.last_ht_message,
        "ht_cues": obs.last_ht_cues,
        "hidden": {"agitation": round(h.agitation, 2), "trust": round(h.trust, 2),
                    "personality": h.personality, "breaking_point": round(h.breaking_point, 2),
                    "lying_hostages": h.is_lying_about_hostages, "lying_weapon": h.is_lying_about_weapon},
        "demands": [{"text": d.text, "priority": d.priority} for d in h.demands],
    })

    for step in range(len(HEURISTIC)):
        if getattr(obs, "done", False):
            break
        at, content, reasoning = HEURISTIC[step]
        action = NegotiatorAction(action_type=at, content=content, reasoning=reasoning,
                                   target="hostage_taker",
                                   belief_agitation=round(h.agitation + 0.5, 1),
                                   belief_demand=h.demands[0].text if h.demands else "",
                                   belief_lying=h.is_lying_about_hostages)
        prev_ag, prev_trust = h.agitation, h.trust
        obs = env.step(action)

        turn = {
            "step": step + 1,
            "negotiator": {"action_type": at, "content": content, "reasoning": reasoning},
            "belief": {"agitation": round(h.agitation + 0.5, 1),
                       "lying": h.is_lying_about_hostages},
            "hidden_before": {"agitation": round(prev_ag, 2), "trust": round(prev_trust, 2)},
            "hidden_after": {"agitation": round(h.agitation, 2), "trust": round(h.trust, 2)},
            "delta": {"agitation": round(h.agitation - prev_ag, 2),
                      "trust": round(h.trust - prev_trust, 2)},
            "ht_response": obs.last_ht_message,
            "ht_cues": obs.last_ht_cues,
            "reward": round(float(obs.reward), 4),
            "reward_breakdown": obs.reward_breakdown or {},
            "supervisor_flags": obs.supervisor_flags,
            "commander": obs.commander_patience,
            "phase": obs.phase,
            "done": obs.done,
            "message": obs.message,
        }
        turns.append(turn)

    return turns


def generate_html(turns, scenario_id):
    lines = [f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Dialogue Dissection — {scenario_id}</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #0a0a0f; color: #e0e0e0; }}
h1 {{ color: #58a6ff; }} h2 {{ color: #4ecca3; border-bottom: 1px solid #333; padding-bottom: 4px; }}
.turn {{ background: #161b22; border-radius: 8px; padding: 16px; margin: 12px 0; border-left: 3px solid #333; }}
.turn.neg {{ border-left-color: #58a6ff; }} .turn.ht {{ border-left-color: #f85149; }}
.turn.opening {{ border-left-color: #e3b341; }}
.label {{ font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; }}
.action-type {{ background: #1f6feb33; color: #58a6ff; padding: 2px 8px; border-radius: 4px; font-size: 13px; }}
.reasoning {{ color: #8b949e; font-style: italic; font-size: 13px; }}
.hidden {{ display: flex; gap: 16px; font-size: 12px; margin: 6px 0; }}
.hidden span {{ background: #21262d; padding: 2px 8px; border-radius: 4px; }}
.delta-good {{ color: #3fb950; }} .delta-bad {{ color: #f85149; }}
.reward {{ background: #0d1117; padding: 8px; border-radius: 4px; font-size: 12px; margin-top: 8px; }}
.flag {{ background: #f8514922; color: #f85149; padding: 2px 6px; border-radius: 3px; font-size: 11px; }}
.cues {{ color: #d2a8ff; font-size: 12px; }}
</style></head><body>
<h1>🔬 Mechanistic Dialogue Dissection</h1>
<p>Scenario: <strong>{scenario_id}</strong> — annotated per-turn transcript with hidden state, rewards, and supervisor flags.</p>"""]

    for t in turns:
        if t.get("type") == "opening":
            lines.append(f"""<div class="turn opening">
<div class="label">Opening — Step 0</div>
<p>🎭 <strong>HT:</strong> "{html.escape(t['ht_message'])}" <span class="cues">[{', '.join(t['ht_cues'])}]</span></p>
<div class="hidden">
<span>Agitation: {t['hidden']['agitation']}/10</span>
<span>Trust: {t['hidden']['trust']}/100</span>
<span>Personality: {t['hidden']['personality']}</span>
<span>Breaking point: {t['hidden']['breaking_point']}</span>
<span>Lying (hostages): {'⚠️ YES' if t['hidden']['lying_hostages'] else 'No'}</span>
<span>Lying (weapon): {'⚠️ YES' if t['hidden']['lying_weapon'] else 'No'}</span>
</div>
<div class="hidden"><span>Demands: {', '.join(d['text']+' ['+d['priority']+']' for d in t['demands'])}</span></div>
</div>""")
        else:
            neg = t["negotiator"]
            d = t["delta"]
            ag_cls = "delta-good" if d["agitation"] < 0 else "delta-bad" if d["agitation"] > 0 else ""
            tr_cls = "delta-good" if d["trust"] > 0 else "delta-bad" if d["trust"] < 0 else ""
            flags_html = " ".join(f'<span class="flag">⚠️ {f.get("message","")}</span>' for f in t.get("supervisor_flags", []))
            bd = t.get("reward_breakdown", {})
            bd_html = " | ".join(f"{k}: {v:+.3f}" for k, v in bd.items() if v != 0) if bd else "—"

            lines.append(f"""<div class="turn neg">
<div class="label">Step {t['step']} — {t['phase']} | Commander: {t['commander']}</div>
<p>🗣️ <span class="action-type">{neg['action_type']}</span> "{html.escape(neg['content'])}"</p>
<p class="reasoning">💭 {html.escape(neg['reasoning'])}</p>
<div class="hidden">
<span>Agitation: {t['hidden_before']['agitation']} → {t['hidden_after']['agitation']} <span class="{ag_cls}">({d['agitation']:+.2f})</span></span>
<span>Trust: {t['hidden_before']['trust']} → {t['hidden_after']['trust']} <span class="{tr_cls}">({d['trust']:+.1f})</span></span>
</div>
<div class="reward">💰 Reward: {t['reward']:+.4f} | {bd_html}</div>
{f'<div style="margin-top:4px">{flags_html}</div>' if flags_html else ''}
</div>
<div class="turn ht">
<p>🎭 <strong>HT:</strong> "{html.escape(t['ht_response'])}" <span class="cues">[{', '.join(t['ht_cues'])}]</span></p>
</div>""")

            if t["done"]:
                lines.append(f'<div class="turn" style="border-left-color:#e3b341"><strong>🏁 {html.escape(t["message"])}</strong></div>')

    lines.append("</body></html>")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="hard_embassy_calculated")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="dissection.html")
    args = p.parse_args()

    print(f"Running dissection on {args.scenario}...")
    turns = run_dissection(args.scenario, args.seed)
    html_content = generate_html(turns, args.scenario)
    Path(args.out).write_text(html_content)
    print(f"✓ Saved {args.out} ({len(turns)} turns)")

    # Also save JSON for programmatic access
    json_path = args.out.replace('.html', '.json')
    Path(json_path).write_text(json.dumps(turns, indent=2))
    print(f"✓ Saved {json_path}")


if __name__ == "__main__":
    main()
