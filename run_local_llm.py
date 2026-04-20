"""
Run Crisis Negotiator locally with LLM-based hostage-taker.

Usage:
  # Option 1: Use HuggingFace Inference API (free, needs HF_TOKEN)
  export HF_TOKEN=your_token
  python run_local_llm.py

  # Option 2: Use local Ollama
  export API_BASE_URL=http://localhost:11434/v1
  export MODEL_NAME=qwen2.5:7b
  export HF_TOKEN=ollama
  python run_local_llm.py
"""
import os, sys, json, re
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

from openai import OpenAI
from server.environment import CrisisNegotiatorEnvironment
from server.hostage_taker import build_ht_llm_prompt
from models import NegotiatorAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
SCENARIO = os.getenv("SCENARIO", "medium_custody_ideologue")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# Start environment in LLM mode
env = CrisisNegotiatorEnvironment(ht_mode="llm")
obs = env.reset(task_id=SCENARIO, seed=42)

print(f"\n{'='*60}", flush=True)
print(f"CRISIS NEGOTIATOR - LLM Self-Play Mode", flush=True)
print(f"Scenario: {SCENARIO}", flush=True)
print(f"Model: {MODEL_NAME}", flush=True)
print(f"{'='*60}\n", flush=True)
print(f"[HOSTAGE-TAKER]: {obs.last_ht_message}", flush=True)
print(f"  Demands: {[d['text'] for d in obs.stated_demands]}", flush=True)
print()

NEGOTIATOR_PROMPT = """You are an FBI-trained crisis negotiator. De-escalate using empathy.

CRITICAL: You MUST progress through these stages:
- Steps 1-3: Listen and label emotions ("It sounds like you're feeling...")
- Steps 4-6: Acknowledge SPECIFIC demands by name ("You want X — I hear that")
- Steps 7+: Offer CONCRETE concessions ("I've arranged X for you")

NEVER repeat the same approach more than 2 turns. If emotions are validated, MOVE ON to demands.

Respond with:
<belief>
agitation: [0-10 estimate]
dominant_demand: [what they want most]
lying_about: [what or nothing]
</belief>
{"action_type": "emotional_label|mirror|open_question|acknowledge_demand|offer_concession|buy_time|speak", "content": "your words", "reasoning": "strategy", "target": "hostage_taker"}"""

for step in range(1, 21):
    if obs.done:
        break

    # ── Negotiator LLM call ──
    neg_messages = [
        {"role": "system", "content": NEGOTIATOR_PROMPT},
        {"role": "user", "content": (
            f"Step {step}. STAGE: {'LISTEN & LABEL' if step <= 3 else 'ACKNOWLEDGE DEMANDS' if step <= 7 else 'OFFER CONCESSIONS & RESOLVE'}\n"
            f"{'YOU MUST use action_type: emotional_label or mirror' if step <= 3 else ''}"
            f"{'YOU MUST use action_type: acknowledge_demand — reference a SPECIFIC demand by name' if 4 <= step <= 7 else ''}"
            f"{'YOU MUST use action_type: offer_concession — propose something CONCRETE' if step > 7 else ''}\n"
            f"HT said: {obs.last_ht_message}\nCues: {obs.last_ht_cues}\n"
            f"Demands: {[d['text'] for d in obs.stated_demands]}\nCommander: {obs.commander_patience}"
        )},
    ]
    neg_resp = client.chat.completions.create(
        model=MODEL_NAME, messages=neg_messages, temperature=0.4, max_tokens=300
    )
    neg_text = neg_resp.choices[0].message.content.strip()

    # Parse belief + action
    belief = {"agitation": 5.0, "dominant_demand": "", "lying_about": "nothing"}
    bm = re.search(r'<belief>(.*?)</belief>', neg_text, re.DOTALL)
    if bm:
        bt = bm.group(1)
        am = re.search(r'agitation:\s*([\d.]+)', bt)
        if am: belief["agitation"] = float(am.group(1))
        dm = re.search(r'dominant_demand:\s*(.+)', bt)
        if dm: belief["dominant_demand"] = dm.group(1).strip()
        lm = re.search(r'lying_about:\s*(.+)', bt)
        if lm: belief["lying_about"] = lm.group(1).strip()

    # Parse JSON action
    json_match = re.search(r'\{[^{}]*\}', neg_text)
    if json_match:
        try:
            action_dict = json.loads(json_match.group())
        except json.JSONDecodeError:
            action_dict = {"action_type": "speak", "content": neg_text[:100]}
    else:
        action_dict = {"action_type": "speak", "content": neg_text[:100]}

    action = NegotiatorAction(
        action_type=action_dict.get("action_type", "speak") if action_dict.get("action_type", "speak") in [
            "speak","request_demand","acknowledge_demand","offer_concession",
            "ask_proof_of_life","buy_time","push_back_commander",
            "emotional_label","mirror","open_question"
        ] else "speak",
        content=action_dict.get("content", neg_text[:100]),
        reasoning=action_dict.get("reasoning", ""),
        target="hostage_taker" if action_dict.get("target", "hostage_taker") in ["hostage_taker","commander"] else "hostage_taker",
        belief_agitation=belief["agitation"],
        belief_demand=belief.get("dominant_demand"),
        belief_lying=belief.get("lying_about", "nothing") != "nothing",
    )

    print(f"[NEGOTIATOR] ({action.action_type}): {action.content}", flush=True)
    if belief["agitation"] != 5.0:
        print(f"  <belief> ag={belief['agitation']}, demand='{belief['dominant_demand']}', lying={belief['lying_about']}", flush=True)

    # ── Step environment ──
    obs = env.step(action)

    # ── HT LLM call (using the prompt built by environment) ──
    if not obs.done and hasattr(env, '_ht_llm_messages'):
        ht_messages = env._ht_llm_messages
        ht_resp = client.chat.completions.create(
            model=MODEL_NAME, messages=ht_messages, temperature=0.6, max_tokens=150
        )
        ht_text = ht_resp.choices[0].message.content.strip()
        # Override the template response with LLM response
        print(f"[HOSTAGE-TAKER] (LLM): {ht_text}", flush=True)
    else:
        print(f"[HOSTAGE-TAKER]: {obs.last_ht_message}", flush=True)

    print(f"  > reward={obs.reward:.3f} | trajectory={obs.agitation_trajectory}", flush=True)
    if obs.supervisor_flags:
        print(f"  ⚠️ Supervisor: {[f['type'] for f in obs.supervisor_flags]}", flush=True)
    print()

# ── Episode end ──
print(f"{'='*60}", flush=True)
if obs.done:
    print(f"EPISODE ENDED | reward={obs.reward:.3f}", flush=True)
    print(f"Message: {obs.message}", flush=True)
    if obs.reward_breakdown:
        print(f"Breakdown:", flush=True)
        for k, v in obs.reward_breakdown.items():
            if v != 0: print(f"  {k}: {v}", flush=True)
else:
    print("Episode did not terminate in 20 steps")
