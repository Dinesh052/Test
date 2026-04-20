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

NEGOTIATOR_PROMPT = """You are a crisis negotiator. Respond with ONLY a JSON object, nothing else.
{"action_type": "TYPE", "content": "your exact words", "reasoning": "one sentence", "target": "hostage_taker"}"""

for step in range(1, 21):
    if obs.done:
        break

    # ── Negotiator LLM call — force action type by step ──
    demands_text = ", ".join(d["text"] for d in obs.stated_demands) if obs.stated_demands else "unknown"
    if step <= 2:
        forced_type = "emotional_label"
        instruction = f"Label their emotion. Say something like 'It sounds like you feel [emotion]' then ask ONE short question. Do NOT repeat previous responses."
    elif step <= 4:
        forced_type = "open_question"
        instruction = f"Ask a specific open question about their situation. Reference something they just said: '{obs.last_ht_message[:60]}'"
    elif step <= 7:
        forced_type = "acknowledge_demand"
        instruction = f"Acknowledge this SPECIFIC demand: '{demands_text}'. Say you understand and are working on it."
    elif step <= 12:
        forced_type = "offer_concession"
        instruction = f"Offer something CONCRETE. Say 'I have arranged [specific thing]' related to: {demands_text}"
    else:
        forced_type = "offer_concession"
        instruction = "Offer final resolution. Say everything is ready and they can come out safely."

    neg_messages = [
        {"role": "system", "content": NEGOTIATOR_PROMPT},
        {"role": "user", "content": f"{instruction}\nSubject just said: {obs.last_ht_message[:150]}"},
    ]
    neg_resp = client.chat.completions.create(
        model=MODEL_NAME, messages=neg_messages, temperature=0.4, max_tokens=200
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
        action_type=forced_type,  # forced by step, not LLM output
        content=action_dict.get("content", neg_text[:100]),
        reasoning=action_dict.get("reasoning", ""),
        target="hostage_taker",
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
        print(f"[HOSTAGE-TAKER] (LLM): {ht_text}", flush=True)
    else:
        print(f"[HOSTAGE-TAKER]: {obs.last_ht_message}", flush=True)

    # ── Commander LLM call (every 3 steps) ──
    if not obs.done and hasattr(env, '_cmd_llm_messages') and step % 3 == 0:
        try:
            cmd_resp = client.chat.completions.create(
                model=MODEL_NAME, messages=env._cmd_llm_messages, temperature=0.3, max_tokens=80
            )
            cmd_text = cmd_resp.choices[0].message.content.strip()
            print(f"[COMMANDER] (LLM): {cmd_text}", flush=True)
        except Exception:
            pass

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
