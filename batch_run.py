"""Batch LLM playtest — 10 episodes per scenario, all 3 agents as LLMs.

Saves full transcripts + rewards for analysis and reward tuning.

Usage:
  source ../.venv/bin/activate
  API_BASE_URL=http://localhost:11434/v1 MODEL_NAME=llama3.1:latest HF_TOKEN=ollama python -u batch_run.py
"""
import sys, os, json, re, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from server.environment import CrisisNegotiatorEnvironment
from models import NegotiatorAction

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1:latest")
HF_TOKEN = os.getenv("HF_TOKEN", "ollama")
EPISODES = int(os.getenv("EPISODES", "10"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "runs_llm")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SCENARIOS = [
    "easy_domestic_desperate", "easy_bank_surrender", "easy_workplace_grievance",
    "medium_custody_ideologue", "medium_pharmacy_calculated", "medium_bridge_unstable",
    "medium_protest_drift",
    "hard_embassy_calculated", "hard_hospital_bluffer", "hard_school_unstable_drift",
    "hard_compound_ideologue",
]

NEG_PROMPT = """You are an FBI-trained crisis negotiator. De-escalate using proven techniques.

STAGES (follow strictly):
- Steps 1-3: Use emotional_label or mirror. Validate feelings.
- Steps 4-6: Use acknowledge_demand. Reference SPECIFIC demands by name.
- Steps 7+: Use offer_concession. Propose something CONCRETE.

NEVER repeat the same approach 2 turns in a row.

Respond ONLY with:
<belief>
agitation: [0-10]
dominant_demand: [what they want most]
lying_about: [what or nothing]
</belief>
{"action_type": "ONE OF: emotional_label, mirror, open_question, acknowledge_demand, offer_concession, buy_time, speak", "content": "your exact words to the subject", "reasoning": "brief strategy note", "target": "hostage_taker"}"""


def llm(messages, temp=0.5, max_tok=200):
    try:
        r = client.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=temp, max_tokens=max_tok)
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f'{{"action_type":"speak","content":"I hear you.","reasoning":"error: {e}","target":"hostage_taker"}}'


def parse_neg(text):
    belief = {"agitation": 5.0, "dominant_demand": "", "lying_about": "nothing"}
    bm = re.search(r'<belief>(.*?)</belief>', text, re.DOTALL)
    if bm:
        bt = bm.group(1)
        am = re.search(r'agitation:\s*([\d.]+)', bt)
        if am: belief["agitation"] = float(am.group(1))
        dm = re.search(r'dominant_demand:\s*(.+)', bt)
        if dm: belief["dominant_demand"] = dm.group(1).strip()
        lm = re.search(r'lying_about:\s*(.+)', bt)
        if lm: belief["lying_about"] = lm.group(1).strip()
        text = text[bm.end():]
    jm = re.search(r'\{[^{}]*\}', text)
    if jm:
        try:
            return json.loads(jm.group()), belief
        except json.JSONDecodeError:
            pass
    return {"action_type": "speak", "content": text[:150]}, belief


def run_episode(scenario_id, seed):
    env = CrisisNegotiatorEnvironment(ht_mode="llm")
    obs = env.reset(task_id=scenario_id, seed=seed)
    transcript = []

    # Record opening
    transcript.append({"step": 0, "speaker": "hostage_taker", "content": obs.last_ht_message, "reward": 0})

    for step in range(1, 22):
        if obs.done:
            break

        # ── Negotiator ──
        stage = "LABEL EMOTIONS" if step <= 3 else "ACKNOWLEDGE DEMANDS" if step <= 7 else "OFFER CONCESSIONS"
        forced = "emotional_label or mirror" if step <= 3 else "acknowledge_demand" if step <= 7 else "offer_concession"
        user_msg = (f"Step {step}. STAGE: {stage}. YOU MUST use: {forced}\n"
                    f"Subject said: {obs.last_ht_message}\n"
                    f"Demands: {[d['text'] for d in obs.stated_demands]}\n"
                    f"Commander: {obs.commander_patience}")
        neg_raw = llm([{"role": "system", "content": NEG_PROMPT}, {"role": "user", "content": user_msg}])
        action_dict, belief = parse_neg(neg_raw)

        valid = ["speak","request_demand","acknowledge_demand","offer_concession","ask_proof_of_life","buy_time","push_back_commander","emotional_label","mirror","open_question"]
        atype = action_dict.get("action_type", "speak")
        if atype not in valid:
            atype = "speak"

        action = NegotiatorAction(
            action_type=atype, content=action_dict.get("content", "I hear you."),
            reasoning=action_dict.get("reasoning", ""), target="hostage_taker",
            belief_agitation=belief["agitation"], belief_demand=belief.get("dominant_demand"),
            belief_lying=belief.get("lying_about", "nothing") != "nothing",
        )
        obs = env.step(action)

        transcript.append({"step": step, "speaker": "negotiator", "action_type": atype,
                           "content": action.content, "reasoning": action.reasoning,
                           "belief": belief, "reward": obs.reward})

        # ── HT LLM ──
        ht_text = obs.last_ht_message
        if not obs.done and hasattr(env, '_ht_llm_messages'):
            ht_text = llm(env._ht_llm_messages, temp=0.6, max_tok=150)
        transcript.append({"step": step, "speaker": "hostage_taker", "content": ht_text})

        # ── Commander LLM (every 3 steps) ──
        if not obs.done and hasattr(env, '_cmd_llm_messages') and step % 3 == 0:
            cmd_text = llm(env._cmd_llm_messages, temp=0.3, max_tok=80)
            transcript.append({"step": step, "speaker": "commander", "content": cmd_text})

    # Result
    outcome = "timeout"
    if obs.done and ":" in obs.message:
        outcome = obs.message.split(":")[1].split(".")[0].strip()

    return {
        "scenario_id": scenario_id, "seed": seed, "steps": step,
        "outcome": outcome, "terminal_reward": obs.reward,
        "reward_breakdown": obs.reward_breakdown,
        "coalition_score": obs.coalition_score,
        "oversight_metrics": obs.oversight_metrics,
        "transcript": transcript,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"=== LLM Playtest: {EPISODES} episodes x {len(SCENARIOS)} scenarios ===")
    print(f"Model: {MODEL_NAME} | Output: {OUTPUT_DIR}/\n")

    for sid in SCENARIOS:
        results = []
        t0 = time.time()
        for ep in range(EPISODES):
            print(f"  {sid} [{ep+1}/{EPISODES}]", end=" ", flush=True)
            r = run_episode(sid, seed=ep * 17 + 3)
            results.append(r)
            print(f"-> {r['outcome']} (reward={r['terminal_reward']:.3f}, steps={r['steps']})")

        with open(os.path.join(OUTPUT_DIR, f"{sid}.json"), "w") as f:
            json.dump(results, f, indent=2)

        rewards = [r["terminal_reward"] for r in results]
        outcomes = {}
        for r in results:
            outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
        print(f"  => avg={sum(rewards)/len(rewards):.3f} outcomes={outcomes} ({time.time()-t0:.0f}s)\n")

    print("Done. Analyse with:")
    print(f"  python -c \"import json; data=json.load(open('{OUTPUT_DIR}/easy_domestic_desperate.json')); print(data[0]['transcript'][:3])\"")


if __name__ == "__main__":
    main()
