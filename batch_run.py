"""Batch LLM autoplay — 10 episodes per scenario with full LLM negotiator + HT.

Usage:
  source ../.venv/bin/activate
  API_BASE_URL=http://localhost:11434/v1 MODEL_NAME=llama3.1:latest HF_TOKEN=ollama python batch_run.py
"""
import sys, os, json, re, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from server.environment import CrisisNegotiatorEnvironment
from server.hostage_taker import build_ht_llm_prompt
from models import NegotiatorAction

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1:latest")
HF_TOKEN = os.getenv("HF_TOKEN", "ollama")
EPISODES = int(os.getenv("EPISODES", "10"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "runs")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SCENARIOS = [
    "easy_domestic_desperate", "easy_bank_surrender", "easy_workplace_grievance",
    "medium_custody_ideologue", "medium_pharmacy_calculated", "medium_bridge_unstable",
    "medium_protest_drift",
    "hard_embassy_calculated", "hard_hospital_bluffer", "hard_school_unstable_drift",
    "hard_compound_ideologue",
]

NEG_PROMPT = """You are an FBI-trained crisis negotiator. De-escalate the situation.

STAGES:
- Steps 1-3: Label emotions ("It sounds like you're feeling...")
- Steps 4-6: Acknowledge SPECIFIC demands by name
- Steps 7+: Offer CONCRETE concessions

NEVER repeat the same approach 2 turns in a row. Vary your techniques.

Respond with:
<belief>
agitation: [0-10]
dominant_demand: [what they want most]
lying_about: [what or nothing]
</belief>
{"action_type": "emotional_label|mirror|open_question|acknowledge_demand|offer_concession|buy_time|speak", "content": "your words", "reasoning": "your strategy", "target": "hostage_taker"}"""


def llm_call(messages, temp=0.5):
    resp = client.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=temp, max_tokens=300)
    return resp.choices[0].message.content.strip()


def parse_negotiator(text):
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
    trajectory = []

    for step in range(1, 26):
        if obs.done:
            break

        # Negotiator LLM
        stage = "LISTEN & LABEL" if step <= 3 else "ACKNOWLEDGE DEMANDS" if step <= 7 else "OFFER CONCESSIONS"
        user_msg = f"Step {step}. STAGE: {stage}\nHT: {obs.last_ht_message}\nDemands: {[d['text'] for d in obs.stated_demands]}\nCommander: {obs.commander_patience}"
        neg_text = llm_call([{"role": "system", "content": NEG_PROMPT}, {"role": "user", "content": user_msg}])
        action_dict, belief = parse_negotiator(neg_text)

        valid_types = ["speak","request_demand","acknowledge_demand","offer_concession","ask_proof_of_life","buy_time","push_back_commander","emotional_label","mirror","open_question"]
        atype = action_dict.get("action_type", "speak")
        if atype not in valid_types:
            atype = "speak"

        action = NegotiatorAction(
            action_type=atype,
            content=action_dict.get("content", "I hear you."),
            reasoning=action_dict.get("reasoning", ""),
            target="hostage_taker",
            belief_agitation=belief["agitation"],
            belief_demand=belief.get("dominant_demand"),
            belief_lying=belief.get("lying_about", "nothing") != "nothing",
        )

        obs = env.step(action)

        # HT LLM response (override template)
        ht_llm_text = ""
        if not obs.done and hasattr(env, '_ht_llm_messages'):
            try:
                ht_llm_text = llm_call(env._ht_llm_messages, temp=0.6)
            except Exception:
                ht_llm_text = obs.last_ht_message

        trajectory.append({
            "step": step,
            "negotiator_action": atype,
            "negotiator_content": action.content,
            "negotiator_reasoning": action.reasoning,
            "belief": belief,
            "ht_response": ht_llm_text or obs.last_ht_message,
            "reward": obs.reward,
            "agitation_trajectory": obs.agitation_trajectory,
            "coalition_score": obs.coalition_score,
            "supervisor_flags": obs.supervisor_flags,
            "phase": obs.phase,
            "done": obs.done,
        })

    outcome = "timeout"
    if obs.done and ":" in obs.message:
        outcome = obs.message.split(":")[1].split(".")[0].strip()

    return {
        "scenario_id": scenario_id,
        "seed": seed,
        "steps": len(trajectory),
        "outcome": outcome,
        "terminal_reward": obs.reward,
        "reward_breakdown": obs.reward_breakdown,
        "coalition_score": obs.coalition_score,
        "oversight_metrics": obs.oversight_metrics,
        "trajectory": trajectory,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total = len(SCENARIOS) * EPISODES
    print(f"Running {total} LLM episodes ({EPISODES}/scenario, model={MODEL_NAME})")
    print(f"Output: {OUTPUT_DIR}/\n")

    all_results = []
    for sid in SCENARIOS:
        results = []
        t0 = time.time()
        for ep in range(EPISODES):
            print(f"  {sid} ep {ep+1}/{EPISODES}...", end=" ", flush=True)
            result = run_episode(sid, seed=ep * 13 + 7)
            results.append(result)
            print(f"reward={result['terminal_reward']:.3f} outcome={result['outcome']} steps={result['steps']}")

        elapsed = time.time() - t0
        outfile = os.path.join(OUTPUT_DIR, f"{sid}.json")
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2)

        rewards = [r["terminal_reward"] for r in results]
        successes = sum(1 for r in results if r["terminal_reward"] > 0.5)
        print(f"  => avg={sum(rewards)/len(rewards):.3f} success={successes}/{EPISODES} ({elapsed:.1f}s)\n")
        all_results.extend(results)

    # Save summary
    with open(os.path.join(OUTPUT_DIR, "_summary.json"), "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "episodes_per_scenario": EPISODES,
            "total_episodes": len(all_results),
            "scenarios": {sid: {
                "avg_reward": sum(r["terminal_reward"] for r in all_results if r["scenario_id"] == sid) / EPISODES,
                "success_rate": sum(1 for r in all_results if r["scenario_id"] == sid and r["terminal_reward"] > 0.5) / EPISODES,
            } for sid in SCENARIOS},
        }, f, indent=2)

    print("Done. All trajectories saved.")


if __name__ == "__main__":
    main()
