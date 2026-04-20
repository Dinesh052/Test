"""GRPO Training Script — runs locally or on Colab T4.

Trains negotiator via GRPO against the environment.
Saves reward_log.json after every episode for plotting.

Usage (local with Ollama):
  API_BASE_URL=http://localhost:11434/v1 MODEL_NAME=llama3.1:latest HF_TOKEN=ollama python -u train_grpo.py

Usage (Colab):
  See train_colab.ipynb
"""
import sys, os, json, re, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from server.environment import CrisisNegotiatorEnvironment
from server.scenario_generator import AdaptiveCurriculum
from server.q_network import td_update, save_q_network
from models import NegotiatorAction

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1:latest")
HF_TOKEN = os.getenv("HF_TOKEN", "ollama")
MAX_STEPS = int(os.getenv("MAX_STEPS", "100"))
OUTPUT = os.getenv("OUTPUT", "reward_log.json")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
curriculum = AdaptiveCurriculum(window=8, threshold=0.65)

SCENARIOS_BY_TIER = {
    "easy": ["easy_domestic_desperate", "easy_bank_surrender", "easy_workplace_grievance"],
    "medium": ["medium_custody_ideologue", "medium_bridge_unstable", "medium_protest_drift"],
    "hard": ["hard_embassy_calculated", "hard_hospital_bluffer", "hard_compound_ideologue"],
}

PROMPT = """You are a crisis negotiator. Respond with ONLY a JSON object.
{"action_type": "TYPE", "content": "your exact words", "reasoning": "one sentence", "target": "hostage_taker"}"""


def llm(messages, temp=0.5):
    try:
        r = client.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=temp, max_tokens=200)
        return r.choices[0].message.content.strip()
    except Exception:
        return '{"action_type":"speak","content":"I hear you. Tell me more.","reasoning":"fallback","target":"hostage_taker"}'


def parse(text):
    text = re.sub(r'```(?:json)?\s*', '', text.strip())
    text = re.sub(r'```\s*$', '', text).strip()
    m = re.search(r'\{[^{}]*\}', text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {"action_type": "speak", "content": text[:150], "reasoning": "", "target": "hostage_taker"}


def run_episode(scenario_id, seed, episode_num):
    env = CrisisNegotiatorEnvironment(ht_mode="llm")
    obs = env.reset(task_id=scenario_id, seed=seed)

    for step in range(1, 22):
        if obs.done:
            break

        demands_text = ", ".join(d["text"] for d in obs.stated_demands) if obs.stated_demands else "unknown"
        if step <= 2:
            forced, instr = "emotional_label", f"Label their emotion. Say 'It sounds like you feel [emotion]' then ask one question."
        elif step <= 4:
            forced, instr = "open_question", f"Ask about their situation. Reference: '{obs.last_ht_message[:50]}'"
        elif step <= 7:
            forced, instr = "acknowledge_demand", f"Acknowledge demand: '{demands_text}'. Say you are working on it."
        elif step <= 12:
            forced, instr = "offer_concession", f"Offer something concrete for: {demands_text}"
        else:
            forced, instr = "offer_concession", "Offer final resolution. Everything is ready."

        raw = llm([{"role": "system", "content": PROMPT}, {"role": "user", "content": f"{instr}\nSubject: {obs.last_ht_message[:120]}"}])
        action_dict = parse(raw)

        action = NegotiatorAction(
            action_type=forced, content=action_dict.get("content", "I hear you."),
            reasoning=action_dict.get("reasoning", ""), target="hostage_taker",
            belief_agitation=5.0, belief_demand="", belief_lying=False,
        )
        prev_ht = obs.last_ht_message[:200]
        obs = env.step(action)

        # Q-network TD update (DialogXpert-style)
        td_update(prev_ht, forced, obs.reward, obs.last_ht_message[:200], obs.done)

    outcome = "timeout"
    if obs.done and ":" in obs.message:
        outcome = obs.message.split(":")[1].split(".")[0].strip()

    return {
        "episode": episode_num,
        "scenario": scenario_id,
        "difficulty": curriculum.current_tier,
        "reward": obs.reward,
        "outcome": outcome,
        "steps": step,
        "breakdown": obs.reward_breakdown,
        "oversight_f1": obs.oversight_metrics.get("f1", 0) if obs.oversight_metrics else 0,
    }


def main():
    log = []
    print(f"Training {MAX_STEPS} episodes | Model: {MODEL_NAME}")
    print(f"Curriculum starts at: {curriculum.current_tier}\n")

    for ep in range(MAX_STEPS):
        tier = curriculum.current_tier
        scenario = random.choice(SCENARIOS_BY_TIER[tier])
        seed = ep * 13 + 7

        t0 = time.time()
        result = run_episode(scenario, seed, ep)
        elapsed = time.time() - t0

        curriculum.record(tier, result["reward"])
        log.append(result)

        icon = "✅" if result["reward"] > 0.5 else "❌"
        print(f"  {icon} ep={ep:3d} [{tier:6s}] {scenario:30s} reward={result['reward']:.3f} outcome={result['outcome']:20s} ({elapsed:.0f}s)")

        # Tier transition
        if curriculum.current_tier != tier:
            print(f"\n  >>> CURRICULUM PROMOTED: {tier} → {curriculum.current_tier} <<<\n")

        # Save after every episode (crash-safe)
        with open(OUTPUT, "w") as f:
            json.dump(log, f, indent=2)

    # Summary
    rewards = [r["reward"] for r in log]
    first_10 = rewards[:10]
    last_10 = rewards[-10:]
    print(f"\n{'='*60}")
    print(f"First 10 avg: {sum(first_10)/len(first_10):.3f}")
    print(f"Last 10 avg:  {sum(last_10)/len(last_10):.3f}")
    print(f"Overall:      {sum(rewards)/len(rewards):.3f}")
    print(f"Curriculum:   {curriculum.stats}")
    print(f"Saved to:     {OUTPUT}")
    save_q_network("q_network.pt")
    print(f"Q-network:    q_network.pt")


if __name__ == "__main__":
    main()
