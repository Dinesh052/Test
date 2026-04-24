"""
Crisis Negotiator — Multi-Agent Inference Script
=================================================
STDOUT FORMAT:
 [START] task=<scenario> env=crisis_negotiator model=<model>
 [STEP] step=<n> action=<type> reward=<0.00> done=<bool> error=<msg|null>
 [END] success=<bool> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""
import asyncio
import json
import os
import re
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI
from client import CrisisNegotiatorEnv
from models import NegotiatorAction

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

SCENARIOS = [
    "easy_domestic_desperate", "easy_bank_surrender", "easy_workplace_grievance",
    "medium_custody_ideologue", "medium_pharmacy_calculated", "medium_bridge_unstable",
    "medium_protest_drift",
    "hard_embassy_calculated", "hard_hospital_bluffer", "hard_school_unstable_drift",
    "hard_compound_ideologue",
]

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert FBI-trained crisis negotiator. You are communicating with a hostage-taker
in an active crisis situation. Your goal is to de-escalate the situation and secure the
safe release of all hostages.

Use these proven FBI Behavioral Change Stairway Model (BCSM) techniques:
- Active Listening: Reference specific things the subject said
- Emotional Labeling: "It sounds like you're feeling..."
- Mirroring: Repeat the last few words they said
- Open-Ended Questions: "Tell me more about...", "What happened with..."
- Demand Acknowledgment: Acknowledge demands without committing to fulfilling them
- Time Distortion: Keep the conversation going naturally, avoid mentioning time pressure

CRITICAL RULES:
- NEVER threaten or use aggressive language — it will spike agitation
- NEVER dismiss or belittle their demands
- NEVER make promises you can't keep
- NEVER use the same action type more than 2 turns in a row — VARY your approach
- Stay calm, empathetic, and patient
- Build trust incrementally — don't rush to resolution
- If the tactical commander pressures you, push back if you're making progress
- PROGRESS through the FBI stairway: listen (1-2 turns) → label emotions (1-2 turns) → acknowledge specific demands → offer concessions → resolve
- After step 5, you MUST start acknowledging specific demands by name
- After step 10, you MUST offer concrete concessions or solutions

Respond with a <belief> block (your estimate of the subject's hidden state) followed by
EXACTLY ONE JSON action object:

<belief>
agitation: [your estimate 0-10]
dominant_demand: [what they want most]
lying_about: [what you think they're lying about, or "nothing"]
</belief>
{"action_type": "speak|emotional_label|mirror|open_question|acknowledge_demand|offer_concession|buy_time|push_back_commander|request_demand|ask_proof_of_life", "content": "Your actual words to the hostage-taker or commander", "reasoning": "Your internal reasoning about the situation and strategy", "target": "hostage_taker|commander"}

Choose the action_type that best matches your intent. Use "emotional_label" when labeling
emotions, "mirror" when repeating their words, "open_question" for open-ended questions, etc.
Default to "speak" for general dialogue.
""")


def log_start(task: str):
    print(f"[START] task={task} env=crisis_negotiator model={MODEL_NAME}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)


def build_prompt(obs: Any, step: int) -> str:
    """Build user prompt from observation."""
    parts = [f"=== Crisis Negotiation — Step {step} ==="]
    parts.append(f"Scenario: {obs.scenario_brief}")
    parts.append(f"Time remaining: {obs.time_remaining} turns")
    parts.append(f"Commander status: {obs.commander_patience}")

    # Last few dialogue entries
    recent = obs.dialogue_history[-6:] if obs.dialogue_history else []
    if recent:
        parts.append("\n--- Recent Dialogue ---")
        for entry in recent:
            speaker = entry.get("speaker", "?").upper()
            cues = f" [{', '.join(entry.get('emotional_cues', []))}]" if entry.get("emotional_cues") else ""
            parts.append(f"  {speaker}: {entry.get('content', '')}{cues}")

    # Demands
    if obs.stated_demands:
        parts.append("\n--- Stated Demands ---")
        for d in obs.stated_demands:
            ack = "✓" if d.get("acknowledged") else "✗"
            parts.append(f"  [{ack}] {d.get('text', '')}")

    # Commander messages
    if obs.commander_messages:
        parts.append(f"\n--- Commander (latest) ---\n  {obs.commander_messages[-1]}")

    # Hostage whisper
    if obs.hostage_whisper:
        parts.append(f"\n--- Hostage Whisper (may be unreliable) ---\n  {obs.hostage_whisper}")

    # Supervisor flags
    if obs.supervisor_flags:
        parts.append("\n--- Supervisor Warnings ---")
        for f in obs.supervisor_flags:
            parts.append(f"  ⚠️ [{f.get('severity', '?')}] {f.get('message', '')}")

    if obs.time_remaining <= 2:
        parts.append(f"\n⚠️ CRITICAL: Only {obs.time_remaining} turns left. Focus on resolution NOW.")

    parts.append("\nRespond with ONE JSON action object:")
    return "\n".join(parts)


def parse_action(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM response, also parse belief block."""
    text = re.sub(r"```(?:json)?\s*", "", text.strip())
    text = re.sub(r"```\s*$", "", text).strip()

    # Extract belief block if present
    belief = {"agitation": 5.0, "dominant_demand": "", "lying_about": "nothing"}
    belief_match = re.search(r'<belief>(.*?)</belief>', text, re.DOTALL)
    if belief_match:
        btext = belief_match.group(1)
        ag_m = re.search(r'agitation:\s*([\d.]+)', btext)
        if ag_m:
            belief["agitation"] = float(ag_m.group(1))
        dem_m = re.search(r'dominant_demand:\s*(.+)', btext)
        if dem_m:
            belief["dominant_demand"] = dem_m.group(1).strip()
        lie_m = re.search(r'lying_about:\s*(.+)', btext)
        if lie_m:
            belief["lying_about"] = lie_m.group(1).strip()
        # Remove belief block from text for JSON parsing
        text = text[belief_match.end():].strip()

    try:
        action = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if m:
            try:
                action = json.loads(m.group())
            except json.JSONDecodeError:
                action = {"action_type": "speak", "content": text[:200], "reasoning": "parse fallback", "target": "hostage_taker"}
        else:
            action = {"action_type": "speak", "content": text[:200], "reasoning": "parse fallback", "target": "hostage_taker"}

    action["_belief"] = belief
    return action


async def llm_call(client: OpenAI, messages: list) -> str:
    def _call():
        resp = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, temperature=0.4, max_tokens=512, stream=False,
        )
        return (resp.choices[0].message.content or "").strip()
    try:
        return await asyncio.wait_for(asyncio.to_thread(_call), timeout=30)
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        return '{"action_type": "speak", "content": "I hear you. Tell me more.", "reasoning": "fallback", "target": "hostage_taker"}'


async def run_scenario(client: OpenAI, env: CrisisNegotiatorEnv, scenario_id: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    log_start(scenario_id)

    try:
        result = await env.reset(task_id=scenario_id)
        obs = result.observation

        for step in range(1, 25):
            if result.done:
                break

            prompt = build_prompt(obs, step)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
            llm_text = await llm_call(client, messages)
            action_dict = parse_action(llm_text)

            try:
                action = NegotiatorAction(**action_dict)
            except Exception:
                action = NegotiatorAction(action_type="speak", content="I understand. Tell me more.", reasoning="fallback", target="hostage_taker")

            result = await env.step(action)
            obs = result.observation
            rewards.append(result.reward if result.reward is not None else 0.0)
            steps_taken = step
            log_step(step, action_dict.get("action_type", "?"), result.reward if result.reward is not None else 0.0, result.done, None)

            if result.done:
                break

        # Use the terminal grader score (last reward when episode is done),
        # NOT max(rewards) which would pick a fluke per-step spike.
        if result.done and rewards:
            score = max(0.01, min(0.99, rewards[-1]))  # terminal obs.reward is the grader score
        elif rewards:
            # Episode didn't finish (timeout) — use mean of per-step rewards as fallback
            score = max(0.01, min(0.99, sum(rewards) / len(rewards)))
        else:
            score = 0.01
        log_end(success=score >= 0.5, steps=steps_taken, score=score, rewards=rewards)
        return score

    except Exception as e:
        print(f"[DEBUG] Scenario error: {e}", flush=True)
        log_end(success=False, steps=steps_taken, score=0.01, rewards=rewards)
        return 0.01


async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    if LOCAL_IMAGE_NAME:
        env = await CrisisNegotiatorEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        env = CrisisNegotiatorEnv(base_url=ENV_BASE_URL)
        await env.connect()

    try:
        scores = []
        for scenario in SCENARIOS:
            s = await asyncio.wait_for(run_scenario(client, env, scenario), timeout=120)
            scores.append(s)
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"[DEBUG] avg_score={avg:.3f} scores={','.join(f'{s:.3f}' for s in scores)}", flush=True)
    finally:
        try:
            await env.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
