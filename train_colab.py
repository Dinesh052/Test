"""
Crisis Negotiator — GRPO Training Script (Colab-compatible)
============================================================
Requirements: pip install unsloth trl openenv-core peft

Uses Group Relative Policy Optimization (GRPO) from TRL with
Unsloth 4-bit quantization to train on a free Colab T4 GPU.
"""
import os
import json
import re
import sys
from typing import List

# ── 1. Install (uncomment in Colab) ──────────────────────
# !pip install unsloth trl openenv-core peft accelerate

# ── 2. Load Model with Unsloth 4-bit LoRA ────────────────
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)

# ── 3. Environment Setup ─────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server.environment import CrisisNegotiatorEnvironment
from server.scenario_generator import AdaptiveCurriculum
from models import NegotiatorAction

env = CrisisNegotiatorEnvironment()
curriculum = AdaptiveCurriculum(window=10, threshold=0.7)

SYSTEM_PROMPT = """You are an expert FBI-trained crisis negotiator. De-escalate the situation using:
- Emotional Labeling: "It sounds like you're feeling..."
- Mirroring: Repeat their last words
- Open Questions: "Tell me more about..."
- Demand Acknowledgment: Acknowledge without committing
- Stay calm. Never threaten. Never dismiss demands.

Respond with ONE JSON object:
{"action_type": "emotional_label|mirror|open_question|acknowledge_demand|speak|offer_concession|buy_time", "content": "your words", "reasoning": "your strategy", "target": "hostage_taker"}"""


def build_prompt(obs) -> str:
    """Build training prompt from observation."""
    parts = [f"Scenario: {obs.scenario_brief}"]
    parts.append(f"Step {obs.step}, Time left: {obs.time_remaining}")
    if obs.dialogue_history:
        recent = obs.dialogue_history[-4:]
        for e in recent:
            parts.append(f"{e['speaker'].upper()}: {e['content']}")
    if obs.stated_demands:
        parts.append(f"Demands: {[d['text'] for d in obs.stated_demands]}")
    if obs.commander_patience != "patient":
        parts.append(f"Commander: {obs.commander_patience}")
    return "\n".join(parts)


def parse_action(text: str) -> dict:
    """Parse JSON action from model output."""
    text = re.sub(r"```(?:json)?\s*", "", text.strip())
    text = re.sub(r"```\s*$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\{[^{}]*\}', text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return {"action_type": "speak", "content": text[:200], "reasoning": "", "target": "hostage_taker"}


# ── 4. Reward Function ────────────────────────────────────
def run_episode(model_output: str, scenario_seed: int) -> float:
    """Run one episode with the model's output as the first action. Returns reward."""
    scenario = curriculum.get_scenario(seed=scenario_seed)
    obs = env.reset(task_id="generate:" + scenario["difficulty"], seed=scenario_seed)

    action_dict = parse_action(model_output)
    try:
        action = NegotiatorAction(**action_dict)
    except Exception:
        action = NegotiatorAction(action_type="speak", content=model_output[:200], reasoning="", target="hostage_taker")

    obs = env.step(action)
    total_reward = obs.reward

    # Continue episode with greedy policy for remaining steps
    for _ in range(20):
        if obs.done:
            break
        # Simple heuristic continuation
        action = NegotiatorAction(
            action_type="emotional_label",
            content="I hear you. Tell me more about how you're feeling.",
            reasoning="maintain rapport",
            target="hostage_taker",
        )
        obs = env.step(action)
        total_reward += obs.reward

    final_reward = obs.reward if obs.done else max(0.01, min(0.99, (total_reward + 1) / 2))
    # Record for curriculum AFTER episode ends
    curriculum.record(scenario.get("difficulty", "medium"), final_reward)
    return final_reward


# ── 5. GRPO Training ─────────────────────────────────────
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

# Generate training prompts from scenarios
def generate_training_data(n: int = 200) -> Dataset:
    """Generate prompts by resetting env with different scenarios."""
    prompts = []
    for i in range(n):
        scenario = curriculum.get_scenario(seed=i)
        obs = env.reset(task_id="generate:" + scenario["difficulty"], seed=i)
        prompt = build_prompt(obs)
        prompts.append({"prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]})
    return Dataset.from_list(prompts)


dataset = generate_training_data(200)


def reward_fn(completions: List[str], **kwargs) -> List[float]:
    """GRPO reward function — runs each completion through the environment."""
    rewards = []
    for i, completion in enumerate(completions):
        try:
            r = run_episode(completion, scenario_seed=i)
        except Exception:
            r = 0.01
        rewards.append(r)
    return rewards


config = GRPOConfig(
    output_dir="./crisis-negotiator-grpo",
    num_generations=4,
    max_new_tokens=256,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=5e-6,
    logging_steps=10,
    save_steps=50,
    bf16=True,
)

trainer = GRPOTrainer(
    model=model,
    config=config,
    reward_funcs=[reward_fn],
    tokenizer=tokenizer,
    train_dataset=dataset,
)

# ── 6. Train ─────────────────────────────────────────────
print("Starting GRPO training...")
print(f"Curriculum: {curriculum.stats}")
trainer.train()

# ── 7. Save & Show Results ────────────────────────────────
model.save_pretrained("./crisis-negotiator-trained")
tokenizer.save_pretrained("./crisis-negotiator-trained")

print("\n=== Training Complete ===")
print(f"Curriculum final state: {curriculum.stats}")
print("Model saved to ./crisis-negotiator-trained")
print("\nTo test: python inference.py --model ./crisis-negotiator-trained")
