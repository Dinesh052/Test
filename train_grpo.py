"""
train_grpo.py — GRPO fine-tuning on Crisis Negotiator.
Runs on free T4 Colab (15GB VRAM). ~30-45 min for 200 steps.
"""
import os, json
from typing import List
from reward_fn import crisis_reward_fn

MODEL_ID = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
OUTPUT_DIR = "./crisis_negotiator_grpo"
MAX_STEPS = int(os.getenv("MAX_STEPS", "200"))
BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 5e-6
MAX_NEW_TOKENS = 256
NUM_GENERATIONS = 4

SYSTEM_PROMPT = """You are an expert FBI-trained crisis negotiator.
Use de-escalation techniques: emotional labeling, mirroring, open questions, demand acknowledgment.
Respond with ONE JSON object:
{"action_type": "speak|emotional_label|mirror|open_question|acknowledge_demand|offer_concession|buy_time|push_back_commander", "content": "your words", "reasoning": "your strategy", "target": "hostage_taker"}"""

TRAINING_SCENARIOS = [
    "Crisis step 3/20. HT Marcus (domestic, armed): 'Nobody cares. My kids are everything and she's taking them.' Cues: [grief, desperation]. Demands: [custody, no charges]. Commander: patient",
    "Crisis step 7/20. HT in pharmacy (calculated, armed): 'Insulin was $400. My mother is diabetic. What choice did I have?' Cues: [anger, justification]. Demands: [insulin supply, media]. Commander: restless",
    "Crisis step 12/20. Embassy standoff, ideologue: 'You still haven't addressed my core demand. I think you're stalling.' Cues: [suspicion, controlled anger]. Demands: [TV statement, prisoner release]. Commander: urgent. Agitation: 8.2/10",
    "Crisis step 5/20. School, unstable HT: 'I just want someone to listen. Everyone dismisses me.' Cues: [abandonment, fragility]. Demands: [unclear]. Hostage whisper: 'He's crying.' Commander: patient",
    "Crisis step 15/20. Hospital, bluffer: 'I know you're just saying what you're trained to say. None of this is real.' Cues: [distrust, testing]. Demands: [admin meeting - ack'd]. Trust: 45. Commander: final_warning",
]

def build_dataset(n_repeats=20):
    dataset = []
    for _ in range(n_repeats):
        for s in TRAINING_SCENARIOS:
            dataset.append({"prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": s},
            ]})
    return dataset

def reward_for_trl(completions: List[str], **kwargs) -> List[float]:
    return crisis_reward_fn(completions)

def train():
    try:
        from unsloth import FastLanguageModel
        from trl import GRPOConfig, GRPOTrainer
        from datasets import Dataset
    except ImportError:
        print("Install: pip install unsloth trl datasets"); raise

    print(f"Loading {MODEL_ID}...")
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=MODEL_ID, max_seq_length=1024, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(model, r=16, target_modules=["q_proj","v_proj","k_proj","o_proj"], lora_alpha=16, use_gradient_checkpointing="unsloth")

    print("Building dataset...")
    raw = build_dataset(20)
    dataset = Dataset.from_list(raw)

    config = GRPOConfig(
        output_dir=OUTPUT_DIR, max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR, max_new_tokens=MAX_NEW_TOKENS, num_generations=NUM_GENERATIONS,
        logging_steps=10, save_steps=50, report_to="none", remove_unused_columns=False,
    )

    trainer = GRPOTrainer(model=model, tokenizer=tokenizer, reward_funcs=[reward_for_trl], args=config, train_dataset=dataset)

    print(f"Training for {MAX_STEPS} steps...")
    trainer.train()

    print(f"Saving to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(f"{OUTPUT_DIR}/reward_log.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    print("Done. Run: python plot_rewards.py")
    return trainer.state.log_history

if __name__ == "__main__":
    train()
