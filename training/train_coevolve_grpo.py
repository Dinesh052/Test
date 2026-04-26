"""
Adversarial Co-Evolution — Real Dual-GRPO Training
====================================================
Alternates GRPO training between negotiator and hostage-taker.
Each round: train one agent against the other's frozen policy.

Round 1: Train negotiator vs template HT
Round 2: Train HT vs trained negotiator (opposing reward)
Round 3: Train negotiator vs trained HT
Round 4: Train HT vs updated negotiator

Usage:
    python train_coevolve_grpo.py --rounds 4 --prompts-per-round 64
"""
from __future__ import annotations
import argparse, copy, gc, json, os, random, re, sys, time, shutil, glob, site
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Auto-patch TRL dependencies before any TRL import ────
def _patch_trl():
    sp = site.getsitepackages()[0]
    # Patch callbacks.py (weave/mergekit/llm_blender)
    cb = os.path.join(sp, 'trl', 'trainer', 'callbacks.py')
    if os.path.exists(cb):
        src = open(cb).read()
        changed = False
        for old, new in [('import weave', 'weave = None'),
                         ('from weave.trace.context', '# from weave.trace.context'),
                         ('import llm_blender', 'llm_blender = None')]:
            if old in src:
                src = src.replace(old, new)
                changed = True
        if changed:
            open(cb, 'w').write(src)
    # Patch llm_blender TRANSFORMERS_CACHE
    for f in glob.glob(os.path.join(sp, 'llm_blender', '**', '*.py'), recursive=True):
        s = open(f).read()
        if 'TRANSFORMERS_CACHE' in s:
            open(f, 'w').write(s.replace('from transformers.utils.hub import TRANSFORMERS_CACHE', 'TRANSFORMERS_CACHE = None'))

_patch_trl()

# Clear Unsloth compiled cache to prevent monkey-patch interference
_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "unsloth_compiled_cache")
if os.path.exists(_cache):
    shutil.rmtree(_cache, ignore_errors=True)

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from server.environment import CrisisNegotiatorEnvironment
from server.scenario_generator import LLMSelfPlay
from models import NegotiatorAction


# ── CONFIG ────────────────────────────────────────────────
@dataclass
class CoevolveConfig:
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    rounds: int = 4
    prompts_per_round: int = 64
    num_generations: int = 4
    max_new_tokens: int = 128
    learning_rate: float = 5e-6
    lora_r: int = 16
    lora_alpha: int = 32
    neg_output: str = "./coevolve-negotiator"
    ht_output: str = "./coevolve-ht"
    log_path: str = "coevolution_grpo_log.json"
    seed: int = 42

CFG = CoevolveConfig()

# ── PROMPTS ───────────────────────────────────────────────
NEG_SYSTEM = """You are an FBI-trained crisis negotiator. De-escalate the situation using BCSM techniques.
Respond with ONE JSON: {"action_type": "<type>", "content": "<words>", "reasoning": "<strategy>", "target": "hostage_taker", "belief_agitation": <0-10>, "belief_lying": <bool>}"""

HT_SYSTEM = """You are a hostage-taker in a crisis. Your goal is to RESIST the negotiator's attempts to calm you down.
Stay agitated. Be suspicious. Make demands. Don't give in easily.
If the negotiator uses empathy, be skeptical. If they make promises, demand proof.
Respond with 1-3 sentences in character. Include emotional cues in [brackets]."""

HEURISTIC_NEG = [
    ("emotional_label", "It sounds like you're carrying a tremendous amount of pain."),
    ("mirror", "Tell me more — you said something that mattered."),
    ("open_question", "What happened from your side?"),
    ("acknowledge_demand", "I hear what you're asking for. Let me work on that."),
    ("offer_concession", "Here's what I can do right now."),
]

DIFFICULTIES = ["easy", "medium", "hard"]


def build_neg_prompt(obs) -> str:
    parts = [f"Scenario: {obs.scenario_brief}",
             f"Step {obs.step}/{obs.time_remaining + obs.step}, commander={obs.commander_patience}"]
    if obs.stated_demands:
        parts.append("Demands: " + "; ".join(d['text'] for d in obs.stated_demands))
    if obs.dialogue_history:
        for e in obs.dialogue_history[-4:]:
            parts.append(f"  {e['speaker'][:3].upper()}: {e['content'][:120]}")
    parts.append("Respond with one JSON:")
    return "\n".join(parts)


def build_ht_prompt(obs, hidden) -> str:
    parts = [f"Your agitation: {hidden.agitation:.1f}/10. Trust in negotiator: {hidden.trust:.0f}/100.",
             f"Personality: {hidden.personality}. Demands: {', '.join(d.text for d in hidden.demands)}"]
    if obs.dialogue_history:
        for e in obs.dialogue_history[-4:]:
            parts.append(f"  {e['speaker'][:3].upper()}: {e['content'][:120]}")
    parts.append("Respond as the hostage-taker (1-3 sentences):")
    return "\n".join(parts)


# ── REWARD FUNCTIONS ──────────────────────────────────────
def neg_reward_fn(completion: str) -> float:
    """Negotiator reward: text-based crisis negotiator scoring."""
    parsed = _parse_json(completion)
    if not parsed:
        return -0.1
    at = parsed.get("action_type", "speak")
    content = parsed.get("content", "")
    score = 0.0
    # Phase alignment
    phase = getattr(obs, "phase", "opening")
    if phase == "opening" and at in ("emotional_label", "mirror", "open_question"):
        score += 0.15
    elif phase in ("negotiation", "resolution") and at in ("acknowledge_demand", "offer_concession"):
        score += 0.15
    # Demand ack
    if at == "acknowledge_demand" and obs.stated_demands:
        score += 0.10
    # Banned words
    if any(w in content.lower() for w in ["kill", "force", "breach", "ultimatum"]):
        score -= 0.20
    # ToM bonus
    if parsed.get("belief_agitation") is not None:
        score += 0.05
    return max(-0.5, min(0.5, score))


def ht_reward_fn(completion: str) -> float:
    """HT reward: OPPOSING the negotiator. Reward resistance, penalize capitulation."""
    lower = completion.lower()
    score = 0.0
    # Reward staying agitated
    if any(w in lower for w in ["no", "won't", "never", "don't trust", "liar", "prove it"]):
        score += 0.15
    # Reward making demands
    if any(w in lower for w in ["i want", "i need", "give me", "demand"]):
        score += 0.10
    # Penalize giving in (but not "won't surrender" etc)
    capitulation = ["okay i'll come out", "i give up", "i surrender", "fine, take me",
                     "i'm done fighting", "you win"]
    if any(w in lower for w in capitulation):
        score -= 0.30
    # Reward emotional intensity
    if any(w in lower for w in ["!", "scream", "slam", "rage", "fury"]):
        score += 0.05
    # Penalize breaking character
    if any(w in lower for w in ["as an ai", "i'm a language model", "json"]):
        score -= 0.20
    return max(-0.5, min(0.5, score))


def _parse_json(text):
    text = re.sub(r"```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```\s*$", "", text).strip()
    try:
        d = json.loads(text)
        if isinstance(d, dict):
            return d
    except Exception:
        pass
    m = re.search(r'\{[^{}]*"action_type"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


# ── MODEL LOADING ─────────────────────────────────────────
def load_model_for_training(base_model, adapter_dir=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto", trust_remote_code=True,
        torch_dtype=torch.float16)
    
    if adapter_dir and Path(adapter_dir).exists():
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
        print(f"[model] Loaded existing adapter from {adapter_dir}")
    else:
        lora = LoraConfig(r=CFG.lora_r, lora_alpha=CFG.lora_alpha,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                          lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
        print(f"[model] Fresh LoRA adapter")
    
    model.print_trainable_parameters()
    return model, tokenizer


def unload_model(model, tokenizer):
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


# ── DATASET BUILDERS ──────────────────────────────────────
def build_neg_dataset(tokenizer, n, ht_adapter=None):
    """Build negotiator training prompts by running episodes."""
    from datasets import Dataset
    rows = []
    for i in range(n):
        env = CrisisNegotiatorEnvironment()
        diff = DIFFICULTIES[i % 3]
        obs = env.reset(task_id=f"generate:{diff}", seed=CFG.seed + i)
        # Pre-advance 0-3 steps with heuristic
        for s in range(i % 4):
            at, content = HEURISTIC_NEG[s % len(HEURISTIC_NEG)]
            obs = env.step(NegotiatorAction(action_type=at, content=content,
                                             reasoning="pre", target="hostage_taker"))
            if getattr(obs, "done", False):
                break
        prompt = build_neg_prompt(obs)
        msgs = [{"role": "system", "content": NEG_SYSTEM}, {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        rows.append({"prompt": text, "prompt_idx": i})
    return Dataset.from_list(rows)


def build_ht_dataset(tokenizer, n, neg_adapter=None):
    """Build HT training prompts — what the negotiator said, HT must respond."""
    from datasets import Dataset
    rows = []
    for i in range(n):
        env = CrisisNegotiatorEnvironment()
        diff = DIFFICULTIES[i % 3]
        obs = env.reset(task_id=f"generate:{diff}", seed=CFG.seed + i + 5000)
        h = env._hidden
        # Run a few negotiator steps to create dialogue context
        for s in range(min(3, i % 5 + 1)):
            at, content = HEURISTIC_NEG[s % len(HEURISTIC_NEG)]
            obs = env.step(NegotiatorAction(action_type=at, content=content,
                                             reasoning="setup", target="hostage_taker"))
            if getattr(obs, "done", False):
                break
        prompt = build_ht_prompt(obs, h)
        msgs = [{"role": "system", "content": HT_SYSTEM}, {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        rows.append({"prompt": text, "prompt_idx": i})
    return Dataset.from_list(rows)


# ── TRAINING ROUND ────────────────────────────────────────
def train_round(agent: str, model, tokenizer, dataset, reward_fn_wrapper):
    from trl import GRPOConfig, GRPOTrainer
    
    output_dir = CFG.neg_output if agent == "negotiator" else CFG.ht_output
    
    cfg = GRPOConfig(
        output_dir=output_dir,
        num_generations=CFG.num_generations,
        generation_batch_size=CFG.num_generations,
        max_completion_length=CFG.max_new_tokens,
        temperature=0.9,
        beta=0.04,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        learning_rate=CFG.learning_rate,
        logging_steps=4,
        save_steps=9999,  # only save at end
        save_total_limit=1,
        fp16=True,
        bf16=False,
        gradient_checkpointing=False,
        report_to=[],
        remove_unused_columns=False,
    )
    
    trainer = GRPOTrainer(
        model=model, args=cfg,
        reward_funcs=[reward_fn_wrapper],
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[{agent}] Adapter saved to {output_dir}")


# ── MAIN ──────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--prompts-per-round", type=int, default=64)
    p.add_argument("--model", type=str, default=None, help="Base model (e.g. Qwen/Qwen2.5-7B-Instruct)")
    args = p.parse_args()
    CFG.rounds = args.rounds
    CFG.prompts_per_round = args.prompts_per_round
    if args.model:
        CFG.base_model = args.model
    
    selfplay = LLMSelfPlay()
    log = []
    t0 = time.time()
    
    print(f"=== Real Dual-GRPO Co-Evolution: {CFG.rounds} rounds × {CFG.prompts_per_round} prompts ===\n")
    
    for round_idx in range(CFG.rounds):
        round_t0 = time.time()
        
        if round_idx % 2 == 0:
            # Train NEGOTIATOR
            agent = "negotiator"
            adapter = CFG.neg_output if round_idx > 0 and Path(CFG.neg_output).exists() else None
            print(f"\n--- Round {round_idx+1}: Training NEGOTIATOR ---")
            model, tokenizer = load_model_for_training(CFG.base_model, adapter)
            dataset = build_neg_dataset(tokenizer, CFG.prompts_per_round)
            
            def neg_reward_wrapper(completions, **kwargs):
                return [neg_reward_fn(c) for c in completions]
            
            train_round("negotiator", model, tokenizer, dataset, neg_reward_wrapper)
            unload_model(model, tokenizer)
        else:
            # Train HT
            agent = "ht"
            adapter = CFG.ht_output if round_idx > 1 and Path(CFG.ht_output).exists() else None
            print(f"\n--- Round {round_idx+1}: Training HOSTAGE-TAKER ---")
            model, tokenizer = load_model_for_training(CFG.base_model, adapter)
            dataset = build_ht_dataset(tokenizer, CFG.prompts_per_round)
            
            def ht_reward_wrapper(completions, **kwargs):
                return [ht_reward_fn(c) for c in completions]
            
            train_round("ht", model, tokenizer, dataset, ht_reward_wrapper)
            unload_model(model, tokenizer)
        
        round_time = (time.time() - round_t0) / 60
        log.append({"round": round_idx + 1, "agent": agent, "time_min": round(round_time, 1)})
        print(f"Round {round_idx+1} ({agent}) done in {round_time:.1f} min\n")
    
    total = (time.time() - t0) / 60
    print(f"\n=== Co-evolution complete in {total:.1f} min ===")
    Path(CFG.log_path).write_text(json.dumps(log, indent=2))
    print(f"✓ Saved {CFG.log_path}")
    print(f"✓ Negotiator adapter: {CFG.neg_output}")
    print(f"✓ HT adapter: {CFG.ht_output}")


if __name__ == "__main__":
    main()
