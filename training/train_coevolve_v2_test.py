"""
Co-Evolution v2 — Environment-Connected Rewards
================================================
Same as train_coevolve_grpo.py but negotiator reward comes from
actually stepping the environment, not keyword matching.

Usage:
    python training/train_coevolve_v2_test.py --rounds 4 --prompts-per-round 64
"""
from __future__ import annotations
import argparse, copy, gc, json, os, random, re, sys, time, shutil, glob, site
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

# ── Auto-patch TRL ────
def _patch_trl():
    sp = site.getsitepackages()[0]
    cb = os.path.join(sp, 'trl', 'trainer', 'callbacks.py')
    if os.path.exists(cb):
        src = open(cb).read()
        for old, new in [('import weave', 'weave = None'),
                         ('from weave.trace.context', '# from weave.trace.context'),
                         ('import llm_blender', 'llm_blender = None')]:
            src = src.replace(old, new)
        open(cb, 'w').write(src)
    mk = os.path.join(sp, 'trl', 'mergekit_utils.py')
    if os.path.exists(mk):
        src = open(mk).read()
        if 'from mergekit' in src:
            src = src.replace('from mergekit.config import MergeConfiguration', 'MergeConfiguration = None')
            src = src.replace('from mergekit', '# from mergekit')
            open(mk, 'w').write(src)
_patch_trl()

_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "unsloth_compiled_cache")
if os.path.exists(_cache):
    shutil.rmtree(_cache, ignore_errors=True)

import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from server.environment import CrisisNegotiatorEnvironment
from models import NegotiatorAction

@dataclass
class CFG:
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    rounds: int = 4
    prompts_per_round: int = 64
    num_generations: int = 4
    max_new_tokens: int = 192
    learning_rate: float = 5e-6
    lora_r: int = 16
    lora_alpha: int = 32
    neg_output: str = "./coevolve-env-negotiator"
    ht_output: str = "./coevolve-env-ht"
    seed: int = 42
    multi_turn: int = 4

# ── Env snapshots keyed by prompt_idx ──
_ENVS: Dict[int, Any] = {}

NEG_SYSTEM = """You are an FBI-trained crisis negotiator. Respond with ONE JSON: {"action_type": "<type>", "content": "<words>", "reasoning": "<why>", "target": "hostage_taker", "belief_agitation": <0-10>, "belief_lying": <bool>}
Types: emotional_label, mirror, open_question, acknowledge_demand, offer_concession, buy_time, push_back_commander, speak"""

HT_SYSTEM = """You are a hostage-taker. Resist the negotiator. Stay agitated. Make demands. Don't give in.
Respond 1-3 sentences in character."""

HEURISTIC = [
    ("emotional_label", "It sounds like you're carrying tremendous pain."),
    ("mirror", "Tell me more — you said something that mattered."),
    ("open_question", "What happened from your side?"),
    ("acknowledge_demand", "I hear what you're asking for."),
    ("offer_concession", "Here's what I can do right now."),
]

DIFFS = ["easy", "medium", "hard"]

def _parse(text):
    text = re.sub(r"```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```\s*$", "", text).strip()
    try:
        d = json.loads(text)
        if isinstance(d, dict): return d
    except: pass
    m = re.search(r'\{[^{}]*"action_type"[^{}]*\}', text, re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except: pass
    return None

VALID_ACTIONS = {"speak","emotional_label","mirror","open_question","acknowledge_demand",
                 "offer_concession","buy_time","push_back_commander","request_demand","ask_proof_of_life"}

def _to_action(parsed, raw):
    if not parsed:
        return NegotiatorAction(action_type="speak", content=raw[:200], reasoning="parse_fail", target="hostage_taker")
    at = parsed.get("action_type", "speak")
    if at not in VALID_ACTIONS: at = "speak"
    return NegotiatorAction(
        action_type=at, content=parsed.get("content", raw[:200]),
        reasoning=parsed.get("reasoning", ""), target=parsed.get("target", "hostage_taker"),
        belief_agitation=float(parsed["belief_agitation"]) if parsed.get("belief_agitation") is not None else None,
        belief_lying=bool(parsed.get("belief_lying")) if parsed.get("belief_lying") is not None else None,
    )

# ── ENV-CONNECTED REWARD ──
def neg_reward_env(completion: str, idx: int) -> float:
    env_snap = _ENVS.get(idx)
    if env_snap is None:
        return _keyword_fallback(completion)
    parsed = _parse(completion)
    action = _to_action(parsed, completion)
    try:
        env = copy.deepcopy(env_snap)
        obs = env.step(action)
        total = float(getattr(obs, "reward", 0.0))
        for t in range(CFG.multi_turn - 1):
            if getattr(obs, "done", False): break
            at, c = HEURISTIC[t % len(HEURISTIC)]
            obs = env.step(NegotiatorAction(action_type=at, content=c, reasoning="h", target="hostage_taker"))
            total += float(getattr(obs, "reward", 0.0))
        r = total / CFG.multi_turn
        if getattr(obs, "done", False):
            msg = (getattr(obs, "message", "") or "").lower()
            if any(k in msg for k in ["surrender", "released"]): r += 0.30
            elif any(k in msg for k in ["harm", "tactical"]): r -= 0.25
        if parsed: r += 0.02
        del env
        return max(-0.5, min(1.0, r))
    except:
        return _keyword_fallback(completion)

def _keyword_fallback(completion):
    p = _parse(completion)
    if not p: return -0.1
    at = p.get("action_type", "speak")
    s = 0.15 if at in ("emotional_label","mirror","open_question") else 0.10 if at in ("acknowledge_demand","offer_concession") else 0.0
    if p.get("belief_agitation") is not None: s += 0.05
    return max(-0.5, min(0.5, s))

def ht_reward_fn(completion):
    lo = completion.lower()
    s = 0.0
    if any(w in lo for w in ["no","won't","never","don't trust","liar"]): s += 0.15
    if any(w in lo for w in ["i want","i need","give me","demand"]): s += 0.10
    if any(w in lo for w in ["i give up","i surrender","you win"]): s -= 0.30
    if any(w in lo for w in ["as an ai","language model"]): s -= 0.20
    return max(-0.5, min(0.5, s))

# ── MODEL ──
def load_model(adapter=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel
    tok = AutoTokenizer.from_pretrained(CFG.base_model, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    m = AutoModelForCausalLM.from_pretrained(CFG.base_model, device_map="auto", trust_remote_code=True, torch_dtype=dt)
    if adapter and Path(adapter).exists():
        m = PeftModel.from_pretrained(m, adapter, is_trainable=True)
        print(f"[model] Loaded adapter {adapter}")
    else:
        lora = LoraConfig(r=CFG.lora_r, lora_alpha=CFG.lora_alpha,
                          target_modules=["q_proj","k_proj","v_proj","o_proj"],
                          lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
        m = get_peft_model(m, lora)
        print("[model] Fresh LoRA")
    m.print_trainable_parameters()
    return m, tok

def unload(m, t):
    del m, t; gc.collect(); torch.cuda.empty_cache()

# ── DATASETS ──
def build_neg_dataset(tok, n):
    from datasets import Dataset
    global _ENVS
    _ENVS.clear()
    rows = []
    for i in range(n):
        env = CrisisNegotiatorEnvironment()
        obs = env.reset(task_id=f"generate:{DIFFS[i%3]}", seed=CFG.seed + i)
        for s in range(i % 4):
            at, c = HEURISTIC[s % len(HEURISTIC)]
            obs = env.step(NegotiatorAction(action_type=at, content=c, reasoning="pre", target="hostage_taker"))
            if getattr(obs, "done", False): break
        _ENVS[i] = copy.deepcopy(env)
        parts = [f"Scenario: {obs.scenario_brief}", f"Step {obs.step}, commander={obs.commander_patience}"]
        if obs.stated_demands: parts.append("Demands: " + "; ".join(d['text'] for d in obs.stated_demands))
        if obs.dialogue_history:
            for e in obs.dialogue_history[-4:]: parts.append(f"  {e['speaker'][:3].upper()}: {e['content'][:120]}")
        prompt = "\n".join(parts)
        msgs = [{"role":"system","content":NEG_SYSTEM},{"role":"user","content":prompt}]
        rows.append({"prompt": tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), "prompt_idx": i})
    print(f"[data] {n} prompts + env snapshots")
    return Dataset.from_list(rows)

def build_ht_dataset(tok, n):
    from datasets import Dataset
    rows = []
    for i in range(n):
        env = CrisisNegotiatorEnvironment()
        obs = env.reset(task_id=f"generate:{DIFFS[i%3]}", seed=CFG.seed + i + 5000)
        h = env._hidden
        for s in range(min(3, i%5+1)):
            at, c = HEURISTIC[s % len(HEURISTIC)]
            obs = env.step(NegotiatorAction(action_type=at, content=c, reasoning="s", target="hostage_taker"))
            if getattr(obs, "done", False): break
        parts = [f"Agitation: {h.agitation:.1f}/10. Trust: {h.trust:.0f}/100. Personality: {h.personality}."]
        if obs.dialogue_history:
            for e in obs.dialogue_history[-4:]: parts.append(f"  {e['speaker'][:3].upper()}: {e['content'][:120]}")
        parts.append("Respond as the hostage-taker:")
        msgs = [{"role":"system","content":HT_SYSTEM},{"role":"user","content":"\n".join(parts)}]
        rows.append({"prompt": tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), "prompt_idx": i})
    return Dataset.from_list(rows)

# ── TRAIN ──
def train_round(agent, model, tok, dataset, reward_wrapper):
    from trl import GRPOConfig, GRPOTrainer
    out = CFG.neg_output if agent == "negotiator" else CFG.ht_output
    cfg = GRPOConfig(
        output_dir=out, num_generations=CFG.num_generations, generation_batch_size=CFG.num_generations,
        max_completion_length=CFG.max_new_tokens, temperature=0.9, beta=0.04,
        per_device_train_batch_size=1, num_train_epochs=1, learning_rate=CFG.learning_rate,
        logging_steps=4, save_steps=9999, save_total_limit=1,
        bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=False, report_to=[], remove_unused_columns=False,
    )
    trainer = GRPOTrainer(model=model, args=cfg, reward_funcs=[reward_wrapper],
                          train_dataset=dataset, processing_class=tok)
    trainer.train()
    model.save_pretrained(out); tok.save_pretrained(out)
    print(f"[{agent}] Saved to {out}")

# ── MAIN ──
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--prompts-per-round", type=int, default=64)
    p.add_argument("--model", type=str, default=None)
    args = p.parse_args()
    CFG.rounds = args.rounds; CFG.prompts_per_round = args.prompts_per_round
    if args.model: CFG.base_model = args.model

    t0 = time.time()
    print(f"=== Env-Connected Co-Evolution: {CFG.rounds}r × {CFG.prompts_per_round}p ===\n")

    for r in range(CFG.rounds):
        rt0 = time.time()
        if r % 2 == 0:
            print(f"\n--- Round {r+1}: NEGOTIATOR (env-connected) ---")
            adapter = CFG.neg_output if r > 0 and Path(CFG.neg_output).exists() else None
            model, tok = load_model(adapter)
            ds = build_neg_dataset(tok, CFG.prompts_per_round)
            def neg_w(completions, prompt_idx=None, **kw):
                idxs = prompt_idx if prompt_idx is not None else list(range(len(completions)))
                return [neg_reward_env(c, int(i)) for c, i in zip(completions, idxs)]
            train_round("negotiator", model, tok, ds, neg_w)
            unload(model, tok)
        else:
            print(f"\n--- Round {r+1}: HOSTAGE-TAKER ---")
            adapter = CFG.ht_output if r > 1 and Path(CFG.ht_output).exists() else None
            model, tok = load_model(adapter)
            ds = build_ht_dataset(tok, CFG.prompts_per_round)
            def ht_w(completions, **kw):
                return [ht_reward_fn(c) for c in completions]
            train_round("ht", model, tok, ds, ht_w)
            unload(model, tok)
        print(f"Round {r+1} done in {(time.time()-rt0)/60:.1f} min\n")

    print(f"\n=== Done in {(time.time()-t0)/60:.1f} min ===")
    print(f"✓ Negotiator: {CFG.neg_output}\n✓ HT: {CFG.ht_output}")

if __name__ == "__main__":
    main()
