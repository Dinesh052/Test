"""Quick sanity check that base 3B model produces sane JSON-shaped output."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

m = "Qwen/Qwen2.5-3B-Instruct"
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
tok = AutoTokenizer.from_pretrained(m)
model = AutoModelForCausalLM.from_pretrained(
    m, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16
)

msgs = [
    {"role": "system", "content": "You are a crisis negotiator. Reply with one JSON object only."},
    {"role": "user", "content": 'Output exactly: {"action_type":"emotional_label","content":"It sounds hard.","reasoning":"build trust","target":"hostage_taker"}'},
]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print("===PROMPT TAIL===")
print(text[-400:])
ids = tok(text, return_tensors="pt").to(model.device)
out = model.generate(
    **ids, max_new_tokens=80, do_sample=False,
    pad_token_id=tok.pad_token_id or tok.eos_token_id,
)
print("===GREEDY OUTPUT===")
print(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True))
print("===SAMPLED OUTPUT (T=0.7)===")
out2 = model.generate(
    **ids, max_new_tokens=80, do_sample=True, temperature=0.7, top_p=0.9,
    pad_token_id=tok.pad_token_id or tok.eos_token_id,
)
print(tok.decode(out2[0][ids.input_ids.shape[1]:], skip_special_tokens=True))
