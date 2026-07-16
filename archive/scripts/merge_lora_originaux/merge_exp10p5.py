import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "/home/criteo/rl-gym-workout/models/qwen25_3b_exp10p3_step414_35pct"
ADAPTER = "/home/criteo/rl-gym-workout/saves/trl_grpo/exp10.5_resume35_lr_div1.5_best"
OUT = "/home/criteo/rl-gym-workout/models/qwen25_3b_exp10p5_step368_51pct"

print("Loading base model...", flush=True)
m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
print("Loading adapter...", flush=True)
m = PeftModel.from_pretrained(m, ADAPTER)
print("Merging...", flush=True)
m = m.merge_and_unload()
print("Saving merged model...", flush=True)
m.save_pretrained(OUT, safe_serialization=True)
AutoTokenizer.from_pretrained(BASE).save_pretrained(OUT)
print("DONE MERGE OK", flush=True)
