import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Base = modèle fusionné exp10.7 (53%), base du run exp10.8.
BASE = "/home/criteo/rl-gym-workout/models/qwen25_3b_exp10p7_step92_53pct"
# Adapter = best LoRA de exp10.8 (step 368, 58/100).
ADAPTER = "/home/criteo/rl-gym-workout/saves/trl_grpo/exp10.8_warmstart53_lr_div3_best"
# Sortie sur le scratchpad (disque local plein) — à déplacer si on veut le garder.
OUT = ("/tmp/claude-10001/-home-criteo-rl-gym-workout/"
       "4a0c68bc-ec8b-4b50-9e9d-a9650c78108d/scratchpad/models/qwen25_3b_exp10p8_step368_58pct")

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
