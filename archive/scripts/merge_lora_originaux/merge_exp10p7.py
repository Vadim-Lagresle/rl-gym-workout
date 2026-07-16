import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Base = modèle complet fusionné de exp10.5 (51%), qui sert déjà de base à exp10.7.
BASE = "/home/criteo/rl-gym-workout/models/qwen25_3b_exp10p5_step368_51pct"
# Adapter = best LoRA de exp10.7 (step 92, 53/100), sauvé adapter-only.
ADAPTER = "/home/criteo/rl-gym-workout/saves/trl_grpo/exp10.7_warmstart51_best"
# Sortie = base warm-start pour exp10.8.
OUT = "/home/criteo/rl-gym-workout/models/qwen25_3b_exp10p7_step92_53pct"

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
