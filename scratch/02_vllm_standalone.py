"""
Test minimal : peut-on charger Qwen2.5-3B-Instruct dans vLLM "standard" (pas le fork
verl) et générer une simple completion ?

But : isoler si le crash de l'eval verl vient de (A) vLLM lui-même / driver CUDA, ou
(B) du fork `verl.third_party.vllm` / de l'intégration Ray.

Lancement :
    conda activate agentgym-rl
    python scratch/02_vllm_standalone.py
"""

import os

os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_USE_MODELSCOPE", "0")

from vllm import LLM, SamplingParams


MODEL_PATH = "/home/v.lagresle/rl-gym-workout/models/Qwen2.5-3B-Instruct"


def main() -> None:
    print("Loading vLLM (standard, not the verl fork)...")
    llm = LLM(
        model=MODEL_PATH,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        max_model_len=16384,
        dtype="bfloat16",
        load_format="safetensors",
        tensor_parallel_size=1,
    )
    print("Model loaded. Running a short generation...")

    sampling = SamplingParams(temperature=1.0, max_tokens=64)
    prompts = [
        "You are a helpful assistant. Respond briefly.\n\nUser: What is 2+2?\nAssistant:",
    ]
    out = llm.generate(prompts, sampling)
    print("=" * 32, "GENERATION", "=" * 32)
    for o in out:
        print("PROMPT:", o.prompt)
        print("OUTPUT:", o.outputs[0].text)
    print("=" * 76)


if __name__ == "__main__":
    main()
