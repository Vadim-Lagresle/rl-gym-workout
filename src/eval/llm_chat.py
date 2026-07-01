"""Génération chat pour scripts eval exp16 (vLLM ou HuggingFace)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
VLLM_SERVER_URL = "http://127.0.0.1:8001"


def resolve_model_ref(model: str) -> str:
    p = Path(model)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return str(p) if p.exists() else model


def vllm_supports_model(model_ref: str) -> bool:
    sys.path.insert(0, str(REPO_ROOT / "src" / "utils"))
    from vllm_supports import vllm_supports  # noqa: WPS433

    return vllm_supports(model_ref)


def pick_backend(model_ref: str, backend: str) -> str:
    if backend in ("vllm", "hf"):
        return backend
    return "vllm" if vllm_supports_model(model_ref) else "hf"


def check_vllm_server(vllm_url: str = VLLM_SERVER_URL) -> None:
    try:
        requests.get(f"{vllm_url}/health", timeout=5).raise_for_status()
    except Exception as e:
        raise SystemExit(
            f"Serveur vLLM non disponible sur {vllm_url}.\n"
            f"Lance : bash src/utils/start_vllm_server.sh <model>\nErreur : {e}"
        )


def generate_reply_vllm(
    model_name: str,
    messages: list[dict],
    vllm_url: str = VLLM_SERVER_URL,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "stream": False,
    }
    r = requests.post(f"{vllm_url}/v1/chat/completions", json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def generate_reply_hf(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    enable_thinking: bool = False,
) -> str:
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if temperature <= 0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 1.0
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


@dataclass
class ChatGenerator:
    model_ref: str
    backend: str = "auto"
    vllm_url: str = VLLM_SERVER_URL
    no_thinking: bool = False
    _hf_model: Any = None
    _hf_tokenizer: Any = None
    _backend_resolved: str = ""

    def __post_init__(self) -> None:
        self.model_ref = resolve_model_ref(self.model_ref)
        self._backend_resolved = pick_backend(self.model_ref, self.backend)
        if self._backend_resolved == "vllm":
            check_vllm_server(self.vllm_url)
            print(f"[llm] backend=vllm url={self.vllm_url} model={self.model_ref}")
        else:
            print(f"[llm] backend=hf loading {self.model_ref} ...", flush=True)
            self._hf_tokenizer = AutoTokenizer.from_pretrained(
                self.model_ref, trust_remote_code=True
            )
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_ref,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            self._hf_model.eval()
            print("[llm] backend=hf model loaded.", flush=True)

    @property
    def backend_name(self) -> str:
        return self._backend_resolved

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        if self._backend_resolved == "vllm":
            return generate_reply_vllm(
                self.model_ref,
                messages,
                vllm_url=self.vllm_url,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        return generate_reply_hf(
            self._hf_model,
            self._hf_tokenizer,
            messages,
            max_new_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=not self.no_thinking,
        )
