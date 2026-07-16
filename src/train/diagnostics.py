"""Télémétrie mémoire GPU pendant l'entraînement."""

from __future__ import annotations

from typing import Any

from transformers import TrainerCallback


class MemDiagCallback(TrainerCallback):
    """Loggue un breakdown mémoire GPU détaillé toutes les 50 steps.

    Distingue :
      - PyTorch alloué/réservé  → torch.cuda.memory_stats()
      - États bitsandbytes      → itère optimizer.state (stockés hors pool PyTorch)
      - nvidia-smi (process)    → mémoire totale vue par le driver CUDA
    Le « phantom » (driver − torch − bnb) couvre le contexte CUDA, NCCL et vLLM.
    """

    def on_step_end(self, targs: Any, state: Any, control: Any,
                    model: Any = None, optimizer: Any = None, **kwargs: Any) -> None:
        if state.global_step % 50 != 0:
            return
        import os
        import subprocess

        import torch
        torch_alloc = torch.cuda.memory_allocated() / 1024**3
        torch_reserv = torch.cuda.memory_reserved() / 1024**3

        # États bitsandbytes (int8) — hors pool PyTorch
        bnb_bytes = 0
        if optimizer is not None:
            for pg in optimizer.param_groups:
                for p in pg["params"]:
                    s = optimizer.state.get(p, {})
                    for v in s.values():
                        if hasattr(v, "nbytes"):
                            bnb_bytes += v.nbytes
                        elif hasattr(v, "element_size") and hasattr(v, "numel"):
                            bnb_bytes += v.element_size() * v.numel()
        bnb_gib = bnb_bytes / 1024**3

        # Mémoire totale du process vue par le driver (nvidia-smi)
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"], text=True)
            pid = os.getpid()
            driver_gib = next(
                (int(row.split(",")[1]) / 1024
                 for row in out.strip().splitlines()
                 if row.split(",")[0].strip() == str(pid)),
                None)
        except Exception:
            driver_gib = None

        phantom = (driver_gib - torch_alloc - bnb_gib) if driver_gib else None
        print(
            f"[mem step={state.global_step}] "
            f"torch_alloc={torch_alloc:.1f} GiB  "
            f"torch_reserved={torch_reserv:.1f} GiB  "
            f"bnb_states={bnb_gib:.1f} GiB  "
            f"driver_total={driver_gib:.1f} GiB  "
            f"phantom(driver-torch-bnb)={phantom:.1f} GiB"
            if phantom is not None else
            f"[mem step={state.global_step}] "
            f"torch_alloc={torch_alloc:.1f} GiB  "
            f"bnb_states={bnb_gib:.1f} GiB",
            flush=True,
        )
