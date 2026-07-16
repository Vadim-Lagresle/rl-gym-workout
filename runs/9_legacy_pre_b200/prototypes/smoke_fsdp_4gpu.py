"""Standalone smoke test: load Qwen-3B on each of 4 GPUs, FSDP-wrap with CPUOffload (same flow as verl ref policy init). NCCL debug enabled."""
import os
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_DEBUG_SUBSYS'] = 'INIT,ENV,COLL'

import torch
import torch.distributed as dist
import datetime
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, CPUOffload, ShardingStrategy
from transformers import AutoModelForCausalLM


def worker(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29512'
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        'nccl', rank=rank, world_size=world_size,
        timeout=datetime.timedelta(seconds=180))
    print(f"[rank{rank}] NCCL init OK, world_size={dist.get_world_size()}", flush=True)

    mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])
    if rank == 0:
        print(f"[rank{rank}] device_mesh OK", flush=True)

    print(f"[rank{rank}] loading Qwen-3B on CPU...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        '/home/v.lagresle/rl-gym-workout/models/Qwen2.5-3B-Instruct',
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
    )
    model.to(torch.bfloat16)
    print(f"[rank{rank}] HF load done; FSDP-wrapping with CPUOffload", flush=True)

    dist.barrier()
    fsdp = FSDP(
        model,
        cpu_offload=CPUOffload(offload_params=True),
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32),
        sync_module_states=True,
        device_mesh=mesh,
        forward_prefetch=False,
    )
    print(f"[rank{rank}] FSDP wrap OK", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    mp.spawn(worker, args=(4,), nprocs=4, join=True)
