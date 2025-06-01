import os
import torch
import torch.distributed as dist

def init_ddp(ddp):
    torch.distributed.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, torch.distributed.get_world_size()

def cleanup_ddp():
    dist.destroy_process_group()
    
