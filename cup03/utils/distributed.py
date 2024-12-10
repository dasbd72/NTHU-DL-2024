import signal
from datetime import timedelta

import torch.distributed as dist


def _handle_sigterm(signum, frame):
    print("Received SIGTERM, cleaning up distributed process group.")
    cleanup_distributed()
    exit(0)


def init_distributed(rank, world_size):
    """Setup the distributed process group."""
    timeout = timedelta(minutes=10)
    dist.init_process_group(
        "nccl", timeout=timeout, rank=rank, world_size=world_size
    )
    signal.signal(signal.SIGTERM, _handle_sigterm)
    print("Setupped process group for rank {}".format(rank))


def cleanup_distributed():
    """Clean up the distributed process group."""
    dist.destroy_process_group()
    print("Cleaned up process group")
