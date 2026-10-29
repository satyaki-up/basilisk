import os
import sys

import torch
import torch.distributed as dist


def dist_env_vars() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def init_process_group() -> None:
    # One thread per process, otherwise each process grabs every core.
    torch.set_num_threads(1)
    # This machine's hostname does not resolve. Ranks talk over loopback.
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    dist.init_process_group(backend="gloo")


def shutdown() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def rank() -> int:
    return dist.get_rank()


def world_size() -> int:
    return dist.get_world_size()
