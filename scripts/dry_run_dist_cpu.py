# Dry run of of collective operations on one CPU via multiple processes.

# Example usage:
# uv run torchrun --standalone --local-addr 127.0.0.1 --nproc_per_node="cpu" scripts/dry_run_dist_cpu.py
import os
import sys

import torch
import torch.distributed as dist


def main() -> None:
    # One thread per process, otherwise each process grabs every core.
    torch.set_num_threads(1)
    # https://sangama.hashnode.dev/chapter-2-what-is-eth0-understanding-network-interfaces-and-loopback
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local = torch.tensor([float(rank)])
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
        expected = world_size * (world_size - 1) / 2
        got = local.item()
        if got != expected:
            raise RuntimeError(f"rank {rank} allreduce sum {got}, expected {expected}")
        print(f"rank {rank}/{world_size} allreduce sum {got:.0f}\n", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
