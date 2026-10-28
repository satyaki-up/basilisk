# Dry run of of collective operations on one CPU via multiple processes.

# Example usage: uv run python scripts/dry_run_dist_cpu.py --nproc 4
import argparse
import os
import socket
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run(rank: int, world_size: int) -> None:
    # One thread per process, otherwise each process grabs every core.
    torch.set_num_threads(1)
    # https://sangama.hashnode.dev/chapter-2-what-is-eth0-understanding-network-interfaces-and-loopback
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        local = torch.tensor([float(rank)])
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
        expected = world_size * (world_size - 1) / 2
        got = local.item()
        if got != expected:
            raise RuntimeError(f"rank {rank} allreduce sum {got}, expected {expected}")
        print(f"rank {rank}/{world_size} allreduce sum {got:.0f}", flush=True)
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="All-reduce across Gloo processes on one CPU.")
    parser.add_argument("--nproc", type=int, default=4)
    args = parser.parse_args()
    if args.nproc < 2:
        raise ValueError(f"--nproc must be at least 2, got {args.nproc}")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(free_port())
    mp.spawn(run, args=(args.nproc,), nprocs=args.nproc, join=True)


if __name__ == "__main__":
    main()
