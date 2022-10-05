import os
import argparse
import torch

import torch.distributed.rpc as rpc

from server import Server
from worker import Worker

def run(rank, worldsize, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "7777"
    backend_options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads = 16,
        rpc_timeout = 0 # infinite timeout
    )
    if rank != 0:
        rpc.init_rpc(f"worker{rank}", rank=rank, world_size=worldsize, rpc_backend_options=backend_options)
        # all the workers wait passively for the server to kick off
    else:
        rpc.init_rpc("server", rank=rank, world_size=worldsize, rpc_backend_options=backend_options)
        server = Server(args)
        workers = []
        for worker_rank in range(1, worldsize):
            workers.append(rpc.rpc_async(f"worker{worker_rank}", Worker, args=(worker_rank, args)))
        torch.futures.wait_all(workers)
        server.optimize()

    rpc.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batchnorm Ensembled Generative INversion')
    parser.add_argument('--modelnames', default=['resnet18', 'resnet50'], type=str, nargs='+',
        help="Model architectures to do BatchNorm inversion")
    parser.add_argument('--devices', default=[-1, -1], type=int, nargs='+',
        help="Devices for each model; -1 for cpu, x for cuda:x")



    args = parser.parse_args()
    assert len(args.modelnames) == len(args.devices)
    world_size = len(args.modelnames) + 1
    torch.multiprocessing.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
