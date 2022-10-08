import os
import argparse
import torch

import torch.distributed.rpc as rpc

from server import Server
from worker import Worker

def run(rank, worldsize, args):
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    if rank != 0:
        rpc.init_rpc(f"worker{rank}", rank=rank, world_size=worldsize)
        # all the workers wait passively for the server to kick off
    else:
        rpc.init_rpc("server", rank=rank, world_size=worldsize)
        server = Server(args)
        workers = []
        for worker_rank in range(1, worldsize):
            workers.append(rpc.rpc_async(f"worker{worker_rank}", Worker, args=(worker_rank, args)))
        for fut in workers:
            worker = fut.wait()
            worker.prepare_device()
        server.optimize()

    rpc.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batchnorm Ensembled Generative INversion')
    # Distributed
    parser.add_argument('--master_addr', default='localhost', type=str,
        help="os.environ['MASTER_ADDR']")
    parser.add_argument('--master_port', default=7777, type=int,
        help="os.environ['MASTER_PORT']")
    parser.add_argument('--modelnames', default=['resnet18', 'resnet50'], type=str, nargs='+',
        help="Model architectures to do BatchNorm inversion")
    parser.add_argument('--devices', default=[0, 1], type=int, nargs='+',
        help="Devices for each model; -1 for cpu, x for cuda:x")

    # Batch
    parser.add_argument('--categories', default=[985, 417, 402, 94, 63, 250, 530, 920, 386, 949], type=int, nargs='+',
        help='Imagenet categories of images in the batch, values taken in [0, 999]')
    parser.add_argument('--samples-per-category', default=4, type=int,
        help='How many samples to generate for each catogory. Batchsize = `samples-per-category` * `categories`')
    
    # Optimization
    parser.add_argument('--ce_scale', default=0.8, type=float,
        help='Cross entropy loss scale')
    parser.add_argument('--bn_scale', default=0.05, type=float,
        help='BN statistics matching loss scale')
    parser.add_argument('--tv_scale', default=3e-3, type=float,
        help='Total variation image prior loss scale')
    parser.add_argument('--lr', default=0.8, type=float,
        help='Learning rate of the optimization process of input batch')
    parser.add_argument('--cos_lr', default=True, type=bool,
        help='Use cosine learning rate step strategy')
    parser.add_argument('--adam_betas', default=[0.3, 0.9], type=float, nargs='+',
        help='Beta parameter of the Adam optimizer')
    parser.add_argument('--warmup_iters', default=500, type=int,
        help='Warm-up iterations')
    parser.add_argument('--iters', default=2000, type=int,
        help='Iterations for optimization')
        
    # Post-processing
    parser.add_argument('--roll', type=int, default=0, 
        help='Roll the image in height and weight after each step (for image stabilization)')
    parser.add_argument('--flip', action='store_true', default=False,
        help='Randomly flip the image after each optimization step')
    parser.add_argument('--clip', type=int, default=0,
        help='Clip the value of image tensor after each optimization step')


    args = parser.parse_args()
    assert len(args.modelnames) == len(args.devices)
    world_size = len(args.modelnames) + 1
    torch.multiprocessing.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
