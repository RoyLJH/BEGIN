import argparse
import os

import torch
import torch.distributed.rpc as rpc
#from torch.distributed.rpc import RRef, rpc_async, remote

INVERTER_NAME = "inverter"
WORKER_NAME = "worker{}"

def run_worker(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '7777'
    # rank0 is always organizer
    if rank == 0:
        rpc.init_rpc(INVERTER_NAME, rank=rank, world_size=world_size)
    else:
        rpc.init_rpc(WORKER_NAME.format(rank), rank=rank, world_size=world_size)
    rpc.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Batchnorm Ensembled Generative INverter")
    parser.add_argument('--models', type=str, nargs='+', default=['resnet18', 'resnet50'],
        help='Select which models to invert')
    parser.add_argument('--categories', type=int, nargs='+', default=[985, 417, 402, 94, 63, 250, 530, 920, 386, 949],
        help='(ImageNet) categories to compose the image batch')
    parser.add_argument('--samples_per_category', type=int, default=4,
        help='How many samples to generate per category. The whole batchsize will be `samples_per_category` x `categories`')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1],
        help='Cuda device for corresponding model; -1 for cpu device')


    
        

if __name__ == "__main__":
    #main()
    world_size = 2
    torch.multiprocessing.spawn(
        run_worker,
        args=(world_size, )
        nprocs=world_size,
        join=True
    )