import os
import argparse
import torch

import torch.distributed.rpc as rpc

class Server(object):
    def __init__(self, args):
        # initialize all the workers
        self.workers = []
        for worker_rank in range(args.modelnames):
            self.workers.append(Worker(worker_rank, args))
        
        pass

    def optimize(self):
        # for each iteration, send current input to each worker and wait all workers update their gradients
        print("Optimize finished")



    
# worker prepares model; compute gradients in every iteration
class Worker(object):
    def __init__(self, rank, args):
        self.rank = rank
        self.worker_name = rpc.get_worker_info().name
        self.modelname = args.modelnames[rank]
        self.device = f"cuda:{args.devices[rank]}" if args.devices[rank] >= 0 else "cpu"
        info = f"{self.worker_name} got {self.modelname} on device {self.device}"
        print(info)
        

    def prepare_model(self):
        # select and initialize the model; Batchnorm hook the model
        pass
        
    def compute_grad(self, input_rref):
        pass



def run(rank, worldsize, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "7777"
    #backend_options = rpc.TensorPipeRpcBackendOptions(
    #    num_worker_threads = 16,
    #    rpc_timeout = 0 # infinite timeout
    #)
    if rank != 0:
        rpc.init_rpc(f"worker{rank}", rank=rank, world_size=worldsize, ) #rpc_backend_options=backend_options)
        # all the workers wait passively for the server to kick off
    else:
        rpc.init_rpc("server", rank=rank, world_size=worldsize, ) #rpc_backend_options=backend_options)
        server = Server(args)
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
