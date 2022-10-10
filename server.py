import os
import time
import torch
import torch.distributed.rpc as rpc
import torchvision.utils
import numpy as np

from worker import Worker

class Server(object):
    def __init__(self, args):
        self.worker_rrefs = []
        worldsize = len(args.modelnames) + 1
        for worker_rank in range(1, worldsize):
            self.worker_rrefs.append(rpc.remote(f"worker{worker_rank}", Worker, args=(worker_rank, args, rpc.RRef(self))))
    
        self.args = args
        self.batchsize = len(args.categories) * args.samples_per_category
        self.total_iters = args.iters
        self.warmup_iters = args.warmup_iters
        self.base_lr = args.lr
        self.adam_betas = args.adam_betas
        self.tv_scale = args.tv_scale
        self.l2_scale = args.l2_scale

        timestamp = time.strftime("%m-%d %H-%M-%S", time.localtime())
        self.result_folder_path = f"results/{timestamp}/"
        if not os.path.exists(self.result_folder_path):
            os.makedirs(self.result_folder_path)
        self.logger = ""
        self.log("Creating server...")
            
    def adjust_lr(self, current_iter):
        if current_iter < self.warmup_iters:
            lr = self.base_lr * (current_iter + 1) / self.warmup_iters
        else:
            lr = 0.5 * (1 + np.cos(np.pi * (current_iter - self.warmup_iters)/(self.total_iters - self.warmup_iters))) * self.base_lr
        self.optimizer.param_groups[0]['lr'] = lr

    def get_tv_loss(self, x):
        diff1 = x[:, :, :, :-1] - x[:, :, :, 1:]
        diff2 = x[:, :, :-1, :] - x[:, :, 1:, :]
        diff3 = x[:, :, 1:, :-1] - x[:, :, :-1, 1:]
        diff4 = x[:, :, :-1, :-1] - x[:, :, 1:, 1:]
        tv_loss = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
        return tv_loss

    def get_l2_loss(self, x):
        return torch.norm(x, 2)

    def optimize(self):
        for worker_rref in self.worker_rrefs:
            worker_rref.rpc_async().prepare_device()
        for trial in range(self.args.trials):
            self.log(f"Starting Optimization for Trial {trial}...")
            self.input = torch.randn(self.batchsize, 3, 224, 224).requires_grad_(True)
            self.optimizer = torch.optim.Adam([self.input], lr=self.base_lr, betas=self.adam_betas)
            worker_futs = []
            for worker_rref in self.worker_rrefs:
                worker_futs.append(worker_rref.rpc_async().receive_input_pointer(self.input))
            torch.futures.wait_all(worker_futs)
            for iter in range(self.args.iters):
                self.adjust_lr(iter)
                self.optimizer.zero_grad()
                # async call distributed workers to compute model loss
                worker_futs.clear()
                for worker_rref in self.worker_rrefs:
                    worker_futs.append(worker_rref.rpc_async().compute_grad(iter))
                # compute image pixel loss
                tv_loss = self.get_tv_loss(self.input)
                l2_loss = self.get_l2_loss(self.input)
                image_loss = self.tv_scale * tv_loss + self.l2_scale * l2_loss
                image_loss.backward()
                
                # wait for all workers to aggregate gradients
                torch.futures.wait_all(worker_futs)

                # optimizer step
                self.optimizer.step()
                if (iter+1) % 100 == 0:
                    self.save_result(trial, iter+1)
        self.log("Optimization finished")
        with open(f"{self.result_folder_path}/log.txt", "w") as logfile:
            logfile.write(self.logger)

    def log(self, text):
        timestamp = time.strftime("%H-%M-%S", time.localtime())
        print(f"{timestamp} {text}")
        self.logger += f"{timestamp} {text}\n"

    def save_result(self, trial, iteration):
        ncols = len(self.args.categories)
        nrows = self.args.samples_per_category
        input_tensor = self.input.clone().detach()
        dataset_mean = np.array([0.485, 0.456, 0.406])
        dataset_std = np.array([0.229, 0.224, 0.225])
        for c in range(3):
            input_tensor[:, c] = torch.clamp(input_tensor[:, c] * dataset_std[c] + dataset_mean[c], 0, 1)
        row_tensors_tuple = torch.split(input_tensor, split_size_or_sections=ncols)
        row_tensors = []
        for row_tensor in row_tensors_tuple:
            row_tensors.append(torch.stack(row_tensor.split(1), dim=3).reshape(3, 224, 224 * ncols))
        reshaped_tensor = torch.stack(row_tensors, dim=1).reshape(3, 224*nrows, 224*ncols)
        img_path = f"{self.result_folder_path}/batch{trial}_iteration{iteration}.png"
        torchvision.utils.save_image(reshaped_tensor, img_path)