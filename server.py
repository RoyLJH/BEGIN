import os
import time
import torch
import torchvision.utils
import numpy as np

class Server(object):
    def __init__(self, args):
        self.args = args
        self.batchsize = len(args.categories) * args.samples_per_category
        self.labels = [l for _ in range(args.samples_per_category) for l in args.categories]

        timestamp = time.strftime("%m-%d %H-%M-%S", time.localtime())
        self.result_folder_path = f"results/{timestamp}/"
        if not os.path.exists(self.result_folder_path):
            os.makedirs(self.result_folder_path)
        self.logger = ""
        self.log("Creating server...")
        
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
            
    def log(self, text):
        timestamp = time.strftime("%H-%M-%S", time.localtime())
        print(f"{timestamp} {text}")
        self.logger += f"{timestamp} {text}\n"

    def get_logger(self):
        return self.logger
        

    def optimize(self):
        for trial in range(self.args.trials):
            self.log(f"Starting Optimization for Trial {trial}...")
            self.input = torch.randn(self.batchsize, 3, 224, 224)
            for iter in range(self.args.iters):
                if (iter+1) % 1000 == 0:
                    self.save_result(trial, iter)
        self.log("Optimization finished")
        with open(f"{self.result_folder_path}/log.txt", "w") as logfile:
            logfile.write(self.logger)