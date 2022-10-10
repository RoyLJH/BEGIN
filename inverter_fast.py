import torch
import torch.nn as nn
import torchvision
import argparse
import os
import numpy as np
from PIL import Image
import time
import threading

class BatchNormStatMatchingHook():
    # Hook for matching BN statistics
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        channels = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([channels, -1]).var(1, unbiased=False)
        self.bn_matching_loss = torch.norm(module.running_mean.data - mean, 2) + \
            torch.norm(module.running_var.data - var, 2)

def prepare_model(modelname, device):
    # prepare the model 
    if modelname == 'resnet18':
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    elif modelname == 'resnet34':
        model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
    elif modelname == 'resnet50':
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    elif modelname == 'resnet101':
        model = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V2)
    elif modelname == 'resnet152':
        model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V2)
    elif modelname == 'resnext50_32x4d':
        model = torchvision.models.resnext50_32x4d(weights=torchvision.models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
    elif modelname == 'resnext101_32x8d':
        model = torchvision.models.resnext101_32x8d(weights=torchvision.models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2)
    elif modelname == 'wide_resnet50_2':
        model = torchvision.models.wide_resnet50_2(weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V2)
    elif modelname == 'wide_resnet101_2':
        model = torchvision.models.wide_resnet101_2(weights=torchvision.models.Wide_ResNet101_2_Weights.IMAGENET1K_V2)
    elif modelname == 'vgg11_bn':
        model = torchvision.models.vgg11_bn(weigths=torchvision.models.VGG11_BN_Weights.IMAGENET1K_V1)
    elif modelname == 'vgg13_bn':
        model = torchvision.models.vgg13_bn(weights=torchvision.models.VGG13_BN_Weights.IMAGENET1K_V1)
    elif modelname == 'vgg16_bn':
        model = torchvision.models.vgg16_bn(weights=torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1)
    elif modelname == 'vgg19_bn':
        model = torchvision.models.vgg19_bn(weights=torchvision.models.VGG19_BN_Weights.IMAGENET1K_V1)
    elif modelname == 'convnext_base':
        model = torchvision.models.convnext_base(weights=torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    elif modelname == 'convnext_small':
        model = torchvision.models.convnext_small(weights=torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
    elif modelname == 'convnext_large':
        model = torchvision.models.convnext_large(weights=torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
    elif modelname == 'convnext_tiny':
        model = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    elif modelname == 'googlenet':
        model = torchvision.models.googlenet(weights=torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1)
    elif modelname == 'inception_v3':
        model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
    else:
        raise NotImplementedError(f"Do not support {modelname}; please add model architecture and pretrained weights to `worker.py: prepare_model()`")

    model = model.to(device).eval()
    # add BN hooks
    bn_hooks = []
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            bn_hooks.append(BatchNormStatMatchingHook(module))
    return model, bn_hooks

class ImagenetInverterWorker(threading.Thread):
    def __init__(self, threadidx, args, inputs, labels, trial):
        super(ImagenetInverterWorker, self).__init__()
        self.threadidx = threadidx
        self.args = args
        self.modelname = args.modelnames[threadidx]
        self.device = args.devices[threadidx]
        self.labels = labels.to(self.device)
        self.inputs_pointer = inputs
        self.inputs = inputs.clone().to(self.device)
        self.model, self.bn_hooks = prepare_model(self.modelname, self.device)
        self.total_iters = args.iters
        self.trial = trial

        # loss scale
        self.ce_scale = args.ce_scale
        self.bn_scale = args.bn_scale
        self.tv_scale = args.tv_scale
        self.l2_scale = args.l2_scale

        if threadidx == 0:
            # optimization related (controlled by thread 0)
            
            self.warmup_iters = args.warmup_iters
            self.base_lr = args.lr
            self.adam_betas = args.adam_betas
            self.optimizer = torch.optim.Adam([inputs], lr=args.lr, betas=args.adam_betas)

            # post-processing related (controlled by thread 0)
            self.roll = args.roll
            self.flip = args.flip
            self.clip = args.clip

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

    def save_result(self, trial, iteration):
        global result_folder_path
        ncols = len(self.args.categories)
        nrows = self.args.samples_per_category
        input_tensor = self.inputs.clone().detach()
        dataset_mean = np.array([0.485, 0.456, 0.406])
        dataset_std = np.array([0.229, 0.224, 0.225])
        for c in range(3):
            input_tensor[:, c] = torch.clamp(input_tensor[:, c] * dataset_std[c] + dataset_mean[c], 0, 1)
        row_tensors_tuple = torch.split(input_tensor, split_size_or_sections=ncols)
        row_tensors = []
        for row_tensor in row_tensors_tuple:
            row_tensors.append(torch.stack(row_tensor.split(1), dim=3).reshape(3, 224, 224 * ncols))
        reshaped_tensor = torch.stack(row_tensors, dim=1).reshape(3, 224*nrows, 224*ncols)
        img_path = f"{result_folder_path}/batch{trial}_iteration{iteration}.png"
        torchvision.utils.save_image(reshaped_tensor, img_path)

    def run(self):
        global worldsize, threadSync
        ce_criterion = torch.nn.CrossEntropyLoss()
        best_loss = float("inf")
        for iter in range(self.total_iters):
            # Thread 0 (1) adjust LR (2) zero_grad (3) image prior loss
            if self.threadidx == 0:
                self.adjust_lr(iter)
                self.optimizer.zero_grad()
                tv_loss = self.get_tv_loss(self.inputs)
                l2_loss = self.get_l2_loss(self.inputs)
                image_loss = self.tv_scale * tv_loss + self.l2_scale * l2_loss
                image_loss.backward()
            # All thread compute model loss
            outputs = self.model(self.inputs)
            bn_loss = sum([hook.bn_matching_loss for hook in self.bn_hooks])
            ce_loss = ce_criterion(outputs, self.labels)
            model_loss = self.ce_scale * ce_loss + self.bn_scale * bn_loss
            if model_loss.item() < best_loss or iter % 10 == 0:
                best_loss = min(best_loss, model_loss.item())
                log(f"{self.modelname} iter {iter} bn_loss {bn_loss.item():.4f} ce_loss {ce_loss.item():.4f}")
            model_loss.backward()

            # Thread sync barrier (wait for all threads)
            threadSync[self.threadidx] = True
            if self.threadidx != 0:  # Thread (id>0) wait passively for thread 0 to kick off next iter
                while threadSync[self.threadidx]:
                    time.sleep(0.001)
                    continue
            else: # Thread 0 save image and kick off
                if (iter + 1) % 200 == 0:
                    self.save_result(self.trial, iter+1)
                while True:
                    syncFlag = True
                    for threadSyncFlag in threadSync:
                        syncFlag = syncFlag and threadSyncFlag
                    # syncFlag == True : All threads finished model_loss.backward()
                    if syncFlag:
                        self.optimizer.step()
                        threadSync = [False] * worldsize
                        break
            
            # Prepare for next iteration
            self.inputs = self.inputs_pointer.clone().to(self.device)

        if self.threadidx == 0:
            log("Optimization ends!")
        return self.inputs_pointer


class ImageNetInverterThread(threading.Thread):
    def __init__(self, threadidx, inputs_ori=None, optimizer=None,
        modelnames=["se_resnet50", "resnet50v15"], devices=["cuda:0", "cuda:1"],
        optimizer_betas=[0.3, 0.9], lr=0.8, iters=2000, warmup_iters=100,
        cos_lr=False, decay_iter=None, decay_coeff=None,
        target_labels=None, sample_per_label=4, 
        main_scale=1.0, bn_scale=0.05, first_bn_scale=8, 
        tvl1_scale=0.0, tvl2_scale=4e-3, l2_scale=1e-5, var_scale=1.0, 
        low_res=False, roll=0, flip=False, clip=0,
        info="", save_rows=8, save_cols=5):

        super(ImageNetInverterThread, self).__init__()
        assert len(modelnames) == len(devices)
        assert len(optimizer_betas) == 2
        self.threadidx = threadidx
        self.modelnames = modelnames
        self.devices = devices
        self.modelname = modelnames[threadidx]
        self.device = devices[threadidx]
        
        self.inputs_ori = inputs_ori
        self.inputs = self.inputs_ori.clone().to(self.device)

        self.optimizer = optimizer
        self.optimizer_betas = optimizer_betas
        self.lr = lr
        self.iters = iters
        self.cos_lr = cos_lr
        if (not decay_iter is None):
            assert not decay_coeff is None
            assert len(decay_coeff) == len(decay_iter)
            self.decay_iter = decay_iter
            self.decay_coeff = decay_coeff
        
        self.target_labels = target_labels # python list
        self.sample_per_label = sample_per_label

        self.main_scale = main_scale
        self.bn_scale = bn_scale
        self.first_bn_scale = first_bn_scale
        self.tvl1_scale = tvl1_scale
        self.tvl2_scale = tvl2_scale
        self.l2_scale = l2_scale

        self.low_res = low_res
        self.roll = roll
        self.flip = flip
        self.clip = clip

        self.info = info
        self.save_rows = save_rows
        self.save_cols = save_cols

        # record time
        now = int(round(time.time() * 1000))
        self.nowstamp = str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(now/1000)))    

        # prepare the model
        self.model = self.prepare_model(self.modelname)

        # register hooks for BN matching
        self.bn_matching_layers = []
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn_matching_layers.append(BatchNormMatchingHook(module, var_scale))
        self.bn_rescale = [self.first_bn_scale] + [1. for _ in range(len(self.bn_matching_layers) - 1)]
        
        # register learning rate scheduler
        self.lr_cos_scheduler = self.lr_cosine_policy(self.lr, warmup_iters, self.iters)
        #self.lr_cos_scheduler = self.lr_cosine_policy(self.lr, self.iters//10, self.iters)

        # generate targets
        targets_extend_list = [target for target in target_labels for _ in range(sample_per_label)]
        targets_list = [target for target in self.target_labels for _ in range(self.sample_per_label)]
        self.round_target_labels = [348] * 8
        #round_target_labels = self.target_labels.copy()
        #round_target_labels.insert(0, self.target_labels[-1])
        #self.round_target_labels = round_target_labels[:len(self.target_labels)]
        self.round_targets_list = [target for target in self.round_target_labels for _ in range(self.sample_per_label)]
        self.targets = torch.LongTensor(targets_list).to(self.device)
        self.round_targets = torch.LongTensor(self.round_targets_list).to(self.device)
        self.targets_mixup = label_mixup(self.targets, self.round_targets, num_classes=1000)
        self.printLabel()
        

    def run_onehot(self):
        global threadSync
        global threadTotal
        global syncLock
        global next_iter
        ce_criterion = nn.CrossEntropyLoss()
        best_loss = float("inf")
        for iter in range(self.iters):
            # thread 0 to 
            #   (1) adjust the learning rate 
            #   (2) zero the gradient 
            #   (3) calculate image prior losses
            if self.threadidx == 0:
                # adjust learning rate
                if self.cos_lr == True:
                    self.lr_cos_scheduler(self.optimizer, iter, iter)
                else:
                    if not self.decay_iter is None:
                        for (idx, decay_it) in enumerate(self.decay_iter):
                            if iter == decay_it:
                                self.optimizer.param_groups[0]['lr'] *= self.decay_coeff[idx]
                # zero the gradient
                self.optimizer.zero_grad()
                # calculate image prior loss (tv loss and l2 loss)
                tvl1_loss, tvl2_loss = get_image_tv_losses(self.inputs)
                l2_loss = torch.norm(self.inputs, 2)
                image_loss = self.tvl1_scale * tvl1_loss + \
                            self.tvl2_scale * tvl2_loss + \
                            self.l2_scale * l2_loss
                image_loss.backward()

            # compute model loss
            outputs = self.model(self.inputs)
            bn_loss = sum([mod.r_feature * self.bn_rescale[idx] for (idx, mod) in enumerate(self.bn_matching_layers)])
            class_loss = ce_criterion(outputs, self.targets)

            model_loss = self.main_scale * class_loss +\
                        self.bn_scale * bn_loss
            
            if model_loss.item() < best_loss:
                best_loss = model_loss.item()
                print(self.modelname, " iter ", iter, " best loss:%.4g , lr:%.8g" % 
                            (best_loss, self.optimizer.param_groups[0]['lr']))
            
            model_loss.backward()

            #print("Thread ", self.threadidx, " has completed computing loss.backward() in iter ", iter)
            threadSync[self.threadidx] = True

            if self.threadidx != 0:
                while threadSync[self.threadidx] == True:
                    time.sleep(0.001)
                    continue
            else:
                # thread 0 to save the current image (from last iteration)
                if (iter + 1) % 200 == 0 :
                    self.save_images(iter, self.optimizer.param_groups[0]['lr'], self.save_rows, self.save_cols)
                # wait for all threadSync to be True
                while True:
                    syncFlag = True
                    for i in range(threadTotal):
                        syncFlag = syncFlag and threadSync[i]
                    if syncFlag == False:
                        continue
                    else: # all the threads have completed loss.backward()
                        self.optimizer.step()
                        #print("Thread ", self.threadidx, " has completed optimizer.step()")
                        next_iter = False                   
                        for i in range(threadTotal):
                            threadSync[i] = False
                        break
            
            
            # cover self.inputs with the updated inputs_ori
            self.inputs = self.inputs_ori.clone().to(self.device)

            # apply clip
            if self.clip != 0 and iter % self.clip == 0:
                with torch.no_grad():
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    for c in range(3):
                        m, s = mean[c], std[c]
                        self.inputs[:, c].clamp_(-m/s, m/s)

            # perform downsample for self.low_res

            # apply random jitter offsets according to self.roll

            # apply flipping

            # wait for all threadSync to be False
            if self.threadidx != 0:
                while next_iter == False:
                    time.sleep(0.001)
                    continue
            # thread 0 wait for all threadSync to be False
            else:
                while True:
                    syncFlag = False
                    for i in range(threadTotal):
                        syncFlag = syncFlag or threadSync[i]
                    if syncFlag == True:
                        continue
                    else:
                        next_iter = True
                        break
        if self.threadidx == 0:
            print("Optimization ends! The labels are: ")
            self.printLabel()
        return self.inputs_ori


    def run(self):
        global threadSync
        global threadTotal
        global syncLock
        global next_iter
        #ce_criterion = nn.CrossEntropyLoss()
        custom_ce_criterion = cross_entropy_for_onehot
        best_loss = float("inf")
        for iter in range(self.iters):
            # thread 0 to 
            #   (1) adjust the learning rate 
            #   (2) zero the gradient 
            #   (3) calculate image prior losses
            if self.threadidx == 0:
                # adjust learning rate
                if self.cos_lr == True:
                    self.lr_cos_scheduler(self.optimizer, iter, iter)
                else:
                    if not self.decay_iter is None:
                        for (idx, decay_it) in enumerate(self.decay_iter):
                            if iter == decay_it:
                                self.optimizer.param_groups[0]['lr'] *= self.decay_coeff[idx]
                # zero the gradient
                self.optimizer.zero_grad()
                # calculate image prior loss (tv loss and l2 loss)
                tvl1_loss, tvl2_loss = get_image_tv_losses(self.inputs)
                l2_loss = torch.norm(self.inputs, 2)
                image_loss = self.tvl1_scale * tvl1_loss + \
                            self.tvl2_scale * tvl2_loss + \
                            self.l2_scale * l2_loss
                image_loss.backward()

            # compute model loss
            outputs = self.model(self.inputs)
            bn_loss = sum([mod.r_feature * self.bn_rescale[idx] for (idx, mod) in enumerate(self.bn_matching_layers)])
            #class_loss = custom_ce_criterion(outputs, self.targets)
            class_loss = custom_ce_criterion(outputs, self.targets_mixup)



            model_loss = self.main_scale * class_loss +\
                        self.bn_scale * bn_loss
            
            if model_loss.item() < best_loss:
                best_loss = model_loss.item()
                print(self.modelname, " iter ", iter, " best loss:%.4g , lr:%.8g" % 
                            (best_loss, self.optimizer.param_groups[0]['lr']))
            
            model_loss.backward()

            #print("Thread ", self.threadidx, " has completed computing loss.backward() in iter ", iter)
            threadSync[self.threadidx] = True

            if self.threadidx != 0:
                while threadSync[self.threadidx] == True:
                    time.sleep(0.001)
                    continue
            else:
                # thread 0 to save the current image (from last iteration)
                if (iter + 1) % 200 == 0 :
                    self.save_images(iter, self.optimizer.param_groups[0]['lr'], self.save_rows, self.save_cols)
                # wait for all threadSync to be True
                while True:
                    syncFlag = True
                    for i in range(threadTotal):
                        syncFlag = syncFlag and threadSync[i]
                    if syncFlag == False:
                        continue
                    else: # all the threads have completed loss.backward()
                        self.optimizer.step()
                        #print("Thread ", self.threadidx, " has completed optimizer.step()")
                        next_iter = False                   
                        for i in range(threadTotal):
                            threadSync[i] = False
                        break
            
            
            # cover self.inputs with the updated inputs_ori
            self.inputs = self.inputs_ori.clone().to(self.device)

            # apply clip
            if self.clip != 0 and iter % self.clip == 0:
                with torch.no_grad():
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    for c in range(3):
                        m, s = mean[c], std[c]
                        self.inputs[:, c].clamp_(-m/s, m/s)

            # perform downsample for self.low_res

            # apply random jitter offsets according to self.roll

            # apply flipping

            # wait for all threadSync to be False
            if self.threadidx != 0:
                while next_iter == False:
                    time.sleep(0.001)
                    continue
            # thread 0 wait for all threadSync to be False
            else:
                while True:
                    syncFlag = False
                    for i in range(threadTotal):
                        syncFlag = syncFlag or threadSync[i]
                    if syncFlag == True:
                        continue
                    else:
                        next_iter = True
                        break
        if self.threadidx == 0:
            print("Optimization ends! The labels are: ")
            self.printLabel()
        return self.inputs_ori

        

    def save_images(self, iter, lr, rows=7, cols=5):
        global save_folder
        # Inputs: GPU tensor. Detransform 
        inputs_clone = self.inputs_ori.clone().detach().cpu()
        # Detransform and Clip
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        for c in range(3):
            m, s = mean[c], std[c]
            inputs_clone[:, c] = torch.clamp(inputs_clone[:, c] * s + m, 0, 1)
        # Convert to numpy, then convert to PIL Image
        img_matrix = None
        for row in range(rows):
            img_row = None
            for col in range(cols):
                img_torch = inputs_clone[row * cols + col] #[C, H, W]
                if img_row is None:
                    img_row = img_torch
                else:
                    img_row = torch.cat((img_row, img_torch), dim=2) # W
            if img_matrix is None:
                img_matrix = img_row
            else:
                img_matrix = torch.cat((img_matrix, img_row), dim=1) # H
        img_np = img_matrix.data.numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((img_np * 255).astype(np.uint8))
        imgpath = './inverted_images_sample/' + str(self.modelnames) + '/' 
        if not os.path.exists(imgpath):
            os.makedirs(imgpath)
        imgpath = imgpath + str(self.nowstamp)[5:] + '_'
        imgpath = imgpath + 'm%.4g_bn%.4g_f%.4g' % (self.main_scale, self.bn_scale, self.first_bn_scale)
        #imgpath = imgpath + '_l2%.4g_tv%.4g' % (self.l2_scale, self.tvl2_scale) 

        if self.optimizer_betas != [0.3, 0.9]:
            imgpath = imgpath + '_betas' + str(self.optimizer_betas)

        #if self.cos_lr == True:
        #    imgpath = imgpath + '_cos'
        imgpath = imgpath + 'lr' + str(self.lr)
        if self.flip == True:
            imgpath = imgpath + '_flip'
        if self.low_res == True:
            imgpath = imgpath + '_lowres'
        if self.clip != 0:
            imgpath = imgpath + '_clip' + str(self.clip)
        if self.roll != 0:
            imgpath = imgpath + '_roll' + str(self.roll)
        imgpath = imgpath + self.info + '_/'
        if not os.path.exists(imgpath):
            os.makedirs(imgpath)
        save_folder = imgpath
        print("================ iter ", iter, " ================")
        if (iter + 1) % 400 == 0 :
            iterpath = imgpath + 'iter_' + str(iter) + '_lr_%.6g.png' % lr
            pil_image.save(iterpath)
            print("save image in ", iterpath)
        else:
            imgpath = imgpath + "current.png"
            pil_image.save(imgpath)
            print("save image in ", imgpath)

    def prepare_model(self, modelname):
        if modelname == "densenet121":
            model = torchvision.models.densenet121(pretrained=False)
            model.load_state_dict(torch.load("~/checkpoint/densenet121.pth"), strict=False)
        elif modelname == "googlenet":
            model = torchvision.models.googlenet(pretrained=False)
            model.load_state_dict(torch.load("~/checkpoint/googlenet.pth"))
        elif modelname == "resnet18":
            model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            #model.load_state_dict(torch.load("~/checkpoint/" + modelname + ".pth"))
        elif modelname == "resnet50" or modelname == "resnet50_noaug":
            model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
            #model = load_model_pytorch(model, "~/checkpoint/" + modelname + ".pth", gpu_n=int(str(self.device)[-1:]))
        elif modelname == "resnet152":
            model = torchvision.models.resnet152(pretrained=False)
            model.load_state_dict(torch.load("~/checkpoint/" + modelname + ".pth"))
        elif modelname == "vgg16-bn":
            model = torchvision.models.vgg16_bn(pretrained=False)
            model.load_state_dict(torch.load("~/checkpoint/" + modelname + ".pth"))
        elif modelname == "vgg19-bn":
            model = torchvision.models.vgg19_bn(pretrained=False)
            model.load_state_dict(torch.load("~/checkpoint/" + modelname + ".pth"))
        elif modelname == "resnet18v15_noaug":
            from models.resnetv15 import build_resnet
            model = build_resnet("resnet18", "classic")
            model = load_model_pytorch(model, "~/checkpoint/"+modelname+".pth", gpu_n=int(str(self.device)[-1:]))
        elif modelname == "resnet34v15_noaug":
            from models.resnetv15 import build_resnet
            model = build_resnet("resnet34", "classic")
            model = load_model_pytorch(model, "~/checkpoint/"+modelname+".pth", gpu_n=int(str(self.device)[-1:]))
        elif modelname == "resnet50v15" or modelname == "resnet50v15_noaug":
            from models.resnetv15 import build_resnet
            model = build_resnet("resnet50", "classic")
            model = load_model_pytorch(model, "~/checkpoint/"+modelname+".pth", gpu_n=int(str(self.device)[-1:]))
            #model.load_state_dict(torch.load("./checkpoint/" + modelname + ".pth"))
        elif modelname == "se_resnet50":
            from models.senet import se_resnet50
            model = se_resnet50()
            model = load_model_pytorch(model, "~/checkpoint/" + modelname + ".pth", gpu_n=int(str(self.device)[-1:]))
        elif modelname == "se_resnet18":
            from models.senet import se_resnet18
            model = se_resnet18()
            model = load_model_pytorch(model, "~/checkpoint/" + modelname + ".pth", gpu_n=int(str(self.device)[-1:]))
        elif modelname == "mobilenetv2":
            from models.mobilenetv2 import mobilenetv2
            model = mobilenetv2()
            model = load_model_pytorch(model, "~/checkpoint/" + modelname + ".pth", gpu_n=int(str(self.device)[-1:]))
        elif modelname == "shufflenetv1":
            from models.shufflenet import shufflenetv1
            model = shufflenetv1()
            model = load_model_pytorch(model, "~/checkpoint/" + modelname + ".pth", gpu_n=int(str(self.device)[-1:]))
        else :
            print("Do not have the pretrained model for ", modelname)
            return None
        torch.backends.cudnn.benchmark = True
        return model.to(self.device).eval()

    def lr_cosine_policy(self, base_lr, warmup_length, epochs):
        def _lr_fn(iteration, epoch):
            if epoch < warmup_length:
                lr = base_lr * (epoch + 1) / warmup_length
            else:
                e = epoch - warmup_length
                es = epochs - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            return lr

        def lr_policy(lr_fn):
            def _alr(optimizer, iteration, epoch):
                lr = lr_fn(iteration, epoch)
                self.optimizer.param_groups[0]['lr'] = lr
            return _alr
        return lr_policy(_lr_fn)

    def printLabel(self, list=None):
        if list is None:
            for id in self.target_labels:
                print(all_targets[id])
        else:
            for id in list:
                print(all_targets[id])


def log(text):
    global logger
    timestamp = time.strftime("%H-%M-%S", time.localtime())
    print(f"{timestamp} {text}")
    logger += f"{timestamp} {text}\n"

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
    parser.add_argument('--trials', default=1, type=int,
        help='How many batches to generate for same setting')

    # Optimization
    parser.add_argument('--ce_scale', default=0.8, type=float,
        help='Cross entropy loss scale')
    parser.add_argument('--bn_scale', default=0.05, type=float,
        help='BN statistics matching loss scale')
    parser.add_argument('--tv_scale', default=4e-3, type=float,
        help='Total variation image prior loss scale')
    parser.add_argument('--l2_scale', default=0, type=float,
        help='L2 image prior loss scale')
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
    assert len(args.adam_betas) == 2
    torch.backends.cudnn.benchmark = True

    # Distributed setting
    worldsize = len(args.modelnames)
    devices = [('cpu' if d==-1 else f'cuda:{d}') for d in args.devices]
    threadSync = [False] * worldsize
    next_iter_flag = False

    # Miscellaneous
    timestamp = time.strftime("%m-%d %H-%M-%S", time.localtime())
    result_folder_path = f"results/{timestamp}/"
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    logger = ""
    log(f"Batchnorm Ensembled Generative INversion (faster impl) from {args.modelnames}")

    # Optimization Setting
    label_list = [l for _ in range(args.samples_per_category) for l in args.categories]
    labels = torch.LongTensor(label_list)
    for trial in range(args.trials):
        inputs = torch.randn(len(labels), 3, 224, 224).requires_grad_(True)
        threads = []
        for i in range(worldsize):
            threads.append(ImagenetInverterWorker(threadidx=i, args=args, inputs=inputs, labels=labels, trial=trial))
        begintime = time.time()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        endtime = time.time()
        log(f"Batch {trial} optimization time: {endtime - begintime} seconds.")
    
    with open(f"{result_folder_path}/log.txt", "w") as logfile:
        logfile.write(logger)