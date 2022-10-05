import torch
import torchvision
import torch.distributed.rpc as rpc
# worker prepares model; compute gradients in every iteration

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
    

def prepare_model(modelname):
    # prepare the model 
    torchvision_pretrained_dict = {
        'convnext_base': 'ConvNeXt_Base_Weights',
        'convnext_large': 'ConvNeXt_Large_Weights',
        'convnext_small': 'ConvNeXt_Small_Weights',
        'convnext_tiny': 'ConvNeXt_Tiny_Weights',
        'densenet121': 'DenseNet121_Weights', 
        'densenet161': 'DenseNet161_Weights',
        'efficientnet_b0': 'EfficientNet_B0_Weights',
        'efficientnet_b1': 'EfficientNet_B1_Weights',
        'efficientnet_b2': 'EfficientNet_B2_Weights',
        'efficientnet_b3': 'EfficientNet_B3_Weights',
        'efficientnet_b4': 'EfficientNet_B4_Weights',
        'efficientnet_b5': 'EfficientNet_B5_Weights',
        'efficientnet_b6': 'EfficientNet_B6_Weights',
        'efficientnet_b7': 'EfficientNet_B7_Weights',
        'googlenet': 'GoogLeNet_Weights',
        'inception_v3': 'Inception_V3_Weights',
        'mnasnet0_5': 'MNASNet0_5_Weights',
        'mnasnet1_0': 'MNASNet1_0_Weights',
        'mobilenet_v2': 'MobileNet_V2_Weights',
        'resnet18': 'ResNet18_Weights',
        'resnet34': 'ResNet34_Weights',
        'resnet50': 'ResNet50_Weights',
        'resnet101': 'ResNet101_Weights',
        'resnext50_32x4d': 'ResNeXt50_32X4D_Weights',
        'resnext101_32x8d': 'ResNeXt101_32X8D_Weights',
        'vgg11_bn': 'VGG11_BN_Weights',
        'vgg13_bn': 'VGG13_BN_Weights',
        'vgg16_bn': 'VGG16_BN_Weights',
        'vgg19_bn': 'VGG19_BN_Weights',
        'wide_resnet50_2': 'Wide_ResNet50_2_Weights',
        'wide_resnet101_2': 'Wide_ResNet101_2_Weights',
    }
    if modelname in torchvision_pretrained_dict.keys():
        model_func = getattr(torchvision.models, modelname)
        model = model_func(weights=torchvision_pretrained_dict)
        return model
    else:
        raise NotImplementedError
    

class Worker(object):
    def __init__(self, rank, args):
        self.rank = rank
        self.worker_name = rpc.get_worker_info().name
        self.modelname = args.modelnames[rank - 1]
        self.device = f"cuda:{args.devices[rank - 1]}" if args.devices[rank - 1] >= 0 else "cpu"
        
        info = f"{self.worker_name} got {self.modelname} on device {self.device}"
        print(info)
        

    def prepare_model(self):
        # Load pretrained weight


        # Batchnorm hook the model
        pass
        
    def compute_grad(self, input_rref):
        pass