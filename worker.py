import torch
import torchvision
import torch.distributed.rpc as rpc

from server import Server
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

    # add BN hooks
    bn_hooks = []
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            bn_hooks.append(BatchNormStatMatchingHook(module))
    return model, bn_hooks


class Worker(object):
    # Do not add any cuda tensors in this __init__ func(): torch.distributed.rpc framework only support cpu tensors
    def __init__(self, rank, args, server_rref):
        self.rank = rank
        self.worker_name = rpc.get_worker_info().name
        self.server_rref = server_rref
        self.modelname = args.modelnames[rank - 1]
        self.device = f"cuda:{args.devices[rank - 1]}" if args.devices[rank - 1] >= 0 else "cpu"
        self.model, self.bn_hooks = prepare_model(self.modelname)

    def log(self, text):
        text = f"{self.worker_name} | {text}"
        self.server_rref.rpc_sync().log(text)

    def prepare_device(self):
        self.model = self.model.to(self.device)
        self.log(f"Successfully load {self.modelname} on {self.device}")

    def compute_grad(self, input_rref):
        pass