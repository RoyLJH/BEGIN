import torchvision
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
        model = torchvision.models.vgg11_bn(weights=torchvision.models.VGG11_BN_Weights.IMAGENET1K_V1)
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
    return model


if __name__ == "__main__":
    modelnames = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 
        'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2',
        'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn', 'convnext_base', 'convnext_small', 'convnext_large',
        'convnext_tiny', 'googlenet', 'inception_v3'
    ]
    for modelname in modelnames:
        model = prepare_model(modelname)
