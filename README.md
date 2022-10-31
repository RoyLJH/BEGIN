# BEGIN: Batchnorm Ensembled Generative Inversion

## Requirements
    torch==1.12.1+cu113
    torchvision==0.13.1+cu113
    
## Usage
    python inverter_rpc.py --modelnames modelA modelB modelC --devices 0 1 2
or 
    
    python inverter_fast.py --modelnames modelD modelE --devices 3 -1
    
The two scripts (`inverter_rpc.py` and `inverter_fast.py`) implements exactly same functionality.
Note that `inverter_rpc.py` is implemented by [PyTorch RPC framework](https://pytorch.org/docs/stable/rpc.html). 
Current Pytorch RPC framework does not support transporting cuda tensors across devices, therefore the communication between agents has high time cost. To accelerate the joint optimization process, a re-implementation is given in `inverter_fast.py`, which uses multi-threading in Python.

### Arguments explained
    modelnames: a list of modelnames to be involved in joint BN inversion.
    devices: non-negative device id represents cuda device, -1 represents gpu. One device corresponds to one model architecture.
    categories: a list of integers indicating ImageNet labels (range from 0 - 999, [ImageNet human readable labels](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a))
    
See the desciption of other arguments using `--help`.
   
Default categories inside the batch: [daisy, balloon, acoustic guitar, hummingbird, Indian cobra, Siberian husky, digital clock, traffic light, Indian elephant, strawberry].
   
### Extension
To add a new model into joint optimization, add your model into `prepare_model()` function, define a unique modelname(different from given ones) and how to load your model.

## Interesting findings

### Shortcut learning
The phenomenon of shortcut learning of DNNs is defined and described in [Geirhos et al](https://www.nature.com/articles/s42256-020-00257-z). In discriminative learning, it would only pick any feature that is sufficient to reliably discriminate on a given dataset. 

Single model BN inversion (Resnet101):

Multiple model joint BN inversion():


### Privacy leakage
BN statistics expose quite much sensitive information of the training data. This is well known as quite many research papers on reconstruction-based 
privacy attacks [Huang et al.](https://proceedings.neurips.cc/paper/2021/hash/3b3fff6463464959dcd1b68d0320f781-Abstract.html), [Yin et al.](https://openaccess.thecvf.com/content/CVPR2021/papers/Yin_See_Through_Gradients_Image_Batch_Recovery_via_GradInversion_CVPR_2021_paper.pdf). 

Here we give another example of how BN statistics can even expose the data augmentation technique used in training. Using the same architectures with different weight parameters, we obtain different results of inversion.

First image shows Resnet50 + Resnet101 BN inversion (both use torchvision pretrained weight IMAGENET1K_V1) 

Second image shows Resnet50 + Resnet101 BN inversion (both use torchvision pretrained weight IMAGENET1K_V2)


Second result shows cutting line inside single image, which can indicate the use of cutmix data augmentation (as indeed used in training [IMAGENET1K_V2 weights](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/)). This can be seen as another example of how Batchnorm statistics encodes strong informative clues of training data, and can be used by malicious attackers to breach the privacy.

### Semantic Information encoded in BN statistics
All the above inverted results uses BatchNorm matching loss (L2 distance of mean and variance of current batch and the restored BN stats of whole dataset) along with classification loss. We show that only uses information from BN layers can give us a lot information of the image distribution.

