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
The phenomenon of shortcut learning of DNNs is defined and described in [Geirhos et al.](https://www.nature.com/articles/s42256-020-00257-z). In discriminative learning, it would only pick any feature that is sufficient to reliably discriminate on a given dataset.  A well known claim is that in image classification task, CNNs tend to capture **texture** instead of **shape** information [Geirhos et al](https://openreview.net/forum?id=Bygh9j09KX), [Islam et al](https://openreview.net/forum?id=NcFEZOi-rLa). From my perspective, the inversion progress can tell what information is encoded or captured in the network (the network can only capture more information than what we visualize, since we do not use any information other than network parameter itself). A more accurate way of describing this is: The network always looks for as simple patterns(shortcuts) as possible to achieve its objective. 

Let's take the inversion result from a single ResNet 101 as an example:

![res101](demos/res101.png)

What network learn from Imagenet can be quite complex. For some category, it may only learns the texture, but for some other categories, it can learn to capture **shape**(*ballon*) and **color** (*strawberry* and *traffic light*), **object parts** (*snake head*, *husky dog* eyes and body) or even **relevant object parts**(grassland for *daisy*, human face for *mobile phone*, human arm for *guitar*) .


And if we jointly invert from multiple networks, for example a combination of 8 models (GoogleNet, ResNet34, ResNet50, ResNet101, ResNext50_32x4d, WRN50_2, WRN101_2, VGG19_BN):

![8 models joint](demos/joint8.png)

We can see from above that the inversion quality is better than inversion from a single network. That makes sense since networks with different architecuture and learning process capture different aspects or shortcuts of source data distribution. Joint inversion seeks a "consensus" among all networks, kind of compose all the surfaces to avoid biases learnt by any single network.

### Privacy leakage
BN statistics expose quite much sensitive information of the training data. This is well known as quite many research papers on reconstruction-based 
privacy attacks [Huang et al.](https://proceedings.neurips.cc/paper/2021/hash/3b3fff6463464959dcd1b68d0320f781-Abstract.html), [Yin et al.](https://openaccess.thecvf.com/content/CVPR2021/papers/Yin_See_Through_Gradients_Image_Batch_Recovery_via_GradInversion_CVPR_2021_paper.pdf). 

Here we give another example of how BN statistics can even expose the data augmentation technique used in training. Using the same architectures with different weight parameters, we obtain different results of inversion.

First image shows Resnet50 + Resnet101 BN inversion (both use torchvision pretrained weight IMAGENET1K_V1) 

![res50+res101_v1](demos/res50+res101_V1.png)

Second image shows Resnet50 + Resnet101 BN inversion (both use torchvision pretrained weight IMAGENET1K_V2)

![res50+res101_v2](demos/res50+res101_V2.png)

Second result shows cutting line and trace of stitched image patches inside single image, which can indicate the use of cutmix data augmentation (as indeed used in training [IMAGENET1K_V2 weights](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/)). This can be seen as another example of how Batchnorm statistics encodes strong informative clues of training data, and can be used by malicious attackers to breach the privacy.

### Semantic Information in BN statistics: decoupling BN matching loss and classification loss
All the above inverted results uses BatchNorm matching loss (L2 distance of mean and variance of current batch and the restored BN stats of whole dataset) along with classification loss. We show that only uses information from BN layers can give us quite a lot information of the image distribution. 

