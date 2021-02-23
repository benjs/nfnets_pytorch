# NFNet Pytorch Implementation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benjs/nfnets_pytorch/blob/master/demo.ipynb)

This repo contains pretrained NFNet models F0-F6 with high ImageNet accuracy from the paper *High-Performance Large-Scale Image Recognition Without Normalization*. The small models are as accurate as an EfficientNet-B7, but train 8.7 times faster. The large models set a new SOTA top-1 accuracy on ImageNet.
 
| NFNet  | F0  | F1  | F2  | F3  | F4  | F5  | F6+SAM  |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|  Top-1 accuracy Brock et al. | 83.6 | 84.7 | 85.1 | 85.7 | 85.9 | 86.0 | 86.5 |
|  Top-1 accuracy this implementation | 82.82 | 84.63 | 84.90 | 85.46 | 85.66 | 85.62 | TBD |

All credits go to the authors of the [original paper](https://arxiv.org/abs/2102.06171). This repo is heavily inspired by their nice JAX implementation in the [official repository](https://github.com/deepmind/deepmind-research/blob/master/nfnets/). Visit their repo for citing.

## Get started
```
git clone https://github.com/benjs/nfnets_pytorch.git
pip3 install -r requirements.txt
```
or if you don't need eval and training script
```
pip install git+https://github.com/benjs/nfnets_pytorch
```
Download pretrained weights from the [official repository](https://github.com/deepmind/deepmind-research/blob/master/nfnets/) and call

```python
from nfnets import pretrained_nfnet
model_F0 = pretrained_nfnet('pretrained/F0_haiku.npz')
model_F1 = pretrained_nfnet('pretrained/F1_haiku.npz')
# ...
```

The model variant is automatically derived from the parameter count in the pretrained weights file. 

## Validate yourself
```
python3 eval.py --pretrained pretrained/F0_haiku.npz --dataset path/to/imagenet/valset/
```

You can download the ImageNet validation set from the [ILSVRC2012 challenge site](http://www.image-net.org/challenges/LSVRC/2012/downloads.php#images) after asking for access with, for instance, your .edu mail address or from [AcademicTorrents](https://academictorrents.com/)

## Scaled weight standardization convolutions in your own model 
Simply replace all your `nn.Conv2d` with `WSConv2D` and all your `nn.ReLU` with `VPReLU` or `VPGELU` (variance preserving ReLU/GELU).

``` python
import torch.nn as nn
from nfnets import WSConv2D, VPReLU, VPGELU

# Simply replace your nn.Conv2d layers
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
 
        self.activation = VPReLU(inplace=True) # or VPGELU
        self.conv0 = WSConv2D(in_channels=128, out_channels=256, kernel_size=1, ...)
        # ...

    def forward(self, x):
      out = self.activation(self.conv0(x))
      # ...
```

## SGD with adaptive gradient clipping in your own model
Simply replace your `SGD` optimizer with `SGD_AGC`.
```python
from nfnets import SGD_AGC

optimizer = SGD_AGC(
        named_params=model.named_parameters(), # Pass named parameters
        lr=1e-3,
        momentum=0.9,
        clipping=0.1, # New clipping parameter
        weight_decay=2e-5, 
        nesterov=True)
```

It is important to exclude certain layers from clipping or momentum. The authors recommends to exclude the last fully convolutional from clipping and the bias/gain parameters from weight decay:
```python
import re

for group in optimizer.param_groups:
    name = group['name'] 
    
    # Exclude from weight decay
    if len(re.findall('stem.*(bias|gain)|conv.*(bias|gain)|skip_gain', name)) > 0:
        group['weight_decay'] = 0

    # Exclude from clipping
    if name.startswith('linear'):
        group['clipping'] = None

```

## Train your own NFNet
Adjust your desired parameters in [default_config.yaml](default_config.yaml) and start training.
```
python3 train.py --dataset /path/to/imagenet/
```

There is still some parts missing for complete training from scratch:
- Multi-GPU training
- Data augmentations
- FP16 activations and gradients

## Contribute

The implementation is still in an early stage in terms of usability / testing. 
If you have an idea to improve this repo open an issue, start a discussion or submit a pull request.

The current development status can be seen in [this](https://github.com/benjs/nfnets_pytorch/projects/1) project board. 
