# NFNet Pytorch Implementation

All credits go to the authors of the [original paper](https://arxiv.org/abs/2102.06171): High-Performance Large-Scale Image Recognition Without Normalization. This repo is inspired by their JAX implementation in the [official repository](https://github.com/deepmind/deepmind-research/blob/master/nfnets/nf_resnet.py). Visit their repo for citing.

## Development status
Generally, the development is in a very early stage
and the models have not been thoroughly tested.

- [x] NFNet Models
  - [x] F0-F7
  - [x] Scaled weight standardization
  - [x] Squeeze and excite
  - [x] Stochastic depth
- [x] SGD with unit adaptive gradient clipping (SGD-AGC)
- [x] Training on ImageNet
- [x] Model overfits on small dataset
- [ ] Tensorboard support + General usability
- [ ] Multi-GPU support
- [ ] Training on complete dataset and validation to _Baseline + Second Conv_ (please contact me if you have spare GPUs)
- [ ] Data augmentation
- [ ] Model validation to _Baseline + CutMix_ (please contact me if you have spare GPUs)
- [ ] Pre-trained weights (please contact me if you have spare GPUs)
- [ ] Training on custom data
- [ ] Signal propagation plots (from [first paper](https://arxiv.org/abs/2101.08692))

## Installation
Adjust the CUDA version of the torch and torchvision module in [requirements.txt](requirements.txt) to your existing installation and enter
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## Train on ImageNet
1. You need the ImageNet dataset which you can download at [image-net.org](http://www.image-net.org/challenges/LSVRC/2012/downloads.php#images) after asking for access. Place the dev_kit.tar and unpacked image directory in the same directory.
2. Go to the [config file](default_config.yaml) and change `dataset` to the directory you placed the image directory and dev_kit.tar in.
3. Run `python3 train.py`

### Tested batch sizes for a single GPU with 8GB VRAM
|  F0  |  F1  |  F2  |  F3  |  F4  |  F5  |  F6  |  F7  |  
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|  60  |  20  |  9  |  3  |  1  |  -  |  -  |  -  |