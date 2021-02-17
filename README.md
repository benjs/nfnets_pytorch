# NFNet Pytorch Implementation

All credits go to the authors of the [original paper](https://arxiv.org/abs/2102.06171): High-Performance Large-Scale Image Recognition Without Normalization. This repo is inspired by their JAX implementation in the [official repository](https://github.com/deepmind/deepmind-research/blob/master/nfnets/). Visit their repo for citing.

## Get started
After cloning this repo, adjust the CUDA version of the torch and torchvision module in [requirements.txt](requirements.txt) to your existing installation and enter
```
pip3 install -r requirements.txt
```
Go to the [config file](default_config.yaml) and enter the path to your ImageNet directory (downloadable after asking for access from [image-net.org](http://www.image-net.org/challenges/LSVRC/2012/downloads.php#images)).
The directory needs to contain the unpacked ImageNet archives and the dev_kit.tar file.

Start training with
```
python3 train.py --config default_config.yaml
```

## Development status
Generally, the development is in a very early stage
and the models have not been thoroughly tested.

- [x] NFNet Models
  - [x] F0-F7
  - [x] Scaled weight standardization
  - [x] Squeeze and excite
  - [x] Stochastic depth
  - [x] FP16 Operations
- [x] SGD with unit adaptive gradient clipping (SGD-AGC)
  - [ ] Exclude certain layers from weight-decay, clipping
- [x] Training on ImageNet
- [x] Model overfits on small dataset
- [ ] Tensorboard support + General usability (eval script etc.)
- [ ] Multi-GPU support
- [ ] Baseline augmentation
- [ ] Training on complete dataset and validation to _Baseline + Second Conv_ 
- [ ] Data augmentation
- [ ] Model validation to _Baseline + CutMix_ 
- [ ] Pre-trained weights 
- [ ] Training on custom data
- [ ] Signal propagation plots (from [first paper](https://arxiv.org/abs/2101.08692))
