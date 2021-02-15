# NFNet Pytorch Implementation
 - [Original Repository](https://github.com/deepmind/deepmind-research/blob/master/nfnets/nf_resnet.py)
 - [Original Paper](https://arxiv.org/abs/2102.06171)

## Installation
Install python and opencv if you don't have it already. [Here](https://benjs.de/opencv/2021/02/03/install-opencv-and-python-without-root-access.html) is an instruction.
Adjust the CUDA version of the torch and torchvision module in [requirements.txt](requirements.txt) to your installation and enter
```
pip3 install -r requirements.txt
```

## Demo run
After installation, check if everything is working correctly with
```
python3 demo.py --device 0
```