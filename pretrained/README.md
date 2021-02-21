# Pretrained weights

Download the pretrained weights from the [official repository](https://github.com/deepmind/deepmind-research/tree/master/nfnets#pre-trained-weights) and place them inside this folder.
Then start training with 
```
python3 train.py --pretrained pretrained/F0_haiku.npz
```

or evaluation with 
```
python3 eval.py --pretrained pretrained/F0_haiku.npz --dataset /path/to/imagenet/val/
