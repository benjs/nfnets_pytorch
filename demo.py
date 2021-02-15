import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

if __name__=='__main__':
    x = torch.randn(1, 1, 1000, 1000)
    conv = nn.Conv2d(1, 2, 3)
    nn.init.normal_(conv.weight, 0, 1)

    y = conv.forward(x)

    print(conv.weight)
    print(torch.mean(conv.weight))
    print(torch.mean(conv.weight, (0, 1, 2),keepdim=True))

    print(conv.weight.shape)
    print(np.prod(conv.weight.shape))
    print(conv.weight.shape[:-1])
    print(conv.weight.shape[-1])
    print(conv.weight.shape[1:])


    