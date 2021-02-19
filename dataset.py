from pathlib import Path
from typing import Callable
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageNet

def get_dataset(path:Path, transforms:Callable=None) -> Dataset:
    return ImageNet(str(path), split='val', transform=transforms)