import argparse
import math
import PIL
from pathlib import Path
from PIL.Image import Image

import torch
import torch.nn as nn
import torchvision.transforms.functional as tF
import torchvision.transforms.functional_pil as tF_pil
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.transforms import Compose, Normalize, Resize, ToTensor

from dataset import get_dataset
from nfnets import NFNet, pretrained_nfnet

# Evaluation method used in the paper
# This seems to perform slightly worse than a simple resize
class Pad32CenterCrop(nn.Module):
    def __init__(self, size:int):
        super().__init__()
        self.size = size
        self.scaled_size = (size+32, size+32)
        
    def forward(self, img:Image):
        img = tF_pil.resize(img=img, size=self.scaled_size, interpolation=PIL.Image.BICUBIC)
        return tF.center_crop(img, self.size)

def evaluate_on_imagenet(model:NFNet, dataset_dir:Path, batch_size=50, device='cuda:0'): 
    transforms = Compose([
        #Pad32CenterCrop(model.test_imsize),
        ToTensor(),
        Resize((model.test_imsize, model.test_imsize), PIL.Image.BICUBIC),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"Starting evaluation from {dataset_dir}")
    dataset = get_dataset(dataset_dir, transforms=transforms)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size, # F0: 120, F1: 100, F2: 80
        shuffle=False,
        pin_memory=False,
        num_workers=8
    )

    print(f"Validation set contains {len(dataset)} images.")

    model.to(device)
    model.eval()

    processed_imgs = 0
    correct_labels = 0
    for step, data in enumerate(dataloader):
        with torch.no_grad():
            inputs = data[0].to(device)
            targets = data[1].to(device)

            output = model(inputs).type(torch.float32)

            processed_imgs += targets.size(0)
            _, predicted = torch.max(output, 1)
            correct_labels += (predicted == targets).sum().item()

            batch_padding = int(math.log10(len(dataloader.dataset)) + 1)
            print(f"\rProcessing {processed_imgs:{batch_padding}d}/{len(dataloader.dataset)}. Accuracy: {100.0*correct_labels/processed_imgs:6.4f}", sep=' ', end='', flush=True)

    print(f"\nFinished eval. Accuracy: {100.0*correct_labels/processed_imgs:6.4f}")


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate NFNets.')
    parser.add_argument('--dataset', type=Path, help='Path to dataset root directory', required=True)
    parser.add_argument('--pretrained', type=Path, help='Path to pre-trained weights in haiku format', required=True)
    parser.add_argument('--batch-size', type=int, help='Validation batch size', default=50)
    parser.add_argument('--device', type=str, help='Validation device. Either \'cuda:0\' or \'cpu\'', default='cuda:0')
    args = parser.parse_args()

    if not args.pretrained.exists():
        raise FileNotFoundError(f"Could not find file {args.pretrained.absolute()}")

    model = pretrained_nfnet(args.pretrained)

    evaluate_on_imagenet(model, dataset_dir=args.dataset, batch_size=args.batch_size, device=args.device)
    