import pathlib
import torch
import yaml
import argparse
from pathlib import Path

from model import NFNet

from torch import optim
import torch.nn as nn

import torch.nn.functional as F
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.transforms import Compose, Normalize, Resize, ToTensor
from torchvision.models.resnet import resnet50
from dataset import get_dataset

def train(variant:str, num_classes:int, batch_size:int, 
    epochs:int, learning_rate:float, dataset_path:Path, device:str) -> None:
    
    model = NFNet(num_classes=num_classes, variant=variant)

    transforms = Compose([
        Resize((model.train_imsize, model.train_imsize)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = get_dataset(path=dataset_path, transforms=transforms)
    subset = Subset(dataset, range(batch_size))

    dataloader = DataLoader(
        dataset=subset, 
        batch_size=batch_size, #=batch_size
        shuffle=False,
        num_workers=0, 
        pin_memory=True)

    model.to(device)

    optimizer = optim.SGD(
        model.parameters(), 
        lr=learning_rate, 
        momentum=0.9, 
        #weight_decay=2e-5, 
        #nesterov=True
        )

    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()

        for _, data in enumerate(dataloader):
            inputs = data[0].to(device)
            targets = data[1].to(device)

            optimizer.zero_grad()

            output = model(inputs)
            loss = ce_loss(output, targets)
            loss.backward()
            optimizer.step()

            print(f"\rEpoch {epoch+1}/{epochs}: l={loss.item()}", sep=' ', end='', flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train NFNets.')
    parser.add_argument('--config', type=Path, help='Path to config.yaml', default='default_config.yaml')
    args = parser.parse_args()
    
    if not args.config.exists():
        print(f"Config file \"{args.config}\" does not exist!\n")
        exit()

    with args.config.open() as file:
        config = yaml.safe_load(file)

    train(
        variant=config['variant'],
        dataset_path=config['dataset'],
        num_classes=config['num_classes'], 
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        device=config['device'],
        )
