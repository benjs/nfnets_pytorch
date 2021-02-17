import pathlib
from torch.tensor import Tensor
import yaml
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.transforms import Compose, Normalize, Resize, ToTensor

import optim
from model import NFNet
from dataset import get_dataset

def train(variant:str, num_classes:int, batch_size:int, epochs:int, dataset_path:Path, 
    learning_rate:float=0.1, device:str='cpu', stochdepth_rate:float=None, alpha:float=0.2,
    group_size:int=1, se_ratio:float=0.5, scale_lr:bool=True, momentum:float=0.9,
    weight_decay:float=2e-5, nesterov:bool=True, clipping:float=0.1, num_workers:int=0,
    overfit:bool=False) -> None:
    
    model = NFNet(
        num_classes=num_classes, 
        variant=variant, 
        stochdepth_rate=stochdepth_rate, 
        alpha=alpha,
        se_ratio=se_ratio,
        groups=group_size,
        )

    # Use FP16 operations to fasten training
    model.half()

    transforms = Compose([
        Resize((model.train_imsize, model.train_imsize)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = get_dataset(path=dataset_path, transforms=transforms)
    
    if overfit:
        dataset = Subset(dataset, range(batch_size))

    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=True)

    model.to(device)

    if scale_lr:
        learning_rate = learning_rate*batch_size/256

    optimizer = optim.SGD_AGC(
        model.parameters(), 
        lr=learning_rate*batch_size/256 if scale_lr else learning_rate, 
        momentum=momentum,
        clipping=clipping,
        weight_decay=weight_decay, 
        nesterov=nesterov
        )

    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()

        for step, data in enumerate(dataloader):
            inputs = data[0].half().to(device)
            targets = data[1].to(device)

            optimizer.zero_grad()

            output = model(inputs).type(torch.float32)
            loss = ce_loss(output, targets)
            loss.backward()
            optimizer.step()

            print(f"\rEpoch {epoch+1:04d}/{epochs}\tImg: {step*batch_size:5d}/{len(dataloader.dataset)}\tLoss: {loss.item():8.6f}                 ", 
                sep=' ', end='', flush=True)

        if not overfit:
            cp_dir = Path("checkpoints")
            cp_dir.mkdir(exist_ok=True)

            cp_path = cp_dir / ("checkpoint_epoch" + str(epoch))

            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'loss': loss
            }, str(cp_path))

            print(f"Saved checkpoint to {str(cp_path)}")

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train NFNets.')
    parser.add_argument('--config', type=Path, help='Path to config.yaml', default='default_config.yaml')
    parser.add_argument('--batch-size', type=int, help='Training batch size', default=None)
    parser.add_argument('--overfit', const=True, default=False, nargs='?', help='Crop the dataset to the batch size and force model to (hopefully) overfit')
    parser.add_argument('--variant', type=str, help='NFNet variant to train', default=None)
    args = parser.parse_args()
    
    if not args.config.exists():
        print(f"Config file \"{args.config}\" does not exist!\n")
        exit()

    with args.config.open() as file:
        config = yaml.safe_load(file)

    # Override config.yaml settings with command line settings
    for arg in vars(args):
        if getattr(args, arg) is not None and arg in config:
            config[arg] = getattr(args, arg)

    train(
        variant=config['variant'] if args.variant is None else args.variant,
        dataset_path=config['dataset'],
        num_classes=config['num_classes'], 
        batch_size=config['batch_size'] if args.batch_size is None else args.batch_size,
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        device=config['device'],
        alpha=config['alpha'],
        stochdepth_rate=config['stochdepth_rate'],
        se_ratio=config['se_ratio'],
        group_size=config['groups'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay'],
        nesterov=config['nesterov'],
        clipping=config['clipping'],
        num_workers=config['num_workers'],
        overfit=args.overfit
        )
