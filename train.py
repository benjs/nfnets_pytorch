import argparse
import math
import os
import PIL
import time
from numpy import mod
import yaml
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.transforms import Compose, Normalize, Resize, ToTensor, RandomHorizontalFlip, RandomCrop

from dataset import get_dataset
from nfnets import NFNet, SGD_AGC, optim, pretrained_nfnet, exclude_from_clipping, exclude_from_weight_decay
from utils import logging, checkpointing

def train(rank:int, config:dict) -> None:
    torch.manual_seed(0)

    runs_dir = Path('runs')
    logger = logging.TensorboardLogger(runs_dir, rank)

    checkpoints_dir = logger.dir / 'checkpoints'
    cp = checkpointing.CheckpointManager(checkpoints_dir, rank)

    if config['device'].startswith('cuda'):
        if torch.cuda.is_available():
            print(f"Using CUDA{torch.version.cuda} with cuDNN{torch.backends.cudnn.version()}.")
        else:
            raise ValueError("You specified to use cuda device, but cuda is not available.")


    if config['pretrained'] is not None:
        # There is still a overhead
        # when using this in ddp mode
        model = pretrained_nfnet(
            path=config['pretrained'], 
            stochdepth_rate=config['stochdepth_rate'],
            alpha=config['alpha'],
            activation=config['activation']
        )
    else:
        model = NFNet(
            num_classes=config['num_classes'], 
            variant=config['variant'], 
            stochdepth_rate=config['stochdepth_rate'], 
            alpha=config['alpha'],
            se_ratio=config['se_ratio'],
            activation=config['activation']
        )

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=config['world_size'])

    transforms = Compose([
        Resize((model.train_imsize, model.train_imsize), InterpolationMode.BICUBIC),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    device = rank if config['ddp'] else config['device'] 
    dataset = get_dataset(path=config['dataset'], transforms=transforms)

    if config['overfit']:
        dataset = Subset(dataset, [i*50 for i in range(0,1000)] )

    sampler = DistributedSampler(dataset, shuffle=True) if config['ddp'] else None

    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=config['batch_size'],
        shuffle=(sampler is None), # Data already shuffled with distributed sampler
        num_workers=config['num_workers'], 
        pin_memory=config['pin_memory'])

    if config['scale_lr']:
        learning_rate = config['learning_rate']*config['batch_size']/256
    else:
        learning_rate = config['learning_rate']

    if not config['do_clip']:
        config['clipping'] = None

    if config['use_fp16']:
        model.half()

    model.to(device) # "memory_format=torch.channels_last" TBD

    if(config['ddp']):
        model = DistributedDataParallel(model, device_ids=[rank])

    optimizer = SGD_AGC(
        # The optimizer needs all parameter names 
        # to filter them by hand later
        named_params=model.named_parameters(), 
        lr=learning_rate,
        momentum=config['momentum'],
        clipping=config['clipping'],
        weight_decay=config['weight_decay'], 
        nesterov=config['nesterov']
        )
    
    # Find desired parameters and exclude them 
    # from weight decay and clipping
    for group in optimizer.param_groups:
        name = group['name'] 
        
        if exclude_from_weight_decay(name):
            group['weight_decay'] = 0

        if exclude_from_clipping(name):
            group['clipping'] = None

    criterion = nn.CrossEntropyLoss()

    scaler = amp.GradScaler()
    
    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        processed_imgs = 0
        correct_labels = 0
        epoch_time = time.time()

        if config['ddp']:
            # This shuffles the data indices
            # such that every process sees 
            # different images every epoch
            sampler.set_epoch(epoch)

        for step, data in enumerate(dataloader):
            inputs = data[0].half().to(device) if config['use_fp16'] else data[0].to(device)
            targets = data[1].to(device)

            optimizer.zero_grad()

            with amp.autocast(enabled=config['amp']):
                output = model(inputs)
                loss = criterion(output, targets)
            
            # Gradient scaling
            # https://www.youtube.com/watch?v=OqCrNkjN_PM
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            processed_imgs += targets.size(0)
            _, predicted = torch.max(output, 1)
            correct_labels += (predicted == targets).sum().item()

            epoch_padding = int(math.log10(config['epochs']) + 1)
            batch_padding = int(math.log10(len(dataloader.dataset)) + 1)
            logger.print(f"\rEpoch {epoch+1:0{epoch_padding}d}/{config['epochs']}"
                f"\tImg {processed_imgs:{batch_padding}d}/{len(dataloader.dataset)}"
                f"\tLoss {running_loss / (step+1):6.4f}"
                f"\tAcc {100.0*correct_labels/processed_imgs:5.3f}%\t",
                sep=' ', end='', flush=True)

        elapsed = time.time() - epoch_time
        logger.print (f"({elapsed:.3f}s, {elapsed/len(dataloader):.3}s/step, {elapsed/len(dataset):.3}s/img)")

        logger.log('training/loss', running_loss/(step+1))
        logger.log('training/accuracy', 100.0*correct_labels/processed_imgs)
        logger.step()

        #if not config['overfit']:
        if epoch % 10 == 0 and epoch != 0:
            cp.save(epoch, model, optim)
        
    dist.destroy_process_group()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train NFNets.')
    parser.add_argument('--config', type=Path, help='Path to config.yaml', default='default_config.yaml')
    parser.add_argument('--batch-size', type=int, help='Training batch size', default=None)
    parser.add_argument('--overfit', const=True, default=False, nargs='?', help='Crop the dataset to the batch size and force model to (hopefully) overfit')
    parser.add_argument('--variant', type=str, help='NFNet variant to train', default=None)
    parser.add_argument('--pretrained', type=Path, help='Path to pre-trained weights in haiku format', default=None)
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

    config['pretrained'] = args.pretrained

    if config['ddp']:
        mp.spawn(fn=train, args=(config,), nprocs=config['world_size'], join=True)
    else:
        train(0, config=config)
