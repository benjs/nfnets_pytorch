import argparse
import math
import PIL
import time
import yaml
from pathlib import Path
from PIL.Image import Image

import matplotlib.pyplot as plt
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.transforms import Compose, Normalize, Resize, ToTensor, RandomHorizontalFlip, RandomCrop

from dataset import get_dataset
from nfnets import NFNet, SGD_AGC, pretrained_nfnet

def train(config:dict) -> None:
    if config['device'].startswith('cuda'):
        if torch.cuda.is_available():
            print(f"Using CUDA{torch.version.cuda} with cuDNN{torch.backends.cudnn.version()}")
        else:
            raise ValueError("You specified to use cuda device, but cuda is not available.")
    
    if config['pretrained'] is not None:
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

    transforms = Compose([
        RandomHorizontalFlip(),
        Resize((model.train_imsize, model.train_imsize), PIL.Image.BICUBIC),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    device = config['device']
    dataset = get_dataset(path=config['dataset'], transforms=transforms)
    
    if config['overfit']:
        dataset = Subset(dataset, [i*50 for i in range(0,1000)] )

    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
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
        
        if model.exclude_from_weight_decay(name):
            group['weight_decay'] = 0

        if model.exclude_from_clipping(name):
            group['clipping'] = None

    criterion = nn.CrossEntropyLoss()

    runs_dir = Path('runs')
    run_index = 0
    while (runs_dir / ('run' + str(run_index))).exists():
        run_index += 1
    runs_dir = runs_dir / ('run' + str(run_index))
    runs_dir.mkdir(exist_ok=False, parents=True)
    checkpoints_dir = runs_dir / 'checkpoints'
    checkpoints_dir.mkdir()

    writer = SummaryWriter(str(runs_dir))
    scaler = amp.GradScaler()

    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        processed_imgs = 0
        correct_labels = 0
        epoch_time = time.time()

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
            print(f"\rEpoch {epoch+1:0{epoch_padding}d}/{config['epochs']}"
                f"\tImg {processed_imgs:{batch_padding}d}/{len(dataloader.dataset)}"
                f"\tLoss {running_loss / (step+1):6.4f}"
                f"\tAcc {100.0*correct_labels/processed_imgs:5.3f}%\t",
            sep=' ', end='', flush=True)

        elapsed = time.time() - epoch_time
        print (f"({elapsed:.3f}s, {elapsed/len(dataloader):.3}s/step, {elapsed/len(dataset):.3}s/img)")

        global_step = epoch*len(dataloader) + step
        writer.add_scalar('training/loss', running_loss/(step+1), global_step)
        writer.add_scalar('training/accuracy', 100.0*correct_labels/processed_imgs, global_step)

        #if not config['overfit']:
        if epoch % 10 == 0 and epoch != 0:
            cp_path = checkpoints_dir / ("checkpoint_epoch" + str(epoch+1) + ".pth")

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

    train(config=config)
