import yaml
import argparse
import math
from pathlib import Path
from pretrained import from_pretrained_haiku

import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.transforms import Compose, Normalize, Resize, ToTensor

import optim
from model import NFNet
from dataset import get_dataset

def train(config:dict) -> None:
    
    if config['pretrained'] is not None:
        model = from_pretrained_haiku(
            config['pretrained'], 
            stochdepth_rate=config['stochdepth_rate'],
            alpha=config['alpha']
            )
    else:
        model = NFNet(
            num_classes=config['num_classes'], 
            variant=config['variant'], 
            stochdepth_rate=config['stochdepth_rate'], 
            alpha=config['alpha'],
            se_ratio=config['se_ratio']
            )

    transforms = Compose([
        Resize((model.train_imsize, model.train_imsize)),
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

    model.to(device)

    optimizer = optim.SGD_AGC(
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

    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        processed_imgs = 0
        correct_labels = 0

        for step, data in enumerate(dataloader):
            inputs = data[0].half().to(device) if config['use_fp16'] else data[0].to(device)
            targets = data[1].to(device)

            optimizer.zero_grad()

            output = model(inputs).type(torch.float32)

            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

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

        global_step = epoch*len(dataloader) + step
        writer.add_scalar('training/loss', running_loss/(step+1), global_step)
        writer.add_scalar('training/accuracy', 100.0*correct_labels/processed_imgs, global_step)

        #if not config['overfit']:
        if epoch % 10 == 0:
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
