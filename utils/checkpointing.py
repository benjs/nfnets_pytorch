import torch
import torch.nn as nn
from pathlib import Path
from torch.optim import Optimizer

class CheckpointManager:
    def __init__(self, dir:Path, rank=0) -> None:
        self.dir = dir
        self.rank = rank
        if rank == 0:
            self.dir.mkdir()

    def save(self, epoch:int, model:nn.Module, optimizer:Optimizer) -> None:
        if self.rank == 0:
            cp_path = self.dir / ("epoch" + str(epoch+1) + ".pth")

            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optim': optimizer.state_dict()
            }, str(cp_path))

            print(f"Saved checkpoint to {str(cp_path)}")
