from pathlib import Path
from sys import path
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

class TensorboardLogger:
    def __init__(self, root_dir:Path, rank:int=0) -> None:
        self.global_step = 0
        self.rank = rank
        
        if self.rank == 0:
            run_index = 0
            while (root_dir / ('run' + str(run_index))).exists():
                run_index += 1
            
            runs_dir = root_dir / ('run' + str(run_index))
            runs_dir.mkdir(exist_ok=False, parents=True)
            
            self.dir = runs_dir
            self.writer = SummaryWriter(str(self.dir))

    def step(self):
        self.global_step += 1

    def log(self, tag:str, value):
        if self.rank == 0:
            self.writer.add_scalar(tag, value, self.global_step)

    def print(self, *values: object, sep:str=" ", end:str="\n", flush:bool=False) -> None:
        if self.rank==0:
            print(*values, sep=sep, end=end, flush=flush)