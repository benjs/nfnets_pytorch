import re
import dill
import torch
import argparse
import numpy as np
from pathlib import Path

from nfnets import NFNet

def pretrained_nfnet(path, stochdepth_rate:float=0.5, alpha:float=0.2, activation:str='gelu') -> NFNet:
    if isinstance(path, str):
        path = Path(path)
    
    with path.open('rb') as f:
        params = dill.load(f)

    layers_to_variant = {
        94: 'F0',
        178: 'F1',
        262: 'F2',
        346: 'F3',
        430: 'F4',
        514: 'F5'
    }

    if not len(params) in layers_to_variant:
        raise RuntimeError(f"Cannot load file {path.absolute()}."
                    f" File contains invalid parameter count {len(params)}!")

    model = NFNet(
        variant=layers_to_variant[len(params)],
        num_classes=1000, 
        alpha=alpha,
        stochdepth_rate=stochdepth_rate,
        se_ratio=0.5,
        activation=activation)

    state_dict = {}

    for layer_name in params:
        for param_name in params[layer_name]:
            l = layer_name
            l = l.replace("NFNet/~/", "")
            l = re.sub("(nf_block_(\d*))", r"body.\2", l)
            l = re.sub("(nf_block)", r"body.0", l)
            l = re.sub("stem_*", "stem.", l)
            l = l.replace("/~/", ".")
            
            p = str(param_name)
            p = "weight" if p == "w" else p
            p = "bias" if p == "b" else p
            
            param = params[layer_name][param_name]
            
            if len(param.shape) == 4: 
                # Conv layers, HWIO -> OIHW
                param = param.swapaxes(0,3).swapaxes(1,2).swapaxes(2,3)

            elif len(param.shape) == 2:
                # Linear layers, OI -> IO
                param = param.swapaxes(0,1)
            
            if p == 'gain':
                param = np.expand_dims(param, axis=(1,2,3))
            
            #if "conv" in l:
            #    state_dict[f"{l}.eps"] = torch.tensor(1e-4, requires_grad=False)

            with torch.no_grad():
                t = torch.from_numpy(param)
                complete_name =  f'{l}.{p}'
                if not complete_name in model.state_dict():
                    raise ValueError(
                        f"Parameter {complete_name} not found in state dict!"
                        " Please report an issue.")
                
                state_dict[complete_name] = t

    model.load_state_dict(state_dict, strict=True)
    return model

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Load haiku weights and convert them to .pth file.')  
    parser.add_argument('--pretrained', type=Path, help='Path to pre-trained weights in haiku format')
    args = parser.parse_args()

    if not args.pretrained.exists():
        raise FileNotFoundError(f"Could not find file {args.pretrained.absolute()}")

    model = from_pretrained_haiku(args.pretrained)
    
    torch.save({
        'model': model.state_dict()
    }, str(args.pretrained.with_suffix('.pth')))