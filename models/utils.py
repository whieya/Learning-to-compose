import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import yaml
from ignite.engine import Engine
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from models.base_model import BaseModel

def _downsample(batch: dict) -> dict:
    out = batch.copy()
    out['image'] = nn.functional.interpolate(batch['image'], scale_factor=0.5)  
    # out['mask'] = nn.functional.interpolate(
    #         batch['mask'].squeeze(2), scale_factor=0.5).unsqueeze(2)  
    return out

@dataclass
class ForwardPass:
    model: nn.Module
    model_sa: nn.Module
    device: Union[torch.device, str]
    preprocess_fn: Optional[Callable] = None
    downsample: bool = False

    def __call__(self, batch: dict, use_losses: dict, eval_mode: bool, visualize_comp = False) -> Tuple[dict, dict]:
        for key in batch.keys():
            if key == 'iters':
                continue
            batch[key] = batch[key].to(self.device, non_blocking=True)
        if self.preprocess_fn is not None:
            batch = self.preprocess_fn(batch)
        if self.downsample:
            batch = _downsample(batch)

        output = self.model(batch['image'], use_losses, \
                eval_mode=eval_mode, visualize_comp=visualize_comp)
        return batch, output

class TrainCheckpointHandler:
    def __init__(
        self, checkpoint_path: Union[str, Path], device: Union[torch.device, str]
    ):
        if isinstance(checkpoint_path, str):
            checkpoint_path = Path(checkpoint_path)
        self.checkpoint_train_path = checkpoint_path / "train_checkpoint.pt"
        self.model_path = checkpoint_path / "model.pt"
        self.train_yaml_path = checkpoint_path / "train_state.yaml"
        self.device = device

    def save_checkpoint(self, state_dicts: dict, step: int = 0, config: dict = None):
        """Saves a checkpoint.

        If the state contains the key "model", the model parameters are saved
        separately to model.pt, and they are not saved to the checkpoint file.
        """
        _model_path = Path(str(self.model_path).replace('model.pt',f'model_{step}.pt'))
        _ckpt_path = Path(str(self.checkpoint_train_path).replace('train_checkpoint.pt',f'train_checkpoint_{step}.pt'))
        torch.save({
            'states' : state_dicts,
            'config' : config,
            }, _ckpt_path)


    def load_checkpoint(self, objects: dict):
        """Loads checkpoint into the provided dictionary."""

        # Load checkpoint without model
        state = torch.load(self.checkpoint_train_path, self.device)
        for varname in state:
            logging.debug(f"Loading checkpoint: variable name '{varname}'")
            try:
                objects[varname].load_state_dict(state[varname])
            except:
                print(varname)

        # Load model
        if "model" in objects:
            logging.debug(f"Loading checkpoint: model")
            model_state_dict = torch.load(self.model_path, self.device)
            objects["model"].load_state_dict(model_state_dict)


def linear_warmup_exp_decay(
    warmup_steps: Optional[int] = None,
    exp_decay_rate: Optional[float] = None,
    exp_decay_steps: Optional[int] = None,
) -> Callable[[int], float]:
    assert (exp_decay_steps is None) == (exp_decay_rate is None)
    use_exp_decay = exp_decay_rate is not None
    if warmup_steps is not None:
        assert warmup_steps > 0

    def lr_lambda(step):
        multiplier = 1.0
        if warmup_steps is not None and step < warmup_steps:
            multiplier *= step / warmup_steps
        if use_exp_decay:
            multiplier *= exp_decay_rate ** (step / exp_decay_steps)
        return multiplier

    return lr_lambda

def infer_model_type(model_name: str) -> str:
    if model_name.startswith("baseline_vae"):
        return "distributed"
    if model_name in [
        "slot-attention",
        "monet",
        "genesis",
        "space",
        "monet-big-decoder",
        "slot-attention-big-decoder",
    ]:
        return "object-centric"
    raise ValueError(f"Could not infer model type for model '{model_name}'")
