from dataclasses import dataclass
from math import sqrt
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from models.base_model import BaseModel
from models.base_trainer import BaseTrainer
from models.nn_utils import init_xavier_
from models.utils import linear_warmup_exp_decay


@dataclass
class SlotAttentionTrainer(BaseTrainer):

    use_exp_decay: bool
    exp_decay_rate: Optional[float]
    exp_decay_steps: Optional[int]
    use_warmup_lr: bool
    warmup_steps: Optional[int]

    # lambdas
    lambda_composition : float = 0.0
    lambda_oneshot : float = 0.0
    lambda_mask_reg : float = 0.0
    lambda_slot_diffusion : float = 0.0

    @property
    def loss_terms(self) -> List[str]:
        return [
                "loss_oneshot", 
                "loss_composition", 
                "loss_slot_diffusion",
                "loss_mask_reg",
                "lr",
                ]

    @property
    def val_loss_terms(self) -> List[str]:
        return ["loss"]

    @property
    def param_groups(self) -> List[str]:
        return ["encoder", "decoder", "slot_attention"]

    def _post_init(self, model: nn.Module, model_sa: nn.Module, dataloaders: List[DataLoader]):
        super()._post_init(model, model_sa, dataloaders)

    @torch.no_grad()
    def eval_one_step(self, batch: dict) -> Tuple[dict, Tuple[dict, dict]]:
        # loss coefficients
        lambda_composition = self.lambda_composition
        lambda_oneshot = self.lambda_oneshot
        lambda_mask_reg = self.lambda_mask_reg
        lambda_slot_diffusion = self.lambda_slot_diffusion

        use_losses = {
            'use_loss_composition' : lambda_composition != 0.0,
            'use_loss_oneshot' : lambda_oneshot != 0.0,
            'use_loss_mask_reg' : lambda_mask_reg != 0.0,
            'use_slot_diffusion' : lambda_slot_diffusion != 0.0,
        }

        batch, output = self.eval_step(batch, use_losses, eval_mode=True)
        return batch, output


    def train_one_step(self, batch: dict, visualize_comp: bool=False, steps: int=0) -> Tuple[dict, Tuple[dict, dict]]:

        optimizer = self.optimizers[0]
        input_batch = batch

        # loss coefficients
        lambda_composition = self.lambda_composition
        lambda_oneshot = self.lambda_oneshot
        lambda_mask_reg = self.lambda_mask_reg
        lambda_slot_diffusion = self.lambda_slot_diffusion

        use_losses = {
            'use_loss_composition' : lambda_composition != 0.0,
            'use_loss_oneshot' : lambda_oneshot != 0.0,
            'use_loss_mask_reg' : lambda_mask_reg != 0.0,
            'use_slot_diffusion' : lambda_slot_diffusion != 0.0,
        }

        batch, output = self.eval_step(input_batch, use_losses, eval_mode=False, visualize_comp=visualize_comp)

        # final losses
        loss_composition = output['loss_composition']
        loss_oneshot = output['loss_oneshot']
        loss_slot_diffusion = output['loss_slot_diffusion']
        loss_mask_reg = output['loss_mask_reg']

        total_loss = lambda_composition * loss_composition +\
                lambda_oneshot * loss_oneshot +\
                lambda_slot_diffusion * loss_slot_diffusion +\
                lambda_mask_reg * loss_mask_reg

        # train 
        optimizer.zero_grad()
        if self.config.use_accel:
            self.accelerator.backward(total_loss)
        else:
            total_loss.backward()
        optimizer.step()

        # lr scheduler
        self.lr_schedulers[0].step()
        
        all_lrs = self.lr_schedulers[0].get_lr()
        output['lr'] = all_lrs[0]
        return batch, output



