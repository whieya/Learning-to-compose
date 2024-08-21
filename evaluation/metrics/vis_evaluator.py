import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from torch.utils.data import DataLoader

from data.datasets import MultiObjectDataset
from evaluation.metrics.ari import ari as ari_
from evaluation.metrics.segmentation_covering import segmentation_covering
from models.base_model import BaseModel
from models.utils import ForwardPass, ForwardPass_Compose
from utils.utils import dict_tensor_mean
from utils.viz import make_recon_img

_DEFAULT_METRICS = [
    "ari",
    "mean_segcover",
    "scaled_segcover",
    "mse",
    "mse_unmodified_fg",
    "mse_fg",
]


@dataclass
class CompositionVisualizer:
    dataloader: DataLoader
    device: str
    num_slots : int
    num_imgs : int
    _forward_pass: ForwardPass = field(init=False)

    def __post_init__(self):
        # This should be a MultiObjectDataset.
        dataset: MultiObjectDataset = self.dataloader.dataset  # type: ignore

    @torch.no_grad()
    def eval(
            self, model: nn.Module, model_sa: nn.Module , steps: Optional[int] = None
    ) -> Tuple[Dict[str, float], Dict[str, float]]:

        self._forward_pass = ForwardPass_Compose(model, model_sa, self.device)
        for idx, batch in enumerate(self.dataloader):
            if idx > 0:
                break

            batch, output = self._forward_pass(batch)
        return output

