import functools
import logging
import sys
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import List, Optional, Union

import torch
from ignite.engine import Engine
from ignite.engine.events import CallableEventWithFilter
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from models.base_model import MANDATORY_FIELDS
from models.nn_utils import (
    global_norm,
    grad_global_norm,
    group_global_norm,
    group_grad_global_norm,
)
from utils.utils import ExitResubmitException, filter_dict
from utils.viz import apply_color_map, make_recon_img
from einops import rearrange
from evaluation.metrics.ari import ari as ari_
from evaluation.metrics.segmentation_covering import segmentation_covering
from accelerate import Accelerator
import wandb

@dataclass
class Logger:
    working_dir: Path
    model: nn.Module
    model_sa: nn.Module
    loss_terms: List[str]
    val_loss_terms: List[str]
    scalar_params: List[str]  # names of scalar parameters in output
    param_groups: Optional[List[str]] = None
    num_images: int = 3 
    num_slots: int = 12
    accelerator: Accelerator = None
    log_method: str = 'wandb'
    cnn_downsample: int=1

    def __post_init__(self):
        if self.log_method == 'wandb':
            self.add_scalar = self.log_wandb_scalar
            self.add_image = self.log_wandb_image
            self.add_images = self.log_wandb_image

        elif self.log_method == 'tensorboard':
            if self.accelerator is None:
                self.add_scalar = self.writer.add_scalar
                self.add_image = self.writer.add_image
                self.add_images = self.writer.add_images
                self.flush = self.writer.flush
            else:
                self.add_scalar = self.accelerator.get_tracker('tensorboard').add_scalar
                self.add_image = self.accelerator.get_tracker('tensorboard').add_image
                self.add_images = self.accelerator.get_tracker('tensorboard').add_images
                self.flush = self.accelerator.get_tracker('tensorboard').flush

    @torch.no_grad()
    def log_wandb_scalar(self, key:str, value:str, step:int):
        return self.accelerator.log({key : value}, step=step)
    def log_wandb_image(self, key:str, value:str, step:int):
        return self.accelerator.log({key : wandb.Image(value.float())}, step=step)

    @torch.no_grad()
    def log_dict(self, metrics: dict, iteration_num: int, group_name: str):
        for metric_name in metrics:
            self.add_scalar(
                f"{group_name}/{metric_name}", metrics[metric_name], iteration_num
            )

    @torch.no_grad()
    def _compute_metrics(self, output: dict, step: int):
        batch, output_G = output
        true_mask = batch["mask"].cpu().argmax(dim=1)
        pred_mask = output_G["mask"].cpu().argmax(dim=1, keepdim=True).squeeze(2)

        ########### TODO##############
        self.skip_background = True
        self.num_bg_objects = 1
        self.num_ignored_objects = self.num_bg_objects if self.skip_background else 0

        if output_G["mask"].shape[1] == 1:  # not an object-centric model
            ari = mean_segcover = scaled_segcover = torch.full(
                (true_mask.shape[0],), fill_value=torch.nan
            )
        else:
            # Num background objects should be equal (for each sample) to:
            # batch["visibility"].sum([1, 2]) - batch["num_actual_objects"].squeeze(1)
            ari = ari_(true_mask, pred_mask, self.num_ignored_objects)

        # Mask shape (B, O, 1, H, W), is_foreground (B, O, 1), is_modified (B, O), where
        # O = max num objects. Expand the last 2 to (B, O, 1, 1, 1) for broadcasting.
        unsqueezed_shape = (*batch["is_foreground"].shape, 1, 1)
        is_fg = batch["is_foreground"].view(*unsqueezed_shape)
        is_modified = batch["is_modified"].view(*unsqueezed_shape)

        # Mask with foreground objects: shape (B, 1, H, W)
        fg_mask = (batch["mask"] * is_fg).sum(1)

        # Mask with unmodified foreground objects: shape (B, 1, H, W)
        unmodified_fg_mask = (batch["mask"] * is_fg * (1 - is_modified)).sum(1)

        # Return with shape (batch_size, )
        dict_metrics = dict(
                ari=ari,
        )

        return dict_metrics

    @torch.no_grad()
    def _log_metrics(self, dict_metrics: dict, step: int, prefix:str):
        self.log_dict(dict_metrics, step, f'{prefix}_metrics')

    @torch.no_grad()
    def log_img(self, img, step: int, prefix: str):
        x, gt = img
        bs = x.size(0)
        x = (x/2 + 0.5).clamp(0.0, 1.0)
        gt = (gt/2 + 0.5).clamp(0.0,1.0)

        sqrt_nrow = bs
        final_img = make_grid(torch.cat([gt,x], dim=0), nrow=sqrt_nrow)
        self.add_image(f"{prefix}-ddpm-decoding", final_img, step)
        del img, x, final_img

    @torch.no_grad()
    def log_img_comp(self, output: dict, step: int, prefix: str):
        # return
        batch, output_G = output
        x = output_G['x']
        bs = x.size(0)
        x = x/2 + 0.5
        comp = (output_G['comp_interp']/2 + 0.5)

        num_slots = comp.size(0) // (bs//2)
        comp = rearrange(comp, '(b s) c h w -> b s c h w', s=num_slots, c=3)

        img_lists = [x[:bs//2]]
        comp_lists = [comp[:,i] for i in range(num_slots)]
        img_lists = img_lists + comp_lists + [x[bs//2:]]
        stacked_img = torch.stack(img_lists, dim=1) #    

        sqrt_nrow = bs//2
        x_recon = _flatten_slots(stacked_img, nrow=1)
        self.add_image(f'{prefix}-comp-reconstruction', x_recon.clamp(0.0, 1.0), step)
        
        # # log denoised images
        # all_t = [50,200,500]
        # if f'comp_interp_denoised_{all_t[0]}' in output_G.keys():
        #     for t in all_t:
        #         denoised_comp = (output_G[f'comp_interp_denoised_{t}']/2 + 0.5)
        #         denoised_comp = rearrange(denoised_comp, '(b s) c h w -> b s c h w', s=num_slots, c=3)
        #         img_lists = [x[:bs//2]]
        #         comp_lists = [denoised_comp[:,i] for i in range(num_slots)]
        #         img_lists = img_lists + comp_lists + [x[bs//2:]]
        #         stacked_img = torch.stack(img_lists, dim=1) #    

        #         sqrt_nrow = bs//2
        #         x_recon = _flatten_slots(stacked_img, nrow=1)
        #         self.add_image(f'{prefix}_denoised/comp-denoised_{t}', x_recon.clamp(0.0, 1.0), step)
        #     del denoised_comp
        del batch, output_G, x_recon, stacked_img, img_lists, comp_lists, comp, x, output


    @torch.no_grad()
    def _flush(self):
        if self.log_method=='tensorboard':
            self.flush()

    @torch.no_grad()
    def _log_only_images(self, output: dict, step: int, prefix:str):
        n_img = self.num_images
        batch, output_G = output

        bs = batch['image'].size(0)        
        sqrt_nrow = int(sqrt(n_img))

        x_1 = batch["image"][:n_img]/2 + 0.5
        x_2 = batch['image'][bs//2:bs//2+n_img]/2 + 0.5

        pred_mask_1 = output_G["mask"][:n_img]
        pred_mask_1 = pred_mask_1.repeat_interleave(self.cnn_downsample, dim=-1).repeat_interleave(self.cnn_downsample, dim=-2)
        pred_mask_2 = output_G["mask"][bs//2:bs//2+n_img]
        pred_mask_2 = pred_mask_2.repeat_interleave(self.cnn_downsample, dim=-1).repeat_interleave(self.cnn_downsample, dim=-2)
        pred_mask_1 = (batch['image'][:n_img].unsqueeze(1) * pred_mask_1 + (1 - pred_mask_1))/2 + 0.5 
        pred_mask_2 = (batch['image'][bs//2:bs//2 + n_img].unsqueeze(1) * pred_mask_2 + (1 - pred_mask_2))/2 + 0.5

        pred_mask = torch.cat([pred_mask_1, pred_mask_2], dim=0)
        pred_mask = _flatten_slots(pred_mask, sqrt_nrow)
        pred_mask = make_grid(pred_mask, nrow=sqrt_nrow)
        self.add_image(f"{prefix}-mask: pred", pred_mask, step)

        # decoder mask
        if 'dec_mask' in output_G.keys():
            dec_mask = output_G['dec_mask'].transpose(-1,-2)
            # TODO : hard coded 16 should be changed
            dec_mask = dec_mask.reshape(bs, self.num_slots, 1, 16, 16).repeat_interleave(8, dim=-2).repeat_interleave(8, dim=-1)
            dec_mask_1 = dec_mask[:n_img]
            dec_mask_2 = dec_mask[bs//2:bs//2+n_img]
            dec_mask_1 = (batch['image'][:n_img].unsqueeze(1) * dec_mask_1 + (1 - dec_mask_1))/2 + 0.5 
            dec_mask_2 = (batch['image'][bs//2:bs//2 + n_img].unsqueeze(1) * dec_mask_2 + (1 - dec_mask_2))/2 + 0.5

            dec_mask = torch.cat([dec_mask_1, dec_mask_2], dim=0)
            dec_mask = _flatten_slots(dec_mask, sqrt_nrow)
            dec_mask = make_grid(dec_mask, nrow=sqrt_nrow)
            self.add_image(f"{prefix}-dec_mask: pred", dec_mask, step)
            del dec_mask, dec_mask_1, dec_mask_2

        del batch, output, output_G, x_1, x_2 # stacked_img, x_recon
        del pred_mask_1, pred_mask_2, pred_mask


    @torch.no_grad()
    def _log_images(self, output: dict, step: int, prefix:str):
        n_img = self.num_images
        batch, output_G = output

        bs = batch['image'].size(0)       

        x_1 = batch["image"][:n_img]/2 + 0.5
        x_2 = batch['image'][bs//2:bs//2+n_img]/2 + 0.5

        print(output_G.keys())

        sqrt_nrow = int(sqrt(n_img))
    
        flat_mask = _flatten_slots(batch["mask"][:n_img], sqrt_nrow)
        mask = make_grid(flat_mask, nrow=sqrt_nrow).float()
        self.add_image(f"{prefix}-mask: true", mask, step)

        pred_mask_1 = output_G["mask"][:n_img].repeat_interleave(self.cnn_downsample, dim=-1).repeat_interleave(self.cnn_downsample, dim=-2)
        pred_mask_2 = output_G["mask"][bs//2:bs//2+n_img].repeat_interleave(self.cnn_downsample, dim=-1).repeat_interleave(self.cnn_downsample, dim=-2)
        pred_mask_1 = (batch['image'][:n_img].unsqueeze(1) * pred_mask_1 + (1 - pred_mask_1))/2 + 0.5 
        pred_mask_2 = (batch['image'][bs//2:bs//2 + n_img].unsqueeze(1) * pred_mask_2 + (1 - pred_mask_2))/2 + 0.5

        pred_mask = torch.cat([pred_mask_1, pred_mask_2], dim=0)
        pred_mask = _flatten_slots(pred_mask, sqrt_nrow)
        pred_mask = make_grid(pred_mask, nrow=sqrt_nrow)
        self.add_image(f"{prefix}-mask: pred", pred_mask, step)

        if 'dec_mask' in output_G.keys():
            # decoder mask
            dec_mask = output_G['dec_mask'].transpose(-1,-2)
            # TODO : hard coded 16 should be changed
            height=width=int(dec_mask.shape[-1]**0.5)
            dec_mask = dec_mask.reshape(bs, self.num_slots, 1, height, width).repeat_interleave(8, dim=-2).repeat_interleave(8, dim=-1)
            dec_mask_1 = dec_mask[:n_img]
            dec_mask_2 = dec_mask[bs//2:bs//2+n_img]
            dec_mask_1 = (batch['image'][:n_img].unsqueeze(1) * dec_mask_1 + (1 - dec_mask_1))/2 + 0.5 
            dec_mask_2 = (batch['image'][bs//2:bs//2 + n_img].unsqueeze(1) * dec_mask_2 + (1 - dec_mask_2))/2 + 0.5

            dec_mask = torch.cat([dec_mask_1, dec_mask_2], dim=0)
            dec_mask = _flatten_slots(dec_mask, sqrt_nrow)
            dec_mask = make_grid(dec_mask, nrow=sqrt_nrow)
            self.add_image(f"{prefix}-dec_mask: pred", dec_mask, step)
            del dec_mask, dec_mask_1, dec_mask_2

        if prefix=='val':
            mask_segmap, pred_mask_segmap = _compute_segmentation_mask(batch, n_img, output_G)
            self.add_images(f"{prefix}-segmentation: true", mask_segmap, step)
            self.add_images(f"{prefix}-segmentation: pred", pred_mask_segmap, step)
            del mask_segmap, pred_mask_segmap 
            
        del batch, output, output_G, x_1, x_2 # stacked_img, x_recon
        del pred_mask_1, pred_mask_2, pred_mask



    @torch.no_grad()
    def _log_train_losses(self, output, epoch, prefix):
        batch, output_G = output
        self.log_dict(
            filter_dict(output_G, allow_list=self.loss_terms, inplace=False),
            epoch,
            prefix,
        )
        del batch, output_G, output

    @torch.no_grad()
    def _log_scalar_params(self, output, epoch):
        batch, output_G = output
        self.log_dict(
            filter_dict(output_G, allow_list=self.scalar_params, inplace=False),
            epoch,
            "model params",
        )

    @torch.no_grad()
    def _log_stats(self, output, step):
        batch, output_G = output
        output = output_G
        for metric_name in output:
            if metric_name in self.loss_terms:  # already logged in _log_train_losses()
                continue
            if (
                metric_name in self.scalar_params
            ):  # already logged in _log_scalar_params()
                continue
            prefix = "model outputs"
            if metric_name not in MANDATORY_FIELDS:
                prefix += f" ({self.model.name})"
            self._log_tensor(f"{prefix}/{metric_name}", output[metric_name], step)
        del batch, output_G, output

    @torch.no_grad()
    def _log_params(self, step):
        """Logs the global norm of all parameters and of their gradients."""
        self.add_scalar(
            "param grad norms/model",
            grad_global_norm(self.model.parameters()),
            step,
        )
        self.add_scalar(
            "param norms/model",
            global_norm(self.model.parameters()),
            step,
        )

        self.add_scalar(
            "param grad norms/model_sa",
            grad_global_norm(self.model_sa.parameters()),
            step,
        )
        self.add_scalar(
            "param norms/model_sa",
            global_norm(self.model_sa.parameters()),
            step,
        )

    @torch.no_grad()
    def _log_grouped_params(self, engine):
        """Logs the global norm of parameters and their gradients, by group."""
        if self.param_groups is None:
            return
        assert isinstance(self.param_groups, list)
        for name in self.param_groups:
            self.add_scalar(
                f"param grad norms/group: {name}",
                group_grad_global_norm(self.model, name),
                engine.state.iteration,
            )
            self.add_scalar(
                f"param norms/group: {name}",
                group_global_norm(self.model, name),
                engine.state.iteration,
            )

    @torch.no_grad()
    def _log_tensor(self, name, tensor, step):
        if not isinstance(tensor, Tensor):
            return
        if tensor.numel() == 1:
            stats = ["item"]
        else:
            stats = ["min", "max", "mean"]
        for stat in stats:
            value = getattr(tensor, stat)()
            if stat == "item":
                name_ = name
            else:
                name_ = f"{name} [{stat}]"
            self.add_scalar(name_, value, step)


def _compute_segmentation_mask(batch, num_images, output):
    # [bs, ns, 1, H, W] to [bs, 1, H, W]
    mask_segmap = batch["mask"][:num_images].argmax(1)

    # [bs, ns, 1, H, W] to [bs, 1, H, W]
    pred_mask_segmap = output["mask"][:num_images].argmax(1)

    # If shape is [bs, H, W], turn it into [bs, 1, H, W]
    if mask_segmap.shape[1] != 1:
        mask_segmap = mask_segmap.unsqueeze(1)

    # If shape is [bs, H, W], turn it into [bs, 1, H, W]
    if pred_mask_segmap.shape[1] != 1:
        pred_mask_segmap = pred_mask_segmap.unsqueeze(1)

    mask_segmap = apply_color_map(mask_segmap)
    pred_mask_segmap = apply_color_map(pred_mask_segmap)
    return mask_segmap, pred_mask_segmap


def _flatten_slots(images: Tensor, nrow: int):
    image_lst = images.split(1, dim=0)
    image_lst = [
        make_grid(image.squeeze(0), nrow=images.shape[1]) for image in image_lst
    ]
    images = torch.stack(image_lst, dim=0)
    pad_value = 255 if isinstance(images, torch.LongTensor) else 1.0
    return make_grid(images, nrow=nrow, pad_value=pad_value, padding=4)

def _flatten_slots_masks(img1, img2, img3, nrow: int):
    imgs = [img1, img2, img3]
    result_list = []
    
    img_lst1 = img1.split(1, dim=0)
    img_lst2 = img2.split(1, dim=0)
    img_lst3 = img3.split(1, dim=0)
    image_lst = []
    for i1, i2, i3 in zip(img_lst1, img_lst2, img_lst3):
        image = torch.cat((i1,i2,i3), dim=1)
        image_lst.append(
            make_grid(image.squeeze(0), nrow=i1.shape[1]))
    images = torch.stack(image_lst, dim=0)
    pad_value = 255 if isinstance(images, torch.LongTensor) else 1.0
    return make_grid(images, nrow=nrow, pad_value=pad_value, padding=4)


def log_tensor_stats(tensor: Tensor, name: str, prefix: str = ""):
    """Logs stats of a tensor."""
    if tensor.numel() == 0:
        return
    logging.debug(f"{prefix}{name} stats:")
    for stat_name in ["min", "max", "mean", "std"]:
        try:
            stat_value = getattr(tensor, stat_name)()
        except RuntimeError as e:
            logging.warning(f"Could not log stat '{stat_name}' of tensor '{name}': {e}")
        else:
            logging.debug(f"   {stat_name}: {stat_value}")
    logging.debug(f"   {name}.shape: {tensor.shape}")


def log_dict_stats(d: dict, prefix: str = ""):
    """Logs stats of tensors in a dict."""
    for k in d:
        if isinstance(d[k], Tensor):
            log_tensor_stats(d[k].float(), k, prefix=prefix)


def log_engine_stats(engine: Engine):
    """Logs stats of all tensors in an engine's state (inputs and outputs)."""
    batch, output = engine.state.output
    log_dict_stats(batch, "[input] ")
    log_dict_stats(output, "[output] ")



class PaddingFilter(logging.Filter):
    def __init__(self, pad_len: int = 8, pad_char: str = " "):
        super().__init__()
        assert len(pad_char) == 1
        self.pad_len = pad_len
        self.pad_char = pad_char

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str) and "\n" in record.msg:
            parts = record.msg.split("\n")
            padding = self.pad_char * self.pad_len
            record.msg = f"\n{padding}".join(parts)
        return super().filter(record)


class IgniteFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str) and "terminating due to exception" in record.msg:
            return False
        return super().filter(record)


def filter_ignite_logging():
    engine_logger = logging.getLogger("ignite.engine.engine.Engine")
    engine_logger.addFilter(IgniteFilter())
    engine_logger.setLevel(logging.WARNING)  # Lower this for debugging


def set_logger(
    level: Optional[Union[int, str]] = logging.INFO,
    log_dir: Optional[Path] = None,
    log_fname: Optional[str] = None,
    capture_warnings: bool = True,
):
    """Sets up the default logger.

    Args:
        level: logging level.
        log_dir: log directory. If None (default), it defaults to `${CWD}/logs`.
        log_fname: log file name. If None (default), logging to file is disabled.
        capture_warnings: captures UserWarnings from the warnings package.
    """

    logging.captureWarnings(capture_warnings)

    def formatting_wrapper(format_):
        return f"[{format_}] %(message)s"

    prefix = "%(levelname)s:%(filename)s:%(lineno)s"
    logging.basicConfig(
        format=formatting_wrapper(prefix),
        level=level,
    )

    logging.root.addFilter(PaddingFilter())  # `logging.root` is the default logger

    # Skip logging to file.
    if log_fname is None:
        logging.info("Completed default logger setup. Logging to file disabled.")
        return

    # Else, setup logging to file below.
    if log_dir is None:
        log_dir = Path.cwd() / "logs"
    if not log_dir.exists():
        logging.info(f"Required log dir {log_dir} does not exist: will be created")
        log_dir.mkdir(parents=True)
    elif not log_dir.is_dir():  # exists but is a file
        raise FileExistsError(
            f"Required log dir {log_dir} exists and is not a directory."
        )
    log_path = log_dir / log_fname
    formatter = logging.Formatter(fmt=formatting_wrapper("%(asctime)s " + prefix))
    # If the file exists and is not empty, append a separator to denote different runs.
    if log_path.exists() and log_path.stat().st_size > 0:
        with open(log_path, "a") as fh:
            fh.write("\n======\n\n")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logging.root.addHandler(fh)  # `logging.root` is the default logger
    logging.info(f"Completed default logger setup. Logging to file: {log_path}")


def setup_logging(
    level: Optional[Union[int, str]] = logging.INFO,
    log_dir: Optional[Path] = None,
    log_fname: Optional[str] = None,
):
    """Sets up the default logger and silences most ignite logging.

    Args:
        level: logging level.
        log_dir: log directory. If None (default), it defaults to `${CWD}/logs`.
        log_fname: log file name. If None (default), logging to file is disabled.
    """
    set_logger(level, log_dir, log_fname)
    filter_ignite_logging()


def logging_wrapper(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            fn(*args, **kwargs)
        except ExitResubmitException:
            # Customize this depending on the job scheduler. E.g., this works for HTCondor.
            sys.exit(3)
        except BaseException as e:
            logging.exception(e)
            raise e

    return wrapper
