import time
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm
import os

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from data.datasets import MultiObjectDataset
from evaluation.metrics.metrics_evaluator import MetricsEvaluator
from models.utils import ForwardPass, TrainCheckpointHandler, infer_model_type
from utils.logging import Logger, log_engine_stats
from utils.utils import ExitResubmitException, SkipTrainingException
from models.nn_utils import summary_num_params
from models.utils import linear_warmup_exp_decay
from torch.optim.lr_scheduler import LambdaLR

from torch.distributions import Normal
from torch.optim.optimizer import Optimizer, required
from utils.utils import ExitResubmitException, filter_dict
from accelerate import Accelerator
from evaluation.metrics.ari import ari as ari_
from evaluation.metrics.iou import *


@dataclass
class BaseTrainer:
    config: dict
    device: str
    steps: int
    clip_grad_norm: Optional[float]
    debug: bool
    working_dir: Path
    num_slots : int
 
    model: nn.Module = field(init=False)
    model_sa: nn.Module = field(init=False)
    dataloaders: List[DataLoader] = field(init=False)
    optimizers: List[Optimizer] = field(init=False)
    evaluator: MetricsEvaluator = field(init=False)
    eval_step: ForwardPass = field(init=False)
    checkpoint_handler: TrainCheckpointHandler = field(init=False)
    lr_schedulers: List[_LRScheduler] = field(init=False)  # optional schedulers

    def __post_init__(self):
        pass  

    def _make_optimizers(self, **kwargs):
        """Makes default optimizer on all model parameters.

        Called at the end of `_post_init()`. Override to customize.
        """
        alg = kwargs.pop("alg")  # In this base implementation, alg is required.
        opt_class = getattr(torch.optim, alg)
        
        # Group the parameters into two groups
        params_group = [
            {"params": self.model.model_phi.parameters(), "lr": self.config.lr_lsd},
            {"params": self.model.phi_slot_proj.parameters(), "lr": self.config.lr_lsd},
            {"params": self.model_sa.parameters(), "lr": self.config.lr_sa},
        ]

        self.optimizers = [opt_class(params_group)]

    def _setup_lr_scheduling(self):
        """Registers hook that steps all LR schedulers at each iteration.

        Called at the beginning of `_setup_training(). Override to customize.
        """

    def lr_scheduler_step(self):
        for scheduler in self.lr_schedulers:
            scheduler.step()

    @property
    def scalar_params(self) -> List[str]:
        """List of scalar model parameters that should be logged.

        They must be in the model's output dictionary. Empty list by default.
        """
        return []

    @property
    @abstractmethod
    def loss_terms(self) -> List[str]:
        ...

    @property
    @abstractmethod
    def val_loss_terms(self) -> List[str]:
        ...

    @property
    def param_groups(self) -> List[str]:
        """Parameter groups whose norm and gradient norm will be logged separately to tensorboard."""
        return []

    def setup(
        self,
        model: nn.Module,
        model_sa: nn.Module,
        dataloaders: List[DataLoader],
        load_ckpt_path: str = None,
    ):
        self._post_init(model, model_sa, dataloaders)
        self._setup_training(load_ckpt_path)

    def _post_init(self, model: nn.Module, model_sa: nn.Module, 
            dataloaders: List[DataLoader]):
        """Adds model and dataloaders to the trainer.

        Overriding methods should call this base method first.

        This method adds model and dataloaders to the Trainer object. It creates
        an evaluation step, the optimizer, and sets up tensorboard, but does not
        create a trainer engine. Anything that goes in the checkpoints must be
        created here. Anything that requires a trainer (e.g. callbacks) must be
        defined in `_setup_training()`.
        """
        assert model.training is True  # don't silently set it to train
        self.model = model
        self.model_sa = model_sa
        self.dataloaders = dataloaders

        training_dataset: MultiObjectDataset = self.dataloaders[0].dataset  # type: ignore

        # set up logger
        logging_dir = (
            self.working_dir
            / 'logs'
            / f"{self.config.exp_name}"
        )

        # set up model checkpointing
        model_save_dir = (
            self.working_dir
            / 'ckpts'
            / f"{self.config.exp_name}"
        )

        self.model_save_dir = model_save_dir
        if self.config.use_accel:
            self.accelerator = Accelerator(
                    log_with=self.config.log_method,
                    project_dir=model_save_dir,
                    )

            self.device = self.accelerator.device
        else:
            self.device = self.config.device

        self.model = self.model.to(self.device)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            run = os.path.split(__file__)[-1].split(".")[0]

            # use wandb
            if self.config.log_method == 'wandb':
                # please change entity here?!?jedi=0, ?!?   (*_*project_name: str*_*, config: Optional[dict]=None, init_kwargs: Optional[dict]={}) ?!?jedi?!?
                self.accelerator.init_trackers(
                        project_name = 'compSA_release',
                        config = self.config,
                        init_kwargs={'wandb':{'name':f'{self.config.exp_name}', 
                            'entity':'whieya'}}
                        )
            else:
                self.accelerator.init_trackers(run)

            self.logger = Logger(
                logging_dir,
                model,
                model_sa,
                loss_terms=self.loss_terms,
                val_loss_terms=['loss'],
                scalar_params=self.scalar_params,
                param_groups=self.param_groups,
                num_images=self.config.log_n_imgs,
                num_slots=self.num_slots,
                accelerator=self.accelerator,
                log_method=self.config.log_method,
                cnn_downsample=self.config.cnn_downsample
            )
            self.logger.__post_init__()

            summary_string, num_params = summary_num_params(self.model, max_depth=10)
            summary_string, num_params_sa = summary_num_params(self.model_sa, max_depth=10)
            self.logger.add_scalar("num. parameters", num_params, 0)
            self.logger.add_scalar("num. parameters disc", num_params_sa, 0)
    

        os.makedirs(model_save_dir, exist_ok=True)
        self.checkpoint_handler = TrainCheckpointHandler(model_save_dir, self.device)
        self.lr_schedulers = []  # No scheduler by default - subclasses append to this.

        # Here we only do training and validation.
        if len(self.dataloaders) < 2:
            raise ValueError("At least 2 dataloaders required (train and validation)")

        self.training_dataloader = self.dataloaders[0]
        self.validation_dataloader = self.dataloaders[1]

        # Make the optimizers here because we need to save them in the checkpoints.
        optim_config = {
                'alg' : 'Adam',
                }
        self._make_optimizers(**optim_config)

        # Set unused variables to None, necessary for LR scheduler setup
        if not self.config.use_exp_decay:
            self.exp_decay_steps = self.exp_decay_rate = None
        if not self.config.use_warmup_lr:
            self.warmup_steps = None

        lr_scheduler = LambdaLR(
            self.optimizers[0],
            lr_lambda=linear_warmup_exp_decay(
                self.warmup_steps, self.exp_decay_rate, self.exp_decay_steps
            ),
        )
        self.lr_schedulers.append(lr_scheduler)


        if self.config.use_accel:
            # it supports multi-gpu setting
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.model, self.optimizers[0], self.training_dataloader, self.validation_dataloader, self.lr_schedulers[0] =\
                    self.accelerator.prepare(model, self.optimizers[0], self.training_dataloader, self.validation_dataloader,
                            self.lr_schedulers[0])

        self.eval_step = ForwardPass(self.model, self.model_sa, self.device)

    def _setup_training(self, load_ckpt_path: str = None):
        """Completes the setup of the trainer.

        Overriding methods should call this base method first.

        Args:
            load_checkpoint: Whether a checkpoint should be loaded.
        """

        # Add to the trainer the hooks to step the schedulers. By default, all
        # schedulers are stepped at each training iteration.
        self._setup_lr_scheduling()

        self.load_ckpt_path = load_ckpt_path
        if load_ckpt_path is not None:
            self.load_checkpoint(load_ckpt_path)
            print(f"Restored checkpoint from {load_ckpt_path}")

        # Force state to avoid error in case we change number of training steps.
        self.epoch_length = self.steps

        # Setting epoch to 0 is necessary because, if the previous run was completed,
        # the current state has epoch=1 so training will not start.
        self.epoch = 0

        # # Initial training iteration (maybe after loading checkpoint)
        #iter_start = self.trainer.state.iteration
        iter_start = 0
        print(f"Current training iteration: {iter_start}")

    @torch.no_grad()
    def vis_comp(self, batch, global_steps):

        torch.cuda.empty_cache()
        self.model.set_eval_mode()
        self.model_sa.eval()

        output = self.eval_one_step(batch)
        self.logger._log_images(output, global_steps)
        self.logger._log_train_losses(output, global_steps)
        self.logger.log_img_comp(output, global_steps)
        self.model.set_train_mode()
        self.model_sa.eval()

        torch.cuda.empty_cache()
        del output


    @torch.no_grad()
    def compute_metrics(self, outputs: dict, step: int, compute_miou:bool):
        batch, output = outputs
        true_mask = batch["mask"].argmax(dim=1)

        # rescale the output mask with cnn_downsampling factor
        if self.config.cnn_downsample > 1:
            true_mask = F.interpolate(true_mask.float(), 
                    scale_factor=1 / self.config.cnn_downsample, mode="nearest").long().cpu()
        else:
            true_mask = true_mask.long().cpu()

        pred_mask = output["mask"].cpu().argmax(dim=1, keepdim=True).squeeze(2)

        self.skip_background = True
        self.num_bg_objects = 1
        self.num_ignored_objects = self.num_bg_objects if self.skip_background else 0

        # Compute metrics
        ari = ari_(true_mask, pred_mask, self.num_ignored_objects)

        if compute_miou:
            # Mask shape (B, O, 1, H, W), is_foreground (B, O, 1), is_modified (B, O), where
            # O = max num objects. Expand the last 2 to (B, O, 1, 1, 1) for broadcasting.
            unsqueezed_shape = (*batch["is_foreground"].shape, 1, 1)
            is_fg = batch["is_foreground"].view(*unsqueezed_shape)
            is_modified = batch["is_modified"].view(*unsqueezed_shape)

            # Mask with foreground objects: shape (B, 1, H, W)
            fg_mask = (batch["mask"] * is_fg).sum(1)
            # Mask with unmodified foreground objects: shape (B, 1, H, W)
            unmodified_fg_mask = (batch["mask"] * is_fg * (1 - is_modified)).sum(1)

            ####### compute mious #############
            visibility = batch['visibility']
            is_fg = batch['is_foreground']

            miou, mbo = compute_total_ious(true_mask, pred_mask, is_fg, self.num_slots)

            miou = torch.cat(miou)
            mbo = torch.cat(mbo)

        else:
            miou = torch.tensor(0)
            mbo = torch.tensor(0)

        # Return with shape (batch_size, )
        dict_metrics = dict(
                ari=ari,
                miou=miou,
                mbo=mbo,
        )

        return dict_metrics

    @torch.no_grad()
    def log_qual(self, global_steps, output, main_process=False, mode='val'):
        if mode == 'val':
            prefix='val_losses'
        elif mode == 'train':
            prefix='train_losses'
        elif mode == 'train_step':
            prefix='train_losses_step'

        if main_process:
            self.logger.log_img_comp(output, global_steps, prefix=mode)
            self.logger._log_train_losses(output, global_steps, prefix=mode)
            if mode == 'val':
                self.logger._log_only_images(output, global_steps, prefix=mode)
            self.logger._flush()

        del output

    @torch.no_grad()
    def log_img(self, global_steps, img, main_process=False, mode='val'):
        if main_process:
            self.logger.log_img(img, global_steps, prefix=mode)
            self.logger._flush()


    @torch.no_grad()
    def eval_qual(self, global_steps, main_process=False, mode='val'):

        if mode == 'val':
            dataloader = self.validation_dataloader
            prefix='val_losses'
        elif mode == 'train':
            dataloader = self.training_dataloader
            prefix='train_losses'
        elif mode == 'train_step':
            dataloader = self.training_dataloader
            prefix='train_losses_step'


        torch.cuda.empty_cache()
        self.model.set_eval_mode()

        count = 0
        for idx, batch in tqdm(enumerate(dataloader), disable=not main_process):
            if self.config.debug or 'train' in mode:
                if idx > 5:
                    break

            output = self.eval_one_step(batch)

        if main_process:
            self.logger.log_img_comp(output, global_steps, prefix=mode)
            self.logger._log_train_losses(output, global_steps, prefix=mode)
            if mode == 'val':
                self.logger._log_only_images(output, global_steps, prefix=mode)
            self.logger._flush()

        self.model.set_train_mode()
        torch.cuda.empty_cache()
        del output



    @torch.no_grad()
    def eval_metrics(self, global_steps, main_process=False, compute_miou=False, mode='val'):
        if mode == 'val':
            dataloader = self.validation_dataloader
            prefix='val_losses'
        elif mode == 'train':
            dataloader = self.training_dataloader
            prefix='train_losses'
        elif mode == 'train_step':
            dataloader = self.training_dataloader
            prefix='train_losses_step'


        torch.cuda.empty_cache()
        self.model.set_eval_mode()

        dict_metrics = {'ari':0.0}
        count = 0
        ari, miou, mbo = [], [], []

        for idx, batch in tqdm(enumerate(dataloader), disable=not main_process):
            if self.config.debug or 'train' in mode:
                if idx > 10:
                    break

            output = self.eval_one_step(batch)

            if mode=='val':
                _dict_metrics = self.compute_metrics(output, global_steps, compute_miou)
                ari.append(_dict_metrics['ari'])
                miou.append(_dict_metrics['miou'])
                mbo.append(_dict_metrics['mbo'])

        if mode=='val':
            ari = torch.cat(ari).to(self.device)
            miou = torch.cat(miou).to(self.device)
            mbo = torch.cat(mbo).to(self.device)

            # padding for different length
            max_objects = 24
            if miou.size(0) != ari.size(0):
                n_pad = ari.size(0)*max_objects - miou.size(0)
                pad = -torch.ones((n_pad,), device=miou.device).float()
                miou = torch.cat([miou, pad])

                n_pad = ari.size(0)*max_objects - mbo.size(0)
                pad = -torch.ones((n_pad,), device=mbo.device).float()
                mbo = torch.cat([mbo, pad])

            ari_total = self.accelerator.gather(ari)
            miou_total = self.accelerator.gather(miou)
            mbo_total = self.accelerator.gather(mbo)

            # remove padding
            miou_total = miou_total[miou_total>=0]
            mbo_total = mbo_total[mbo_total>=0]

        if main_process:
            self.logger.log_img_comp(output, global_steps, prefix=mode)
            self.logger._log_train_losses(output, global_steps, prefix=mode)
            if mode == 'val':
                dict_metrics['ari'] = ari_total.mean().item()
                dict_metrics['miou'] = miou_total.mean().item()
                dict_metrics['mbo'] = mbo_total.mean().item() 

                self.logger._log_metrics(dict_metrics, global_steps, prefix=mode)
                self.logger._log_images(output, global_steps, prefix=mode)
                print(dict_metrics)
            self.logger._flush()

        self.model.set_train_mode()
        torch.cuda.empty_cache()
        del output

    def save_checkpoint(self, step):
        # ## TODO : save checkpoints
        self.accelerator.save_state(os.path.join(self.model_save_dir, f'checkpoint-{step}'))

    def load_checkpoint(self, ckpt_path):

        if self.config.use_accel:
            print('========== loading pretrained ckpt ===================')
            if self.config.eval_miou:
                # Load model on single gpu
                model_state_dict = torch.load(ckpt_path, self.device)['module']
                self.model.module.load_state_dict(model_state_dict)

            else:
                self.accelerator.load_state(ckpt_path)


            print('========== loading pretrained ckpt done ==============')
        else:
            state_dicts = self._get_checkpoint_state()

            # Load checkpoint without model
            state = torch.load(ckpt_path, self.device)
            states = state['states']
            for varname in states:
                try:
                    state_dicts[varname].load_state_dict(states[varname])
                except:
                    print(varname)

    def train_one_step(self):
        # abstract
        pass

    def train(self):
        log_freq = self.config.log_freq  # logging frequency
        save_freq = self.config.save_freq  # checkpoint saving frequency

        if self.config.eval_miou:
            assert self.load_ckpt_path is not None
            if self.accelerator.is_main_process:
                print(f'========= Evaluation =======')
            self.eval_metrics(1, main_process=self.accelerator.is_main_process, compute_miou=True, mode='val')

        else:
            start_steps = 0
            if self.load_ckpt_path is not None:
                start_steps = int(self.load_ckpt_path.split('checkpoint-')[-1])
            global_steps = start_steps 

            with tqdm(total=self.steps-start_steps, disable=not self.accelerator.is_main_process) as p_bar:
                run_iter = True
                while run_iter:
                    for idx, batch in enumerate(self.training_dataloader):
                        if global_steps >= self.steps:
                            run_iter=False
                            break

                        # train one step
                        if (global_steps+1) % log_freq == 0:
                            output = self.train_one_step(batch, visualize_comp=True, steps=global_steps+1)
                            self.log_qual(global_steps + 1, output, main_process=self.accelerator.is_main_process, mode='train')
                        else:
                            output = self.train_one_step(batch, steps=global_steps+1)

                        # save checkpoints
                        if (global_steps+1) % save_freq == 0:
                            if not self.config.debug:
                                self.save_checkpoint(global_steps + 1)

                        # p_bar update
                        global_steps += 1
                        p_bar.update(1)

                        if self.config.use_accel:
                            # validation
                            if (global_steps+1) % log_freq == 0:
                                imgs = self.model.ddpm_decoding(batch['image'])
                                self.log_img(global_steps + 1, imgs, self.accelerator.is_main_process, mode='train')
                                if 'ffhq' in self.config.dataset_name:
                                    self.eval_qual(global_steps + 1, main_process=self.accelerator.is_main_process, mode='val')
                                else:
                                    self.eval_metrics(global_steps + 1, main_process=self.accelerator.is_main_process, compute_miou=True, mode='val')


        if self.accelerator.is_main_process:
            print("================== Training completed =====================")
        self.accelerator.end_training()


    def _get_checkpoint_state(self) -> dict:
        state = dict(model=self.model)
        state.update({f"opt_{i}": opt for i, opt in enumerate(self.optimizers)})

        # LR schedulers are not necessarily present
        if hasattr(self, "lr_schedulers"):
            state.update(
                {
                    f"lr_scheduler_{i}": scheduler
                    for i, scheduler in enumerate(self.lr_schedulers)
                }
            )
        return state

def extract_state_dicts(state: dict) -> dict:
    return {name: state[name].state_dict() for name in state}
