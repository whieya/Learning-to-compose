from dataclasses import dataclass, field
from math import sqrt
from typing import Dict, List, Literal, Optional, Tuple, Union
import copy
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import torch
from torch import Tensor, nn
import torch.functional as F

from models.nn_utils import get_conv_output_shape, make_sequential_from_config
from models.shared.nn import PositionalEmbedding
from models.transformer import TransformerDecoder, PositionalEncoding
from models.unet_model import UNet as simple_unet
from models.networks.stylegan2_layers import *
import numpy as np
from einops import repeat, rearrange
from models.networks.swapae_layers import *
from models.networks.vq_layers import Encoder_Resnet, Decoder_Resnet
import functools

### diffusion
import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel, DiffusionPipeline, UNet2DConditionModel
from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.models import AutoencoderKL
from diffusers.utils import randn_tensor
from diffusers import UNet2DEncoder, UNet2DConditionModelWithPE
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_tensorboard_available, is_wandb_available

from torch.optim.lr_scheduler import LambdaLR
from .slate_utils import * 
from scipy.optimize import linear_sum_assignment




class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )
        weight = torch.softmax(weight, dim=-1)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = torch.softmax(weight, dim=-1)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")



class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = nn.GroupNorm(32, channels)
        #self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)



class View(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, input):
        return input.view(self.size)



class Comp_Model(nn.Module):
    '''
    Main model for composing the slots.

    '''
    def __init__(self, config,
            device='cuda', 
            resolution=64, 
            num_slots=11, 
            name='Comp_Model', 
            log_n_imgs=4, 
            dataset_name=None, 
            max_steps=500, 
            ddim_steps=10, 
            scale_latent=1.0, 
            slot_dim=192, 
            diff_dim=192,
            share_slot_init=False, 
            ):
        super().__init__()
        
        self.dataset_name = dataset_name
        self.device = device

        ################ load pretrained vae model ####################
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
        self.vae.requires_grad_(False) # freeze vae
        self.scaling_factor = 0.18215
        ###############################################################

        #################### Diffusion Model ##########################
        self.image_size = resolution
        self.eta = 1.0
        self.num_inference_steps = ddim_steps # number of inference step for decoding comp
        self.scale_latent = scale_latent # legacy. Keep it 1. 
        self.share_slot_init = share_slot_init # it determines whether to align slot inits.
        self.weighting_choice = 'sigma^2'  # weighting function for SDS loss

        # conditional version
        self.base_dim=diff_dim
        self.slot_dim=slot_dim
        self.num_slots = num_slots
        self.log_n_imgs = log_n_imgs 
        base_dim, slot_dim = self.base_dim, self.slot_dim

        # projection layer
        self.phi_slot_proj = nn.Linear(slot_dim, base_dim)

        # Diffusion Model
        # we use UNet + Positional Embedding following LSD.
        self.vae_downratio = 8
        self.model_phi = UNet2DConditionModelWithPE(
                sample_size=self.image_size//self.vae_downratio, # vae downsample to 1/8
                in_channels=4,
                out_channels=4,
                layers_per_block=2,
                block_out_channels=(base_dim, base_dim*2, base_dim*4, base_dim*4),
                attention_head_dim=8,
                down_block_types=(
                    "CrossAttnDownBlock2DWithPE",
                    "CrossAttnDownBlock2DWithPE",
                    "CrossAttnDownBlock2DWithPE",
                    "CrossAttnDownBlock2DWithPE",
                ),
                up_block_types=(
                    "CrossAttnUpBlock2DWithPE",
                    "CrossAttnUpBlock2DWithPE",
                    "CrossAttnUpBlock2DWithPE",
                    "CrossAttnUpBlock2DWithPE",
                    ),
                mid_block_type = "UNetMidBlock2DCrossAttnWithPE",
                cross_attention_dim=base_dim)
        

        # noise scheduler
        beta_start, beta_end = 0.00085, 0.0120
        self.noise_scheduler = DDPMScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            num_train_timesteps=1000,
            beta_schedule='linear',
            prediction_type='epsilon',
            clip_sample=False,
        )

        self.ddim_scheduler = DDIMScheduler(
                beta_start=self.noise_scheduler.beta_start,
                beta_end=self.noise_scheduler.beta_end,
                num_train_timesteps=self.noise_scheduler.num_train_timesteps,
                beta_schedule=self.noise_scheduler.beta_schedule,
                prediction_type=self.noise_scheduler.prediction_type,
                clip_sample=False,
                steps_offset=1,
                set_alpha_to_one=False,
            )
        self.ddim_scheduler.set_timesteps(self.num_inference_steps)
        self.num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = max_steps
        self.alphas = self.noise_scheduler.alphas_cumprod # for convenience
        
    def load_pretrained(self, ckpt_path):
        self.model.from_pretrained(ckpt_path)
       
    def init_SA_module(self, model):
        self.model_sa = model
        print("initializing SA module..")

    def get_last_layer(self):
        return self.model_sa.out.weight

    def random_mix(self, slot1, slot2):
        bs, ns, dim = slot1.size()

        # mixing
        with torch.no_grad():
            noise = torch.rand((bs, ns, 1), requires_grad=False, device=slot1.device)
            b_mask = (noise > 0.5).long()

        slot_mix = slot1 * b_mask + slot2 * (1 - b_mask)
        return slot_mix, b_mask

    
    @torch.no_grad()
    def vae_encode(self, img):
        x = self.vae.encode(img).latent_dist.sample()

        # scaling 
        x = x * self.scaling_factor
        return x
    
    # @torch.no_grad()
    def vae_decode(self, latent, detach=False):
        latent = latent / self.scaling_factor
        if detach:
            with torch.no_grad():
                out = self.vae.decode(latent).sample.detach()
        else:
            out = self.vae.decode(latent).sample
        return out

    @torch.no_grad()
    def ddpm_decoding(self, img):
        n_imgs=4
        img = img[:n_imgs]
        x = self.vae_encode(img)
        bs, c, height, width = x.size()

        if self.model_sa.slot_encode_RGB:
            slot_input, _  = self.model_sa.slot_encode(img)
        else:
            slot_input, _  = self.model_sa.slot_encode(x)

        _slot = self.phi_slot_proj(slot_input)

        # ddim decoding
        pipeline = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=None,
            tokenizer=None,
            unet=self.model_phi,
            scheduler=self.noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=None,
        )

        generator = torch.Generator(device=img.device).manual_seed(42)

        image = pipeline(
            prompt_embeds=_slot,
            height=128,
            width=128,
            num_inference_steps=25,
            generator=generator,
            guidance_scale=1.,
            output_type="pt",
        ).images
       
        final_img = torch.tensor(image).permute(0,3,1,2).clamp(0.0,1.0).to(img.device)
        return final_img, img


    def forward(self, img, use_losses, eval_mode, visualize_comp=False):
        with torch.set_grad_enabled(not eval_mode):
            out={
                'loss_composition' : 0.0,
                'loss_oneshot' : 0.0,
                'loss_slot_diffusion': 0.0,
                'loss_mask_reg' : 0.0,
            }

            ############## encoding with pretrained vae #################
            x = self.vae_encode(img)
            bs, c, height, width = x.size()

            ############### slot encoding #################
            # slot encoding
            if self.share_slot_init:
                # We will split the batch into half and mix them. 
                # We share slot initializations between those two sets of slots. 
                slot_noise = torch.randn(bs//2, self.num_slots, self.model_sa.latent_size).to(self.device)
                slot_noise = torch.cat([slot_noise, slot_noise], dim=0)

            else:
                slot_noise = torch.randn(bs, self.num_slots, self.model_sa.latent_size).to(self.device)

            if self.model_sa.slot_encode_RGB:
                slot_input, attns_mask  = self.model_sa.slot_encode(img, slot_noise)
            else:
                slot_input, attns_mask  = self.model_sa.slot_encode(x, slot_noise)
            
            out['mask'] = attns_mask.unsqueeze(2).detach()
            out['latent_input'] = x.detach()

            ############### reconstruction path #################
            if use_losses['use_loss_oneshot']:
                # slot decoder
                if self.model_sa.autoregressive:
                    gt_input = x.view(bs, c, -1)
                    gt_input = gt_input.transpose(-2, -1)
                    dict_slot_recon = self.model_sa.slot_decode(slot_input, gt_input=gt_input, inference=eval_mode)
                else:
                    dict_slot_recon = self.model_sa.slot_decode(slot_input)

                output = dict_slot_recon['reconstruction']
                out['dec_mask'] = dict_slot_recon['dec_mask']

                loss_oneshot = nn.MSELoss()(output, x)
                out['loss_oneshot'] = loss_oneshot.item() if eval_mode else loss_oneshot
                out['output'] = self.vae_decode(output, detach=True)
            ############### reconstruction path end #################
            
            ################ composition path ###################
            if use_losses['use_loss_composition'] or use_losses['use_loss_mask_reg']:

                # shift bs//2 for  slot_input in batch dimension
                slot_input_shifted = torch.cat([slot_input[bs//2:], slot_input[:bs//2]], dim=0)
                slot_mix, b_mask =  self.random_mix(slot_input, slot_input_shifted)

                # Mixed Decoding
                if not eval_mode and not self.model_sa.autoregressive:
                    ############# trick for freezing oneshot decoder ############
                    # for composition, we do not update decoder with sds loss 
                    decoder_parts = [self.model_sa.positional_encoder,
                            self.model_sa.bi_tf_dec,
                            self.model_sa.out,
                            self.model_sa.slot_proj,
                            ]

                    # Turn off gradient
                    self.model_sa.mask_token.requires_grad_(False)
                    for _model in decoder_parts:
                        for param in _model.parameters():
                            param.requires_grad_(False)

                    # decode slots
                    dict_slot_comp = self.model_sa.slot_decode(slot_mix)

                    # Turn on gradient
                    self.model_sa.mask_token.requires_grad_(True)
                    for _model in decoder_parts:
                        for name, param in _model.named_parameters():
                            if param.dtype==torch.bool:
                                continue
                            param.requires_grad_(True)
                    #############################################################

                    comp = dict_slot_comp['reconstruction']
                    detach_vae_decode = not use_losses['use_loss_mask_reg']
                    out['comp'] = self.vae_decode(comp, detach=detach_vae_decode)                

                else: # eval_mode or autoregressive
                    if self.model_sa.autoregressive and not eval_mode:
                        out['comp'] = None
                    else:
                        with torch.no_grad():
                            if self.model_sa.autoregressive:
                                dict_slot_comp = self.model_sa.slot_decode(slot_mix, gt_input=gt_input, inference=True)
                            else:
                                dict_slot_comp = self.model_sa.slot_decode(slot_mix)

                            comp = dict_slot_comp['reconstruction']
                            out['comp'] = self.vae_decode(comp, detach=True)

                ############# mask regularization on composition #################
                if use_losses['use_loss_mask_reg']:
                    # Assumption : comp comes from either x1 or x2.
                    mse_loss = nn.MSELoss(reduction='none')
                    x_shifted = torch.cat([x[bs//2:], x[:bs//2]], dim=0)

                    # according to b_mask, compute total attn mask from each image
                    attns_mask_shifted = torch.cat([attns_mask[bs//2:], attns_mask[:bs//2]], dim=0)
                    attns_mask_1 = attns_mask * b_mask.unsqueeze(-1)
                    attns_mask_2 = attns_mask * (1 - b_mask).unsqueeze(-1)
                    attns_mask_1 = attns_mask_1.sum(dim=1)
                    attns_mask_2 = attns_mask_2.sum(dim=1) 

                    # upsample mask.
                    if self.model_sa.cnn_downsample > 1:
                        # upsample masks with scale factor of cnn_downsample.
                        # [bs, h, w] -> [bs, h*s, w*s]
                        _s = self.model_sa.cnn_downsample
                        attns_mask_1 = attns_mask_1.repeat_interleave(_s, dim=1).repeat_interleave(_s, dim=2)
                        attns_mask_2 = attns_mask_2.repeat_interleave(_s, dim=1).repeat_interleave(_s, dim=2)

                    if not self.model_sa.slot_encode_RGB:
                        loss_cond_1 = mse_loss(comp, x).mean(1) * attns_mask_1
                        loss_cond_2 = mse_loss(comp, x_shifted).mean(1) * attns_mask_2
                    else:
                        img_shifted = torch.cat([img[bs//2:], img[:bs//2]], dim=0)
                        loss_cond_1 = mse_loss(out['comp'].detach(), img).mean(1) * attns_mask_1
                        loss_cond_2 = mse_loss(out['comp'].detach(), img_shifted).mean(1) * attns_mask_2
                        loss_cond_1 = loss_cond_1.view(bs, -1)
                        loss_cond_2 = loss_cond_2.view(bs, -1)
                        loss_mask_reg = loss_cond_1.mean() * 0.5 + loss_cond_2.mean() * 0.5

                    out['loss_mask_reg'] = loss_mask_reg.item() if eval_mode else loss_mask_reg

                ############ sds loss for composition validity #############
                if use_losses['use_loss_composition']:
                    # hyper-parameters determining the denoising level
                    min_t, max_t = self.min_step, self.max_step

                    if self.alphas.device != x.device:
                        self.alphas = self.alphas.to(x.device)

                    with torch.no_grad():
                        if eval_mode:
                            t = torch.tensor(200, device=self.device)
                        else:
                            t = torch.randint(min_t, max_t, [1], dtype=torch.long, device=self.device)

                        _alphas = self.alphas[t]
                        if self.weighting_choice == 'sigma^2':  # original_version
                            w = 0.5 * (_alphas * (1 - _alphas)) ** 0.5 
                        elif self.weighting_choice == 'alpha^0.5*sigma^-1':  # 'uniform'
                            w = 1.0 * (_alphas / (1 - _alphas))
                        elif self.weighting_choice == 'uniform':
                            w = 0.5 * (_alphas / (1 - _alphas)) ** 0.5

                        noise_out = torch.randn_like(output)

                        # forward process
                        output_noisy = self.noise_scheduler.add_noise(comp * self.scale_latent, noise_out, t)

                        # one step decoding 
                        _slot_mix = self.phi_slot_proj(slot_mix)
                        model_output_phi = self.model_phi(output_noisy, t, _slot_mix).sample

                        # predict noise model_output
                        noise_pred = model_output_phi

                        # Compute SDS loss.
                        # We use alternative loss trick to track the magnitude of the loss. 
                        # This loss is equivalent to SDS loss
                        # Be carefull for the w calculation
                        denoised_pretrain = 1 / _alphas**0.5 * (output_noisy - (1-_alphas)**0.5 * noise_pred)
                        denoised_pretrain = denoised_pretrain / self.scale_latent

                    loss_composition = w * nn.MSELoss()(comp, denoised_pretrain.detach())
                    out['loss_composition'] = loss_composition.item() if eval_mode else out['loss_composition'] + loss_composition
            ################ composition path end ###################              

            ##########################  Slot Diffusion path  #############################              
            if use_losses['use_slot_diffusion']:
                _slot = self.phi_slot_proj(slot_input)

                # slot diffusion 
                if eval_mode:
                    t = torch.tensor(500, dtype=torch.long, device=self.device)
                else:
                    t = torch.randint(0, 1000, (bs,), dtype=torch.long, device=self.device)

                # forward process
                noise = torch.randn_like(x)
                x_noisy = self.noise_scheduler.add_noise(x*self.scale_latent, noise, t).detach()
                noise_pred_phi = self.model_phi(x_noisy.detach(), t, _slot).sample
                loss_slot_diffusion = torch.nn.functional.mse_loss(noise_pred_phi, noise)

                out['loss_slot_diffusion'] = out['loss_slot_diffusion'] + loss_slot_diffusion.item() \
                        if eval_mode else out['loss_slot_diffusion'] + loss_slot_diffusion

            ########################## Slot Duiffusion path end #############################              

            out['loss_composition'] = out['loss_composition']
            out['loss_slot_diffusion'] = out['loss_slot_diffusion']

            ########## Composed generation for Visualization ##############
            if eval_mode or visualize_comp:
                # visualize composed masks
                # composed_img, img1, stuffs, img2
                with torch.no_grad():
                    # we will use only small portion of the images for logging
                    n_imgs = min(self.log_n_imgs, bs//2) # 4
                    out['x'] = torch.cat([img[:n_imgs], img[bs//2:bs//2+n_imgs]], dim=0)
                    dim_slot = slot_input.size(2)

                    # composition path
                    slot_1 = slot_input[:n_imgs]
                    slot_input_shifted = torch.cat([slot_input[bs//2:], slot_input[:bs//2]], dim=0)
                    slot_2 = slot_input_shifted[:n_imgs]
                
                    # repeat the tenso
                    n_s, d_s = self.num_slots, dim_slot
                    slot_1 = slot_1.unsqueeze(1).repeat(1, n_s + 1, 1, 1)
                    slot_2 = slot_2.unsqueeze(1).repeat(1, n_s + 1, 1, 1) 
                    slot_1 = slot_1.view(-1, n_s, d_s)
                    slot_2 = slot_2.view(-1, n_s, d_s)

                    # we will mix slot_1 and slot_2 with mixing ratio from 0 to 1.
                    # To this end, we prepare multiple triu_mask corresponding to different mixing ratio.  
                    mat = torch.ones((n_s+1, n_s), device=slot_1.device) # [num_slots + 1, num_slots]
                    triu_mask = torch.triu(mat, diagonal=0) # [num_slots, num_slots]
                    triu_mask = triu_mask.unsqueeze(0).repeat(n_imgs, 1, 1) # [B//2, num_slots + 1, num_slots]
                    triu_mask = triu_mask.view(-1, n_s).unsqueeze(-1) # [B//2*(num_slots+1), num_slots]
                    
                    slot_mix = slot_1 * triu_mask + slot_2 * (1-triu_mask) # [B//2*(n_s+1), n_s, 256]

                    if self.model_sa.autoregressive:
                        dict_slot_comp = self.model_sa.slot_decode(slot_mix, gt_input=gt_input, inference=True)
                    else:
                        dict_slot_comp = self.model_sa.slot_decode(slot_mix)

                    comp_interp = dict_slot_comp['reconstruction']
                    out['comp_interp'] = self.vae_decode(comp_interp, detach=True)
            return out

    def set_eval_mode(self):
        self.model_sa.eval()

    def set_train_mode(self):
        self.model_sa.train()

class Encoder(nn.Module):
    def __init__(
        self,
        width: int,
        height: int,
        channels: List[int] = (32, 32, 32, 32),
        kernels: List[int] = (5, 5, 5, 5),
        strides: List[int] = (1, 1, 1, 1),
        paddings: List[int] = (2, 2, 2, 2),
        input_channels: int = 3,
        batchnorms: List[bool] = tuple([False] * 4),
    ):
        super().__init__()
        assert len(kernels) == len(strides) == len(paddings) == len(channels)
        self.conv_bone = make_sequential_from_config(
            input_channels,
            channels,
            kernels,
            batchnorms,
            False,
            paddings,
            strides,
            "relu",
            try_inplace_activation=True,
        )
        output_channels = channels[-1]
        output_width, output_height = get_conv_output_shape(
            width, height, kernels, paddings, strides
        )
        self.pos_embedding = PositionalEmbedding(
            output_width, output_height, output_channels
        )
        self.lnorm = nn.GroupNorm(1, output_channels, affine=True, eps=0.001)
        self.conv_1x1 = [
            nn.Conv1d(output_channels, output_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(output_channels, output_channels, kernel_size=1),
        ]
        self.conv_1x1 = nn.Sequential(*self.conv_1x1)

    def forward(self, x: Tensor) -> Tensor:
        conv_output = self.conv_bone(x)
        conv_output = self.pos_embedding(conv_output)
        conv_output = conv_output.flatten(2, 3)  # bs x c x (w * h)
        conv_output = self.lnorm(conv_output)
        return self.conv_1x1(conv_output)


class Slate(nn.Module):
    def __init__(self, 
                 image_size: int = 128,
                 latent_size: int = 128,
                 input_channels: int = 3,
                 eps: float = 1e-8,
                 mlp_size: int = 128,
                 attention_iters: int = 3,
                 num_slots: int = 0,
                 per_slot_init: bool = False,
                 slot_encode_RGB: bool = False,
                 num_dec_blocks: int= 4,
                 d_tf: int=192,
                 num_heads: int=8,
                 autoregressive: bool=False,
                 cnn_enc_type: str='unet',
                 cnn_downsample: int=1,
                 ):
        super().__init__()
      
        self.image_size = image_size if slot_encode_RGB else image_size // 8
        self.latent_size = latent_size
        self.input_channels = input_channels
        self.eps = eps
        self.mlp_size = mlp_size
        self.attention_iters = attention_iters
        self.num_slots = num_slots
        self.per_slot_init = per_slot_init
        self.autoregressive = autoregressive
        self.cnn_downsample = cnn_downsample
        self.loss_fn = nn.MSELoss()

        # slot encoding layers
        self.vae_downratio = 8 # downsample ratio of pretrained vae model
        self.width = image_size // self.vae_downratio
        self.height = image_size // self.vae_downratio
        self.enc_channels = 32
        self.slot_encode_RGB=slot_encode_RGB # whether to encode RGB img or latent features
        self.cnn_enc_type = cnn_enc_type # determines the type of encoder, e.g., cnn or unet

        # cnn encoder
        if not self.slot_encode_RGB:
            assert 0 # currently, we use RGB encoder following LSD implementation.
            if self.cnn_enc_type == 'cnn':
                output_channels = d_tf
                self.encoder = nn.Sequential(
                    nn.Conv2d(self.input_channels, 64, (3, 3), (1, 1), 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    nn.Conv2d(64, 128, (3, 3), (1, 1), 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),
                    nn.Conv2d(128, 192, (3, 3), (1, 1), 1),
                    nn.BatchNorm2d(192),
                    nn.ReLU(True),
                    nn.Conv2d(192, output_channels, (3, 3), (1, 1), 1),
                    )

            elif self.cnn_enc_type == 'unet':
                output_channels = d_tf
                self.encoder = simple_unet(
                        input_channels = self.input_channels,
                        output_channels = output_channels,
                        base_ch = 64,
                        mult_chs=[1,2,4,8],
                        bilinear=True,
                        )

            self.pos_embedding = PositionalEmbedding(
                self.width, self.height, output_channels
            )

        else:
            if self.cnn_enc_type == 'cnn':
                output_channels = d_tf
                first_cnn = nn.Conv2d(self.input_channels, 128, self.cnn_downsample, self.cnn_downsample, 0)
                cnn = nn.Sequential(
                    nn.Conv2d(128, 128, (5, 5), (1, 1), 2),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),
                    nn.Conv2d(128, 128, (5, 5), (1, 1), 2),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),
                    nn.Conv2d(128, 128, (5, 5), (1, 1), 2),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),                                                                                                                                                 
                    nn.Conv2d(128, 128, (5, 5), (1, 1), 2),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),
                    nn.Conv2d(128, output_channels, (5, 5), (1, 1), 2),
                    )
                self.encoder = nn.Sequential(first_cnn, cnn)

            elif self.cnn_enc_type == 'unet':
                output_channels = self.latent_size
                self.first_cnn = nn.Conv2d(self.input_channels, 128, self.cnn_downsample, self.cnn_downsample, 0)
                
                self.unet_backbone = UNet2DEncoder(
                        sample_size=image_size//self.cnn_downsample,
                        in_channels=128,
                        out_channels=output_channels,
                        layers_per_block=2,
                        block_out_channels=(128, 128, 128*2, 128*4),
                        attention_head_dim=8,
                        down_block_types=(
                            "DownBlock2D",
                            "DownBlock2D",
                            "DownBlock2D",
                            "DownBlock2D",
                        ),
                        up_block_types=(
                            "UpBlock2D",
                            "UpBlock2D",
                            "UpBlock2D",
                            "UpBlock2D",
                            ),
                        )
                self.encoder = nn.Sequential(
                    self.first_cnn,
                    self.unet_backbone,
                    )

            self.pos_embedding = PositionalEmbedding(
                self.image_size//self.cnn_downsample, self.image_size//self.cnn_downsample, output_channels
            )

        # slot projection
        self.d_tf = d_tf
        self.slot_proj = nn.Linear(self.latent_size, self.d_tf, bias=False)
        if self.autoregressive:
            self.BOS_token = nn.Parameter(torch.randn(1, 1, self.d_tf))
            self.input_proj = nn.Linear(4, self.d_tf)
        else:
            self.mask_token = nn.Parameter(torch.randn(1, 1, self.d_tf))

        # transformer hyperparameters
        self.num_dec_blocks = num_dec_blocks
        self.num_heads = num_heads
        self.dropout = 0.1

        # transformer layers
        if self.autoregressive:
            self.positional_encoder = PositionalEncoding(1 + self.width * self.height, self.d_tf, trunc_emb=True)
            self.ar_tf_dec = TransformerDecoder(
                self.num_dec_blocks,  self.width * self.height, self.d_tf, self.num_heads, self.dropout, causal_mask=True)
            self.out = nn.Linear(self.d_tf, 4, bias=True)

        else:
            self.positional_encoder = PositionalEncoding(self.width * self.height, self.d_tf)
            self.bi_tf_dec = TransformerDecoder(
                self.num_dec_blocks,  self.width * self.height, self.d_tf, self.num_heads, self.dropout)
            self.out = nn.Linear(self.d_tf, 4, bias=True)

        self.slot_attention = SlotAttentionEncoder(
                num_iterations=self.attention_iters,
                num_slots=self.num_slots,
                input_channels=output_channels,
                slot_size=self.latent_size,
                mlp_hidden_size=self.mlp_size,
                num_heads=1,
                )

    def initialize_weights(self):
        # track all layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                
    def get_last_layer(self):
        return self.decoder.get_last_layer()

    @property
    def slot_size(self) -> int:
        return self.latent_size

    def spatial_broadcast(self, slot: Tensor) -> Tensor:
        slot = slot.unsqueeze(-1).unsqueeze(-1)
        return slot.repeat(1, 1, self.w_broadcast, self.h_broadcast)

    def slot_encode(self, x: Tensor, slot_noise=None) -> Tensor:

        H=W=self.image_size
        bs, c, H_enc, W_enc = x.size()

        encoded = self.encoder(x) 
        encoded = self.pos_embedding(encoded) 
        encoded = encoded.flatten(2, 3).permute(0,2,1)
        z, attns, slot_noise, attns_temp = self.slot_attention(encoded, slot_noise)

        # `attn` has shape: [batch_size, enc_height * enc_width, num_slots].
        attns = attns.transpose(-1, -2)
        attns_temp = attns_temp.transpose(-1, -2)


        downratio = self.cnn_downsample if self.slot_encode_RGB else self.vae_downratio
        # attns_temp = attns_temp.reshape(bs, self.num_slots, H_enc//downratio, W_enc//downratio)
        attns = attns.reshape(bs, self.num_slots, H_enc//downratio, W_enc//downratio)

        # return z, attns_temp, attns
        return z, attns

    def slot_decode(self, slots: Tensor, mr: Tensor = None, gt_input=None, inference=False) -> dict:
        # apply transformer
        slots = self.slot_proj(slots)
        
        if self.autoregressive:
            # expand mask_token
            if inference:
                with torch.no_grad():
                    bos_token = self.BOS_token.expand(slots.size(0), -1, -1)
                    z_input = None

                    for i in range(gt_input.size(1)):
                        if z_input is None:
                            tf_input = bos_token
                        else:
                            tf_input = torch.cat([bos_token, self.input_proj(z_input)], dim=1)

                        tf_input = self.positional_encoder(tf_input)
                        decoder_output, dec_mask = self.ar_tf_dec(tf_input, slots)
                        recon_latent_next = self.out(decoder_output)[:, -1:] # (B, h*w, latent dim)

                        if z_input is None:
                            z_input = recon_latent_next
                        else:
                            z_input = torch.cat([z_input, recon_latent_next], dim=1)
                    
                    recon_latent = z_input 

            else:
                assert gt_input is not None
                bos_token = self.BOS_token.expand(slots.size(0), -1, -1)
                gt_input = self.input_proj(gt_input)
                tf_input = torch.cat([bos_token, gt_input], dim=1)
                tf_input = self.positional_encoder(tf_input)
                decoder_output, dec_mask = self.ar_tf_dec(tf_input[:, :-1], slots)
                recon_latent = self.out(decoder_output) # (B, h*w, latent dim)

        else:
            # bidirectional decoding
            mask_token = self.mask_token.expand(slots.size(0), -1, -1)
            mask_token = self.positional_encoder(mask_token)
            decoder_output, dec_mask = self.bi_tf_dec(mask_token, slots)
            recon_latent = self.out(decoder_output) # (B, h*w, latent dim)

        # reshape into (B, latent_dim ,h ,w )
        try:
            recon_latent = recon_latent.view(slots.size(0), self.height, self.width, 4)
        except:
            print(recon_latent.shape)
        recon_latent = recon_latent.permute(0, 3, 1, 2)

        return {
            "reconstruction": recon_latent, 
            "dec_mask": dec_mask
        }

    def patchify(self,x):
        '''
        x : (B, C, H, W)
        out : (B, H*W/(patch_size**2), patch_size ** 2 * C)
        '''
        B, C, H, W = x.size()
        h_new = H//self.patch_size
        w_new = W//self.patch_size

        x = x.reshape(B, C, h_new, self.patch_size, w_new, self.patch_size)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(B, h_new*w_new, self.patch_size**2*C)
        return x

    def unpatchify(self,x):
        '''
        x: (B, H*W/(patch_size**2), patch_size ** 2 * C)
        out : (B, C, H, W)
        '''
        B, L, D = x.size()
        w=h = int(L**0.5)
        assert w*h == L
        C = D // (self.patch_size**2)
        
        x = x.reshape(B, h, w, self.patch_size, self.patch_size, C)
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(B, C, h*self.patch_size, w*self.patch_size)
        return x

   

class SlotAttention(nn.Module):
    
    def __init__(self, num_iterations, num_slots,
                 input_size, slot_size, mlp_hidden_size, heads,
                 epsilon=1e-8):
        super().__init__()
        
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.input_size = input_size
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon
        self.num_heads = heads

        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)
        
        # Linear maps for the attention module.
        self.project_q = linear(slot_size, slot_size, bias=False)
        self.project_k = linear(input_size, slot_size, bias=False)
        self.project_v = linear(input_size, slot_size, bias=False)
        
        # Slot update functions.
        self.gru = gru_cell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            linear(slot_size, mlp_hidden_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(mlp_hidden_size, slot_size))

    def forward(self, inputs, slots):
        # `inputs` has shape [batch_size, num_inputs, input_size].
        # `slots` has shape [batch_size, num_slots, slot_size].

        B, N_kv, D_inp = inputs.size()
        B, N_q, D_slot = slots.size()

        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs).view(B, N_kv, self.num_heads, -1).transpose(1, 2)
        v = self.project_v(inputs).view(B, N_kv, self.num_heads, -1).transpose(1, 2) 
        k = ((self.slot_size // self.num_heads) ** (-0.5)) * k

        slot_init = slots
        
        # Multiple rounds of attention.
        for it in range(self.num_iterations):
            # ISA : implicit slot attention
            if it == self.num_iterations - 1:
                slots = slots.detach()

            slots_prev = slots
            slots = self.norm_slots(slots)
            
            # Attention.
            q = self.project_q(slots).view(B, N_q, self.num_heads, -1).transpose(1, 2)  
            attn_logits = torch.matmul(k, q.transpose(-1, -2))
            attn = F.softmax(
                attn_logits.transpose(1, 2).reshape(B, N_kv, self.num_heads * N_q)
            , dim=-1).view(B, N_kv, self.num_heads, N_q).transpose(1, 2) 
            attn_vis = attn.sum(1)
            
            # attention mask with temp
            attn_temp = F.softmax(
                attn_logits.transpose(1, 2).reshape(B, N_kv, self.num_heads * N_q)
            , dim=-1).view(B, N_kv, self.num_heads, N_q).transpose(1, 2)
            attn_temp = attn_temp.sum(1)

            # Weighted mean.
            attn = attn + self.epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True)
            updates = torch.matmul(attn.transpose(-1, -2), v)                              
            updates = updates.transpose(1, 2).reshape(B, N_q, -1)
            
            # Slot update.
            slots = self.gru(updates.view(-1, self.slot_size),
                             slots_prev.view(-1, self.slot_size))
            slots = slots.view(-1, self.num_slots, self.slot_size)
            slots = slots + self.mlp(self.norm_mlp(slots))
        
        return slots, attn_vis, attn_temp

class SlotAttentionEncoder(nn.Module):
    
    def __init__(self, num_iterations, num_slots,
                 input_channels, slot_size, mlp_hidden_size, num_heads):
        super().__init__()
        
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.input_channels = input_channels
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.layer_norm = nn.LayerNorm(input_channels)
        self.mlp = nn.Sequential(
            linear(input_channels, input_channels, weight_init='kaiming'),
            nn.ReLU(),
            linear(input_channels, input_channels))
        
        # Parameters for Gaussian init (shared by all slots).
        self.slot_mu = nn.Parameter(torch.zeros(1, 1, slot_size))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_size))
        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.xavier_uniform_(self.slot_log_sigma)

        self.slot_attention = SlotAttention(
            num_iterations, num_slots,
            input_channels, slot_size, mlp_hidden_size, num_heads)

    def forward(self, x, slot_noise=None):
        # `image` has shape: [batch_size, img_channels, img_height, img_width].
        # `encoder_grid` has shape: [batch_size, pos_channels, enc_height, enc_width].
        B, *_ = x.size()
        x = self.mlp(self.layer_norm(x))

        if slot_noise is None:
            noise = x.new_empty(B, self.num_slots, self.slot_size).normal_()
        else:
            noise = slot_noise

        slots = self.slot_mu + torch.exp(self.slot_log_sigma) * noise
        slots, attn, attn_temp = self.slot_attention(x, slots)

        return slots, attn, noise, attn_temp

