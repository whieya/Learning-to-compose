import math
import torch
import util
import torch.nn as nn
import torch.nn.functional as F
from models.networks.stylegan2_layers import ConvLayer, ToRGB, EqualLinear, StyledConv

import argparse
from einops import rearrange, repeat

#################
# Borrowed from set Transformer repo : 
# https://github.com/juho-lee/set_transformer/blob/73432c640ac78140496d6738416c54d32c686d65/modules.py

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)
#################

class UpsamplingBlock(torch.nn.Module):
    def __init__(self, inch, outch, styledim,
                 blur_kernel=[1, 3, 3, 1], use_noise=False):
        super().__init__()
        self.inch, self.outch, self.styledim = inch, outch, styledim
        self.conv1 = StyledConv(inch, outch, 3, styledim, upsample=True,
                                blur_kernel=blur_kernel, use_noise=use_noise)
        self.conv2 = StyledConv(outch, outch, 3, styledim, upsample=False,
                                use_noise=use_noise)

    def forward(self, x, style):
        return self.conv2(self.conv1(x, style), style)


class ResolutionPreservingResnetBlock(torch.nn.Module):
    def __init__(self, opt, inch, outch, styledim, use_noise):
        super().__init__()
        self.conv1 = StyledConv(inch, outch, 3, styledim, upsample=False, use_noise=use_noise)
        self.conv2 = StyledConv(outch, outch, 3, styledim, upsample=False, use_noise=use_noise)
        if inch != outch:
            self.skip = ConvLayer(inch, outch, 1, activate=False, bias=False)
        else:
            self.skip = torch.nn.Identity()

    def forward(self, x, style):
        skip = self.skip(x)
        res = self.conv2(self.conv1(x, style), style)
        return (skip + res) / math.sqrt(2)


class UpsamplingResnetBlock(torch.nn.Module):
    def __init__(self, inch, outch, styledim, blur_kernel=[1, 3, 3, 1], use_noise=False):
        super().__init__()
        self.inch, self.outch, self.styledim = inch, outch, styledim
        self.conv1 = StyledConv(inch, outch, 3, styledim, upsample=True, blur_kernel=blur_kernel, use_noise=use_noise)
        self.conv2 = StyledConv(outch, outch, 3, styledim, upsample=False, use_noise=use_noise)
        if inch != outch:
            self.skip = ConvLayer(inch, outch, 1, activate=True, bias=True)
        else:
            self.skip = torch.nn.Identity()

    def forward(self, x, style):
        skip = F.interpolate(self.skip(x), scale_factor=2, mode='bilinear', align_corners=False)
        res = self.conv2(self.conv1(x, style), style)
        return (skip + res) / math.sqrt(2)


class GeneratorModulation(torch.nn.Module):
    def __init__(self, styledim, outch):
        super().__init__()
        self.scale = EqualLinear(styledim, outch)
        self.bias = EqualLinear(styledim, outch)

    def forward(self, x, style):
        if style.ndimension() <= 2:
            return x * (1 * self.scale(style)[:, :, None, None]) + self.bias(style)[:, :, None, None]
        else:
            style = F.interpolate(style, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
            return x * (1 * self.scale(style)) + self.bias(style)


class StyleGAN2ResnetGenerator(nn.Module):
    """ The Generator (decoder) architecture described in Figure 18 of
        Swapping Autoencoder (https://arxiv.org/abs/2007.00653).
        
        At high level, the architecture consists of regular and 
        upsampling residual blocks to transform the structure code into an RGB
        image. The global code is applied at each layer as modulation.
        
        Here's more detailed architecture:
        
        1. SpatialCodeModulation: First of all, modulate the structure code 
        with the global code.
        2. HeadResnetBlock: resnets at the resolution of the structure code,
        which also incorporates modulation from the global code.
        3. UpsamplingResnetBlock: resnets that upsamples by factor of 2 until
        the resolution of the output RGB image, along with the global code
        modulation.
        4. ToRGB: Final layer that transforms the output into 3 channels (RGB).
        
        Each components of the layers borrow heavily from StyleGAN2 code,
        implemented by Seonghyeon Kim.
        https://github.com/rosinality/stylegan2-pytorch
    """
    def _set_hyper_params(self):
        self.netG_scale_capacity = 1.0
        self.netG_num_base_resnet_layers = 2
        self.netG_use_noise = False
        self.netG_resnet_ch = 256
        self.use_antialias = True
        self.num_classes = 0
        self.num_slots = 6

        self.netE_num_downsampling_sp = 2
        self.global_code_ch = 256 # 2048
        self.spatial_code_ch = 8
        self.latent_size = 64

    def __init__(self):
        super().__init__()
        self._set_hyper_params()
        num_upsamplings = self.netE_num_downsampling_sp
        blur_kernel = [1, 3, 3, 1] if self.use_antialias else [1]

        self.global_code_ch = self.global_code_ch + self.num_classes

        self.add_module(
            "SpatialCodeModulation",
            GeneratorModulation(self.global_code_ch, self.spatial_code_ch))

        in_channel = self.spatial_code_ch 
        for i in range(self.netG_num_base_resnet_layers):
            # gradually increase the number of channels
            out_channel = (i + 1) / self.netG_num_base_resnet_layers * 64
            #out_channel = (i + 1) / self.netG_num_base_resnet_layers * self.nf(0)
            out_channel = max(self.spatial_code_ch, round(out_channel))
            layer_name = "HeadResnetBlock%d" % i
            new_layer = ResolutionPreservingResnetBlock(
                self, in_channel, out_channel, self.global_code_ch, self.netG_use_noise)
            self.add_module(layer_name, new_layer)
            in_channel = out_channel

        for j in range(num_upsamplings):
            out_channel = self.nf(j + 1)
            layer_name = "UpsamplingResBlock%d" % (2 ** (3 + j))
            new_layer = UpsamplingResnetBlock(
                in_channel, out_channel, self.global_code_ch,
                blur_kernel, self.netG_use_noise)
            self.add_module(layer_name, new_layer)
            in_channel = out_channel

        last_layer = ToRGB(out_channel, self.global_code_ch,
                           blur_kernel=blur_kernel)
        self.add_module("ToRGB", last_layer)

        self.fc_layers = nn.Sequential(
                nn.Linear(self.latent_size, 8*8*8),
                nn.Linear(8*8*8, 8*8*8),
                )
      
        # instead of inject global code from input, we learn it.
        self.global_code = nn.Parameter(torch.randn((self.global_code_ch,)))
       
        dim_hidden = 128
        dim_output = 128


        # self.set_decoder = nn.Sequential(
        #         PMA(dim_hidden, num_heads=4, num_seeds=1, ln=True),
        #         SAB(dim_hidden, dim_hidden, num_heads=4, ln=True),
        #         SAB(dim_hidden, dim_hidden, num_heads=4, ln=True),
        #         nn.Linear(dim_hidden, dim_output))

    def nf(self, num_up):
        ch = [64, 64, 32]

        # ch = 128 * (2 ** (self.opt.netE_num_downsampling_sp - num_up))
        # ch = int(min(512, ch) * self.opt.netG_scale_capacity)

        return ch[num_up]

    def fix_and_gather_noise_parameters(self):
        params = []
        device = next(self.parameters()).device
        for m in self.modules():
            if type(m).__name__ == "NoiseInjection":
                assert m.image_size is not None, "One forward call should be made to determine size of noise parameters"
                m.fixed_noise = torch.nn.Parameter(torch.randn(m.image_size[0], 1, m.image_size[2], m.image_size[3], device=device))
                params.append(m.fixed_noise)
        return params

    def remove_noise_parameters(self, name):
        for m in self.modules():
            if type(m).__name__ == "NoiseInjection":
                m.fixed_noise = None


    def set_decode(self, spatial_code):
        ''' perform permutation invariant decoding
            Possible Options : 
            1) Pooling
            2) Set Attention Block
        '''
        return out 

    def forward(self, spatial_code, global_code = None):
        #############
        # For now, instead of use  global code from input,
        # we use learnable codes.
        #############
        bs = spatial_code.size(0)
        if global_code is None:
            global_code = self.global_code 
            global_code = repeat(global_code, 'd -> b d', b=bs)

        # # normalize
        # spatial_code = util.normalize(spatial_code)
        # global_code = util.normalize(global_code)

        # # we need to make it permutation invariant 
        # spatial_code = self.set_decoder(spatial_code)

        spatial_code = self.fc_layers(spatial_code)

        # reshape 
        spatial_code = rearrange(spatial_code, 'b (c h w) -> b c h w', 
                h=8, w=8, c=self.spatial_code_ch)

        x = spatial_code
#         x = self.SpatialCodeModulation(spatial_code, global_code)

        for i in range(self.netG_num_base_resnet_layers):
            resblock = getattr(self, "HeadResnetBlock%d" % i)
            x = resblock(x, global_code)

        for j in range(self.netE_num_downsampling_sp):
            key_name = 2 ** (3 + j)
            upsampling_layer = getattr(self, "UpsamplingResBlock%d" % key_name)
            x = upsampling_layer(x, global_code)
        rgb = self.ToRGB(x, global_code, None)

        return rgb


