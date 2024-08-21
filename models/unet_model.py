'''
borrowed from https://github.com/milesial/Pytorch-UNet/blob/master/unet
'''

""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, input_channels, output_channels, base_ch=64, mult_chs=[1,2,4,8], bilinear=True):
        super(UNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.bilinear = bilinear

        self.base_ch = base_ch
        self.mult_chs = mult_chs

        self.ch = [base_ch, 
                base_ch*mult_chs[0],
                base_ch*mult_chs[1],
                base_ch*mult_chs[2],
                base_ch*mult_chs[3],
                ]

        self.inc = DoubleConv(self.input_channels, self.ch[0])
        self.down1 = Down(self.ch[0], self.ch[1])
        self.down2 = Down(self.ch[1], self.ch[2])
        self.down3 = Down(self.ch[2], self.ch[3])
        factor = 2 if bilinear else 1
        self.down4 = (Down(self.ch[3], self.ch[4] // factor))
        self.up1 = (Up(self.ch[4]//factor + self.ch[3], self.ch[3] // factor, bilinear))
        self.up2 = (Up(self.ch[3]//factor + self.ch[2], self.ch[2] // factor, bilinear))
        self.up3 = (Up(self.ch[2]//factor + self.ch[1], self.ch[1] // factor, bilinear))
        self.up4 = (Up(self.ch[1]//factor + self.ch[0], self.ch[1], bilinear))
        self.outc = OutConv(self.ch[1], output_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


