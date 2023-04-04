import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import *
from .unet_blocks import *


class MGUNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_out_channels = (16,16,16) ,conv_depths=((64, 128, 256, 512, 1024), (64, 128, 256, 512, 1024), (64, 128, 256, 512, 1024), (64, 128, 256, 512, 1024))):
        assert len(conv_depths) > 2, 'conv_depths must have at least 3 members'
        assert len(mid_out_channels) == len(conv_depths)-1

        super(MGUNet2D, self).__init__()



        # defining encoder layers
        encoder_MG_layers = []
        layers = []
        layers.append(UNet2DFirst(in_channels=in_channels, out_channels=mid_out_channels[0], conv_depths=conv_depths[0]))
        layers.extend([UNet2DMiddle(in_channels=mid_out_channels[i], out_channels=mid_out_channels[i+1], conv_depths=conv_depths[i+1], conv_depths_prev=conv_depths[i]) for i in range(len(conv_depths)-2)])
        layers.append(UNet2DLast(in_channels=mid_out_channels[-1], out_channels=out_channels, conv_depths=conv_depths[-1], conv_depths_prev=conv_depths[-2]))


        self.layers = nn.Sequential(*layers)


    def forward(self, x, return_all=False):
        x_enc = x
        for layer in self.layers:
            x_enc = layer(x_enc)

        if not return_all:
            return x_enc[0] #segmentor only
        else:
            return x_enc #segmentor, classifier

