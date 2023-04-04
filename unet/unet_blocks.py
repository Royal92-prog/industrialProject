import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import *


class UNet2DFirst(nn.Module):
    def __init__(self, in_channels, out_channels, conv_depths=(64, 128, 256, 512, 1024)):
        assert len(conv_depths) > 2, 'conv_depths must have at least 3 members'

        super(UNet2DFirst, self).__init__()

        # defining encoder layers
        encoder_layers = []
        encoder_layers.append(First2D(in_channels, conv_depths[0], conv_depths[0]))
        encoder_layers.extend([Encoder2D(conv_depths[i], conv_depths[i + 1], conv_depths[i + 1])
                               for i in range(len(conv_depths)-2)])
        #self, in_channels, middle_channels, out_channels,  dropout=False, downsample_kernel=2 ):

        # defining decoder layers
        decoder_layers = []
        decoder_layers.extend([Decoder2D(2 * conv_depths[i + 1], 2 * conv_depths[i], 2 * conv_depths[i], conv_depths[i])
                               for i in reversed(range(len(conv_depths)-2))])
        #(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False)
        # decoder_layers.append(Last2D(conv_depths[1], conv_depths[0], out_channels))
        decoder_layers.append(Last2D(2*conv_depths[0], conv_depths[0], out_channels))
        #BUG: should be decoder_layers.append(Last2D(2*conv_depths[0], conv_depths[0], out_channels))
        #If 2*conv_depths[0] == conv_depths[1], as it usually is, it will work, but it doesn't really have to be so, so this might break
        # (self, in_channels, middle_channels, out_channels, softmax=False):

        # encoder, center and decoder layers
        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.center = Center2D(conv_depths[-2], conv_depths[-1], conv_depths[-1], conv_depths[-2])
        #(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        self.decoder_layers = nn.Sequential(*decoder_layers)

    def forward(self, x, return_all=False):
        x_enc = [x]
        for enc_layer in self.encoder_layers:
            x_enc.append(enc_layer(x_enc[-1]))

        x_dec = [self.center(x_enc[-1])]
        for dec_layer_idx, dec_layer in enumerate(self.decoder_layers):
            x_opposite = x_enc[-1-dec_layer_idx]
            x_cat = torch.cat(
                [pad_to_shape(x_dec[-1], x_opposite.shape), x_opposite],
                dim=1
            )
            x_dec.append(dec_layer(x_cat))

        if not return_all:
            # return x_dec[-1]
            return x_dec #Return all decoder outputs for subsequent input
        else:
            return x_enc + x_dec


class UNet2DMiddle(nn.Module):
    def __init__(self, in_channels, out_channels, conv_depths=(64, 128, 256, 512, 1024), conv_depths_prev=(64, 128, 256, 512, 1024)):
        '''

        :param in_channels: output of the previous UNet layer's "Last" block
        :param out_channels: output of this layer's "Last" block
        :param conv_depths:
        :param conv_depths_prev: conv_depths of the previous unet layer
        '''
        assert len(conv_depths) > 2, 'conv_depths must have at least 3 members'
        assert len(conv_depths_prev) > 2, 'conv_depths_prev must have at least 3 members'

        super(UNet2DMiddle, self).__init__()

        # defining encoder layers
        encoder_layers = []
        encoder_layers.append(Encoder2D(in_channels+conv_depths[0], conv_depths[1], conv_depths[1]))
        encoder_layers.extend([Encoder2D(conv_depths[i]+conv_depths_prev[i], conv_depths[i + 1], conv_depths[i + 1])
                               for i in range(1,len(conv_depths)-2)])

        # defining decoder layers
        decoder_layers = []
        decoder_layers.extend([Decoder2D(2 * conv_depths[i + 1], 2 * conv_depths[i], 2 * conv_depths[i], conv_depths[i])
                               for i in reversed(range(len(conv_depths)-2))])
        decoder_layers.append(Last2D(in_channels+conv_depths[0], conv_depths[0], out_channels))
        # (self, in_channels, middle_channels, out_channels, softmax=False):

        # encoder, center and decoder layers
        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.center = Center2D(2*conv_depths[-2], conv_depths[-1], conv_depths[-1], conv_depths[-2])
        self.decoder_layers = nn.Sequential(*decoder_layers)

    def forward(self, x, return_all=False):
        x_enc = [x[-1]]
        for enc_layer_idx,enc_layer in enumerate(self.encoder_layers):
            x_prev = x[-1-enc_layer_idx-1]
            x_cat = torch.cat(
                [pad_to_shape(x_enc[-1], x_prev.shape), x_prev],
                dim=1
            )
            x_enc.append(enc_layer(x_cat))

        x_prev = x[0]
        x_cat = torch.cat(
            [pad_to_shape(x_enc[-1], x_prev.shape), x_prev],
            dim=1
        )
        x_dec = [self.center(x_cat)]

        for dec_layer_idx, dec_layer in enumerate(self.decoder_layers):
            x_opposite = x_enc[-1-dec_layer_idx]
            x_cat = torch.cat(
                [pad_to_shape(x_dec[-1], x_opposite.shape), x_opposite],
                dim=1
            )
            x_dec.append(dec_layer(x_cat))

        if not return_all:
            return x_dec
        else:
            return x_enc + x_dec
        #done

class UNet2DLast(nn.Module):
    def __init__(self, in_channels, out_channels, conv_depths=(64, 128, 256, 512, 1024), conv_depths_prev=(64, 128, 256, 512, 1024)):
        assert len(conv_depths) > 2, 'conv_depths must have at least 3 members'
        assert len(conv_depths_prev) > 2, 'conv_depths_prev must have at least 3 members'

        super(UNet2DLast, self).__init__()

        # defining encoder layers
        encoder_layers = []
        encoder_layers.append(Encoder2D(in_channels + conv_depths[0], conv_depths[1], conv_depths[1]))
        encoder_layers.extend([Encoder2D(conv_depths[i] + conv_depths_prev[i], conv_depths[i + 1], conv_depths[i + 1])
                               for i in range(1, len(conv_depths) - 2)])

        # defining decoder layers
        decoder_layers = []
        decoder_layers.extend([Decoder2D(2 * conv_depths[i + 1], 2 * conv_depths[i], 2 * conv_depths[i], conv_depths[i])
                               for i in reversed(range(len(conv_depths) - 2))])
        decoder_layers.append(Last2D(in_channels + conv_depths[0], conv_depths[0], out_channels))

        # encoder, center and decoder layers
        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.center = Center2D(2*conv_depths[-2], conv_depths[-1], conv_depths[-1], conv_depths[-2])
        self.decoder_layers = nn.Sequential(*decoder_layers)

    def forward(self, x, return_all=False):
        x_enc = [x[-1]]
        for enc_layer_idx, enc_layer in enumerate(self.encoder_layers):
            x_prev = x[-1 - enc_layer_idx - 1]
            x_cat = torch.cat(
                [pad_to_shape(x_enc[-1], x_prev.shape), x_prev],
                dim=1
            )
            x_enc.append(enc_layer(x_cat))

        x_prev = x[0]
        x_cat = torch.cat(
            [pad_to_shape(x_enc[-1], x_prev.shape), x_prev],
            dim=1
        )
        x_dec = [self.center(x_cat)]

        for dec_layer_idx, dec_layer in enumerate(self.decoder_layers):
            x_opposite = x_enc[-1 - dec_layer_idx]
            x_cat = torch.cat(
                [pad_to_shape(x_dec[-1], x_opposite.shape), x_opposite],
                dim=1
            )
            x_dec.append(dec_layer(x_cat))

        if not return_all:
            return x_dec[-1], x_dec[0]
        else:
            return x_enc + x_dec
        # done

def pad_to_shape(this, shp):
    """
    Pads this image with zeroes to shp.
    Args:
        this: image tensor to pad
        shp: desired output shape

    Returns:
        Zero-padded tensor of shape shp.
    """
    if len(shp) == 4:
        pad = (0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2])
    elif len(shp) == 5:
        pad = (0, shp[4] - this.shape[4], 0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2])
    return F.pad(this, pad)
