import torch
import torch.nn as nn
import numpy as np


class Skip(nn.Module):
    def __init__(self, in_ch, out_ch, skip_channels=(4, 4, 4, 4), channels=(8, 16, 32, 64)):
        super(Skip, self).__init__()
        self.scales = len(channels)
        assert(len(channels) == len(skip_channels))

        self.down = nn.ModuleList()
        self.up = nn.ModuleList()

        cur_ch = in_ch
        channels = [cur_ch] + list(channels)
        for i in range(self.scales):
            self.down.append(nn.Sequential(
                ConvolutionBlock(cur_ch, channels[i+1], stride=2),
                ConvolutionBlock(channels[i+1], channels[i+1], stride=1)))
            cur_ch = channels[i+1]

        for i in range(self.scales):
            self.up.append(UpBlock(
                in_ch=cur_ch,
                out_ch=channels[-i-1],
                skip_in_ch=channels[-i-2],
                skip_out_ch=skip_channels[-i-1]))
            cur_ch = channels[-i-1]

        self.outc = ConvolutionBlock(
            cur_ch, out_ch, kernel_size=1, use_bn=False, use_act=False)

    def forward(self, x0):
        xs = [x0]
        for i in range(self.scales):
            xs.append(self.down[i](xs[-1]))
        x = xs[-1]
        for i in range(self.scales):
            x = self.up[i](x, xs[-2-i])
        # return self.outc(x)
        return torch.sigmoid(self.outc(x))


class ConvolutionBlock(nn.Module):
    def __init__(self, in_f, out_f, kernel_size=3, stride=1, use_bias=True,
                 pad='reflection', use_bn=True, use_act=True):
        super(ConvolutionBlock, self).__init__()
        to_pad = int((kernel_size - 1) / 2)
        if pad == 'reflection':
            self.pad = nn.ReflectionPad2d(to_pad)
        else:
            self.pad = nn.ZeroPad2d(to_pad)
        self.convolution = nn.Conv2d(
            in_f, out_f, kernel_size, stride, bias=use_bias)
        self.bn = nn.BatchNorm2d(out_f)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.use_bn = use_bn
        self.use_act = use_act

    def forward(self, x):
        x = self.convolution(self.pad(x))
        if self.use_bn:
            x = self.bn(x)
        if self.use_act:
            x = self.act(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_in_ch, skip_out_ch=4,
                 kernel_size=3):
        super(UpBlock, self).__init__()
        self.skip = skip_out_ch > 0
        if skip_out_ch == 0:
            skip_out_ch = 1
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.skip_conv = ConvolutionBlock(
            skip_in_ch, skip_out_ch, kernel_size=1)
        self.concat = Concat()
        self.conv = nn.Sequential(ConvolutionBlock(in_ch + skip_out_ch, out_ch,
                                                   kernel_size=kernel_size),
                                  ConvolutionBlock(out_ch, out_ch,
                                                   kernel_size=1))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.skip_conv(x2)
        if not self.skip:
            x2 = x2 * 0
        x = self.concat(x1, x2)
        x = self.conv(x)
        return x


class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, *inputs):
        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(
                    inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])
        return torch.cat(inputs_, dim=1)
