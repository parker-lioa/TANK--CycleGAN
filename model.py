import torch
import torch.nn as nn
import utils


def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True, ln=False, activation=True):
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    elif ln:
        layers.append(nn.InstanceNorm2d(c_out))
    if activation:
        layers.append(nn.LeakyReLU(0.02, inplace=True))
    return nn.Sequential(*layers)


def deconv(c_in, c_out, k_size, stride=2, pad=1, output_padding=0, bn=True, ln=False, activation=True):
    layers = []
    layers.append(nn.ConvTranspose2d(
        c_in, c_out, k_size, stride, pad, output_padding=output_padding, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    elif ln:
        layers.append(nn.InstanceNorm2d(c_out))
    if activation:
        layers.append(nn.LeakyReLU(0.02, inplace=True))
    return nn.Sequential(*layers)


def residual(filter):
    layers = []
    layers.append(nn.Conv2d(filter, filter, 3, 1, 1, bias=False))
    layers.append(nn.BatchNorm2d(filter))
    layers.append(nn.LeakyReLU(0.02, inplace=True))
    layers.append(nn.Conv2d(filter, filter, 3, 1, 1, bias=False))
    layers.append(nn.BatchNorm2d(filter))
    return nn.Sequential(*layers)


class Disc(nn.Module):

    def __init__(self, in_dim, dim=64, bn=False, ln=True):
        super(Disc, self).__init__()

        self.model = nn.Sequential(
            conv(in_dim, dim, 4, bn=False, ln=False),
            conv(dim, dim * 2, 4, bn=bn, ln=ln),
            conv(dim * 2, dim * 4, k_size=4, bn=bn, ln=ln),
            conv(dim * 4, dim * 8, k_size=4, stride=1, bn=bn, ln=ln),
            conv(dim * 8, 1, k_size=4, stride=1, pad=1,
                 bn=False, ln=False, activation=False)
        )

    def forward(self, x):
        return self.model(x)


class Gen(nn.Module):

    def __init__(self, in_dim=3, out_dim=3, conv_dim=64, bn=False, ln=True):
        super(Gen, self).__init__()

        self.conv1 = conv(in_dim, conv_dim, 7, 1, 3, bn=bn, ln=ln)
        self.conv2 = conv(conv_dim, conv_dim * 2, 3, 2, 1, bn=bn, ln=ln)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 3, 2, 1, bn=bn, ln=ln)

        self.residual1 = residual(conv_dim * 4)
        self.residual2 = residual(conv_dim * 4)
        self.residual3 = residual(conv_dim * 4)
        self.residual4 = residual(conv_dim * 4)
        self.residual5 = residual(conv_dim * 4)
        self.residual6 = residual(conv_dim * 4)
        self.residual7 = residual(conv_dim * 4)
        self.residual8 = residual(conv_dim * 4)
        self.residual9 = residual(conv_dim * 4)

        self.deconv1 = deconv(conv_dim * 4, conv_dim * 2, 3, 2, 1, 1, bn=bn, ln=ln)
        self.deconv2 = deconv(conv_dim * 2, conv_dim, 3, 2, 1, 1, bn=bn, ln=ln)
        self.deconv3 = deconv(conv_dim, out_dim, 7, 1, 3,
                              bn=False, ln=False, activation=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + self.residual1(x)
        x = x + self.residual2(x)
        x = x + self.residual3(x)
        x = x + self.residual4(x)
        x = x + self.residual5(x)
        x = x + self.residual6(x)
        x = x + self.residual7(x)
        x = x + self.residual8(x)
        x = x + self.residual9(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = torch.tanh(x)

        return x


def define_G(in_dim, out_dim, conv_dim, norm_type):
    net = None

    if norm_type == "batch":
        net = Gen(in_dim, out_dim, conv_dim, bn=True, ln=False)
    elif norm_type == "instance":
        net = Gen(in_dim, out_dim, conv_dim, bn=False, ln=True)
    else:
        net = Gen(in_dim, out_dim, conv_dim, bn=False, ln=False)

    utils.init_net(net)

    return net


def define_D(in_dim, conv_dim, norm_type):
    net = None

    if norm_type == "batch":
        net = Disc(in_dim, dim=conv_dim, bn=True, ln=False)
    elif norm_type == "instance":
        net = Disc(in_dim, dim=conv_dim, bn=False, ln=True)
    else:
        net = Disc(in_dim, dim=conv_dim, bn=False, ln=False)

    utils.init_net(net)

    return net
