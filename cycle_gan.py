import torch
from torch import optim
from torch.nn import init
from torch.autograd import grad
import numpy as np
import os
import time
from PIL import Image

from model import define_D, define_G
from utils import set_requires_grad, ImagePool


# build model

class CycleGAN():

    def __init__(self, opt):

        # initialize hyper-parameters

        self.cycle_weight = opt.cycle_weight
        self.idt_weight = opt.idt_weight
        self.lr = opt.lr
        self.clip = opt.clip_value

        self.G1 = define_G(in_dim=opt.A_dim, out_dim=opt.B_dim,
                           conv_dim=64, norm_type="instance")  # generate fake B

        self.G2 = define_G(in_dim=opt.B_dim, out_dim=opt.A_dim,
                           conv_dim=64, norm_type="instance")  # generate fake A

        self.D1 = define_D(in_dim=opt.B_dim, conv_dim=64,
                           norm_type="instance")  # judge input is B or not

        self.D2 = define_D(in_dim=opt.A_dim, conv_dim=64,
                           norm_type="instance")  # judge input is A or not


        self.cycle_loss = torch.nn.L1Loss()
        self.idt_loss = torch.nn.L1Loss()

        self.fake_A_pool = ImagePool(opt.batch_size-1)
        self.fake_B_pool = ImagePool(opt.batch_size-1)

        self.activate_gp = False
        self.loss_option = opt.loss

        if opt.loss == 'lsgan':
            self.d_loss_function = torch.nn.MSELoss()
            self.g_loss_function = torch.nn.MSELoss()
            self.needlabel = True

        elif opt.loss == 'wgan':
            self.d_loss_function = torch.mean
            self.g_loss_function = torch.mean
            self.needlabel = False

        elif opt.loss == 'wgan-gp':
            self.d_loss_function = torch.mean
            self.g_loss_function = torch.mean
            self.needlabel = False
            self.activate_gp = True

        if opt.loss != 'wgan':

            self.g_optimizer = optim.Adam(
                list(self.G1.parameters())+list(self.G2.parameters()), self.lr, betas=[0, 0.9])

            self.d_optimizer = optim.Adam(
                list(self.D1.parameters())+list(self.D2.parameters()), self.lr, betas=[0, 0.9])

        elif opt.loss == 'wgan':

            self.g_optimizer = optim.RMSprop(
                list(self.G1.parameters())+list(self.G2.parameters()), self.lr)

            self.d_optimizer = optim.RMSprop(
                list(self.D1.parameters())+list(self.D2.parameters()), self.lr)

    def set_inputs(self, domainA, domainB):

        self.real_A = domainA
        self.real_B = domainB

    def forward(self):

        self.fake_B = self.G1(self.real_A)
        self.rec_A = self.G2(self.fake_B)
        self.fake_A = self.G2(self.real_B)
        self.rec_B = self.G1(self.fake_A)

    def gradient_penalty(self, Disc, real_data, fake_data):

        if torch.cuda.is_available():
            alpha = torch.rand(real_data.size()[0], 1, 1, 1).cuda()
        else:
            alpha = torch.rand(real_data.size()[0], 1, 1, 1)

        x_hat = alpha * real_data + (1-alpha) * fake_data
        x_hat.requires_grad_(True)

        disc_out = Disc(x_hat)

        if torch.cuda.is_available():
            grad_outputs = torch.ones(disc_out.size()).cuda()
        else:
            grad_outputs = torch.ones(disc_out.size())

        gradient = torch.autograd.grad(outputs=disc_out, inputs=x_hat, grad_outputs=grad_outputs,
                                       create_graph=True, retain_graph=True)[0].view(x_hat.size(0), -1)

        gradient_penalty = ((gradient.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def backward_G(self):

        d1_out = self.D1(self.fake_B)
        d2_out = self.D2(self.fake_A)

        if self.needlabel:

            g1_adv_loss = self.g_loss_function(d1_out, torch.ones_like(d1_out))
            g2_adv_loss = self.d_loss_function(d2_out, torch.ones_like(d2_out))

        else:

            g1_adv_loss = - \
                self.g_loss_function(d1_out.view(d1_out.size()[0], -1))
            g2_adv_loss = - \
                self.g_loss_function(d2_out.view(d2_out.size()[0], -1))

        forward_cycle_loss = self.cycle_loss(
            self.rec_A, self.real_A)*self.cycle_weight
        backward_cycle_loss = self.cycle_loss(
            self.rec_B, self.real_B)*self.cycle_weight

        if self.idt_weight > 0:
            g1_idt_loss = self.idt_loss(
                self.fake_B, self.real_A)*self.idt_weight
            g2_idt_loss = self.idt_loss(
                self.fake_A, self.real_B)*self.idt_weight
            self.g_loss = g1_adv_loss + g2_adv_loss + forward_cycle_loss + \
                backward_cycle_loss + g1_idt_loss + g2_idt_loss
        else:
            self.g_loss = g1_adv_loss + g2_adv_loss + \
                forward_cycle_loss + backward_cycle_loss

        self.g_loss.backward()

    def backward_D(self):

        d1_real_out = self.D1(self.real_B)
        d2_real_out = self.D2(self.real_A)
        d1_fake_out = self.D1(self.fake_B)
        d2_fake_out = self.D2(self.fake_A)

        if self.needlabel:

            d1_loss = self.d_loss_function(d1_real_out, torch.ones_like(
                d1_real_out)) + self.d_loss_function(d1_fake_out, torch.zeros_like(d1_fake_out))
            d2_loss = self.d_loss_function(d2_real_out, torch.ones_like(
                d2_real_out)) + self.d_loss_function(d2_fake_out, torch.zeros_like(d2_fake_out))

        else:

            d1_loss = - \
                self.d_loss_function(d1_real_out) + \
                self.d_loss_function(d1_fake_out)
            d2_loss = - \
                self.d_loss_function(d2_real_out) + \
                self.d_loss_function(d2_fake_out)

        self.d_loss = d1_loss + d2_loss

        if self.activate_gp:

            d2_gradient_penalty = self.gradient_penalty(
                self.D2, self.real_A, self.fake_A)
            d1_gradient_penalty = self.gradient_penalty(
                self.D1, self.real_B, self.fake_B)

            self.d_loss += d1_gradient_penalty + d2_gradient_penalty

        self.d_loss.backward()

    def weight_clip(self):
        for param in self.D1.parameters():
            param.data.clamp_(-self.clip, self.clip)
        for param in self.D2.parameters():
            param.data.clamp_(-self.clip, self.clip)

    def optimize_parameters(self, only_D):
        if only_D == False:
            self.forward()
            # uneccesery when training G
            set_requires_grad([self.D1, self.D2], requires_grad=False)
            self.g_optimizer.zero_grad()
            self.backward_G()
            self.g_optimizer.step()
        set_requires_grad([self.D1, self.D2], True)
        self.d_optimizer.zero_grad()
        self.forward()
        self.backward_D()
        self.d_optimizer.step()
        if self.loss_option == 'wgan':
            self.weight_clip()
