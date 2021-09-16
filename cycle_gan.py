import functools
import math
from collections import OrderedDict

import torch
import torchvision.utils
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import init
from torch.autograd import grad
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import time
from PIL import Image

from model import define_D, define_G
from utils import set_requires_grad, ImagePool, warmup_cosine


# build model

class CycleGAN:

    def __init__(self, opt):

        # initialize hyper-parameters

        self.cycle_weight = opt.cycle_weight
        self.idt_weight = opt.idt_weight
        self.lr = opt.lr
        self.clip = opt.clip_value
        self.if_amp = opt.amp_train
        self.model_path = opt.model_path

        if self.if_amp:
            self.scaler = GradScaler()

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

        self.fake_A_pool = ImagePool(50)
        self.fake_B_pool = ImagePool(50)

        self.activate_gp = False
        self.loss = opt.loss
        self.loss_names = ['g1', 'g2', 'idt1', 'idt2', 'cycle1', 'cycle2', 'd1', 'd2']

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
                list(self.G1.parameters()) + list(self.G2.parameters()), self.lr, betas=[0, 0.99])

            self.d_optimizer = optim.Adam(
                list(self.D1.parameters()) + list(self.D2.parameters()), self.lr, betas=[0, 0.99])

        elif opt.loss == 'wgan':

            self.g_optimizer = optim.RMSprop(
                list(self.G1.parameters()) + list(self.G2.parameters()), self.lr)

            self.d_optimizer = optim.RMSprop(
                list(self.D1.parameters()) + list(self.D2.parameters()), self.lr)

        self.total_epoch = opt.epochs
        lr_func = functools.partial(warmup_cosine, up_period=self.total_epoch * 0.1, y_intercept=1e-5, peak=self.lr,
                                    total_period=self.total_epoch, alpha=0.2)

        # self.scheduler_g = LambdaLR(self.g_optimizer, lr_func)
        # self.scheduler_d = LambdaLR(self.d_optimizer, lr_func)

    def forward(self, domainA, domainB):

        self.real_A = domainA
        self.real_B = domainB
        self.fake_B = self.G1(self.real_A)
        self.rec_A = self.G2(self.fake_B)
        self.fake_A = self.G2(self.real_B)
        self.rec_B = self.G1(self.fake_A)

    def gradient_penalty(self, Disc, real_image, fake_image):

        if torch.cuda.is_available():
            alpha = torch.rand(real_image.size()[0], 1, 1, 1).cuda()
        else:
            alpha = torch.rand(real_image.size()[0], 1, 1, 1)

        x_hat = alpha * real_image + (1 - alpha) * fake_image
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

            self.loss_g1 = self.g_loss_function(d1_out, torch.ones_like(d1_out))
            self.loss_g2 = self.d_loss_function(d2_out, torch.ones_like(d2_out))

        else:

            self.loss_g1 = - \
                self.g_loss_function(d1_out.view(d1_out.size()[0], -1))
            self.loss_g2 = - \
                self.g_loss_function(d2_out.view(d2_out.size()[0], -1))

        self.loss_cycle1 = self.cycle_loss(
            self.rec_A, self.real_A) * self.cycle_weight
        self.loss_cycle2 = self.cycle_loss(
            self.rec_B, self.real_B) * self.cycle_weight

        if self.idt_weight > 0:
            self.loss_idt1 = self.idt_loss(
                self.fake_B, self.real_A) * self.idt_weight
            self.loss_idt2 = self.idt_loss(
                self.fake_A, self.real_B) * self.idt_weight
        else:
            self.loss_idt1 = 0
            self.loss_idt2 = 0

        loss_g = self.loss_g1 + self.loss_g2 + self.loss_cycle1 + self.loss_cycle2 + self.loss_idt1 + self.loss_idt2

        if self.if_amp:
            self.scaler.scale(loss_g).backward()
        else:
            loss_g.backward()

    def backward_basic_D(self, netD, real, fake):

        real_score = netD(real)
        fake_score = netD(fake.detach())

        if self.needlabel:

            d_loss = self.d_loss_function(real_score, torch.ones_like(
                real_score)) + self.d_loss_function(fake_score, torch.zeros_like(fake_score))

        else:

            d_loss = -self.d_loss_function(real_score) + self.d_loss_function(fake_score)

        d_loss = d_loss * 0.5

        if self.activate_gp:
            gp_loss = self.gradient_penalty(netD, real, fake.detach())
            d_loss += gp_loss

        if self.if_amp:
            self.scaler.scale(d_loss).backward()
        else:
            d_loss.backward()

        return d_loss

    def backward_D1(self):

        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_d1 = self.backward_basic_D(self.D1, self.real_B, fake_B)

    def backward_D2(self):

        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_d2 = self.backward_basic_D(self.D2, self.real_A, fake_A)

    def weight_clip(self):

        for param in self.D1.parameters():
            param.data.clamp_(-self.clip, self.clip)
        for param in self.D2.parameters():
            param.data.clamp_(-self.clip, self.clip)

    def optimize_D(self):

        set_requires_grad([self.D1, self.D2], True)

        if self.if_amp:
            with autocast():
                self.d_optimizer.zero_grad()
                self.backward_D1()
                self.backward_D2()
            self.scaler.step(self.d_optimizer)
            self.scaler.update()
        else:
            self.d_optimizer.zero_grad()
            self.backward_D1()
            self.backward_D2()
            self.d_optimizer.step()

        if self.loss == 'wgan':
            self.weight_clip()

    def optimize_G(self):

        set_requires_grad([self.D1, self.D2], requires_grad=False)

        if self.if_amp:
            with autocast():
                self.g_optimizer.zero_grad()
                self.backward_G()
            self.scaler.step(self.g_optimizer)
            self.scaler.update()

        else:
            self.g_optimizer.zero_grad()
            self.backward_G()
            self.g_optimizer.step()

    def save_model(self, model_path, epoch):

        torch.save(self.G1.state_dict(), model_path + '/G1-' + str(epoch) + '.pkl')
        torch.save(self.G2.state_dict(), model_path + '/G2-' + str(epoch) + '.pkl')
        torch.save(self.D1.state_dict(), model_path + '/D1-' + str(epoch) + '.pkl')
        torch.save(self.D2.state_dict(), model_path + '/D2-' + str(epoch) + '.pkl')

    def load_model(self, model_path, epoch=None):

        if epoch is not None:

            if torch.cuda.is_available():
                self.G1.load_state_dict(torch.load(model_path + '/G1-' + str(epoch) + '.pkl'))
                self.G2.load_state_dict(torch.load(model_path + '/G2-' + str(epoch) + '.pkl'))
                self.D1.load_state_dict(torch.load(model_path + '/D1-' + str(epoch) + '.pkl'))
                self.D2.load_state_dict(torch.load(model_path + '/D2-' + str(epoch) + '.pkl'))
            else:
                self.G1.load_state_dict(
                    torch.load(model_path + '/G1-' + str(epoch) + '.pkl', map_location=torch.device('cpu')))
                self.G2.load_state_dict(torch.load(model_path + '/G2-' + str(epoch) + '.pkl',
                                                   map_location=torch.device('cpu')))
                self.D1.load_state_dict(torch.load(model_path + '/D1-' + str(epoch) + '.pkl',
                                                   map_location=torch.device('cpu')))
                self.D2.load_state_dict(torch.load(model_path + '/D2-' + str(epoch) + '.pkl',
                                                   map_location=torch.device('cpu')))
        else:
            if torch.cuda.is_available():
                self.G1.load_state_dict(torch.load(model_path + '/G1' + '.pkl'))
                self.G2.load_state_dict(torch.load(model_path + '/G2' + '.pkl'))
                self.D1.load_state_dict(torch.load(model_path + '/D1' + '.pkl'))
                self.D2.load_state_dict(torch.load(model_path + '/D2' + '.pkl'))
            else:
                self.G1.load_state_dict(torch.load(model_path + '/G1' + '.pkl',
                                                   map_location=torch.device('cpu')))
                self.G2.load_state_dict(torch.load(model_path + '/G2' + '.pkl',
                                                   map_location=torch.device('cpu')))
                self.D1.load_state_dict(torch.load(model_path + '/D1' + '.pkl',
                                                   map_location=torch.device('cpu')))
                self.D2.load_state_dict(torch.load(model_path + '/D2' + '.pkl',
                                                   map_location=torch.device('cpu')))

    def get_losses(self):

        losses = OrderedDict()

        for name in self.loss_names:
            temp = getattr(self, 'loss_' + name)
            if torch.is_tensor(temp):
                losses[name] = temp.item()
            else:
                losses[name] = temp

        return losses

    def tensorboard_scalar_log(self, writer, step):

        for name in self.loss_names:
            temp = getattr(self, 'loss_' + name)
            if torch.is_tensor(temp):
                writer.add_scalar(name, temp.item(), step)
            else:
                writer.add_scalar(name, temp, step)

    def tensorboard_image_log(self, writer, epoch, sample_number):

        sample_a = self.fake_A_pool.sample(sample_number)
        sample_b = self.fake_B_pool.sample(sample_number)

        grid_a = torchvision.utils.make_grid(sample_a, nrow=math.floor(math.sqrt(sample_number)), normalize=True)
        grid_b = torchvision.utils.make_grid(sample_b, nrow=math.floor(math.sqrt(sample_number)), normalize=True)

        writer.add_image('Fake A', grid_a, global_step=epoch)
        writer.add_image('Fake B', grid_b, global_step=epoch)

    def update_lr(self):

        self.scheduler_g.step()
        self.scheduler_d.step()

    def get_lr(self):
        return self.scheduler_d.get_last_lr()

    def eval(self, real_A, real_B):

        set_requires_grad([self.D1, self.D2, self.G1, self.G2], requires_grad=False)

        fake_A = self.G2(real_B)
        fake_B = self.G1(real_A)

        rec_A = self.G2(fake_A)
        rec_B = self.G1(fake_B)

        return (fake_A, fake_B, rec_A, rec_B)
