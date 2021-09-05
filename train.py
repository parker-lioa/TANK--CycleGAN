import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from torchvision import transforms
import torchvision
from torchvision.utils import make_grid
import argparse
import time
import os
import numpy as np
from PIL import Image

from cycle_gan import CycleGAN
from dataset import CycleGANDataSet
from utils import make_grid_and_save_image, cpu_or_gpu, CycleGAN_tensorboard
import imageio


def train(config):

    # define transform

    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([127.5], [127.5])
    ])

    # load dataset

    dataset = CycleGANDataSet(config.data_path1, config.data_path2, transform)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, pin_memory=torch.cuda.is_available())

    # create model

    model = CycleGAN(config)

    if config.pre_train:

        if torch.cuda.is_available():

            model.G1.load_state_dict(torch.load(config.pre_train_G1))
            model.G2.load_state_dict(torch.load(config.pre_train_G2))
            model.D1.load_state_dict(torch.load(config.pre_train_D1))
            model.D2.load_state_dict(torch.load(config.pre_train_D2))

        else:

            model.G1.load_state_dict(torch.load(
                config.pre_train_G1, map_location=torch.device('cpu')))
            model.G2.load_state_dict(torch.load(
                config.pre_train_G2, map_location=torch.device('cpu')))
            model.D1.load_state_dict(torch.load(
                config.pre_train_D1, map_location=torch.device('cpu')))
            model.D2.load_state_dict(torch.load(
                config.pre_train_D2, map_location=torch.device('cpu')))

    # initialize SummaryWriter

    writer = SummaryWriter()

    steps = 0

    for epoch in range(config.epochs):

        start_time = time.time()

        g_loss = 0
        d_loss = 0

        for imgA, imgB in dataloader:

            # customized code -> rgb to grayscale

            imgB = imgB.cpu()
            imgB = [transforms.ToPILImage()(x) for x in imgB]
            imgB = [transforms.Grayscale()(x) for x in imgB]
            imgB = [transforms.ToTensor()(x) for x in imgB]
            imgB = torch.stack(imgB)

            cpu_or_gpu(imgB)

            # end of customized section

            model.set_inputs(cpu_or_gpu(imgA), cpu_or_gpu(imgB))

            if steps % 2 == 0:
                model.optimize_parameters(only_D=False)
            else:
                model.optimize_parameters(only_D=True)

            d_loss += model.d_loss.data
            g_loss += model.g_loss.data

            # if torch.cuda.is_available():
            #     model.set_inputs(imgA.cuda(), imgB.cuda())
            # else:
            #     model.set_inputs(imgA, imgB)
            steps += 1

        print('Epoch:{}/{} in {} sec , loss_G: {} , loss_D: {}'.format(epoch+1,
              config.epochs, time.time()-start_time, g_loss, d_loss))

        CycleGAN_tensorboard(writer, epoch+1, model.fake_A,
                             model.fake_B, g_loss, d_loss)

        # if (epoch+1) % 10 == 0:
        #     make_grid_and_save_image(
        #         model.real_A, config.log_path+'/real_A'+'/real_A_'+str(epoch+1)+'.png')
        #     make_grid_and_save_image(
        #         model.fake_B, config.log_path+'/fake_B'+'/fake_B_'+str(epoch+1)+'.png')
        #     make_grid_and_save_image(
        #         model.real_B, config.log_path+'/real_B'+'/real_B_'+str(epoch+1)+'.png')
        #     make_grid_and_save_image(
        #         model.fake_A, config.log_path+'/fake_A'+'/fake_A_'+str(epoch+1)+'.png')

        if (epoch+1) % 500 == 0:

            torch.save(model.G1.state_dict(), config.model_path + '/G1' +
                       '/G1-'+str(epoch+1)+'.pkl')
            torch.save(model.G2.state_dict(), config.model_path + '/G2' +
                       '/G2-'+str(epoch+1)+'.pkl')
            torch.save(model.D1.state_dict(), config.model_path + '/D1' +
                       '/D1-'+str(epoch+1)+'.pkl')
            torch.save(model.D2.state_dict(), config.model_path + '/D2' +
                       '/D2-'+str(epoch+1)+'.pkl')


if __name__ == '__main__':

    cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    # path

    parser.add_argument('--model_path', type=str, default='.\models')
    parser.add_argument('--log_path', type=str, default='.\logs')
    parser.add_argument('--data_path1', type=str, default='.\datasets\A')
    parser.add_argument('--data_path2', type=str, default='.\datasets\B')
    parser.add_argument('--pre_train', action='store_true')
    parser.add_argument('--pre_train_G1', type=str)
    parser.add_argument('--pre_train_G2', type=str)
    parser.add_argument('--pre_train_D1', type=str)
    parser.add_argument('--pre_train_D2', type=str)

    # model-parameters

    parser.add_argument('--A_dim', type=int, default=1)
    parser.add_argument('--B_dim', type=int, default=1)

    # hyper-parameters

    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='wgan-gp')
    parser.add_argument('--clip_value', type=float, default=0.01)
    parser.add_argument('--cycle_weight', type=float, default=10)
    parser.add_argument('--idt_weight', type=float, default=0)

    config = parser.parse_args()

    train(config)
