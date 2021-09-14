import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from torchvision import transforms
from torchvision.transforms.transforms import Grayscale
from torchvision.utils import make_grid
import argparse
import time
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from cycle_gan import CycleGAN
from dataset import CycleGANDataSet
from utils import make_grid_and_save_image, cpu_or_gpu, CycleGAN_tensorboard, Denormalize
import imageio



def train(config):

    # define transform

    transform_A = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.RandomCrop((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_B = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((200, 200)),
        transforms.RandomCrop((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    # load dataset

    dataset = CycleGANDataSet(
        config.data_path1, config.data_path2, transform_A, transform_B)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, pin_memory=torch.cuda.is_available())

    # create model

    model = CycleGAN(config)

    if config.pre_train:
        model.load_model(config.pre_epoch)

    # initialize SummaryWriter

    writer = SummaryWriter(log_dir=config.log_path)

    steps = 0

    for i in range(config.epochs):

        epoch = i + 1

        start_time = time.time()

        progress_bar = tqdm(dataloader)

        for imgA, imgB in progress_bar:

            model.forward(cpu_or_gpu(imgA), cpu_or_gpu(imgB))

            if steps % config.n_critic == 0:
                model.optimize_G()

            model.optimize_D()

            steps += 1

            loss_description = ''
            losses = model.get_losses()

            for loss_key in losses:
                loss_description += "%s: %.2f  " % (loss_key, losses[loss_key])

            progress_bar.set_description(loss_description)
            model.tensorboard_scalar_log(writer, steps)

        model.update_lr()

        if epoch % config.save_image_period == 0:
            model.tensorboard_image_log(writer, epoch, sample_number=16)

        if epoch % config.save_period == 0:
            model.save_model(epoch)


if __name__ == '__main__':
    cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    # special options

    parser.add_argument('--amp_train', action='store_true')

    # path

    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--save_period', type=int, default=100)
    parser.add_argument('--save_image_period', type=int, default=20)
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--data_path1', type=str, default='./datasets/A')
    parser.add_argument('--data_path2', type=str, default='./datasets/B')
    parser.add_argument('--pre_train', action='store_true')
    parser.add_argument('--pre_epoch', type=int, default=0)

    # model-parameters

    parser.add_argument('--A_dim', type=int, default=1)
    parser.add_argument('--B_dim', type=int, default=1)

    # hyper-parameters

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='wgan-gp')
    parser.add_argument('--n_critic', type=int, default=1)
    parser.add_argument('--clip_value', type=float, default=0.01)
    parser.add_argument('--cycle_weight', type=float, default=10)
    parser.add_argument('--idt_weight', type=float, default=0)

    config = parser.parse_args()

    train(config)
