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
from options import train_opt


def train(config):
    # load dataset

    dataset = CycleGANDataSet(
        config.data_path1, config.data_path2)

    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, pin_memory=torch.cuda.is_available())

    # create model

    model = CycleGAN(config)

    if config.pre_train:
        model.load_model(config.pre_epoch)

    # initialize SummaryWriter

    writer = SummaryWriter()

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

            loss_description = 'Epoch: %s/%s ' % (epoch, config.epochs)
            losses = model.get_losses()

            for loss_key in losses:
                loss_description += "%s: %.2f  " % (loss_key, losses[loss_key])

            progress_bar.set_description(loss_description)
            model.tensorboard_scalar_log(writer, steps)

        # model.update_lr()
        # print(model.get_lr())

        if epoch % config.save_image_period == 0:
            model.tensorboard_image_log(writer, epoch, sample_number=16)

        if epoch % config.save_period == 0:
            model.save_model(epoch, config.model_path)


if __name__ == '__main__':
    cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    train_opt(parser)

    config = parser.parse_args()

    train(config)
