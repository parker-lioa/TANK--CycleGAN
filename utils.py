import random
import torch
import torchvision
import numpy as np
import imageio
from torch.nn import init


def init_weight(m):

    classname = m.__class__.__name__
    if classname.find('conv') != -1 or classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.2)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Norm') != -1:
        init.normal_(m.weight.data, 1, 0.2)


def init_net(net):

    if torch.cuda.is_available():
        net.cuda()

    net.apply(init_weight)

    return net


def set_requires_grad(nets, requires_grad=True):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(
                        0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        # collect all the images and return
        return_images = torch.cat(return_images, 0)
        return return_images


def make_grid_and_save_image(tensor, path):

    grid_image = (torchvision.utils.make_grid(
        tensor, normalize=True)*255).cpu().numpy().astype(np.float32)
    grid_image = np.transpose(grid_image, (1, 2, 0))
    imageio.imwrite(path, grid_image)


def cpu_or_gpu(tensor):

    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


def normalize_img(tensor):

    max_num = torch.max(tensor)
    min_num = torch.min(tensor)

    difference = max_num - min_num
    rate = 2/difference

    tensor = rate * tensor
    new_min = torch.min(tensor)

    intercept = -1 - new_min

    return tensor + intercept


def Denormalize(tensor):

    min = torch.min(tensor)
    max = torch.max(tensor)

    gap = max - min
    deviation_rate = 1/gap

    new_tensor = tensor * deviation_rate
    new_min = torch.min(new_tensor)

    new_tensor = new_tensor - new_min

    return new_tensor


def CycleGAN_tensorboard(writer, epoch, fake_A, fake_B, lossG, lossD):

    writer.add_scalar("G-Loss", lossG, epoch+1)
    writer.add_scalar("D-loss", lossD, epoch+1)

    if epoch % 10 == 0:

        if fake_A.size()[0] > 16:

            fake_A = torch.narrow(fake_A, 0, 0, 15)

        if fake_B.size()[0] > 16:

            fake_B = torch.narrow(fake_B, 0, 0, 15)

        fake_A_grid = torchvision.utils.make_grid(
            Denormalize(fake_A), nrow=4)
        fake_B_grid = torchvision.utils.make_grid(
            Denormalize(fake_B), nrow=4)

        writer.add_image('Fake A', fake_A_grid, global_step=epoch+1)
        writer.add_image('Fake B', fake_B_grid, global_step=epoch+1)
