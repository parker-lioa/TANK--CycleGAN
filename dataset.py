import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
from utils import normalize_img
import os


class CycleGANDataSet(Dataset):

    def __init__(self, directory_A, directory_B, transform_A, transform_B):

        self.transform_A = transform_A
        self.transform_B = transform_B
        self.directory_A = directory_A
        self.directory_B = directory_B
        self.DomainA = os.listdir(directory_A)
        self.DomainB = os.listdir(directory_B)

        assert len(self.DomainA) == len(self.DomainB), 'missing data in A or B'

        self.length = len(self.DomainA)

        self.img_A = [Image.open(
            self.directory_A+'/' + x).copy() for x in self.DomainA]
        self.img_B = [Image.open(
            self.directory_B+'/' + x).copy() for x in self.DomainB]

    def __getitem__(self, index):

        img_A = self.img_A[index]
        img_B = self.img_B[index]

        if self.transform_A is not None:
            img_A = self.transform_A(img_A)
        if self.transform_B is not None:
            img_B = self.transform_B(img_B)
            
        normalize_img(img_A)
        normalize_img(img_B)

        return img_A, img_B

    def __len__(self):

        return self.length
