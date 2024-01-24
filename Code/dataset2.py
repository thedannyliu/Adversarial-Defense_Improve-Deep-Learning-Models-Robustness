import glob
import numpy as np
import torch, os, pdb
import random as rn
from torch.utils import data
from torchvision import transforms
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
import torch.multiprocessing
from PIL import Image
import cv2


class dataset_h5(torch.utils.data.Dataset):
    def __init__(self, X, root='Task3/original/', image_size=480, crop_size=480, mode='Train'):
        super(dataset_h5, self).__init__()

        self.fns = X['imgName'].values
        self.y = X['Fovea_Y'].values
        self.x = X['Fovea_X'].values
        self.mode = mode
        self.root = root

        self.image_size = image_size
        self.crop_size = crop_size
        self.mean_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        ])
        self.transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        ])

        self.n_images = len(self.fns)
        
        
    def __getitem__(self, index):
        im = Image.open(self.root + self.fns[index])
        if self.mode=='Train':
            im = self.transform(im)
        else:
            im = self.mean_transform(im)

        return im, self.x[index], self.y[index]

    def __len__(self):
        return self.n_images