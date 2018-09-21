# EE838 Special Topics on Image Engineering <Image Restoration and Quality Enhancement>
# Semester: Fall 2018, School of EE, KAIST
# Student: Tung M. Luu
# ----------------------------- Homework Assignment 1 -----------------------------
# ---- Implementation and Verification of Single Image Super-Resolution (SISR) ----

from torch.utils.data import Dataset
from utils import *

import os

IMAGE_FORMAT = '*.png'

class SRDataset(Dataset):
    def __init__(self, path, patch_size, scale, aug=False, crop=False, transform=None):
        self.path = path
        self.patch_size = patch_size
        self.scale = scale
        self.crop = crop
        self.transform = transform
        self.aug = aug

        if not os.path.exists(self.path):
            print('Dataset path is not exist: ', self.path)
        elif not (os.path.exists(os.path.join(path, 'lr_images.npy')) or
                  os.path.exists(os.path.join(path, 'hr_images.npy'))):
            print('Not exist compressed data, creating...')
            create_binary_data(path)

        self.lr_images = np.load(os.path.join(path, 'lr_images.npy'))
        self.hr_images = np.load(os.path.join(path, 'hr_images.npy'))
        self.n_images = len(self.lr_images)

    def __len__(self):
        return self.n_images

    def __getitem__(self, item):
        lr_im = Image.fromarray(self.lr_images[item])
        hr_im = Image.fromarray(self.hr_images[item])

        if self.transform is not None:
            lr_im = self.transform(lr_im)
            hr_im = self.transform(hr_im)

        if self.crop:
            lr_im, hr_im = random_cropping(lr_im, hr_im, self.patch_size, self.scale)

        if self.aug:
            rand_aug = random.randint(0, 2) # 0: Horizontal flip, 1: Vertical flip, 2: Keep original
            if rand_aug == 0:
                lr_im = flip(lr_im, dim=2)  # Horizontal flip
                hr_im = flip(hr_im, dim=2)
            elif rand_aug == 1:
                lr_im = flip(lr_im, dim=3)  # Vertical flip
                hr_im = flip(hr_im, dim=3)

        return lr_im, hr_im
