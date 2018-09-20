# EE838 Special Topics on Image Engineering <Image Restoration and Quality Enhancement>
# Semester: Fall 2018, School of EE, KAIST
# Student: Tung M. Luu
# ----------------------------- Homework Assignment 1 -----------------------------
# ---- Implementation and Verification of Single Image Super-Resolution (SISR) ----

from torch.autograd import Variable
from tqdm import tqdm
from PIL import Image

import numpy as np
import random
import torch
import pytorch_ssim
import glob
import os
import time

def random_cropping(lr_images, hr_images, patch_size, scale):
    _, heigh_lr, width_lr = lr_images.shape  # shape of lr images, channel first format

    lr_path_size = patch_size
    hr_path_size = lr_path_size * scale

    # Index of LR image
    h = random.randint(0, heigh_lr - lr_path_size)
    w = random.randint(0, width_lr - lr_path_size)
    # Index of HR image
    H = h * scale
    W = w * scale

    cropped_lr = lr_images[:, h:h + lr_path_size, w:w + lr_path_size]
    cropped_hr = hr_images[:, H:H + hr_path_size, W:W + hr_path_size]

    return cropped_lr, cropped_hr

def compute_psnr(sr_im, hr_im):
    # Note: function only apply for single image (not batch)
    MAX_PIXEL_VALUE = 1
    diff = sr_im - hr_im
    diff = diff.view(1, -1)
    mse = torch.mean(diff**2, dim=1)
    rmse = torch.sqrt(mse)
    psnr = 20*torch.log10(MAX_PIXEL_VALUE/rmse)
    return psnr

def compute_ssim(sr_im, hr_im, device):
    sr_im = Variable(torch.Tensor(sr_im)).to(device)
    hr_im = Variable(torch.Tensor(hr_im)).to(device)
    return pytorch_ssim.ssim(sr_im, hr_im)

def create_binary_data(path):
    file_lr = glob.glob(os.path.join(path, 'LR/*.jpg'))
    file_hr = glob.glob(os.path.join(path, 'HR/*.jpg'))
    file_lr.sort()
    file_hr.sort()
    n_lr_images = len(file_lr)
    n_hr_images = len(file_hr)
    assert n_lr_images == n_hr_images, "Number of lr images != number of hr images"
    lr_images = []
    hr_images = []
    print("Found %d images in %s\n" % (n_lr_images, path))
    print("Compressing dataset...\n")
    time_start = time.time()
    for i in tqdm(range(n_lr_images)):
        lr_im = Image.open(file_lr[i])
        hr_im = Image.open(file_hr[i])
        tmp = lr_im.getpixel((0, 0))
        if isinstance(tmp, int) or len(tmp) != 3:
            lr_im = lr_im.convert("RGB")
        tmp = hr_im.getpixel((0, 0))
        if isinstance(tmp, int) or len(tmp) != 3:
            hr_im = hr_im.convert("RGB")
        lr_images.append(np.asarray(lr_im))
        hr_images.append(np.asarray(hr_im))
    np.save(os.path.join(path, 'lr_images.npy'), lr_images)
    np.save(os.path.join(path, 'hr_images.npy'), hr_images)
    print("Compress data done in: %.5f (s)\n" % (time.time() - time_start))

def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
             else torch.arange(x.size(i)-1, -1, -1).long()
             for i in range(x.dim()))]