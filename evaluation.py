#!/usr/bin/env python
import sys
import os
import os.path
import numpy as np

from PIL import Image
import scipy.misc
from myssim import compare_ssim as ssim

SCALE = 1

def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr

def _open_img(img_p):
    F = scipy.misc.fromimage(Image.open(img_p)).astype(float)/255.0
    h, w, c = F.shape
    F = F[:h-h%SCALE, :w-w%SCALE, :]
    boundarypixels = SCALE
    F = F[boundarypixels:-boundarypixels,boundarypixels:-boundarypixels,:]
    return F

def _open_img_ssim(img_p):
    F = scipy.misc.fromimage(Image.open(img_p))#.astype(float)
    h, w, c = F.shape
    F = F[:h-h%SCALE, :w-w%SCALE, :]
    boundarypixels = SCALE 
    F = F[boundarypixels:-boundarypixels,boundarypixels:-boundarypixels,:]
    return F


def compute_psnr(ref_im, res_im):
    return output_psnr_mse(
        _open_img(os.path.join(input_dir,'ref',ref_im)),
        _open_img(os.path.join(input_dir,'res',res_im))
        )

def compute_mssim(ref_im, res_im):
    ref_img = _open_img_ssim(os.path.join(input_dir,'ref',ref_im))
    res_img = _open_img_ssim(os.path.join(input_dir,'res',res_im))
    channels = []
    for i in range(3):
        channels.append(ssim(ref_img[:,:,i],res_img[:,:,i], gaussian_weights=True, use_sample_covariance=False))
    return np.mean(channels)


# as per the metadata file, input and output directories are the arguments
[_, input_dir, output_dir] = sys.argv

res_dir = os.path.join(input_dir, 'res')
ref_dir = os.path.join(input_dir, 'ref')

ref_jpgs = sorted([p for p in os.listdir(ref_dir) if p.lower().endswith('png')])
res_jpgs = sorted([p for p in os.listdir(res_dir) if p.lower().endswith('png')])

scores = []
for (ref_im, res_im) in zip(ref_jpgs, res_jpgs):
    print(ref_im,res_im)
    scores.append(
        compute_psnr(ref_im,res_im)
    )
    # print(scores[-1])
psnr = np.mean(scores)


scores_ssim = []
for (ref_im, res_im) in zip(ref_jpgs, res_jpgs):
    print(ref_im,res_im)
    scores_ssim.append(
        compute_mssim(ref_im,res_im)
        )
    # print(scores_ssim[-1])
mssim = np.mean(scores_ssim)


with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
    output_file.write("PSNR:%f\n"%psnr)
    output_file.write("SSIM:%f\n"%mssim)
