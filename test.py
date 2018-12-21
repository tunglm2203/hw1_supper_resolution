# EE838 Special Topics on Image Engineering <Image Restoration and Quality Enhancement>
# Semester: Fall 2018, School of EE, KAIST
# Student: Tung M. Luu
# ----------------------------- Homework Assignment 1 -----------------------------
# ---- Implementation and Verification of Single Image Super-Resolution (SISR) ----

from data_loader import *
from models import  *
from torchvision import transforms

import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import argparse
import os
import glob

parser = argparse.ArgumentParser(description='Super Resolution using simple CNN')
parser.add_argument('--use_gpu', action='store_false', help='Use gpu or not (Default True)')
parser.add_argument('--checkpoint', type=str, help='Path to pretrained model')
parser.add_argument('--lr_path', type=str, help='Low resolution image path')
parser.add_argument('--sr_path', type=str, help='Output of super resolution image path')

# Get argument from command line
args = parser.parse_args()
print('--------- YOUR SETTING ---------')
for arg in vars(args):
    print("%15s: %s" %(str(arg), str(getattr(args, arg))))
print("")

# ---------- Model configuration ----------
channel_in = 3
opt_head = {'out_channels':64, 'kernel_size':7, 'stride':1}
opt_body = {'n_resblocks':4, 'n_features':64, 'kernel_size':3, 'stride':1, 'bias':True}
opt_tail = {'out_channels':256, 'kernel_size':3, 'stride':1, 'upscale_factor':2,
            'final_kernel_size':7, 'final_out_channels':3, 'final_stride':1}

model = Model_SR(channel_in=channel_in, opt_head=opt_head, opt_body=opt_body, opt_tail=opt_tail)

# ---------- Device checking ----------
if args.use_gpu:
    use_gpu = torch.cuda.is_available()
else:
    use_gpu = False

if use_gpu:
    n_GPUs = torch.cuda.device_count()
    device = torch.device('cuda')
    cudnn.benchmark = True
    print("Loading model using %d GPU(s)..." % (n_GPUs))
    if n_GPUs > 1:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)
else:
    device = torch.device('cpu')
    print("Loading model using  CPU(s)...")
    model = model.to(device)
print("Load model done.\n")

ckpt = torch.load(os.path.join(args.checkpoint, 'ckpt.pth'))
model.load_state_dict(ckpt['model'])
model.eval()

lr_file = glob.glob(os.path.join(args.lr_path, '*.png'))
lr_file.sort()
n_file = len(lr_file)

print('Starting infer super-resolution images...')
with torch.no_grad():
    for i in tqdm(range(n_file)):
        lr_image = Image.open(lr_file[i])
        transf = transforms.ToTensor()
        lr_image = transf(lr_image)
        lr_image.unsqueeze_(0)
        lr_image = Variable(lr_image).to(device)
        sr_images = model(lr_image)
        vutils.save_image(sr_images, '%s/%04d.png' % (args.sr_path, i + 1),
                          normalize=False,
                          padding=0)
print('Test done.')
print('SR image is saved in ', args.sr_path)
