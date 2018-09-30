# EE838 Special Topics on Image Engineering <Image Restoration and Quality Enhancement>
# Semester: Fall 2018, School of EE, KAIST
# Student: Tung M. Luu
# ----------------------------- Homework Assignment 1 -----------------------------
# ---- Implementation and Verification of Single Image Super-Resolution (SISR) ----

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import *
from models import  *

import torch.optim.lr_scheduler as lr_scheduler
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import torch
import argparse
import random
import os

parser = argparse.ArgumentParser(description='Super Resolution using simple CNN')
parser.add_argument('--n_iter', type=int, default=1000, help='Number of epochs (Default: 10000)')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size (Default: 16)')
parser.add_argument('--period_check', type=int, default=1, help='Check after N iter (Default 1)')
parser.add_argument('--use_gpu', action='store_false', help='Use gpu or not (Default True)')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (Default 0)')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate (Default 1e-4)')
parser.add_argument('--force_lr', type=float, default=None, help='Learning rate (Default 1e-4)')
parser.add_argument('--checkpoint', type=str, default='checkpoint', help='Path to checkpoint')
parser.add_argument('--finetune', action='store_true', help='Finetune model  (Default False)')
parser.add_argument('--model_path', type=str, help='Path to pretrained model')
parser.add_argument('--step', type=int, default=200, help='Step decay learning rate  (Default 1000)')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers (Default 8)')
parser.add_argument('--manualSeed', type=int, default=1, help='Manually set seed')

# Get argument from command line
args = parser.parse_args()
print('--------- YOUR SETTING ---------')
for arg in vars(args):
    print("%15s: %s" %(str(arg), str(getattr(args, arg))))
print("")

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_gpu:
    torch.cuda.manual_seed(args.manualSeed)

def main():
    training_path = './data/train'
    valid_path = './data/valid'
    scale = 2
    patch_size = 64

    if not os.path.exists(args.checkpoint):
        os.mkdir(args.checkpoint)

    data_transforms = {
        'train': transforms.Compose([
        transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.ToTensor()
        ])
    }

    # ---------- Loading dataset ----------
    print('Loading data...')
    time_start = time.time()
    training_data = SRDataset(path=training_path,
                              patch_size=patch_size,
                              scale=scale,
                              crop=True,
                              aug=True,
                              transform=data_transforms['train'])
    valid_data = SRDataset(path=valid_path,
                           patch_size=patch_size,
                           scale=scale,
                           crop=False,
                           aug=False,
                           transform=data_transforms['val'])
    print("Load data done: %.4f (s)\n" % (time.time() - time_start))
    print("Number training data: %d" % (len(training_data)))
    print("Number validation data: %d" % (len(valid_data)))

    train_loader = DataLoader(dataset=training_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)
    valid_loader = DataLoader(dataset=valid_data,
                              batch_size=1,
                              shuffle=False,
                              num_workers=args.num_workers)

    # ---------- Device checking ----------
    if args.use_gpu:
        use_gpu = torch.cuda.is_available()
    else:
        use_gpu = False

    # ---------- Log ----------
    tensor_board = SummaryWriter(args.checkpoint)

    # ---------- Model configuration ----------
    channel_in = 3
    opt_head = {'out_channels':64, 'kernel_size':7, 'stride':1}
    opt_body = {'n_resblocks':4, 'n_features':64, 'kernel_size':3, 'stride':1, 'bias':True}
    opt_tail = {'out_channels':256, 'kernel_size':3, 'stride':1, 'upscale_factor':2,
                'final_kernel_size':7, 'final_out_channels':3, 'final_stride':1}

    model = Model_SR(channel_in=channel_in, opt_head=opt_head, opt_body=opt_body, opt_tail=opt_tail)
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

    # ---------- Optimizer ----------
    if args.force_lr is not None:
        args.learning_rate = args.force_lr
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.learning_rate,
                           weight_decay=args.weight_decay)
    model_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.5)

    # ---------- Loss ----------
    criterion = nn.L1Loss()

    # ---------- Load pretrained model ----------
    if args.finetune:
        ckpt = torch.load(os.path.join(args.checkpoint, 'ckpt.pth'))
        start_iter = ckpt['iter']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if args.force_lr is None:
            model_lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        best_psnr = ckpt['best_psnr']
        print('Load done, finetune from: ')
        print("%10s: %d" % ('iter', start_iter))
        print("%10s: %.4f\n" % ('PSNR', best_psnr))
    else:
        best_psnr = 0.0
        start_iter = 0

    # ---------- Training ----------
    model.train()
    n_train, n_valid = len(train_loader), len(valid_loader)
    n_iter = args.n_iter
    for iter in range(start_iter, n_iter):
        model_lr_scheduler.step()
        cur_lr = optimizer.param_groups[0]['lr']
        print("Training @ Iter: %s,  learning rate: %s" % (iter, cur_lr))

        model.train()
        train_loss = 0.0
        for batch_idx, (lr_images, hr_images) in enumerate(tqdm(train_loader)):
            lr_images, hr_images = Variable(lr_images).to(device), Variable(hr_images).to(device)

            optimizer.zero_grad()
            sr_images = model(lr_images)
            loss = criterion(sr_images, hr_images)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        epoch_loss = train_loss / n_train

        # ---------- Log to Tensorboard ----------
        print("\nIter: %d. Loss: %.4f\n" % (iter, epoch_loss))
        tensor_board.add_scalar('Training loss', epoch_loss, iter)
        tensor_board.add_scalar('Learning rate', cur_lr, iter)

        # ---------- Validation ----------
        if (iter + 1) % args.period_check == 0:
            model.eval()
            valid_psnr = 0.0
            with torch.no_grad():
                for batch_idx, (lr_images, hr_images) in enumerate(tqdm(valid_loader)):
                    lr_images, hr_images = Variable(lr_images).to(device), Variable(hr_images).to(device)
                    sr_images = model(lr_images)
                    sr_images = sr_images.clamp(0, 1.0)
                    valid_psnr += compute_psnr(sr_images, hr_images)
                    tensor_board.add_image(str(batch_idx + 1) + '_LR', lr_images, iter)
                    tensor_board.add_image(str(batch_idx + 1) + '_HR', hr_images, iter)
                    tensor_board.add_image(str(batch_idx + 1) + '_SR', sr_images, iter)
            valid_psnr = valid_psnr / n_valid
            print("Validation PSNR: %.4f\n" % (valid_psnr))
            tensor_board.add_scalar('Validation PSNR', valid_psnr, iter)

            # ---------- save model ----------
            if valid_psnr > best_psnr:
                best_psnr = valid_psnr
                torch.save({
                    'iter' : iter + 1,
                    'model' : model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'lr_scheduler' : model_lr_scheduler.state_dict(),
                    'best_psnr' : best_psnr
                }, '%s/ckpt.pth' % (args.checkpoint))
                print("Save model: %s/ckpt.pth\n" % (args.checkpoint))
                if not os.path.exists('%s/output' % (args.checkpoint)):
                    os.mkdir('%s/output' % (args.checkpoint))
                print("Writing sr images...\n")
                with torch.no_grad():
                    for batch_idx, (lr_images, hr_images) in enumerate(tqdm(valid_loader)):
                        lr_images, hr_images = Variable(lr_images).to(device), Variable(hr_images).to(device)
                        sr_images = model(lr_images)
                        vutils.save_image(sr_images, '%s/output/%04d.png' % (args.checkpoint, batch_idx+1),
                                          normalize=False,
                                          padding=0)

if __name__ == '__main__':
    main()
