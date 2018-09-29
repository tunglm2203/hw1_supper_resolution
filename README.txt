Requirement packages:
    tensorflow (newest)
    torch==0.4.0
    torchvision
    tensorboard/tensorboardX
    tqdm
    pillow
    scipy

Training command:
CUDA_VISIBLE_DEVICES=your_id_gpu python main --checkpoint checkpoint/training_1 --n_iters 1000

where,
--checkpoint: Directory path to save best model and output images
--n_iters: Number of iteration for training (default is 1000)

Test command:
CUDA_VISIBLE_DEVICES=your_id_gpu python test --checkpoint checkpoint/training_1 --lr_path data/valid/LR --sr_path benchmark/res/

where,
--checkpoint: Directory path contains saved model (ckpt.pth)
--lr_path: Path to LR images
--sr_path: Path to save SR images output from model

Note: You can change ID of GPU depend on your resources, this code support trains with multiple GPUs.

