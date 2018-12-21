Requirement:
    tensorflow (newest)
    torch==0.4.0
    torchvision
    tensorboard/tensorboardX
    tqdm
    pillow
    scipy

Data: Extract dataset, copy training (HR, LR) and validation (HR, LR) data into directory ./data/train and ./data/valid, respectively.

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

NOTE: First time training, the compressed binary data will be created for speed up training. You can change ID of GPU depend on your resources, this code support trains with multiple GPUs.

Structure of project:
+data: Directory contains data for train and test
+checkpoint: Directory contains checkpoint for training, output SR images for best model
+benchmark: Directory used to benchmark, run script 'test_evaluate.sh' will auto generating images to this directory
-data_loader.py: File contains class used to load data for training and testing
-models.py: File contains the model
-utils.py: Fiel contains ultility functions (cropping images, create compressed binary data, compute psnr, etc.)
-main.py: File contains the framework for training
-test.py: File contains the framework for testing
-evaluation.py: File used to compute PSNR, SSIM
-myssim: Support to compute SSIM

