CUDA_VISIBLE_DEVICES=7 python test.py --checkpoint checkpoint/training_3/ --lr_path data/valid/LR/ --sr_path benchmark/res/
python evaluation.py benchmark benchmark

