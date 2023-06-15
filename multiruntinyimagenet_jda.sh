#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python C100/train_student_fakd.py --checkpoint-dir ./store/ \
  --gpu-id 0  --i 3 --data /home/sst/dataset/c100 --arch wrn_16_2_fakd --tarch wrn_40_2   \
  --tcheckpoint /home/Bigdata/ckpt/ccd/cifar100/wrn_40_2.pth --dirac_ratio 0.0

CUDA_VISIBLE_DEVICES=0 python C100/train_student_fakd.py --checkpoint-dir ./store/ \
  --gpu-id 0  --i 3 --data /home/sst/dataset/c100 --arch wrn_16_2_fakd --tarch wrn_40_2   \
  --tcheckpoint /home/Bigdata/ckpt/ccd/cifar100/wrn_40_2.pth --dirac_ratio 0.25

CUDA_VISIBLE_DEVICES=0 python C100/train_student_fakd.py --checkpoint-dir ./store/ \
  --gpu-id 0  --i 3 --data /home/sst/dataset/c100 --arch wrn_16_2_fakd --tarch wrn_40_2   \
  --tcheckpoint /home/Bigdata/ckpt/ccd/cifar100/wrn_40_2.pth --dirac_ratio 0.5

CUDA_VISIBLE_DEVICES=0 python C100/train_student_fakd.py --checkpoint-dir ./store/ \
  --gpu-id 0  --i 3 --data /home/sst/dataset/c100 --arch wrn_16_2_fakd --tarch wrn_40_2   \
  --tcheckpoint /home/Bigdata/ckpt/ccd/cifar100/wrn_40_2.pth --dirac_ratio 0.75