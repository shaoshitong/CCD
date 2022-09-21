#!/bin/sh

python C100/train_student_singleccd.py      \
  --checkpoint-dir ./store/  \
  --gpu 0  --i 1 --data /data/data/cifar100/ \
  --tcheckpoint /home/Bigdata/ckpt/ilsvrc2012/teacher/wrn_40_2.pth

python C100/train_student_singleccd.py      \
  --checkpoint-dir ./store/  \
  --gpu 0  --i 2 --data /data/data/cifar100/ \
  --tcheckpoint /home/Bigdata/ckpt/ilsvrc2012/teacher/wrn_40_2.pth

python C100/train_student_singleccd.py      \
  --checkpoint-dir ./store/  \
  --gpu 0  --i 3 --data /data/data/cifar100/ \
  --tcheckpoint /home/Bigdata/ckpt/ilsvrc2012/teacher/wrn_40_2.pth

