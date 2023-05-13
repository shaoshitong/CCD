#!/bin/sh

python C100/train_student_kd.py      \
  --checkpoint-dir ./store/  \
  --gpu 0  --i 1 --data /data/data/cifar100/

python C100/train_student_kd.py      \
  --checkpoint-dir ./store/  \
  --gpu 0  --i 2 --data /data/data/cifar100/

python C100/train_student_kd.py      \
  --checkpoint-dir ./store/  \
  --gpu 0  --i 3 --data /data/data/cifar100/