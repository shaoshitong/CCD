#!/bin/sh

python ./TinyImageNet/train_student_kd.py      \
  --checkpoint-dir ./store/  \
  --gpu 0  --i 1 --data /home/Bigdata/tiny-imagenet/tiny-imagenet-200/
#
#python ./TinyImageNet/train_student_kd.py      \
#  --checkpoint-dir ./store/  \
#  --gpu 0  --i 2 --data /home/Bigdata/tiny-imagenet/tiny-imagenet-200/
#
#python ./TinyImageNet/train_student_kd.py      \
#  --checkpoint-dir ./store/  \
#  --gpu 0  --i 3 --data /home/Bigdata/tiny-imagenet/tiny-imagenet-200/