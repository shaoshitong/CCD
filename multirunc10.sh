#!/bin/sh
python C10/train_student_cifar.py     --tarch resnet56_aux     --arch resnet20_aux     --tcheckpoint /home/Bigdata/kd/store/train_teacher_cifar_arch_resnet56_aux_dataset_cifar10_seed0/resnet56_aux.pth.tar     --checkpoint-dir /home/Bigdata/kd/store/     --data /data/data     --gpu 0 --manual 0 --aux-weight 1. --i 0

python C10/train_student_cifar.py     --tarch resnet56_aux     --arch resnet20_aux     --tcheckpoint /home/Bigdata/kd/store/train_teacher_cifar_arch_resnet56_aux_dataset_cifar10_seed0/resnet56_aux.pth.tar     --checkpoint-dir /home/Bigdata/kd/store/     --data /data/data     --gpu 0 --manual 0 --aux-weight 1. --i 1

python C10/train_student_cifar.py     --tarch resnet56_aux     --arch resnet20_aux     --tcheckpoint /home/Bigdata/kd/store/train_teacher_cifar_arch_resnet56_aux_dataset_cifar10_seed0/resnet56_aux.pth.tar     --checkpoint-dir /home/Bigdata/kd/store/     --data /data/data     --gpu 0 --manual 0 --aux-weight 1. --i 2