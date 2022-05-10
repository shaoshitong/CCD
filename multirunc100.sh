#!/bin/sh
python C100/train_student_cifar.py     --tarch wrn_40_2_aux     --arch wrn_16_2_aux     --tcheckpoint /home/Bigdata/kd/store/train_teacher_cifar_arch_wrn_40_2_aux_dataset_cifar100_seed0/wrn_40_2_aux.pth.tar     --checkpoint-dir /home/Bigdata/kd/store/     --data /data/data     --gpu 0 --manual 0 --aux-weight 0. --i 0

python C100/train_student_cifar.py     --tarch wrn_40_2_aux     --arch wrn_16_2_aux     --tcheckpoint /home/Bigdata/kd/store/train_teacher_cifar_arch_wrn_40_2_aux_dataset_cifar100_seed0/wrn_40_2_aux.pth.tar     --checkpoint-dir /home/Bigdata/kd/store/     --data /data/data     --gpu 0 --manual 0 --aux-weight 0. --i 1

python C100/train_student_cifar.py     --tarch wrn_40_2_aux     --arch wrn_16_2_aux     --tcheckpoint /home/Bigdata/kd/store/train_teacher_cifar_arch_wrn_40_2_aux_dataset_cifar100_seed0/wrn_40_2_aux.pth.tar     --checkpoint-dir /home/Bigdata/kd/store/     --data /data/data     --gpu 0 --manual 0 --aux-weight 0. --i 2
