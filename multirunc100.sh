#!/bin/sh
python C100/train_student_singleccd.py      --checkpoint-dir ./store/   --gpu 0  --i 1

python C100/train_student_singleccd.py      --checkpoint-dir ./store/   --gpu 0  --i 2

python C100/train_student_singleccd.py      --checkpoint-dir ./store/   --gpu 0  --i 3