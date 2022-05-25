python C100/train_student_crd.py      --checkpoint-dir ./store/   --gpu 0  --i 1

python C100/train_student_crd.py      --checkpoint-dir ./store/   --gpu 0  --i 2

python C100/train_student_crd.py  --tarch vgg13_bn_crd  --arch mobilenetV2_crd --tcheckpoint /home/Bigdata/ckpt/ilsvrc2012/teacher/vgg13_bn.pth --checkpoint-dir ./store/   --gpu 0  --i 1

python C100/train_student_crd.py  --tarch vgg13_bn_crd  --arch mobilenetV2_crd --tcheckpoint /home/Bigdata/ckpt/ilsvrc2012/teacher/vgg13_bn.pth --checkpoint-dir ./store/   --gpu 0  --i 2



#python C100/train_student_crd.py      --checkpoint-dir ./store/   --gpu 0  --i 1

