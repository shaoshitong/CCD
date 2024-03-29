import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch.utils.data import SubsetRandomSampler
import os
import shutil
import argparse
import numpy as np
import models
import torchvision
import torchvision.transforms as transforms
from C10.utils import cal_param_size, cal_multi_adds
import losses
from data.datasets import PolicyDatasetImageNet
from data.cached_image_folder import CachedImageFolder

from bisect import bisect_right
import time
import math
scaler=torch.cuda.amp.GradScaler()
# sb sst
# sb sst
# sb sst

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='./ilsvrc2012/', type=str, help='Dataset directory')
parser.add_argument('--dataset', default='imagenet', type=str, help='Dataset name')
parser.add_argument('--arch', default='resnet18_imagenet', type=str, help='student network architecture')
parser.add_argument('--tarch', default='resnet34_imagenet', type=str, help='teacher network architecture')
parser.add_argument('--tcheckpoint', default='./checkpoint/resnet34-imagenet.pth', type=str, help='pre-trained weights of teacher')
parser.add_argument('--init-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--milestones', default=[60,120,180], type=list, help='milestones for lr-multistep')
parser.add_argument('--warmup-epoch', default=0, type=int, help='warmup epoch')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=1536, help='batch size')
parser.add_argument('--accumulate-step', type=int, default=2, help='accumulate the grad')
parser.add_argument('--num-workers', type=int, default=6, help='the number of workers')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
parser.add_argument('--kd_alpha', type=float, default=0.5, help='alpha for KD distillation')
parser.add_argument('--kd_weight', type=float, default=2, help='weight for KD distillation')
parser.add_argument('--ccd', action='store_true',help="if use CCD")
parser.add_argument('--cutmix', action='store_true',help="if use CutMix")
parser.add_argument('--zip', action='store_true',help="if use zip to read image")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')
parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='directory fot storing checkpoints')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')                    


def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    port_id = 10000 + np.random.randint(0, 1000)
    args.dist_url = 'tcp://127.0.0.1:' + str(port_id)
    args.log_txt = 'result/'+ str(os.path.basename(__file__).split('.')[0]) + '_'+\
            'arch' + '_' +  args.arch + '_'+\
            'dataset' + '_' +  args.dataset + '_'+\
            'seed'+ str(args.manual_seed) +'.txt'


    args.log_dir = str(os.path.basename(__file__).split('.')[0]) + '_'+\
            'arch'+ '_' + args.arch + '_'+\
            'dataset' + '_' +  args.dataset + '_'+\
            'seed'+ str(args.manual_seed)


    args.traindir = os.path.join(args.data, 'train')
    args.valdir = os.path.join(args.data, 'val')

    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    torch.set_printoptions(precision=4)

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.log_dir)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    with open(args.log_txt, 'a+') as f:
        f.write("==========\nArgs:{}\n==========".format(args) + '\n')
    print('==> Building model..')
    args.distributed = args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    print('ngpus_per_node', ngpus_per_node)
    args.world_size = ngpus_per_node * args.world_size
    print('multiprocessing_distributed')
    torch.multiprocessing.set_start_method('spawn')
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    best_acc = 0.  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    args.gpu = gpu
    print("Use GPU: {} for training".format(args.gpu))

    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                        world_size=args.world_size, rank=args.rank)

    print('load pre-trained teacher weights from: {}'.format(args.tcheckpoint))     
    checkpoint = torch.load(args.tcheckpoint, map_location='cuda:{}'.format(args.gpu))
    torch.cuda.set_device(args.gpu)

    num_classes = 1000
    model = getattr(models, args.arch)
    net = model(num_classes=num_classes).cuda(args.gpu)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])

    tmodel = getattr(models, args.tarch)
    tnet = tmodel(num_classes=num_classes).cuda(args.gpu)
    tnet = torch.nn.parallel.DistributedDataParallel(tnet, device_ids=[args.gpu])
    tnet.module.load_state_dict(checkpoint)
    tnet.eval()
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.batch_size =int(args.batch_size//(2*args.accumulate_step)+args.batch_size%(2*args.accumulate_step))
    print(f"local batch size is {args.batch_size}")
    args.workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
    cudnn.benchmark = True
    cudnn.fastest=True
    criterion_cls = nn.CrossEntropyLoss().cuda(args.gpu)
    if args.ccd:
        criterion_div = losses.CCDLoss(temperature=args.kd_T,alpha=args.kd_alpha)
    else:
        criterion_div = losses.KDLoss(temperature=args.kd_T,alpha=args.kd_alpha)
    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.cuda()

    trainable_list = nn.ModuleList([])
    trainable_list.append(net)
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=0.1, momentum=0.9, weight_decay=args.weight_decay)


    if args.resume:
        print('load intermediate weights from: {}'.format(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar')))
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                map_location='cuda:{}'.format(args.gpu))
        net.module.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
    if args.zip:
        ann_file = "train_map.txt"
        prefix = "train.zip@/"
        train_set = CachedImageFolder(args.traindir, ann_file, prefix,   transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),cache_mode='part')
        train_set = PolicyDatasetImageNet(train_set)
        indices = np.arange(dist.get_rank(), len(train_set), dist.get_world_size())
        train_sampler = SubsetRandomSampler(indices)
    else:
        train_set = torchvision.datasets.ImageFolder(
        args.traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]))
        train_set=PolicyDatasetImageNet(train_set)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    test_set = torchvision.datasets.ImageFolder(
        args.valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ]))

    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    
    def train(epoch, criterion_list, optimizer,iter_number):
        train_loss = 0.
        train_loss_div = 0.
        top1_num = 0
        top5_num = 0
        total = 0
        if epoch >= args.warmup_epoch:
            lr = adjust_lr(optimizer, epoch, args)
        start_time = time.time()
        criterion_div = criterion_list[1]
        net.train()
        for batch_idx, (input, target) in enumerate(trainloader):
            batch_start_time = time.time()
            input = input.float().cuda()
            if input.ndim == 5:
                b, m, c, h, w = input.shape
                input = input.view(-1, c, h, w)
                target = target.view(-1)
            target = target.cuda()
            if epoch < args.warmup_epoch:
                lr = adjust_lr(optimizer, epoch, args, batch_idx, len(trainloader))
            if iter_number%args.accumulate_step==0:
                optimizer.zero_grad()
                iter_number=0
            with torch.cuda.amp.autocast(enabled=True):
                logits = net(input).float()
            with torch.no_grad():
                t_logits = tnet(input)
            loss_div = torch.tensor(0.).cuda()
            loss_div = loss_div + criterion_div(logits, t_logits, target) * args.kd_weight
            loss = loss_div/args.accumulate_step
            scaler.scale(loss).backward()
            if (iter_number+1)%args.accumulate_step==0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 20.)
                scaler.step(optimizer)
                scaler.update()
            iter_number=iter_number+1
            train_loss += loss.item() / len(trainloader)
            train_loss_div += loss_div.item() / len(trainloader)

            top1, top5 = correct_num(logits, target, topk=(1, 5))
            top1_num += top1
            top5_num += top5
            total += target.size(0)
            if batch_idx%100==0:
                print('Epoch:{}, batch_idx:{}/{}, lr:{:.5f}, Duration:{:.2f}, Top-1 Acc:{:.4f}'.format(
                    epoch, batch_idx, len(trainloader), lr, time.time()-batch_start_time, (top1_num/(total)).item()))
        with open(args.log_txt, 'a+') as f:
            f.write('Epoch:{}\t lr:{:.5f}\t duration:{:.3f}'
                    '\n train_loss:{:.5f}\t train_loss_div:{:.5f}'
                    'Train Top-1 class_accuracy: {}\nTrain Top-5 class_accuracy: {}\n'
                    .format(epoch, lr, time.time() - start_time,
                            train_loss, train_loss_div,str((top1_num/(total)).item()), str((top5_num/(total)).item())))


    def test(epoch, criterion_cls, net):
        test_loss_cls = 0.
        top1_num = 0
        top5_num = 0
        total = 0
        
        net.eval()
        with torch.no_grad():
            for batch_idx, (inputs, target) in enumerate(testloader):
                batch_start_time = time.time()
                input, target = inputs.cuda(), target.cuda()
                logits = net(input)
                top1, top5 = correct_num(logits, target, topk=(1, 5))
                top1_num += top1
                top5_num += top5
                total += target.size(0)

                print('Epoch:{}, batch_idx:{}/{}, Duration:{:.2f}, Top-1 Acc:{:.4f}'.format(
                    epoch, batch_idx, len(testloader), time.time()-batch_start_time, (top1_num/(total)).item()))
            with open(args.log_txt, 'a+') as f:
                f.write('test epoch:{}\t test_loss_cls:{:.5f}\nTop-1 class_accuracy: {}\nTop-5 class_accuracy: {}\n'
                        .format(epoch, test_loss_cls, str((top1_num/(total)).item()), str((top5_num/(total)).item())))
            print('test epoch:{}\t test_loss_cls:{:.5f}\nTop-1 class_accuracy: {}\nTop-5 class_accuracy: {}\n'
                        .format(epoch, test_loss_cls, str((top1_num/(total)).item()), str((top5_num/(total)).item())))

        return float((top1_num/(total)).item())

    if args.evaluate: 
        print('load pre-trained weights from: {}'.format(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'))     
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'),
                                map_location='cuda:{}'.format(args.gpu))
        net.module.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        test(start_epoch, criterion_cls, net) 
    else:
        #print('Evaluate Teacher:')
        #acc = test(0, criterion_cls, tnet)
        #print('teacher accuracy:{}'.format(acc))
        #with open(args.log_txt, 'a+') as f:
        #    f.write('teacher accuracy:{}'.format(acc))
        iter_number=0
        best_acc=0.
        for epoch in range(start_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            train(epoch, criterion_list, optimizer,iter_number)
            acc = test(epoch, criterion_cls, net)

            if args.rank == 0:
                state = {
                    'net': net.module.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict()
                }
                torch.save(state, os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'))

                is_best = False
                if best_acc < acc:
                    best_acc = acc
                    is_best = True

                if is_best:
                    shutil.copyfile(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                    os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'))

        print('Evaluate the best model:')
        print('load pre-trained weights from: {}'.format(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar')))
        args.evaluate = True
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'),
                                map_location='cuda:{}'.format(args.gpu))
        net.module.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        top1_acc = test(start_epoch, criterion_cls, net)

        with open(args.log_txt, 'a+') as f:
            f.write('best_accuracy: {} \n'.format(best_acc))
        print('best_accuracy: {} \n'.format(best_acc))
        os.system('cp ' + args.log_txt + ' ' + args.checkpoint_dir)



def correct_num(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:, :k].float().sum()
        res.append(correct_k)
    return res


def adjust_lr(optimizer, epoch, args, step=0, all_iters_per_epoch=0, eta_min=0.):
    cur_lr = 0.
    if epoch < args.warmup_epoch:
        cur_lr = args.init_lr * float(1 + step + epoch*all_iters_per_epoch)/(args.warmup_epoch *all_iters_per_epoch)
    else:
        epoch = epoch - args.warmup_epoch
        cur_lr = args.init_lr * 0.1 ** bisect_right(args.milestones, epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr

if __name__ == '__main__':
    main()
    
        


        

        
