#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import os
import random
import time
import warnings

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models import *
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--num_cls', default=10, type=int, metavar='N',
                    help='number of classes in dataset (output dimention of models)')
parser.add_argument('-p', '--print-freq', default=500, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--experiments', default=20, type=int, metavar='N',
                    help='number of experiments to run')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--model', default='', type=str, help='path to model checkpoint')


best_acc1 = 0


def main():
    args = parser.parse_args()
    print(vars(args))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    ngpus_per_node = torch.cuda.device_count()
    # Simply call main_worker function
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model")
    # model = models.__dict__[args.arch]()
    model = ResNet18()

    # load from model checkpoint
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model, map_location="cpu")
        state_dict = checkpoint['net']

        new_state_dict = dict()
        for old_key, value in state_dict.items():
            if old_key.startswith('module'):
                new_key = old_key.replace('module.', '')
                new_state_dict[new_key] = value

        msg = model.load_state_dict(new_state_dict, strict=False)
        print(msg)
        # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        print("=> loaded pre-trained model '{}'".format(args.model))
    else:
        print("=> no checkpoint found at '{}'".format(args.model))

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    # Data loading code
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(args.data, train=True, transform=transform_test, download=True)
    valset = datasets.CIFAR10(args.data, train=False, transform=transform_test, download=True)

    train_loader = DataLoader(trainset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.workers,
                              pin_memory=True,
                              drop_last=False)
    val_loader = DataLoader(valset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.workers,
                            pin_memory=True,
                            drop_last=False)

    compute_probabilities(train_loader, val_loader, model, args)


def compute_probabilities(train_loader, val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    train_results = np.zeros((50000, args.experiments, args.num_cls))
    test_results = np.zeros((10000, args.experiments, args.num_cls))
    train_labels = np.zeros(50000)
    test_labels = np.zeros(10000)
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()

        for i, (images, target) in enumerate(train_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            train_labels[i * args.batch_size:i * args.batch_size + len(images)] = target.cpu().numpy()

            for num_exp in range(args.experiments):
                # compute output
                output = model(images)
                if i == 0:
                    print("New output")
                    print(target)
                    print(output.cpu().numpy()[0])

                train_results[i * args.batch_size:i * args.batch_size + len(images), num_exp] = output.cpu().numpy()

        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            test_labels[i * args.batch_size:i * args.batch_size + len(images)] = target.cpu().numpy()

            for num_exp in range(args.experiments):
                # compute output
                output = model(images)

                test_results[i * args.batch_size:i * args.batch_size + len(images), num_exp] = output.cpu().numpy()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        res = [[] for i in range(args.num_cls)]
        for i in range(50000):
            lloc = int(train_labels[i])
            res[lloc].append(train_results[i])
        res = np.array(res)

        for i in range(args.num_cls):
            for j in range(5000):
                for k in range(args.experiments):
                    res[i][j][k] = np.exp(res[i][j][k]) / np.sum(np.exp(res[i][j][k]))

        np.save('/tmp/original_train', res)

        res = [[] for i in range(args.num_cls)]
        for i in range(10000):
            lloc = int(test_labels[i])
            res[lloc].append(test_results[i])
        res = np.array(res)

        for i in range(args.num_cls):
            for j in range(1000):
                for k in range(args.experiments):
                    res[i][j][k] = np.exp(res[i][j][k]) / np.sum(np.exp(res[i][j][k]))

        np.save('/tmp/original_test', res)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
