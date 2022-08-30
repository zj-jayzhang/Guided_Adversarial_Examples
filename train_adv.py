import argparse
import os
import random
import time
import warnings
import sys
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm

import models
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from utils import *
from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from losses import LDAMLoss, FocalLoss
import torch.nn.functional as F

cudnn.benchmark = True
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
    parser.add_argument('--dataset', default='cifar10', help='dataset setting')
    parser.add_argument('--data_dir', default='/mnt/lustre/share_data/zhangjie', 
                        type=str, help='dataset setting')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet32)')
    parser.add_argument('--loss_type', default="CE", type=str, help='loss type')
    parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
    parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
    parser.add_argument('--num_steps', default=8, type=int, help='number of steps')
    parser.add_argument('--train_rule', default='None', type=str, help='data sampling strategy for train loader')
    parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
    parser.add_argument('--exp_str', default='0', type=str, help='number to indicate which experiment it is')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--root_log', type=str, default='log')
    parser.add_argument('--root_model', type=str, default='checkpoint')

    args = parser.parse_args()
    return args


def get_dataset():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        train_dataset = IMBALANCECIFAR10(root=args.data_dir, imb_type=args.imb_type,
                                         imb_factor=args.imb_factor, rand_number=args.rand_number, train=True,
                                         download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True,
                                       transform=transform_val)
    elif args.dataset == 'cifar100':
        train_dataset = IMBALANCECIFAR100(root=args.data_dir, imb_type=args.imb_type,
                                          imb_factor=args.imb_factor, rand_number=args.rand_number, train=True,
                                          download=True, transform=transform_train)
        val_dataset = datasets.CIFAR100(root=args.data_dir, train=False, download=True,
                                        transform=transform_val)
    else:
        warnings.warn('Dataset is not listed')
        return
    cls_num_list = train_dataset.get_cls_num_list()
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=4, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False,
        num_workers=4, pin_memory=True)
    return train_loader, val_loader, cls_num_list


def main_worker():
    global best_acc1
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        adjust_learning_rate(optimizer, epoch, args)
        if args.train_rule == 'None':
            train_sampler = None
            per_cls_weights = None
        elif args.train_rule == 'DRW':
            train_sampler = None
            no_cross_idx = epoch // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[no_cross_idx], cls_num_list)
            per_cls_weights = (1.0 - betas[no_cross_idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        else:
            warnings.warn('Sample rule is not listed')

        if args.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'LDAM':
            criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(args.gpu)
        else:
            warnings.warn('Loss type is not listed')
            return
        if epoch == 0:
            validate(val_loader, model, criterion, args, log_testing)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, log_training)

        # evaluate on validation set
        test_acc, test_loss = validate(val_loader, model, criterion, args, log_testing)

        tf_writer.add_scalar('test_acc', test_acc.item(), epoch)
        tf_writer.add_scalar('test_loss', test_loss, epoch)

        # remember best acc@1 and save checkpoint
        is_best = test_acc > best_acc1
        best_acc1 = max(test_acc, best_acc1)

        output_best = 'Best Prec@1: %.3f\n' % best_acc1
        print(output_best)
        log_testing.write(output_best + '\n')
        log_testing.flush()

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)


def GA_PGD(model, data, target, epsilon, step_size, num_steps, rand_init):
    model.eval()
    Kappa = torch.zeros(len(data))
    x_adv = data.detach() + torch.from_numpy(
        np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        predict = output.max(1, keepdim=True)[1]
        # Update Kappa
        """
        攻击成功，则距离++
        """
        # Kappa = [if predict[p] == target[p] ]
        idx = torch.where(predict==target)[0]  # correctly predict the label
        Kappa[idx] += 1  # steps ++

        # for p in range(len(x_adv)):
        #     if predict[p] == target[p]:
        #         Kappa[p] += 1
        model.zero_grad()
        with torch.enable_grad():
            loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)

        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        # Update adversarial data
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    x_adv = Variable(x_adv, requires_grad=False)
    return x_adv, Kappa




def train(train_loader, model, criterion, optimizer, epoch, args, log):

    model.train()
    args.epsilon = 1 / 255.0  # a small perturbation
    args.step_size = args.epsilon / 8
    no_cross_sum = np.array([0] * 10)
    cross_sum = np.array([0] * 10)
    for i, (input, target) in enumerate(train_loader):
        input, target = input.cuda(), target.cuda()
        loss_all = criterion(model(input), target)  # CE loss
        # ------------------------------------------
        # only select samples in tail classes
        tail_list = torch.where(target >= 5)[0]
        if tail_list.shape[0] != 0:
            input, target = input[tail_list], target[tail_list]
            adv_img, dist = GA_PGD(model, input, target, args.epsilon,
                                   args.step_size, args.num_steps, rand_init=True)

            no_cross_idx = torch.where(dist >= args.num_steps)[0]   # 多轮次未越过分类边界的样本
            if no_cross_idx.shape[0] != 0:
                # advs_no_cross = adv_img[no_cross_idx]
                freq1 = [torch.sum(target[no_cross_idx] == i).item() for i in range(10)]
                no_cross_sum += np.array(freq1)
                # loss_no_cross = criterion(model(advs_no_cross), target[no_cross_idx])
                # loss_all += loss_no_cross

            cross_idx = torch.where(dist<args.num_steps)[0]  # 少轮次越过的样本做对抗训练
            if cross_idx.shape[0] != 0:
                advs_cross = adv_img[cross_idx]
                freq2 = [torch.sum(target[cross_idx] == i).item() for i in range(10)]
                cross_sum += np.array(freq2)
                loss_cross = criterion(model(advs_cross), target[cross_idx])
                loss_all += 1 * loss_cross
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

        # if i % args.print_freq == 0:
        #     output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
        #               'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #               'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        #               'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        #         epoch, i, len(train_loader), loss=losses, top1=top1, top5=top5,
        #         lr=optimizer.param_groups[-1]['lr'] * 0.1))
        #     print(output)
        #     log.write(output + '\n')
        #     log.flush()
    print("cross_sum: {}".format(cross_sum))
    print("no_cross_sum: {}".format(no_cross_sum))


def validate(val_loader, model, criterion, args, log=None, flag='val'):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.cuda(), target.cuda()
            output = model(input)
            loss = F.cross_entropy(output, target)
            # measure accuracy and record loss
            test_acc, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(test_acc[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), loss=losses,
                    top1=top1, top5=top5))
                print(output)
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                  .format(flag=flag, top1=top1, top5=top5, loss=losses))
        out_cls_acc = '%s Class Accuracy: %s' % (
            flag, (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
        print(output)
        print(out_cls_acc)
        if log is not None:
            log.write(output + '\n')
            log.write(out_cls_acc + '\n')
            log.flush()
    return top1.avg, losses.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 10:
        lr = args.lr
    elif epoch < 30:
        lr = args.lr * 0.1
    elif epoch < 60:
        lr = args.lr * 0.01
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


if __name__ == '__main__':
    setup_seed(2021)
    best_acc1 = 0
    args = get_args()

    args.store_name = '_'.join(
        ['adv', args.dataset, args.arch, args.loss_type, args.train_rule,
         args.imb_type, str(args.imb_factor), args.exp_str, str(args.num_steps), str(args.lr)])
    prepare_folders(args)

    print("=> creating model '{}'".format(args.arch))
    num_classes = 100 if args.dataset == 'cifar100' else 10
    use_norm = True if args.loss_type == 'LDAM' else False
    model = models.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cuda:0')
        model.load_state_dict(checkpoint['state_dict'])

    train_loader, val_loader, cls_num_list = get_dataset()
    print('cls num list:')
    print(cls_num_list)
    args.cls_num_list = cls_num_list

    # init log for training
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    main_worker()
