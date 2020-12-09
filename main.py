from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import models
from flops import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--data_path', type=str, default='../data')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save', default='./checkpoint', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='ResNet56', type=str,
                    help='architecture to use')
parser.add_argument('--sr', type=float)
parser.add_argument('--threshold', type=float)
args = parser.parse_args()
print(args)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if args.num_classes == 10:
    train_set = datasets.CIFAR10(args.data_path, train=True)
    test_set = datasets.CIFAR10(args.data_path, train=False)
else:
    train_set = datasets.CIFAR100(args.data_path, train=True)
    test_set = datasets.CIFAR100(args.data_path, train=False)
train_set.transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
test_set.transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
model = models.__dict__[args.arch](num_classes=args.num_classes)
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
history_score = np.zeros((args.epochs + 1, 3))


def train(epoch):
    model.train()
    global history_score
    avg_loss = 0.
    train_acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        avg_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        ###########learning the shape of filter with filter skeleton################
        if args.sr and args.threshold:
            model.update_skeleton(args.sr, args.threshold)
        ############################################################################
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()))
    history_score[epoch][0] = avg_loss / len(train_loader)
    history_score[epoch][1] = train_acc / float(len(train_loader))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


best_prec1 = 0.
for epoch in range(args.epochs):
    if epoch in [args.epochs * 0.5, args.epochs * 0.75]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    train(epoch)
    prec1 = test()
    history_score[epoch][2] = prec1
    np.savetxt(os.path.join(args.save, 'train_record.txt'), history_score, fmt='%10.5f', delimiter=',')
    if prec1 > best_prec1:
        best_prec1 = prec1
        torch.save(model.state_dict(), os.path.join(args.save, 'best.pth.tar'))
print("Best accuracy: " + str(best_prec1))
history_score[-1][0] = best_prec1
np.savetxt(os.path.join(args.save, 'train_record.txt'), history_score, fmt='%10.5f', delimiter=',')
##############pruning filter in filter without finetuning#################
if args.sr and args.threshold:
    model.load_state_dict(torch.load(os.path.join(args.save, 'best.pth.tar')))
    model.prune(args.threshold)
    test()
    print(model)
    torch.save(model.state_dict(), os.path.join(args.save, 'pruned.pth.tar'))
    print_model_param_nums(model)
    count_model_param_flops(model)
#########################################################
