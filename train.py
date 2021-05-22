from __future__ import print_function
import os
import random
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import network
from dataloader import get_dataloader


def train(args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if args.verbose and batch_idx % args.log_interval == 0:
            print('Train Epoch: [{}] [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, test_loader, cur_epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nEpoch [{}] Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        cur_epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct/len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser('Pretrain P model.')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test_batch_size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--data_root', type=str, required=True, default=None)
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'cifar100', 'caltech101', 'nyuv2'],
                        help='dataset name (default: mnist)')
    parser.add_argument('--model', type=str, default='resnet20', choices=['resnet18', 'resnet50', 'mobilenetv2', 'resnet20', 'vgg19'],
                        help='model name (default: resnet20)')
    parser.add_argument('--step_size', type=int, default=80)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--device', type=str, default='0',
                        help='device for training')
    parser.add_argument('--seed', type=int, default=6786, metavar='S',
                        help='random seed (default: 6786)')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('--download', action='store_true', default=False)
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    args.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    os.makedirs('checkpoint/p_model/', exist_ok=True)

    print(args)

    train_loader, test_loader = get_dataloader(args)
    model = network.get_model(args)

    if args.ckpt is not None and args.pretrained:
        model.load_state_dict(torch.load(args.ckpt))
    
    model = model.to(args.device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    best_acc = 0
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size, 0.1)

    if args.test_only:
        acc = test(args, model, test_loader, 0)
        return

    for epoch in range(1, args.epochs + 1):
        # print("Lr = %.6f"%(optimizer.param_groups[0]['lr']))
        train(args, model, train_loader, optimizer, epoch)
        acc = test(args, model, test_loader, epoch)
        scheduler.step()
        if acc>best_acc:
            best_acc = acc
            print('Saving a best checkpoint ...')
            torch.save(model.state_dict(),"checkpoint/p_model/%s-%s.pt"%(args.dataset, args.model))
    print("Best Acc=%.6f" % best_acc)

if __name__ == '__main__':
    main()