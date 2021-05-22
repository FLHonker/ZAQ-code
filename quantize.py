import os
import copy
import random
import numpy as np
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import network
from dataloader import get_dataloader
from collections import OrderedDict
from quantization import quantize_model


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
    parser = argparse.ArgumentParser(description='PyTorch Quantization')

    parser.add_argument('--model', type=str, default='resnet20', help='model name (default: mnist)')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                            help='dataset name (default: cifar10)')
    parser.add_argument('--data_root', required=True, default=None, help='data path')
    parser.add_argument('--ckpt', default='', help='the path of pre-trained parammeters')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=6786, metavar='S', help='random seed (default: 6786)')
    parser.add_argument('--scheduler', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--step_size', type=int, default=80, help='step size')
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training')
    parser.add_argument('--device', default="0", help='device to use')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('--download', action='store_true', default=False)
    # quantization
    parser.add_argument('--weight_bit', type=int, default=6, help='bit-width for parameters')
    parser.add_argument('--activation_bit', type=int, default=8, help='bit-width for act')
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    args.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    os.makedirs('checkpoint/q_model/', exist_ok=True)

    model = network.get_model(args)
    model.load_state_dict(torch.load(args.ckpt))
    model.to(args.device)

    train_loader, test_loader = get_dataloader(args)

    best_acc = test(args, model, test_loader, 0)

    q_model = quantize_model(model, args)
    quant_acc = test(args, q_model, test_loader, 0)
    print("Quant Acc=%.6f"%quant_acc)
    print("Best Acc=%.6f"%best_acc)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    retrain_acc = 0
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size, 0.1)

    if args.test_only:
        return

    for epoch in range(1, args.epochs + 1):
        train(args, q_model, train_loader, optimizer, epoch)
        acc = test(args, q_model, test_loader, epoch)
        scheduler.step()
        if acc > retrain_acc:
            retrain_acc = acc
            print('Saving a best checkpoint ...')
            torch.save(model.state_dict(),"checkpoint/q_model/%s-%s-Q.pt"%(args.dataset, args.model))
    print("Retrain Acc=%.6f" % retrain_acc)
    print("Quant Acc=%.6f" % quant_acc)
    print("Best Acc=%.6f" % best_acc)

if __name__ == "__main__":
    main()