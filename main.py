from __future__ import print_function
import os, random
import copy
import numpy as np
import argparse
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import network
from utils.visualizer import VisdomPlotter
from utils.loss import *
from dataloader import get_dataloader
from quantization import quantize_model


vp = VisdomPlotter('8097', env='ZAQ-main')

def train(args, p_model, q_model, generator, optimizer, epoch):
    p_model.eval()
    q_model.train()
    generator.train()
    optimizer_Q, optimizer_G = optimizer

    inter_loss = SCRM().to(args.device)

    for i in range(args.epoch_itrs):
        for k in range(5):
            z = torch.randn((args.batch_size, args.nz, 1, 1)).to(args.device)
            optimizer_Q.zero_grad()
            fake = generator(z).detach()
            g_p, p_logit = p_model(fake, True)
            g_q, q_logit = q_model(fake, True)
            loss_Q = F.l1_loss(q_logit, p_logit.detach()) + args.alpha * inter_loss(g_q, g_p)
            
            loss_Q.backward()
            optimizer_Q.step()

        z = torch.randn((args.batch_size, args.nz, 1, 1)).to(args.device)
        optimizer_G.zero_grad()
        generator.train()
        fake = generator(z)
        g_p, p_logit = p_model(fake, True) 
        g_q, q_logit = q_model(fake, True)

        loss_G = - F.l1_loss(q_logit, p_logit) - args.alpha * inter_loss(g_q, g_p) - args.beta * g_p[-1].abs().mean()

        loss_G.backward()
        optimizer_G.step()

        if i % args.log_interval == 0:
            print('Train Epoch: [{}] [{}/{} ({:.0f}%)]\tG_Loss: {:.6f} Q_loss: {:.6f}'.format(
                epoch, i, args.epoch_itrs, 100*float(i)/float(args.epoch_itrs), loss_G.item(), loss_Q.item()))
            vp.add_scalar('Loss_Q', (epoch-1)*args.epoch_itrs+i, loss_Q.item())
            vp.add_scalar('Loss_G', (epoch-1)*args.epoch_itrs+i, loss_G.item())

def test(args, model, test_loader, epoch=0):
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(args.device), target.to(args.device)

            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nEpoch [{}] Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    acc = correct/len(test_loader.dataset)
    return acc


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='ZAQ CIFAR.')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test_batch_size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 128)')
    
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--epoch_itrs', type=int, default=60)
    parser.add_argument('--lr_Q', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--lr_G', type=float, default=1e-3,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--data_root', type=str, required=True, default=None)

    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                        help='dataset name (default: cifar10)')
    parser.add_argument('--model', type=str, default='resnet18', 
                        choices=['mobilenetv2', 'vgg19', 'resnet18', 'resnet20', 'resnet50'],
                        help='model name (default: resnet18)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--device', type=str, default='0',
                        help='device for training')
    parser.add_argument('--seed', type=int, default=6786, metavar='S',
                        help='random seed (default: 6786)')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--nz', type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.1)
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    args.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    os.makedirs('checkpoint/q_model/', exist_ok=True)
    print(args)

    _, test_loader = get_dataloader(args)

    args.num_classes = 10 if args.dataset=='cifar10' else 100
    q_model = network.get_model(args)
    generator = network.gan.Generator(nz=args.nz, nc=3, img_size=32)
    
    q_model.load_state_dict(torch.load(args.ckpt))
    print("p_model restored from %s"%(args.ckpt))

    # p_model = p_model.to(device)
    q_model = q_model.to(args.device)
    generator = generator.to(args.device)
    p_model = copy.deepcopy(q_model)

    # quantization
    q_model = quantize_model(q_model, args)
    quant_acc = test(args, q_model, test_loader, 0)
    print('Quat Acc=%0.4f \n' % quant_acc)

    p_model.eval()

    optimizer_Q = optim.SGD(q_model.parameters(), lr=args.lr_Q, weight_decay=args.weight_decay, momentum=0.9)
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_G)
    
    scheduler_Q = optim.lr_scheduler.MultiStepLR(optimizer_Q, [100, 200], args.gamma)
    scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, [100, 200], args.gamma)
    best_acc = 0
    if args.test_only:
        acc = test(args, q_model, test_loader, 0)
        return
    acc_list = []
    for epoch in range(1, args.epochs + 1):
        # Train
        train(args, p_model=p_model, q_model=q_model, generator=generator, optimizer=[optimizer_Q, optimizer_G], epoch=epoch)
        scheduler_Q.step()
        scheduler_G.step()
        # Test
        acc = test(args, q_model, test_loader, epoch)
        acc_list.append(acc)
        if acc>best_acc:
            best_acc = acc
            print('Saving a best checkpoint ...')
            torch.save(q_model.state_dict(),"checkpoint/q_model/ZAQ-%s-%s-%sbit.pt"%(args.dataset, args.model, args.weight_bit))
            torch.save(generator.state_dict(),"checkpoint/q_model/ZAQ-%s-%s-%sbit-generator.pt"%(args.dataset, args.model, args.weight_bit))
        vp.add_scalar('Acc', epoch, acc)
    print("Best Acc=%.6f" % best_acc)

    import csv
    os.makedirs('log', exist_ok=True)
    with open('log/ZAQ-%s-%s-%sbit.csv'%(args.dataset, args.model, args.param_bits), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(acc_list)

if __name__ == '__main__':
    main()