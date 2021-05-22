import torch
from torchvision import datasets, transforms

def get_dataloader(args):
    if args.dataset.lower()=='cifar10':
        train_loader = torch.utils.data.DataLoader( 
            datasets.CIFAR10(args.data_root, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader( 
            datasets.CIFAR10(args.data_root, train=False, download=True,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=4)
    elif args.dataset.lower()=='cifar100':
        train_loader = torch.utils.data.DataLoader( 
            datasets.CIFAR100(args.data_root, train=True, download=True,
                       transform=transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader( 
            datasets.CIFAR100(args.data_root, train=False, download=True,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=4)
   
    return train_loader, test_loader