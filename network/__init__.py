
from . import gan
from . import vgg
from . import mobilenetv2
from . import resnet20
from . import resnet

import torchvision


def get_model(args):
    if args.model.lower()=='resnet20':
        return resnet20.ResNet20(num_classes=args.num_classes)
    elif args.model.lower()=='mobilenetv2':
        return mobilenetv2.MobileNetV2(num_classes=args.num_classes)
    elif args.model.lower()=='vgg19':
        return vgg.VGG19(num_classes=args.num_classes)
    elif args.model.lower()=='resnet18':
        return resnet.ResNet18(num_classes=args.num_classes)
    elif args.model.lower()=='resnet50':
        return resnet.ResNet50(num_classes=args.num_classes)
    else:
        return torchvision.models.__dict__[args.model](num_classes=args.num_classes, pretrained=args.pretrained)
