import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet18, resnet34, resnet50,\
    wide_resnet50_2, wide_resnet101_2, resnext50_32x4d, resnext101_32x8d


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
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


def get_dataset(data_name='cifar10', data_dir='data', train=True, crop_flip=True):
    """
    Get a dataset.
    :param data_name: str, name of dataset.
    :param data_dir: str, base directory of data.
    :param train: bool, return train set if True, or test set if False.
    :param crop_flip: bool, whether use crop_flip as data augmentation.
    :return: pytorch dataset.
    """
    mean_ = [0.5] * 3
    std_ = [0.5] * 3

    transform_3d_crop_flip = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean_, std_)
    ])

    transform_3d = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_, std_)
    ])

    if train:
        # when train is True, we use transform_1d_crop_flip by default unless crop_flip is set to False
        transform = transform_3d if crop_flip is False else transform_3d_crop_flip
    else:
        transform = transform_3d

    if data_name == 'cifar10':
        dataset = datasets.CIFAR10(data_dir, train=train, download=True, transform=transform)
    elif data_name == 'cifar100':
        dataset = datasets.CIFAR100(data_dir, train=train, download=True, transform=transform)
    elif data_name == 'svhn':
        split = 'train' if train else 'test'
        dataset = datasets.SVHN(data_dir, split=split, download=True, transform=transform)
    else:
        raise ('dataset {} is not available'.format(data_name))

    return dataset


def cal_parameters(model):
    """
    Calculate the number of parameters of a Pytorch model.
    :param model: torch.nn.Module
    :return: int, number of parameters.
    """
    return sum([para.numel() for para in model.parameters()])


def get_model(name='resnet18', n_classes=10):
    classifier = eval(name)(pretrained=False)  # load model from torchvision
    classifier.avgpool = nn.AdaptiveAvgPool2d(1)
    classifier.fc = nn.Linear(classifier.fc.in_features, n_classes)
    return classifier

