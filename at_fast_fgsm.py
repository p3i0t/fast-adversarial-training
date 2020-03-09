import hydra
from omegaconf import DictConfig
import logging

import torch
from torch.optim import SGD, lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from advertorch.attacks import LinfPGDAttack
from utils import cal_parameters, get_dataset, get_model, AverageMeter

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logger = logging.getLogger(__name__)

clip_min = 0.
clip_max = 1.


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def train_epoch(classifier, data_loader, args, optimizer, scheduler=None):
    """
    Run one epoch.
    :param classifier: torch.nn.Module representing the classifier.
    :param data_loader: dataloader
    :param args:
    :param optimizer:
    :param scheduler:
    :return: mean of loss, mean of accuracy of this epoch.
    """
    classifier.train()

    eps = eval(args.epsilon)
    eps_iter = eval(args.epsilon_iter)

    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('Acc')

    for batch_idx, (x, y) in enumerate(data_loader):
        x, y = x.to(args.device), y.to(args.device)
        # start with uniform noise
        delta = torch.zeros_like(x).uniform_(-eps, eps)  # set to zero before interations on each mini-batch
        delta.requires_grad_()
        delta = clamp(delta, clip_min - x, clip_max - x)

        loss = F.cross_entropy(classifier(x + delta), y)
        grad_delta = torch.autograd.grad(loss, delta)[0].detach()  # get grad of noise

        # update delta with grad
        delta = (delta + torch.sign(grad_delta) * eps_iter).clamp_(-eps, eps)
        delta = clamp(delta, clip_min - x, clip_max - x)

        # real forward
        logits = classifier(x + delta)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        loss_meter.update(loss.item(), x.size(0))
        acc = (logits.argmax(dim=1) == y).float().mean().item()
        acc_meter.update(acc, x.size(0))

    return loss_meter.avg, acc_meter.avg


def eval_epoch(classifier, data_loader, args, adversarial=False):
    classifier.eval()

    eps = eval(args.epsilon)
    eps_iter = eval(args.epsilon_iter)

    if adversarial is True:
        adversary = LinfPGDAttack(classifier, eps=eps, eps_iter=eps_iter)
    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('Acc')

    for batch_idx, (x, y) in enumerate(data_loader):
        x, y = x.to(args.device), y.to(args.device)
        if adversarial is True:
            x_ = adversary.perturb(x, y)
        else:
            x_ = x
        logits = classifier(x_)
        loss = F.cross_entropy(logits, y)

        loss_meter.update(loss.item(), x.size(0))
        acc = (logits.argmax(dim=1) == y).float().mean().item()
        acc_meter.update(acc, x.size(0))

    return loss_meter.avg, acc_meter.avg


@hydra.main(config_path='configs/fast_fgsm_config.yaml')
def run(args: DictConfig) -> None:
    cuda_available = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = "cuda" if cuda_available and args.device == 'cuda' else "cpu"

    n_classes = args.n_classes
    classifier = get_model(name=args.classifier_name, n_classes=n_classes).to(device)

    logger.info('Classifier: {}, # parameters: {}'.format(args.classifier_name, cal_parameters(classifier)))

    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    train_data = get_dataset(data_name=args.dataset, data_dir=data_dir, train=True, crop_flip=True)
    test_data = get_dataset(data_name=args.dataset, data_dir=data_dir, train=False, crop_flip=False)

    train_loader = DataLoader(dataset=train_data, batch_size=args.n_batch_train, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=args.n_batch_test, shuffle=False)

    optimizer = SGD(classifier.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_steps = args.n_epochs * len(train_loader)
    scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=args.lr_min, max_lr=args.lr_max,
                                      step_size_up=lr_steps/2, step_size_down=lr_steps/2)

    optimal_loss = 1e5
    for epoch in range(1, args.n_epochs + 1):
        loss, acc = train_epoch(classifier, train_loader, args, optimizer, scheduler=scheduler)
        if loss < optimal_loss:
            optimal_loss = loss
            torch.save(classifier.state_dict(), '{}_at.pth'.format(args.classifier_name))
        logger.info('Epoch {}, lr: {:.4f}, loss: {:.4f}, acc: {:.4f}'.format(epoch, lr_scheduler.get_lr()[0], loss, acc))

    clean_loss, clean_acc = eval_epoch(classifier, test_loader, args, adversarial=False)
    adv_loss, adv_acc = eval_epoch(classifier, test_loader, args, adversarial=True)
    logger.info('Clean loss: {:.4f}, acc: {:.4f}'.format(clean_loss, clean_acc))
    logger.info('Adversarial loss: {:.4f}, acc: {:.4f}'.format(adv_loss, adv_acc))


if __name__ == '__main__':
    run()
