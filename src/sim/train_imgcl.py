import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from ray import tune

from . import net
from .utils import cuda

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train(config, options):
    train_batch_size = config['train_batch_size']
    lr_mode = config['lr_mode']
    epochs = config['epochs']
    max_lr = config['max_lr']
    steplr_step = config['steplr_step']

    in_dim = options['in_dim']
    out_dim = options['out_dim']
    in_channels = options['in_channels']
    tier_type = options['tier_type']
    basis = options['basis']
    test_batch_size = options['test_batch_size']
    model_path = options['model_path']
    report_ray = options['report_ray']
    sfix = options['sfix']

    if sfix:
        seed = config['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        g = torch.Generator()
        g.manual_seed(0)
        trainloader = DataLoader(
                        options['dataset']['train'],
                        batch_size=train_batch_size,
                        shuffle=True,
                        worker_init_fn=seed_worker,
                        generator=g)
        valloader = DataLoader(
                        options['dataset']['val'],
                        batch_size=test_batch_size,
                        shuffle=False,
                        worker_init_fn=seed_worker,
                        generator=g)
    else:
        trainloader = DataLoader(
            options['dataset']['train'],
            batch_size=train_batch_size,
            shuffle=True)
        valloader = DataLoader(
            options['dataset']['val'],
            batch_size=test_batch_size,
            shuffle=False)

    if tier_type == 'single':
        tier_dim = options['tier_dim']
        rev_dim = options['rev_dim']
        if basis == 'cnn':
            out_channels = options['out_channels']
            model = net.SingleTierConvNet(
                                in_dim=in_dim,
                                out_dim=out_dim,
                                in_channels=in_channels,
                                out_channels=out_channels,
                                tier_dim=tier_dim,
                                rev_dim=rev_dim
                                )
    elif tier_type == 'two':
        bandwidth = config['bandwidth']
        first_tier_dim = options['first_tier_dim']
        second_tier_dim = options['second_tier_dim']
        rev_dim = options['rev_dim']
        if basis == 'cnn':
            out_channels = options['out_channels']
            model = net.TwoTierConvNet(
                                in_dim=in_dim,
                                out_dim=out_dim,
                                in_channels=in_channels,
                                out_channels=out_channels,
                                first_tier_dim=first_tier_dim,
                                second_tier_dim=second_tier_dim,
                                rev_dim=rev_dim,
                                feature='cossin',
                                kernel='gauss',
                                bandwidth=bandwidth
                                )

    num_params = count_parameters(model)

    optimizer = optim.Adam(model.parameters(), lr=max_lr)
    change_mo = False
    if lr_mode == '1cycle':
        lr_schedule = lambda t: np.interp([t],
                                          [0, (epochs - 5) // 2, epochs - 5, epochs],
                                          [1e-3, max_lr, 1e-3, 1e-3])[0]
        change_mo = True
    elif lr_mode == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, steplr_step, gamma=0.1, last_epoch=-1)
    elif lr_mode != 'constant':
        raise Exception('lr mode one of constant, step, 1cycle')
    if change_mo:
        max_mo = 0.85
        momentum_schedule = lambda t: np.interp([t],
                                                [0, (epochs - 5) // 2, epochs - 5, epochs],
                                                [0.95, max_mo, 0.95, 0.95])[0]

    model = cuda(model)

    for epoch in range(1, 1 + epochs):
        nProcessed = 0
        nTrain = len(trainloader.dataset)
        model.train()
        for batch_idx, batch in enumerate(trainloader):
            if lr_mode == '1cycle':
                lr = lr_schedule(epoch -  1 + batch_idx/ len(trainloader))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            if change_mo:
                beta1 = momentum_schedule(epoch - 1 + batch_idx / len(trainloader))
                for param_group in optimizer.param_groups:
                    param_group['betas'] = (beta1, optimizer.param_groups[0]['betas'][1])

            data, target = cuda(batch[0]), cuda(batch[1])
            optimizer.zero_grad()
            preds = model(data)
            loss = nn.CrossEntropyLoss()(preds, target)
            loss.backward()
            optimizer.step()

            if not report_ray:
                nProcessed += len(data)
                print_freq = 100
                if batch_idx % print_freq == 0 and batch_idx > 0:
                    incorrect = preds.float().argmax(1).ne(target.data).sum()
                    err = 100. * incorrect.float() / float(len(data))
                    partialEpoch = epoch + batch_idx / len(trainloader) - 1
                    print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tError: {:.2f}'.format(
                        partialEpoch, nProcessed, nTrain,
                        100. * batch_idx / len(trainloader),
                        loss.item(), err))

        if lr_mode == 'step':
            lr_scheduler.step()

        if model_path is not None:
            torch.save(model.state_dict(), model_path)

        model.eval()
        test_loss = 0
        incorrect = 0
        with torch.no_grad():
            for batch in valloader:
                data, target = cuda(batch[0]), cuda(batch[1])
                preds = model(data)
                loss = nn.CrossEntropyLoss(reduction='sum')(preds, target)
                test_loss += loss
                incorrect += preds.float().argmax(1).ne(target.data).sum()
            test_loss /= len(valloader.dataset)
            nTotal = len(valloader.dataset)

        latest_test_loss = test_loss.cpu().item()
        latest_test_acc = ((nTotal - incorrect) / nTotal).cpu().item()

        if report_ray:
            tune.report(
                test_loss=latest_test_loss,
                test_acc=latest_test_acc,
                num_params=num_params
                )

    return {
        'test_loss':latest_test_loss,
        'test_acc':latest_test_acc,
        'num_params':num_params
        }
