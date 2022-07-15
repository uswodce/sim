import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune

from . import net
from .utils import cuda

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(config, options):
    lr = config['lr']

    in_dim = options['in_dim']
    out_dim = options['out_dim']
    num_iters = options['num_iters']
    report_ray = options['report_ray']
    sfix = options['sfix']

    if sfix:
        seed = config['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    train_data = options['dataset']['train']
    test_data = options['dataset']['val']

    if options['tier_type'] == 'single':
        tier_dim = config['tier_dim']
        rev_dim = options['rev_dim']
        if options['basis'] == 'fcn':
            basis_dim = config['basis_dim']
            model = net.SingleTierFcNet(
                                in_dim=in_dim,
                                out_dim=out_dim,
                                basis_dim=basis_dim,
                                tier_dim=tier_dim,
                                rev_dim=rev_dim
                                )
        elif options['basis'] == 'rff':
            bandwidth = config['bandwidth']
            model = net.SingleTierFourierNet(
                                    in_dim=in_dim,
                                    out_dim=out_dim,
                                    tier_dim=tier_dim,
                                    rev_dim=rev_dim,
                                    bandwidth=bandwidth
                                    )
    elif options['tier_type'] == 'two':
        first_tier_dim = config['first_tier_dim']
        second_tier_dim = config['second_tier_dim']
        rev_dim = options['rev_dim']
        bandwidth = config['bandwidth']
        if options['basis'] == 'fcn':
            basis_dim = config['basis_dim']
            model = net.TwoTierFcNet(
                                in_dim=in_dim,
                                out_dim=out_dim,
                                basis_dim=basis_dim,
                                first_tier_dim=first_tier_dim,
                                second_tier_dim=second_tier_dim,
                                rev_dim=rev_dim,
                                bandwidth=bandwidth
                                )

    num_params = count_parameters(model)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    model = cuda(model)
    train_input = cuda(train_data[0])
    train_target = cuda(train_data[1])
    test_input = cuda(test_data[0])
    test_target = cuda(test_data[1])

    for _ in range(1, 1 + num_iters):
        model.train()
        optimizer.zero_grad()
        preds = model(train_input)
        loss = nn.MSELoss()(preds, train_target)
        loss.backward()
        optimizer.step()
        train_psnr = - 10 * torch.log10(loss).item() # or 10 * torch.log10(1./loss).item()

        model.eval()
        with torch.no_grad():
            preds = model(test_input)
            test_loss = nn.MSELoss()(preds, test_target)
            test_psnr = - 10 * torch.log10(test_loss).item()
            pred_img = preds

        latest_train_psnr = train_psnr
        latest_test_psnr = test_psnr
        latest_pred_img = pred_img

        if report_ray:
            tune.report(test_psnr=latest_test_psnr, num_params=num_params)

    return {
        'train_psnr':latest_train_psnr,
        'test_psnr':latest_test_psnr,
        'pred_img':latest_pred_img,
        'num_params':num_params
        }
