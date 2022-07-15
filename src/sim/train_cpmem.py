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
    batch_size = config['batch_size']
    lr = config['lr']

    n_classes = options['n_classes']
    epochs = options['epochs']
    eps = options['eps']
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

    n_train = len(train_data[0])
    n_test = len(test_data[0])
    n_steps = train_data[0].size(1)

    if options['tier_type'] == 'single':
        if options['basis'] == 'tcn':
            tier_dim = options['tier_dim']
            rev_dim = options['rev_dim']
            channel_sizes = options['channel_sizes']
            kernel_size = options['kernel_size']
            dropout_rate = options['dropout_rate']
            clip = options['clip']
            multi_gpu = options['multi_gpu']
            model = net.SingleTierTempConvNet(
                                            input_size=1,
                                            output_size=n_classes,
                                            channel_sizes=channel_sizes,
                                            kernel_size=kernel_size,
                                            dropout_rate=dropout_rate,
                                            tier_dim=tier_dim,
                                            rev_dim=rev_dim
                                            )
    elif options['tier_type'] == 'two':
        if options['basis'] == 'tcn':
            first_tier_dim = options['first_tier_dim']
            second_tier_dim = options['second_tier_dim']
            rev_dim = options['rev_dim']
            bandwidth = config['bandwidth']
            channel_sizes = options['channel_sizes']
            kernel_size = options['kernel_size']
            dropout_rate = options['dropout_rate']
            clip = options['clip']
            multi_gpu = options['multi_gpu']
            model = net.TwoTierTempConvNet(
                                            input_size=1,
                                            output_size=n_classes,
                                            channel_sizes=channel_sizes,
                                            kernel_size=kernel_size,
                                            dropout_rate=dropout_rate,
                                            first_tier_dim=first_tier_dim,
                                            second_tier_dim=second_tier_dim,
                                            rev_dim=rev_dim,
                                            bandwidth=bandwidth
                                            )

    num_params = count_parameters(model)

    optimizer = optim.Adam(model.parameters(), lr=lr, eps=eps)

    model = cuda(model)
    if multi_gpu and batch_size >= 50:
        model = nn.DataParallel(model, dim=0).cuda()
    train_x = cuda(train_data[0])
    train_y = cuda(train_data[1])
    test_x = cuda(test_data[0])
    test_y = cuda(test_data[1])

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    train_losses = []
    train_accs = []
    train_times = []
    for ep in range(1, epochs + 1):
        model.train()
        for batch_idx, batch in enumerate(range(0, n_train, batch_size)):
            start_ind = batch
            end_ind = start_ind + batch_size
            x = train_x[start_ind:end_ind]
            y = train_y[start_ind:end_ind]

            optimizer.zero_grad()
            out = model(x.unsqueeze(1)).view(-1, n_classes)
            loss = nn.CrossEntropyLoss()(out, y.view(-1))
            if clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            loss.backward()
            optimizer.step()

            if not report_ray:
                pred = out.data.max(1, keepdim=True)[1]
                correct = pred.eq(y.data.view_as(pred)).cpu().sum()
                counter = out.size(0)
                acc = 100. * correct / counter
                end.record()
                torch.cuda.synchronize()
                elapsed = start.elapsed_time(end)
                print('| Epoch {:3d} | {:5d}/{:5d} batches | lr {:2.5f} | ms {:5.2f} | '
                    'loss {:5.8f} | accuracy {:5.4f}'.format(ep, batch_idx + 1,
                    n_train // batch_size, lr, elapsed, loss.item(), acc))
                train_losses.append(loss.item())
                train_accs.append(acc)
                train_times.append(elapsed)

        if report_ray or ep == epochs:
            model.eval()
            total_loss = 0
            total_len = 0
            correct = 0
            counter = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(range(0, n_test, batch_size)):
                    start_ind = batch
                    end_ind = start_ind + batch_size
                    x = test_x[start_ind:end_ind]
                    y = test_y[start_ind:end_ind]

                    out = model(x.unsqueeze(1)).view(-1, n_classes)
                    test_loss = nn.CrossEntropyLoss()(out, y.view(-1))
                    pred = out.data.max(1, keepdim=True)[1]
                    correct += pred.eq(y.data.view_as(pred)).cpu().sum()
                    counter += out.size(0)
                    total_loss += n_steps * test_loss.item()
                    total_len += n_steps

            latest_test_loss = total_loss / total_len
            latest_test_acc = 100. * correct / counter

        if report_ray:
            tune.report(
                test_loss=latest_test_loss,
                test_acc=latest_test_acc,
                num_params=num_params
                )

    if not report_ray:
        print('\nTest set: Average loss: {:.8f}  |  Accuracy: {:.4f}\n'.format(
            latest_test_loss, latest_test_acc))

    return {
        'test_loss':latest_test_loss,
        'test_acc':latest_test_acc,
        'num_params':num_params,
        'train_losses':train_losses,
        'train_accs':train_accs,
        'train_times':train_times
        }
