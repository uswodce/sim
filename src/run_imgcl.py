import os
import pickle
import argparse
from functools import partial
import random

import numpy as np
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from dataset_loaders import load_data_imgcl

def main(args):
    # fix seed for reproducibility
    if args.sfix:
        seed = 777
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # dataset
    trainval_data, train_data, val_data, test_data = load_data_imgcl(args.dataset, trainval_ratio=0.9)

    # common configuration
    config = {
        'train_batch_size': tune.choice([64, 128, 256, 512]),
        'epochs': tune.choice([20, 40, 60, 80]),
        'max_lr': tune.loguniform(1e-4, 0.05),
        'lr_mode': tune.choice(['1cycle', 'step', 'constant']),
        'steplr_step': tune.qrandint(5, 50, 5)
    }
    options = {
        'dataset': {'train': train_data, 'val': val_data},
        'test_batch_size': 400,
        'model_path': None,
        'report_ray': True
    }

    if args.sfix:
        options['sfix'] = True
        config['seed'] = tune.randint(0, 10000)
    else:
        options['sfix'] = False

    # dataset specific configuration
    if args.dataset == 'mnist':
        options['in_dim'] = 28
        options['out_dim'] = 10
        options['in_channels'] = 1
    elif args.dataset == 'cifar':
        options['in_dim'] = 32
        options['out_dim'] = 10
        options['in_channels'] = 3
    elif args.dataset == 'svhn':
        options['in_dim'] = 32
        options['out_dim'] = 10
        options['in_channels'] = 3

    # method specific configuration
    if args.method == 'sim-single':
        from sim.train_imgcl import train
        options['tier_type'] = 'single'
        options['basis'] = args.basis
        options['tier_dim'] = 50
        options['rev_dim'] = 32
        if args.basis == 'cnn':
            if args.dataset == 'mnist':
                options['out_channels'] = 24
            else:
                options['out_channels'] = 38

    elif args.method == 'sim-two':
        from sim.train_imgcl import train
        options['tier_type'] = 'two'
        config['bandwidth'] = tune.loguniform(1e-3, 1e+3)
        options['basis'] = args.basis
        options['first_tier_dim'] = 50
        options['second_tier_dim'] =  104
        options['rev_dim'] = 32
        if args.basis == 'cnn':
            if args.dataset == 'mnist':
                options['out_channels'] = 21
            else:
                options['out_channels'] = 36

    else:
        raise ValueError('The method is not supported')

    if args.tune:
        # setup for hyperparameter tuning
        scheduler = ASHAScheduler(
            metric='test_acc',
            mode='max',
            max_t=80,
            grace_period=5,
            reduction_factor=2)
        reporter = CLIReporter(
            metric_columns=['test_loss', 'test_acc', 'training_iteration'],
            max_progress_rows=10,
            max_report_frequency=5)
        # train and evaluate
        result = tune.run(
            partial(train, options=options),
            config=config,
            resources_per_trial={'cpu': 1, 'gpu': 1},
            num_samples=200,
            local_dir='../ray_results',
            scheduler=scheduler,
            progress_reporter=reporter)

        # select best trial
        best_trial = result.get_best_trial('test_acc', 'max', 'last')
        print('Best trial config: {}'.format(best_trial.config))
        print('Best trial validation loss: {}'.format(best_trial.last_result['test_loss']))
        print('Best trial validation acc: {}'.format(best_trial.last_result['test_acc']))
        print('Num of parameters: {}'.format(best_trial.last_result['num_params']))
        test_config = best_trial.config

    else:
        if args.dataset == 'cifar':
            if args.method == 'sim-single':
                test_config = {
                    'train_batch_size': 128,
                    'lr_mode': 'step',
                    'epochs': 10,
                    'max_lr': 1e-3,
                    'steplr_step': 10
                }
            if args.method == 'sim-two':
                test_config = {
                    'train_batch_size': 128,
                    'lr_mode': 'step',
                    'epochs': 10,
                    'max_lr': 1e-3,
                    'steplr_step': 10,
                    'bandwidth': 100.0
                }
        elif args.dataset == 'svhn':
            if args.method == 'sim-single':
                test_config = {
                    'train_batch_size': 128,
                    'lr_mode': 'step',
                    'epochs': 10,
                    'max_lr': 1e-3,
                    'steplr_step': 10
                }
            if args.method == 'sim-two':
                test_config = {
                    'train_batch_size': 128,
                    'lr_mode': 'step',
                    'epochs': 10,
                    'max_lr': 1e-3,
                    'steplr_step': 10,
                    'bandwidth': 100.0
                }
        elif args.dataset == 'mnist':
            if args.method == 'sim-single':
                test_config = {
                    'train_batch_size': 128,
                    'lr_mode': 'step',
                    'epochs': 10,
                    'max_lr': 1e-3,
                    'steplr_step': 10
                }
            if args.method == 'sim-two':
                test_config = {
                    'train_batch_size': 128,
                    'lr_mode': 'step',
                    'epochs': 10,
                    'max_lr': 1e-3,
                    'steplr_step': 10,
                    'bandwidth': 100.0
                }
                test_config['bandwidth'] = 10.0

        if args.sfix:
            test_config['seed'] = seed + 777

    # training best configuration and then evaluate
    options['dataset'] = {'train': trainval_data, 'val': test_data}
    options['report_ray'] = False
    num_runs = 3
    test_losses = []
    test_accs = []
    test_num_params = []
    for run_idx in range(num_runs):
        if args.sfix:
            test_config['seed'] += run_idx
        latest_result = train(test_config, options=options)
        test_losses.append(latest_result['test_loss'])
        test_accs.append(latest_result['test_acc'])
        test_num_params.append(latest_result['num_params'])
        print('Run %d:'%run_idx)
        print('Best trial test loss: {}'.format(test_losses[run_idx]))
        print('Best trial test acc: {}'.format(test_accs[run_idx]))
        print('Num of parameters: {:,}'.format(test_num_params[run_idx]))

    mean_loss = np.mean(test_losses)
    mean_acc = np.mean(test_accs)
    mean_num_params = int(np.mean(test_num_params))
    print()
    print('Mean of test set losses: {}'.format(mean_loss))
    print('Mean of test set accs: {}'.format(mean_acc))
    print('Num of parameters: {:,}'.format(mean_num_params))

    # save best configuration and the result
    save_info = {'num_trials': num_runs, 'test_loss':test_losses, 'test_acc':test_accs, 'test_num_params':test_num_params}
    save_info['config'] = test_config
    save_dir = '../result/imgcl/recent'
    if args.tune:
        if args.method == 'sim-single':
            filename = '%s_sim-single_%s.pkl'%(args.dataset, args.basis)
        elif args.method == 'sim-two':
            filename = '%s_sim-two_%s.pkl'%(args.dataset, args.basis)
    else:
        if args.method == 'sim-single':
            filename = '%s_sim-single_%s_off.pkl'%(args.dataset, args.basis)
        elif args.method == 'sim-two':
            filename = '%s_sim-two_%s_off.pkl'%(args.dataset, args.basis)
    path = os.path.join(save_dir, filename)
    with open(path, 'wb') as f:
        pickle.dump(save_info, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='method')

    parser.add_argument('--sfix', action='store_true')
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--dataset', default='mnist')

    parser_sim_single = subparsers.add_parser('sim-single')
    parser_sim_single.add_argument('--basis', default='cnn')

    parser_sim_two = subparsers.add_parser('sim-two')
    parser_sim_two.add_argument('--basis', default='cnn')

    args = parser.parse_args()

    main(args)
