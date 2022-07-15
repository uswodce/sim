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

from dataset_loaders import load_data_cpmem

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
    T = args.T
    seq_len = 10
    n_trainval = 5000
    n_test = 500
    trainval_ratio = 0.9
    trainval_data, train_data, val_data, test_data = load_data_cpmem(T, seq_len, n_trainval, n_test, trainval_ratio)

    # common configuration
    config = {
        'batch_size': tune.choice([1, 2, 5, 10, 20]),
        'lr': tune.loguniform(1e-4, 0.05)
    }
    # NOTE: We set n_classes to 10. Indeed, the copy memory task
    # considers the case when the elements of output sequences
    # only take one of 9 digiits (0-8). But we follow the original
    # source code of https://github.com/locuslab/TCN: the task
    # could be interpreted as the case when we don't know
    # the digit 9 will never be placed in the output sequence.
    options = {
        'n_classes':10,
        'dataset': {'train': train_data, 'val': val_data},
        'epochs':20,
        'report_ray':True
    }

    if args.sfix:
        options['sfix'] = True
        config['seed'] = tune.randint(0, 10000)
    else:
        options['sfix'] = False

    # method specific configuration
    if args.method == 'sim-single':
        from sim.train_cpmem import train
        if args.basis == 'tcn':
            options['tier_type'] = 'single'
            options['basis'] = 'tcn'
            options['tier_dim'] = 32
            options['rev_dim'] = 32
            options['channel_sizes'] = [10] * 8
            options['kernel_size'] = 8
            options['dropout_rate'] = 0.0
            options['clip'] = 1.0
            options['eps'] = 1e-5
            options['multi_gpu'] = True

    elif args.method == 'sim-two':
        from sim.train_cpmem import train
        if args.basis == 'tcn':
            options['tier_type'] = 'two'
            options['basis'] = 'tcn'
            options['first_tier_dim'] = 32
            options['second_tier_dim'] = 32
            options['rev_dim'] = 32
            options['channel_sizes'] = [10] * 8
            options['kernel_size'] = 8
            options['dropout_rate'] = 0.0
            options['clip'] = 1.0
            options['eps'] = 1e-5
            options['multi_gpu'] = True
            config['bandwidth'] = tune.loguniform(1e-3, 1e+3)

    else:
        raise ValueError('The method is not supported')

    if args.tune:
        # setup for hyperparameter tuning
        scheduler = ASHAScheduler(
            metric='test_loss',
            mode='min',
            max_t=6,
            grace_period=3,
            reduction_factor=2)
        reporter = CLIReporter(
            metric_columns=['test_loss', 'training_iteration'],
            max_progress_rows=10,
            max_report_frequency=5)
        # train and evaluate
        result = tune.run(
            partial(train, options=options),
            config=config,
            resources_per_trial={'cpu': 1, 'gpu': 1},
            num_samples=100,
            local_dir='../ray_results',
            scheduler=scheduler,
            progress_reporter=reporter)

        # select best trial
        best_trial = result.get_best_trial('test_loss', 'min', 'last')
        print('Best trial config: {}'.format(best_trial.config))
        print('Best trial validation loss: {}'.format(best_trial.last_result['test_loss']))
        print('Best trial validation acc: {}'.format(best_trial.last_result['test_acc']))
        print('Num of parameters: {}'.format(best_trial.last_result['num_params']))
        test_config = best_trial.config

    else:
        test_config = {}
        if args.method == 'sim-single':
            test_config['batch_size'] = 10
            test_config['lr'] = 0.01
        elif args.method == 'sim-two':
            test_config['batch_size'] = 10
            test_config['lr'] = 0.01
            test_config['bandwidth'] = 500.0

        if args.sfix:
            test_config['seed'] = seed + 777

    # training best configuration and then evaluate
    options['dataset'] = {'train': trainval_data, 'val': test_data}
    options['report_ray'] = False
    num_runs = 1
    test_losses = []
    test_accs = []
    num_params = []
    train_losses = []
    train_accs = []
    train_times = []
    # NOTE: we observe the first run for all methods is likely to be slightly
    # slow compared to other runs. This may be because cuda continues to allocate
    # memory space during the run of this code. For more accurate comparison across runs,
    # it may be better if the code could be modified so that each run is executed separately.
    for run_idx in range(num_runs):
        if args.sfix:
            test_config['seed'] += run_idx
        latest_result = train(test_config, options=options)
        test_losses.append(latest_result['test_loss'])
        test_accs.append(latest_result['test_acc'])
        num_params.append(latest_result['num_params'])
        train_losses.append(latest_result['train_losses'])
        train_accs.append(latest_result['train_accs'])
        train_times.append(latest_result['train_times'])
        print('Best trial train loss at last step: {}'.format(test_losses[run_idx]))
        print('Best trial test loss: {}'.format(test_losses[run_idx]))
        print('Best trial test acc: {}'.format(test_accs[run_idx]))
        print('Num of parameters: {:,}'.format(num_params[run_idx]))

    mean_loss = np.mean(test_losses)
    mean_acc = np.mean(test_accs)
    mean_num_params = int(np.mean(num_params))
    print()
    print('Mean of test losses: {}'.format(mean_loss))
    print('Mean of test accs: {}'.format(mean_acc))
    print('Num of parameters: {:,}'.format(mean_num_params))

    # save best configuration and the result
    save_info = {'test_loss':test_losses, 'test_acc':test_accs, 'num_params':num_params,
                'train_losses':train_losses, 'train_accs':train_accs, 'train_times':train_times}
    save_info['config'] = test_config
    save_dir = '../result/cpmem/recent'
    if args.tune:
        result_filename = 'T%d_%s_%s.pkl'%(args.T, args.method, args.basis)
    else:
        result_filename = 'T%d_%s_%s_off.pkl'%(args.T, args.method, args.basis)
    path = os.path.join(save_dir, result_filename)
    with open(path, 'wb') as f:
        pickle.dump(save_info, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='method')

    parser.add_argument('--sfix', action='store_true')
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--T', type=int, default=1000)

    parser_sim_single = subparsers.add_parser('sim-single')
    parser_sim_single.add_argument('--basis', default='tcn')

    parser_sim_two = subparsers.add_parser('sim-two')
    parser_sim_two.add_argument('--basis', default='tcn')

    args = parser.parse_args()

    main(args)
