import os
import pickle
import argparse
from functools import partial
import random

import numpy as np
import torch
from torchvision.utils import save_image
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from dataset_loaders import load_data_imgreg

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
    train_data, val_data, test_data = load_data_imgreg(args.dataset, args.imgid)

    # common configuration
    config = {
        'lr': tune.loguniform(1e-4, 0.05)
    }
    options = {
        'dataset': {'train': train_data, 'val': val_data},
        'in_dim': 2,
        'out_dim': 3,
        'num_iters': 2000,
        'report_ray': True
    }

    if args.sfix:
        options['sfix'] = True
        config['seed'] = tune.randint(0, 10000)
    else:
        options['sfix'] = False

    # method specific configuration
    if args.method == 'sim-single':
        from sim.train_imgreg import train
        options['tier_type'] = 'single'
        options['basis'] = args.basis
        config['tier_dim'] = tune.choice([32, 64, 128, 256, 512])
        options['rev_dim'] = 256
        if args.basis == 'fcn':
            config['basis_dim'] = tune.choice([32, 64, 128, 256, 512])
        if args.basis == 'rff':
            config['bandwidth'] = tune.loguniform(1e-3, 1e+3)
    elif args.method == 'sim-two':
        from sim.train_imgreg import train
        options['tier_type'] = 'two'
        options['basis'] = args.basis
        config['first_tier_dim'] = tune.choice([32, 64, 128, 256, 512])
        config['second_tier_dim'] = tune.choice([32, 64, 128, 256, 512])
        options['rev_dim'] = 256
        config['bandwidth'] = tune.loguniform(1e-3, 1e+3)
        if args.basis == 'fcn':
            config['basis_dim'] = tune.choice([32, 64, 128, 256, 512])

    if args.tune:
        # setup for hyperparameter tuning
        scheduler = ASHAScheduler(
            metric='test_psnr',
            mode='max',
            max_t=100,
            grace_period=5,
            reduction_factor=2)
        reporter = CLIReporter(
            metric_columns=['test_psnr', 'training_iteration'],
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
        best_trial = result.get_best_trial('test_psnr', 'max', 'last')
        print('Best trial config: {}'.format(best_trial.config))
        print('Best trial validation psnr: {}'.format(best_trial.last_result['test_psnr']))
        print('Num of parameters: {}'.format(best_trial.last_result['num_params']))
        test_config = best_trial.config

    else:
        test_config = {
            'lr': 1e-4
        }
        if args.method == 'sim-single':
            test_config['tier_dim'] = 256
            if args.basis == 'fcn':
                test_config['basis_dim'] = 256
            elif args.basis == 'rff':
                test_config['bandwidth'] = 0.005
        elif args.method == 'sim-two':
            test_config['first_tier_dim'] = 256
            test_config['second_tier_dim'] = 256
            test_config['bandwidth'] = 0.005
            if args.basis == 'fcn':
                test_config['basis_dim'] = 256

        if args.sfix:
            test_config['seed'] = seed + 777

    # training best configuration and then evaluate
    options['dataset'] = {'train': train_data, 'val': test_data}
    options['report_ray'] = False
    if args.sfix:
        test_config['seed'] += 777
    latest_result = train(test_config, options=options)
    train_psnr = latest_result['train_psnr']
    test_psnr = latest_result['test_psnr']
    pred_img = latest_result['pred_img']
    num_param = latest_result['num_params']
    print('Best trial train set psnr: {}'.format(train_psnr))
    print('Best trial test set psnr: {}'.format(test_psnr))
    print('Num of parameters: {:,}'.format(num_param))

    # save best configuration and the result
    save_info = {'train_psnrs':train_psnr, 'test_psnrs':test_psnr, 'pred_imgs':pred_img, 'num_params':num_param}
    info_dir = '../result/imgreg/recent'
    img_dir = '../result/imgreg//recent/imgs'
    if args.tune:
        info_filename = '%s%d_%s_%s.pkl'%(args.dataset, args.imgid, args.method, args.basis)
        img_filename = '%s%d_%s_%s.jpg'%(args.dataset, args.imgid, args.method, args.basis)
    else:
        info_filename = '%s%d_%s_%s_off.pkl'%(args.dataset, args.imgid, args.method, args.basis)
        img_filename = '%s%d_%s_%s_off.jpg'%(args.dataset, args.imgid, args.method, args.basis)

    path = os.path.join(info_dir, info_filename)
    with open(path, 'wb') as f:
        pickle.dump(save_info, f)
    path = os.path.join(img_dir, img_filename)
    save_image(pred_img.permute(0, 3, 1, 2), path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='method')

    parser.add_argument('--sfix', action='store_true')
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--dataset', default='natural')
    parser.add_argument('--imgid', type=int, default=0)

    parser_sim_single = subparsers.add_parser('sim-single')
    parser_sim_single.add_argument('--basis', default='fcn')

    parser_sim_two = subparsers.add_parser('sim-two')
    parser_sim_two.add_argument('--basis', default='fcn')

    args = parser.parse_args()

    main(args)
