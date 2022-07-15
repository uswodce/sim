import imageio
import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import random_split

class ApplyTransform(Dataset):
    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        sample, target = self.dataset[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.dataset)

def load_data_imgcl(dataset, trainval_ratio):
    if dataset == 'mnist':
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        trans_train = transforms.Compose([transforms.ToTensor(), normalize])
        trans_test = transforms.Compose([transforms.ToTensor(), normalize])
        trainvalset = dset.MNIST('../data', train=True, download=True, transform=None)
        testset = dset.MNIST('../data', train=False, transform=trans_test)

    elif dataset == 'cifar':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                            std=[0.2470, 0.2435, 0.2616])
        trans_train = transforms.Compose([transforms.ToTensor(), normalize])
        trans_test = transforms.Compose([transforms.ToTensor(), normalize])
        trainvalset = dset.CIFAR10('../data', train=True, download=True, transform=None)
        testset = dset.CIFAR10('../data', train=False, transform=trans_test)

    elif dataset == 'svhn':
        normalize = transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],
                                        std=[0.1980, 0.2010, 0.1970])
        trans_train = transforms.Compose([transforms.ToTensor(), normalize])
        trans_test = transforms.Compose([transforms.ToTensor(), normalize])
        trainvalset = dset.SVHN('../data', split='train', download=True, transform=None)
        testset = dset.SVHN('../data', split='test', download=True, transform=trans_test)

    else:
        raise ValueError('The dataset is not supported')

    num_train = int(len(trainvalset) * trainval_ratio)
    num_val = int(len(trainvalset)) - num_train
    trainset, valset = random_split(trainvalset, [num_train, num_val])

    trainvalset = ApplyTransform(trainvalset, transform=trans_train)
    trainset    = ApplyTransform(trainset, transform=trans_train)
    valset      = ApplyTransform(valset, transform=trans_test)

    return trainvalset, trainset, valset, testset

def load_data_imgreg(dataset, imgid):
    if dataset == 'natural':
        filename = '../data/data_div2k.npz'
        data = np.load(filename)
        img = data['test_data'][imgid] / 255.

    if dataset == 'text':
        filename = '../data/data_2d_text.npz'
        data = np.load(filename)
        img = data['test_data'][imgid] / 255.

    # Create input pixel coordinates in the unit square
    coords = np.linspace(0, 1, img.shape[0], endpoint=False)
    grid = np.stack(np.meshgrid(coords, coords), -1)

    grid = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    train_data = [grid[:,::2,::2], img[:,::2,::2]]
    val_data   = [grid[:,::2,1::2], img[:,::2,1::2]]
    test_data  = [grid[:,1::2,1::2], img[:,1::2,1::2]]

    return train_data, val_data, test_data

def gen_cpmem_data(T, mem_length, b_size, seed):
    rng = np.random.RandomState(seed=seed)
    seq = torch.from_numpy(rng.randint(1, 9, size=(b_size, mem_length))).float()
    zeros = torch.zeros((b_size, T))
    marker = 9 * torch.ones((b_size, mem_length + 1))
    placeholders = torch.zeros((b_size, mem_length))

    x = torch.cat((seq, zeros[:, :-1], marker), 1)
    y = torch.cat((placeholders, zeros, seq), 1).long()

    return x, y

def load_data_cpmem(T, seq_len, n_trainval, n_test, trainval_ratio):
    seed = 1111 + T
    n_train = int(n_trainval * trainval_ratio)

    trainval_x, trainval_y = gen_cpmem_data(T, seq_len, n_trainval, seed + 100)
    test_x, test_y = gen_cpmem_data(T, seq_len, n_test, seed + 200)

    trainval_data = [trainval_x, trainval_y]
    train_data = [trainval_x[:n_train], trainval_y[:n_train]]
    val_data = [trainval_x[n_train:], trainval_y[n_train:]]
    test_data = [test_x, test_y]

    return trainval_data, train_data, val_data, test_data
