# dataset utils
from tqdm import tqdm
import numpy as np
import torch
import os
from os.path import dirname, realpath
import json
from collections import defaultdict
import time
from torch.utils import data
# project utils
from modules.utils.shared import register_object
from modules.augmentations.basic import ComposeAug
# torch
import torch
import torchvision
import warnings

@register_object("mnist", 'dataset')
class MNIST_Dataset(data.Dataset):
    """A pytorch Dataset for the ImageNet data."""

    def __init__(self, args, augmentations, split_group):
        """Initializes the dataset.

        Constructs a standard pytorch Dataset object which
        can be fed into a DataLoader for batching.

        Arguments:
            args(object): Config.
            augmentations(list): A list of augmentation objects.
            split_group(str): The split group ['train'|'dev'|'test'].
        
        Automatic downloading throws an error, install into args.data_dir manually with
            
            wget www.di.ens.fr/~lelarge/MNIST.tar.gz
            tar -zxvf MNIST.tar.gz
        """

        super(MNIST_Dataset, self).__init__()

        self.args = args
        self.split_group = split_group
        self.composed_all_augmentations = ComposeAug(augmentations)
        
        if self.split_group == 'train':
            self.dataset = torchvision.datasets.MNIST(args.data_dir,
                                          train = True,
                                          download = True)
        else:
            mnist_test = torchvision.datasets.MNIST(args.data_dir, 
                                          train = False,
                                          download = True)
            
            if self.split_group == 'dev':
                self.dataset = [mnist_test[i] for i in range(len(mnist_test) // 2)]
            elif self.split_group == 'test':
                self.dataset = [mnist_test[i] for i in range(len(mnist_test) // 2, len(mnist_test))]
            else:
                raise Exception('Split group must be in ["train"|"dev"|"test"].')


    @staticmethod
    def set_args(args):
        args.num_classes = 10
        args.num_chan = 3
        args.img_size = (32,32)
        args.img_mean = [0.0]
        args.img_std = [1.0]
        args.train_rawinput_augmentations = ["rand_hor_flip", "scale_2d"]
        args.test_rawinput_augmentations = ["scale_2d"]
        args.train_tnsr_augmentations = ["normalize_2d"]
        args.test_tnsr_augmentations = ["normalize_2d"]


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x = np.array(x)
        sample = {'sample_id': '{}_{}_{}'.format(self.split_group, index, y)}
        try:
            sample['x'] = self.composed_all_augmentations({'input': x}, sample)['input']
            sample['x'] = sample['x'].repeat(3,1,1).float()
            sample['y'] = y
            return sample

        except Exception:
            warnings.warn('Could not load sample')

@register_object("mnist_adversarial", 'dataset')
class MNISTAdversarial(MNIST_Dataset):
    """A pytorch Dataset for the ImageNet data."""

    def __init__(self, args, augmentations, split_group):
        """Initializes the dataset.

        Constructs a standard pytorch Dataset object which
        can be fed into a DataLoader for batching.

        Arguments:
            args(object): Config.
            augmentations(list): A list of augmentation objects.
            split_group(str): The split group ['train'|'dev'|'test'].
        
        Automatic downloading throws an error, install into args.data_dir manually with
            
            wget www.di.ens.fr/~lelarge/MNIST.tar.gz
            tar -zxvf MNIST.tar.gz
        """

        super(MNISTAdversarial, self).__init__(args, augmentations, split_group)

        self.args = args
        self.split_group = split_group
        self.composed_all_augmentations = ComposeAug(augmentations)
        self.dataset = []
        if self.split_group == 'train':
            dataset = torchvision.datasets.MNIST(args.data_dir, train = True, download = True)
        else:
            mnist_test = torchvision.datasets.MNIST(args.data_dir, train = False, download = True)
            if self.split_group == 'dev':
                dataset = [mnist_test[i] for i in range(len(mnist_test) // 2)]
            elif self.split_group == 'test':
                dataset = [mnist_test[i] for i in range(len(mnist_test) // 2, len(mnist_test))]
            else:
                raise Exception('Split group must be in ["train"|"dev"|"test"].')
        
        for i in tqdm( range(len(dataset)), total = len(dataset) ):
            x, y = dataset[i]
            if torch.randn(1) > 0.5:
                self.dataset.append( (x, y, 1) )
            else:
                self.dataset.append( (x, y, 0) )


    def __getitem__(self, index):
        x, y, mu = self.dataset[index]
        x = np.array(x)
 
        sample = {'sample_id': '{}_{}_{}'.format(self.split_group, index, y)}
        try:
            x = self.composed_all_augmentations({'input': x}, sample)['input']
            sample['x'] = x + np.random.normal(mu, 1, size = (28,28) )
            sample['x'] = sample['x'].repeat(3,1,1).float()
            sample['y'] = y
            sample['adversary_golds'] = mu
            return sample

        except Exception:
            warnings.warn('Could not load sample')

@register_object("mnist_adversarial_baseline", 'dataset')
class MNISTAdversarialBaseline(MNISTAdversarial):
    """A pytorch Dataset for the ImageNet data."""

    def __init__(self, args, augmentations, split_group):
        """Initializes the dataset.

        Constructs a standard pytorch Dataset object which
        can be fed into a DataLoader for batching.

        Arguments:
            args(object): Config.
            augmentations(list): A list of augmentation objects.
            split_group(str): The split group ['train'|'dev'|'test'].
        
        Automatic downloading throws an error, install into args.data_dir manually with
            
            wget www.di.ens.fr/~lelarge/MNIST.tar.gz
            tar -zxvf MNIST.tar.gz
        """

        super(MNISTAdversarialBaseline, self).__init__(args, augmentations, split_group)
    
    def __getitem__(self, index):
        x, y, mu = self.dataset[index]
        x = np.array(x)
 
        sample = {'sample_id': '{}_{}_{}'.format(self.split_group, index, y)}
        try:
            x = self.composed_all_augmentations({'input': x}, sample)['input']
            sample['x'] = x + np.random.normal(mu, 1, size = (28,28) )
            sample['x'] = sample['x'].repeat(3,1,1).float()
            sample['y'] = mu
            return sample

        except Exception:
            warnings.warn('Could not load sample')
    @staticmethod
    def set_args(args):
        args.num_classes = 2
        args.num_chan = 3
        args.img_size = (32,32)
        args.img_mean = [0.0]
        args.img_std = [1.0]
        args.train_rawinput_augmentations = ["rand_hor_flip", "scale_2d"]
        args.test_rawinput_augmentations = ["scale_2d"]
        args.train_tnsr_augmentations = ["normalize_2d"]
        args.test_tnsr_augmentations = ["normalize_2d"]
