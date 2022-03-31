import torch
import copy
from sybilx.utils.registry import get_object
from sybilx.utils.loading import get_train_dataset_loader
from sybilx.utils.augmentations import get_augmentations
from tqdm import tqdm
from datetime import datetime as dt

def get_dataset_stats(args):
    args = copy.deepcopy(args)
    original_augmentations = get_augmentations( args.train_rawinput_augmentations, args.train_tnsr_augmentations, args )
    new_augmentations = [a for a in original_augmentations if (a._is_cachable or a.name == 'tensorizer')]

    train_data = get_object(args.dataset, 'dataset')(args, 'train')
    data_loader = get_train_dataset_loader(args, train_data,  args.batch_size)

    means, stds = {i:[] for i in range(args.num_chan)}, {i:[] for i in range(args.num_chan)}

    indx = 1
    for batch in tqdm(data_loader):
        tensor = batch['x'].float()
        for channel in range(args.num_chan):
            tensor_chan = tensor[:, channel]
            means[channel].append(torch.mean(tensor_chan))
            stds[channel].append(torch.std(tensor_chan))

        if indx % 0 == 0 :#(len(data_loader)//20) == 0:
            _means = [torch.mean(torch.Tensor(means[channel])).item() for channel in range(args.num_chan)]
            _stds = [torch.mean(torch.Tensor(stds[channel])).item() for channel in range(args.num_chan)]
            print('for indx={}\n mean={}\n std={}\n'.format(indx, _means, _stds))
        indx += 1
        if indx == 5000:
            break

    means = [torch.mean(torch.Tensor(means[channel])) for channel in range(args.num_chan)]
    stds = [torch.mean(torch.Tensor(stds[channel])) for channel in range(args.num_chan)]

    return means, stds
