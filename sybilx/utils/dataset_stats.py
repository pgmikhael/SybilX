import copy
import torch
from tqdm import tqdm
from sybilx.utils.registry import get_object
import sybilx.utils.loading as loaders

def get_dataset_stats(args):
    args = copy.deepcopy(args)

    train_dataset = loaders.get_train_dataset_loader(
        args, get_object(args.dataset, "dataset")(args, "train")
    )

    means, stds = {i:[] for i in range(args.num_chan)}, {i:[] for i in range(args.num_chan)}

    indx = 1
    for batch in tqdm(train_dataset):
        tensor = batch['x']
        for channel in range(args.num_chan):
            tensor_chan = tensor[:, channel]
            means[channel].append(torch.mean(tensor_chan))
            stds[channel].append(torch.std(tensor_chan))

        if indx % (len(train_dataset)//20) == 0:
            _means = [torch.mean(torch.Tensor(means[channel])).item() for channel in range(args.num_chan)]
            _stds = [torch.mean(torch.Tensor(stds[channel])).item() for channel in range(args.num_chan)]
            print('for indx={}\t mean={}\t std={}\t'.format(indx, _means, _stds))
        indx += 1
    means = [torch.mean(torch.Tensor(means[channel])) for channel in range(args.num_chan)]
    stds = [torch.mean(torch.Tensor(stds[channel])) for channel in range(args.num_chan)]

    return means, stds