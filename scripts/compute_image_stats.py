from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(realpath(__file__))))
from sybilx.parsing import parse_args
import sybilx.utils.loading as loaders
from sybilx.utils.registry import get_object
from sybilx.utils.loading import get_sample_loader
import torch
from tqdm import tqdm

def get_dataset_stats(args):
    
    train_dataset = get_object(args.dataset, "dataset")(args, "train")
    train_dataset.input_loader = get_sample_loader('test', args)
    train_loader = loaders.get_train_dataset_loader(args, train_dataset)
    
    means, stds = {i:[] for i in range(args.num_chan)}, {i:[] for i in range(args.num_chan)}

    indx = 1
    for batch in tqdm(train_loader):
        tensor = batch['x']
        for channel in range(args.num_chan):
            tensor_chan = tensor[:, channel]
            means[channel].append(torch.mean(tensor_chan))
            stds[channel].append(torch.std(tensor_chan))

        if indx % (len(train_loader)//20) == 0:
            _means = [torch.mean(torch.Tensor(means[channel])).item() for channel in range(args.num_chan)]
            _stds = [torch.mean(torch.Tensor(stds[channel])).item() for channel in range(args.num_chan)]
            print('for indx={}\t mean={}\t std={}\t'.format(indx, _means, _stds))
        indx += 1
    means = [torch.mean(torch.Tensor(means[channel])) for channel in range(args.num_chan)]
    stds = [torch.mean(torch.Tensor(stds[channel])) for channel in range(args.num_chan)]

    return means, stds

if __name__ == "__main__":

    args = parse_args()

    args.rank = 0
    args.global_rank = 0
    args.world_size = 1
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    means, stds = get_dataset_stats(args)
    for channel, (mean, std) in enumerate(zip(means, stds)):
        print("Channel[{}] Img mean:{}, Img std:{}".format(channel, means, std))
