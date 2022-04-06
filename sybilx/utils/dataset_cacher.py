from sybilx.utils.registry import get_object
import sybilx.utils.loading as loaders
import os
import torch
from tqdm import tqdm


def cache_dataset(args):
    """Loads dataset samples and saves them locally

    Args:
        args (Namespace):
    """
    for split in ["train", "dev", "test"]:
        dataset = loaders.get_eval_dataset_loader(
            args, get_object(args.dataset, "dataset")(args, split)
        )

        for batch in tqdm(dataset):
            for idx, eid in zip(enumerate(batch["exam"])):
                d = (
                    {"x": batch[idx]["x"], "mask": batch[idx]["mask"]}
                    if args.use_annotations
                    else {"x": batch[idx]["x"]}
                )
                filename = get_tensor_path(eid, args.cache_path)
                torch.save(d, filename)


def get_tensor_path(sample_id, cache_path):
    return os.path.join(cache_path, "sample_{}.pt".format(sample_id))
