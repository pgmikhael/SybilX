from ast import arg
from collections import OrderedDict
from argparse import FileType, Namespace
import pickle
import os
import sys
import time
import git
import comet_ml
import pytorch_lightning as pl
from pytorch_lightning import _logger as log

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from sybilx.parsing import parse_args
from sybilx.utils.registry import get_object
from sybilx.datasets.utils import get_censoring_dist
import sybilx.utils.loading as loaders
from sybilx.utils.callbacks import set_callbacks
from utils import get_dataset_stats


def cli_main(args):

    args.checkpoint_callback = False

    trainer = pl.Trainer.from_argparse_args(args)
    # Remove callbacks from args for safe pickling later
    args.callbacks = None
    args.num_nodes = trainer.num_nodes
    args.num_processes = trainer.num_processes
    args.world_size = args.num_nodes * args.num_processes
    args.global_rank = trainer.global_rank
    args.local_rank = trainer.local_rank

    if args.logger_name == "comet":
        tb_logger = pl.loggers.CometLogger(
            api_key=os.environ.get("COMET_API_KEY"),
            project_name=args.project_name,
            experiment_name=args.experiment_name,
            workspace=args.workspace,
            log_env_details=True,
            log_env_cpu=True,
        )
        trainer.logger = tb_logger

    repo = git.Repo(search_parent_directories=True)
    commit = repo.head.object
    log.info(
        "\nProject main running by author: {} \ndate:{}, \nfrom commit: {} -- {}".format(
            commit.author,
            time.strftime("%m-%d-%Y %H:%M:%S", time.localtime(commit.committed_date)),
            commit.hexsha,
            commit.message,
        )
    )

    # print args
    for key, value in sorted(vars(args).items()):
        print("{} -- {}".format(key.upper(), value))

    if args.get_dataset_stats:
        log.info("\nComputing image mean and std...")
        args.img_mean, args.img_std = get_dataset_stats(args)
        log.info('Mean: {}'.format(args.img_mean))
        log.info('Std: {}'.format(args.img_std))

    train_dataset = loaders.get_train_dataset_loader(
        args, get_object(args.dataset, "dataset")(args, "train")
    )
    dev_dataset = loaders.get_eval_dataset_loader(
        args, get_object(args.dataset, "dataset")(args, "dev"), False
    )

    if "survival" in args.metrics:
        # compute censoring distribution
        args.censoring_distribution = get_censoring_dist(train_dataset.dataset)
    

    if args.from_checkpoint:
        if args.snapshot.endswith(".args"):
            snargs = Namespace(**pickle.load(open(args.snapshot, "rb")))
            model = get_object(snargs.lightning_name, "lightning")(snargs)
            modelpath = snargs.model_path
        elif args.snapshot.endswith(".ckpt"):
            model = get_object(args.lightning_name, "lightning")(args)
            modelpath = args.snapshot
        else:
            raise FileType("Snapshot should be an args or ckpt file.")
        model = model.load_from_checkpoint(
            checkpoint_path=modelpath, strict=not args.relax_checkpoint_matching
        )
        model.args = args
    else:
        model = get_object(args.lightning_name, "lightning")(args)

    if args.logger_name == "comet":
        # log to comet
        trainer.logger.experiment.set_model_graph(model)
        trainer.logger.experiment.add_tags(args.comet_tags)
        trainer.logger.experiment.log_parameters(args)

    # add callbacks
    trainer.callbacks = set_callbacks(trainer, args)

    if args.train:
        log.info("\nTraining Phase...")
        trainer.fit(model, train_dataset, dev_dataset)
        args.model_path = trainer.checkpoint_callback.best_model_path

    if args.dev:
        log.info("\Validation Phase...")
        trainer.validate(
            model, dev_dataset, ckpt_path=args.model_path
        ) if args.train else trainer.validate(model, dev_dataset)

    if args.test:
        log.info("\nInference Phase on test set...")
        test_dataset = loaders.get_eval_dataset_loader(
            args, get_object(args.dataset, "dataset")(args, "test"), False
        )
        trainer.test(model, test_dataset)
        trainer.test(
            model, test_dataset, ckpt_path=args.model_path
        ) if args.train else trainer.test(model, test_dataset)

    print("Saving args to {}.args".format(args.results_path))
    pickle.dump(vars(args), open("{}.args".format(args.results_path), "wb"))


if __name__ == "__main__":
    args = parse_args()
    cli_main(args)
