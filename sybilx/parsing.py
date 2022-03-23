import argparse
import torch
import os
import pwd
from pytorch_lightning import Trainer
import itertools
from sybilx.utils.registry import md5
import json
import copy

EMPTY_NAME_ERR = 'Name of augmentation or one of its arguments cant be empty\n\
                  Use "name/arg1=value/arg2=value" format'
POSS_VAL_NOT_LIST = (
    "Flag {} has an invalid list of values: {}. Length of list must be >=1"
)


def parse_augmentations(augmentations):
    """
    Parse the list of augmentations, given by configuration, into a list of
    tuple of the augmentations name and a dictionary containing additional args.

    The augmentation is assumed to be of the form 'name/arg1=value/arg2=value'

    :raw_augmentations: list of strings [unparsed augmentations]
    :returns: list of parsed augmentations [list of (name,additional_args)]

    """
    raw_transformers = augmentations

    transformers = []
    for t in raw_transformers:
        arguments = t.split("/")
        name = arguments[0]
        if name == "":
            raise Exception(EMPTY_NAME_ERR)

        kwargs = {}
        if len(arguments) > 1:
            for a in arguments[1:]:
                splited = a.split("=")
                var = splited[0]
                val = splited[1] if len(splited) > 1 else None
                if var == "":
                    raise Exception(EMPTY_NAME_ERR)
                try:
                    kwargs[var] = float(val)
                except ValueError:
                    kwargs[var] = val

        transformers.append((name, kwargs))

    return transformers


def prepare_training_config_for_eval(train_config):
    """Convert training config to an eval config for testing.

    Parameters
    ----------
    train_config: dict
         config with the following structure:
              {
                   "train_config": ,   # path to train config
                   "log_dir": ,        # log directory used by dispatcher during training
                   "eval_args": {}     # test set-specific arguments beyond default
              }

    Returns
    -------
    experiments: list
    flags: list
    experiment_axies: list
    """

    train_args = json.load(open(train_config["train_config"], "r"))

    experiments, _, _ = parse_dispatcher_config(train_args)
    stem_names = [md5(e) for e in experiments]
    eval_args = copy.deepcopy(train_args)
    eval_args["grid_search_space"].update(train_config["eval_args"])

    # reset defaults
    eval_args["grid_search_space"]["train"] = [False]
    eval_args["grid_search_space"]["test"] = [True]
    eval_args["grid_search_space"]["from_checkpoint"] = [True]
    eval_args["grid_search_space"]["gpus"] = [1]
    eval_args["grid_search_space"]["comet_tags"][0] += " eval"
    eval_args["available_gpus"] = train_config["available_gpus"]
    eval_args["script"] = train_config["script"]

    experiments, flags, experiment_axies = parse_dispatcher_config(eval_args)

    for (idx, e), s in zip(enumerate(experiments), stem_names):
        experiments[idx] += " --snapshot {}".format(
            os.path.join(train_config["log_dir"], "{}.args".format(s))
        )

    return experiments, flags, experiment_axies


def parse_dispatcher_config(config):
    """
    Parses an experiment config, and creates jobs. For flags that are expected to be a single item,
    but the config contains a list, this will return one job for each item in the list.
    :config - experiment_config

    returns: jobs - a list of flag strings, each of which encapsulates one job.
         *Example: --train --cuda --dropout=0.1 ...
    returns: experiment_axies - axies that the grid search is searching over
    """

    grid_search_spaces = config["grid_search_space"]
    paired_search_spaces = config.get("paired_search_space", [])
    flags = []
    arguments = []
    experiment_axies = []

    # add anything outside search space as fixed
    fixed_args = ""
    for arg in config:
        if arg not in [
            "script",
            "grid_search_space",
            "paired_search_space",
            "available_gpus",
        ]:
            if type(config[arg]) is bool:
                if config[arg]:
                    fixed_args += "--{} ".format(str(arg))
                else:
                    continue
            else:
                fixed_args += "--{} {} ".format(arg, config[arg])

    # add paired combo of search space
    paired_args_list = [""]
    if len(paired_search_spaces) > 0:
        paired_args_list = []
        paired_keys = list(paired_search_spaces.keys())
        paired_vals = list(paired_search_spaces.values())
        flags.extend(paired_keys)
        for paired_combo in zip(*paired_vals):
            paired_args = ""
            for i, flg_value in enumerate(paired_combo):
                if type(flg_value) is bool:
                    if flg_value:
                        paired_args += "--{} ".format(str(paired_keys[i]))
                    else:
                        continue
                else:
                    paired_args += "--{} {} ".format(
                        str(paired_keys[i]), str(flg_value)
                    )
            paired_args_list.append(paired_args)

    # add every combo of search space
    product_flags = []
    for key, value in grid_search_spaces.items():
        flags.append(key)
        product_flags.append(key)
        arguments.append(value)
        if len(value) > 1:
            experiment_axies.append(key)

    experiments = []
    exps_combs = list(itertools.product(*arguments))

    for tpl in exps_combs:
        exp = ""
        for idx, flg in enumerate(product_flags):
            if type(tpl[idx]) is bool:
                if tpl[idx]:
                    exp += "--{} ".format(str(flg))
                else:
                    continue
            else:
                exp += "--{} {} ".format(str(flg), str(tpl[idx]))
        exp += fixed_args
        for paired_args in paired_args_list:
            experiments.append(exp + paired_args)

    return experiments, flags, experiment_axies


def parse_args(args_strings=None):
    parser = argparse.ArgumentParser(description="Sybil research repo.")
    # setup
    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="Whether or not to train model",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Whether or not to run model on test set",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        default=False,
        help="Whether or not to run model on dev set",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        default=False,
        help="Whether to run model for pure prediction where labels are not known",
    )
    parser.add_argument(
        "--fine_tune",
        action="store_true",
        default=False,
        help="Whether or not to fine_tune model",
    )
    parser.add_argument(
        "--num_epochs_fine_tune",
        type=int,
        default=1,
        help="Num epochs to finetune model",
    )

    # data
    parser.add_argument(
        "--dataset",
        default="nlst",
        help="Name of dataset from dataset factory to use [default: nlst]",
    )
    parser.add_argument(
        "--class_bal", action="store_true", default=False, help="class balance"
    )
    parser.add_argument(
        "--class_bal_key",
        type=str,
        default="y",
        help="dataset key to use for class balancing",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Width and height of image in pixels. [default: [256,256]]",
    )
    parser.add_argument(
        "--num_chan", type=int, default=3, help="Number of channels for input image"
    )
    parser.add_argument(
        "--img_mean",
        type=float,
        nargs="+",
        default=[128.1722],
        help="Mean of image per channel",
    )
    parser.add_argument(
        "--img_std",
        type=float,
        nargs="+",
        default=[87.1849],
        help="Standard deviation  of image per channel",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="/data/rsg/mammogram/NLST/nlst-ct-png",
        help="Dir of images. Note, image path in dataset jsons should stem from here",
    )
    parser.add_argument(
        "--img_file_type",
        type=str,
        default="png",
        choices=["png", "dicom"],
        help="Type of image. one of [png, dicom]",
    )
    parser.add_argument(
        "--fix_seed_for_multi_image_augmentations",
        action="store_true",
        default=False,
        help="Use same seed for each slice of volume augmentations",
    )
    parser.add_argument(
        "--dataset_file_path",
        type=str,
        default="/Mounts/rbg-storage1/datasets/NLST/full_nlst_google.json",
        help="Path to dataset file either as json or csv",
    )
    parser.add_argument('--data_dir', type = str, default = 'Data/', help="directory for alternate data, used in MNIST toy example")
    parser.add_argument(
        "--num_classes", type=int, default=6, help="Number of classes to predict"
    )

    parser.add_argument('--plco_train_study_yrs', type = int, nargs='+', required=False, default = None, help="filters to only include specific study_yrs when training for PLCO data.")
    parser.add_argument('--plco_test_study_yrs', type = int, nargs='+', required=False, default = None, help="filters to only include specific study_yrs for PLCO data when testing/validating.")
    parser.add_argument('--plco_use_only_visitnum', type = int, nargs='+', required=False, default = None, help="filters to only include specific visitnums for PLCO data.")
    parser.add_argument('--plco_use_only_one_image',
        action="store_true",
        default=False,
        help="filters to only include one image per exam/visitnum for PLCO data."
    )

    # Alternative training/testing schemes
    parser.add_argument(
        "--cross_val_seed",
        type=int,
        default=0,
        help="Seed used to generate the partition.",
    )
    parser.add_argument(
        "--assign_splits",
        action="store_true",
        default=False,
        help="Whether to assign different splits than those predetermined in dataset",
    )
    parser.add_argument(
        "--split_type",
        type=str,
        default="random",
        choices=["random", "institution_split", "cxr_lc"],
        help="How to split dataset if assign_split = True. Usage: ['random', 'institution_split'].",
    )
    parser.add_argument(
        "--split_probs",
        type=float,
        nargs="+",
        default=[0.6, 0.2, 0.2],
        help="Split probs for datasets without fixed train dev test. ",
    )

    # survival analysis setup
    parser.add_argument(
        "--max_followup", type=int, default=6, help="Max followup to predict over"
    )

    # augmentations
    parser.add_argument(
        "--fix_multi_image_augmentation_seed",
        action="store_true",
        default=False,
        help="whether to use same seed for augmentations in the multi-input setting",
    )
    parser.add_argument(
        "--train_rawinput_augmentations",
        nargs="*",
        default=[],
        help='List of image-transformations to use. Usage: "--train_rawinput_augmentations trans1/arg1=5/arg2=2 trans2 trans3/arg4=val"',
    )
    parser.add_argument(
        "--train_tnsr_augmentations",
        nargs="*",
        default=[],
        help='List of image-transformations to use. Usage: "--train_tnsr_augmentations trans1/arg1=5/arg2=2 trans2 trans3/arg4=val"',
    )
    parser.add_argument(
        "--test_rawinput_augmentations",
        nargs="*",
        default=[],
        help="List of image-transformations to use for the dev and test dataset",
    )
    parser.add_argument(
        "--test_tnsr_augmentations",
        nargs="*",
        default=[],
        help="List of image-transformations to use for the dev and test dataset",
    )

    # risk factors
    parser.add_argument(
        "--use_risk_factors",
        action="store_true",
        default=False,
        help="Whether to feed risk factors into last FC of model.",
    )  #
    parser.add_argument(
        "--risk_factor_keys",
        nargs="*",
        default=[],
        help="List of risk factors to include in risk factor vector.",
    )

    # handling CT slices
    parser.add_argument(
        "--fit_to_length",
        action="store_true",
        default=False,
        help="Whether to fit num slices using padding and slice sampling",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=200,
        help="In multi image setting, the number of images per single sample.",
    )
    parser.add_argument(
        "--min_num_images",
        type=int,
        default=0,
        help="In multi image setting, the min number of images per single sample.",
    )
    parser.add_argument(
        "--slice_thickness_filter",
        type=float,
        help="Slice thickness using, if restricting to specific thickness value.",
    )
    parser.add_argument(
        "--cross_section_filter",
        type=str,
        nargs="*",
        help="Restrict to using specific cross sections [transverse, coronal, sagittal, oblique].",
    )
    parser.add_argument(
        "--use_only_thin_cuts_for_ct",
        action="store_true",
        default=False,
        help="Wether to use image series with thinnest cuts only.",
    )

    # region annotations
    parser.add_argument(
        "--use_annotations",
        action="store_true",
        default=False,
        help="whether to use image annotations (pixel labels) in modeling",
    )

    parser.add_argument(
        "--region_annotations_filepath", type=str, help="Path to annotations file"
    )
    parser.add_argument(
        "--annotation_loss_lambda",
        type=float,
        default=1,
        help="Weight of annotation losses",
    )
    parser.add_argument(
        "--image_attention_loss_lambda",
        type=float,
        default=1,
        help="Weight of loss for predicting image attention scores",
    )
    parser.add_argument(
        "--volume_attention_loss_lambda",
        type=float,
        default=1,
        help="Weight of loss for predicting volume attention scores",
    )

    # regularization
    parser.add_argument(
        "--primary_loss_lambda",
        type=float,
        default=1.0,
        help="Lambda to weigh the primary loss.",
    )
    parser.add_argument(
         "--adv_loss_lambda",
         type=float,
         default=1.0,
         help="Lambda to weigh the adversary loss.",
     )

    # Weighted Focal Loss
    parser.add_argument(
        "--wfl_alpha",
        type=float,
        default=0.9,
        help="Alpha to weigh the WFL loss.",
    )
    parser.add_argument(
        "--wfl_gamma",
        type=float,
        default=2,
        help="Lambda to weigh the adversary loss.",
    )
    
    # loader
    parser.add_argument(
        "--input_loader_name",
        type=str,
        default="default_image_loader",
        help="input loader",
    )
    parser.add_argument("--lightning_name", type=str, default="vgg", help="Name of DNN")
    parser.add_argument(
        "--base_model", type=str, default="vgg", help="Name of parent model"
    )

    # losses and metrics
    parser.add_argument(
        "--loss_fns", type=str, nargs="*", default=[], help="Name of loss"
    )
    parser.add_argument(
        "--eval_loss_fns", type=str, nargs="*", default=None, help="Name of loss"
    )
    parser.add_argument(
        "--metrics", type=str, nargs="*", default=[], help="Name of performance metric"
    )
    parser.add_argument(
        "--store_classwise_metrics",
        action="store_true",
        default=False,
        help="Whether to log metrics per class or just log average across classes",
    )

    # learning
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training [default: 128]",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate [default: 0.001]",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.25,
        help="Amount of dropout to apply on last hidden layer [default: 0.25]",
    )
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="Optimizer to use [default: adam]"
    )
    parser.add_argument(
        "--momentum", type=float, default=0, help="Momentum to use with SGD"
    )
    parser.add_argument(
        "--lr_decay",
        type=float,
        default=0.1,
        help="Initial learning rate [default: 0.5]",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="L2 Regularization penaty [default: 0]",
    )
    parser.add_argument(
        "--adv_lr",
        type=float,
        default=0.001,
        help="Initial learning rate for adversary model [default: 0.001]",
    )

    # schedule
    parser.add_argument(
        "--scheduler", type=str, default="reduce_on_plateau", help="Name of scheduler"
    )
    parser.add_argument(
        "--cosine_annealing_period",
        type=int,
        default=10,
        help="length of period of lr cosine anneal",
    )
    parser.add_argument(
        "--cosine_annealing_period_scaling",
        type=int,
        default=2,
        help="how much to multiply each period in successive annealing",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Number of epochs without improvement on dev before halving learning rate and reloading best model [default: 5]",
    )
    parser.add_argument(
        "--num_adv_steps",
        type=int,
        default=1,
        help="Number of steps for domain adaptation discriminator per one step of encoding model [default: 5]",
    )

    # callbacks
    parser.add_argument(
        "--callback_names",
        type=str,
        nargs="*",
        default=["checkpointer", "lr_monitor"],
        help="Lightning callbacks",
    )

    parser.add_argument(
        "--monitor",
        type=str,
        default=None,
        help="Name of metric to use to decide when to save model",
    )

    parser.add_argument(
        "--checkpoint_save_top_k",
        type=int,
        default=1,
        help="the best k models according to the quantity monitored will be saved",
    )
    parser.add_argument(
        "--checkpoint_save_last",
        action="store_true",
        default=False,
        help="save the last model to last.ckpt",
    )

    # stochastic weight averaging
    parser.add_argument(
        "--swa_epoch",
        type=str,
        default="0.8",
        help="when to start swa",
    )

    parser.add_argument(
        "--swa_lr",
        type=float,
        default=None,
        help="lr for swa. None will use existing lr",
    )
    parser.add_argument(
        "--swa_annealing_epochs",
        type=int,
        default=10,
        help="number of epochs in the annealing phase",
    )
    parser.add_argument(
        "--swa_annealing_strategy",
        type=str,
        choices=["cos", "linear"],
        default="cos",
        help="lr annealing strategy",
    )

    # model checkpointing
    parser.add_argument(
        "--turn_off_checkpointing",
        action="store_true",
        default=False,
        help="Do not save best model",
    )

    parser.add_argument(
        "--save_dir", type=str, default="snapshot", help="Where to dump the model"
    )

    parser.add_argument(
        "--from_checkpoint",
        action="store_true",
        default=False,
        help="Whether loading a model from a saved checkpoint",
    )

    parser.add_argument(
        "--relax_checkpoint_matching",
        action="store_true",
        default=False,
        help="Do not enforce that the keys in checkpoint_path match the keys returned by this module’s state dict",
    )

    parser.add_argument(
        "--snapshot",
        type=str,
        default=None,
        help="Filename of model snapshot to load[default: None]",
    )

    # system
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Num workers for each data loader [default: 4]",
    )

    # storing results
    parser.add_argument(
        "--save_hiddens",
        action="store_true",
        default=False,
        help="Save hidden repr from each image to an npz based off results path, git hash and exam name",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        default=False,
        help="Save hidden repr from each image to an npz based off results path, git hash and exam name",
    )
    parser.add_argument(
        "--hiddens_dir",
        type=str,
        default="hiddens/test_run",
        help='Dir to store hiddens npy"s when store_hiddens is true',
    )
    parser.add_argument(
        "--save_attention_scores",
        action="store_true",
        default=False,
        help="Whether to save attention scores when using attention mechanism",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="logs/test.args",
        help="Where to save the result logs",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="defined either automatically by dispatcher.py or time in main.py. Keep without default",
    )

    # cache
    parser.add_argument(
        "--cache_path", type=str, default=None, help="Dir to cache images."
    )
    parser.add_argument(
        "--cache_full_img",
        action="store_true",
        default=False,
        help="Cache full image locally as well as cachable transforms",
    )

    # logger
    parser.add_argument(
        "--logger_name", type=str, default="comet", help="List of tags for comet logger"
    )

    # comet
    parser.add_argument(
        "--comet_tags", nargs="*", default=[], help="List of tags for comet logger"
    )
    parser.add_argument("--project_name", default="CancerCures", help="Comet project")
    parser.add_argument("--workspace", default="pgmikhael", help="Comet workspace")
    parser.add_argument(
        "--log_gen_image",
        action="store_true",
        default=False,
        help="Whether to log sample generated image to comet",
    )

    # run
    parser = Trainer.add_argparse_args(parser)
    if args_strings is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_strings)

    if (isinstance(args.gpus, str) and len(args.gpus.split(",")) > 1) or (
        isinstance(args.gpus, int) and args.gpus > 1
    ):
        args.accelerator = "ddp"
        args.replace_sampler_ddp = False
    else:
        args.accelerator = None
        args.replace_sampler_ddp = False

    args.unix_username = pwd.getpwuid(os.getuid())[0]

    # learning initial state
    args.step_indx = 1

    args.train_rawinput_augmentations = parse_augmentations(
        args.train_rawinput_augmentations
    )
    args.train_tnsr_augmentations = parse_augmentations(args.train_tnsr_augmentations)
    args.test_rawinput_augmentations = parse_augmentations(
        args.test_rawinput_augmentations
    )
    args.test_tnsr_augmentations = parse_augmentations(args.test_tnsr_augmentations)

    return args
