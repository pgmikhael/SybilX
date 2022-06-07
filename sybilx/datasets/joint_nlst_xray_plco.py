import os
import torch
import numpy as np
from copy import copy, deepcopy
from sybilx.parsing import parse_augmentations
from sybilx.utils.registry import register_object
from sybilx.utils.loading import get_sample_loader
from sybilx.datasets.nlst_xray import NLST_XRay_Dataset
from sybilx.datasets.plco_xray import PLCO_XRay_Dataset
from collections import Counter

CT_ITEM_KEYS = [
    "pid",
    "exam",
    "series",
    "y_seq",
    "y_mask",
    "time_at_event",
    "cancer_laterality",
    "has_annotation",
    "origin_dataset",
]

@register_object("nlst_xray+plco", "dataset")
class NLST_Xray_PLCO_Combined_Dataset(data.Dataset):
    """
    NLST and PLCO X-rays Combined.

    Notes:
        - augmentations (and input_loader) is hardcoded between the two datasets

    Options:
        - args.balance_by_dataset can be set in order to sample an equal amount of samples from each dataset during training.
        - args.discard_from_combined_dataset can be set to either 'mgh' or 'nlst', in order to produce a dataset containing
            only samples with the other. This allows you to evaluate on one dataset while using the alignment_ct lightning module.
    """

    def create_dataset(self, split_group):
        """
        Gets the dataset from the paths and labels in the json.
        Arguments:
            split_group(str): One of ['train'|'dev'|'test'].
        Returns:
            The dataset as a dictionary with img paths, label,
            and additional information regarding exam or participant
        """
        dataset = []

        nlst_args = copy(self.args)
        nlst_args.dataset_file_path = (
            "/Mounts/rbg-storage1/datasets/NLST/full_nlst_google.json"
        )
        nlst_args.input_loader_name = "dicom_xray_loader"
        nlst_args.img_file_type = "dicom"
        nlst_args.train_rawinput_augmentations = parse_augmentations("invert_pixels_relative/all_images=0 scale_2d min_max_8bit_scaler histogram_equalize rotate_range/deg=20 random_brightness_contrast/brightness=0.2".split(" "))
        nlst_args.test_rawinput_augmentations = parse_augmentations("invert_pixels_relative/all_images=0 scale_2d min_max_8bit_scaler histogram_equalize".split(" "))
        nlst_dataset = NLST_XRay_Dataset(nlst_args, split_group)
        self.dicom_loader = get_sample_loader(split_group, nlst_args)

        for exam_dict in nlst_dataset.dataset:
            exam_dict["origin_dataset"] = 0
            dataset.append(exam_dict)

        plco_args = copy(self.args)
        plco_args.dataset_file_path = (
            "/storage/ludvig/PLCO_XRAY/metadata_2022_04_05_rotated.json"
        )
        plco_args.input_loader_name = "tif_loader"
        plco_args.img_file_type = "tif"
        plco_args.train_rawinput_augmentations = parse_augmentations("scale_2d min_max_8bit_scaler histogram_equalize rotate_range/deg=20 random_brightness_contrast/brightness=0.2".split(" "))
        plco_args.test_rawinput_augmentations = parse_augmentations("invert_pixels_relative/all_images=0 scale_2d min_max_8bit_scaler histogram_equalize rotate_range/deg=20 random_brightness_contrast/brightness=0.2".split(" "))

        plco_dataset = PLCO_XRay_Dataset(plco_args, split_group)
        self.tif_loader = get_sample_loader(split_group, plco_args)

        for exam_dict in plco_dataset.dataset:
            exam_dict["origin_dataset"] = 1
            exam_dict["cancer_laterality"] = np.array([0,0,0])
            if self.args.use_annotations:
                exam_dict["annotations"] = {"image_annotations": None}
            dataset.append(exam_dict)

        return dataset

    def __getitem__(self, index):
        sample = deepcopy(self.dataset[index])
        item = {}
        if sample["origin_dataset"] == 0: # NLST
            # load and project to 2d
            input_dict = self.dicom_loader.get_image(sample["paths"], sample)
        else:
            assert sample["origin_dataset"] == 1 # PLCO
            # load using tif loader
            input_dict = self.tif_loader.get_image(sample["path"], sample)

        x = input_dict["input"]

        if self.args.use_risk_factors:
            item["risk_factors"] = sample["risk_factors"]

        item["x"] = x
        item["y"] = sample["y"]
        for key in CT_ITEM_KEYS:
            if key in sample:
                item[key] = sample[key]
        return item

    def get_summary_statement(self, dataset, split_group):
        summary = "Constructed Combined NLST + PLCO X-ray Cancer Survival {} dataset with {} records, {} exams, {} patients, and the following class balance \n {}"
        class_balance = Counter([d["y"] for d in dataset])
        exams = set([d["exam"] for d in dataset])
        patients = set([d["pid"] for d in dataset])
        statement = summary.format(
            split_group,
            len(dataset),
            len(exams),
            len(patients),
            class_balance,
        )
        statement += "\n" + "Censor Times: {}".format(
            Counter([d["time_at_event"] for d in dataset])
        )
        return statement