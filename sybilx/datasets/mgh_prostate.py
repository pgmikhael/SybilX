import enum
import os
from posixpath import split
import traceback, warnings
import pickle, json
import numpy as np
from tqdm import tqdm
from collections import Counter
import copy
import torch
import torch.nn.functional as F
from torch.utils import data
from sybilx.serie import Serie
from sybilx.utils.loading import get_sample_loader
from sybilx.datasets.utils import (
    fit_to_length,
    get_scaled_annotation_area,
    METAFILE_NOTFOUND_ERR,
    LOAD_FAIL_MSG,
    DEVICE_ID,
)
from sybilx.utils.registry import register_object
import pandas as pd
import pydicom
import torchio as tio


LABELS_PATH = "/Mounts/rbg-storage1/datasets/MGH_Prostate_Salari/reports/output.csv"

MR_ITEM_KEYS = [
    "pid",
    "accession"
    "series",
    "slice_thickness",
]

@register_object("mgh_prostate", "dataset")
class MGH_Prostate(data.Dataset):
    def __init__(self, args, split_group):
        """
        NLST Dataset
        params: args - config.
        params: transformer - A transformer object, takes in a PIL image, performs some transforms and returns a Tensor
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        """
        super(MGH_Prostate, self).__init__()
        self.args = args

        self._num_images = args.num_images  # number of slices in each volume

        try:
            self.metadata_json = json.load(open(args.dataset_file_path, "r"))
        except Exception as e:
            raise Exception(METAFILE_NOTFOUND_ERR.format(args.dataset_file_path, e))

        self.input_loader = get_sample_loader(split_group, args)
        self.always_resample_pixel_spacing = (args.resample_pixel_spacing) and (
            split_group in ["dev","test"]
        )
        if args.resample_pixel_spacing:
            self.resample_transform = tio.transforms.Resample(
                target=tuple(args.ct_pixel_spacing)
            )
            self.padding_transform = tio.transforms.CropOrPad(
                target_shape=tuple(args.img_size + [args.num_images]), padding_mode=0
            )

        # load labels metadata csv
        self.labels_data = pd.read_csv(LABELS_PATH, low_memory=True)
        self.labels_data.fillna(0, inplace=True)
        self.labels_data.replace(to_replace="No Evidence", value=0, inplace=True)

        self.dataset = self.create_dataset(split_group)
        if len(self.dataset) == 0:
            return

        print(self.get_summary_statement(self.dataset, split_group))

        label_dist = [d[args.class_bal_key] for d in self.dataset]
        label_counts = Counter(label_dist)
        weight_per_label = 1.0 / len(label_counts)
        label_weights = {
            label: weight_per_label / count for label, count in label_counts.items()
        }

        print("Class counts are: {}".format(label_counts))
        print("Label weights are {}".format(label_weights))
        self.weights = [label_weights[d[args.class_bal_key]] for d in self.dataset]

    def create_dataset(self, split_group):
        """
        Gets the dataset from the paths and labels in the json.
        Arguments:
            split_group(str): One of ['train'|'dev'|'test'].
        Returns:
            The dataset as a dictionary with img paths, label,
            and additional information regarding exam or participant
        """

        if self.args.assign_splits:
            np.random.seed(self.args.cross_val_seed)
            self.assign_splits(self.metadata_json)

        dataset = []

        for mrn_row in tqdm(self.metadata_json, position=0):
            #pid, split, exams, pt_metadata = 
            pid, split, exams = (
                mrn_row["pid"],
                mrn_row["split"],
                mrn_row["accessions"],
                #mrn_row["pt_metadata"],
            )
            pt_metadata = None # temporary / until we find a use for it lol

            if not split == split_group:
                continue

            for exam_dict in exams:
                # TODO: update pt_metadata here with patient info for each exam
                for series_id, series_dict in exam_dict["image_series"].items():
                    if self.skip_sample(series_dict, pt_metadata, exam_dict):
                        continue

                    sample = self.get_volume_dict(
                        series_id, series_dict, exam_dict, pt_metadata, pid, split
                    )
                    if len(sample) == 0:
                        continue

                    dataset.append(sample)

        return dataset


    def skip_sample(self, series_dict, pt_metadata, exam_dict):
        series_data = series_dict["series_data"]

        wrong_series = False #TODO: update

        # # check if restricting to specific slice thicknesses
        slice_thickness = series_data["slice_thickness"]
        wrong_thickness = (self.args.slice_thickness_filter is not None) and (
            slice_thickness > self.args.slice_thickness_filter or (slice_thickness < 0)
        )

        # # check if valid biopsy reports exist for this patient
        accession_number = exam_dict["accession_number"]
        df = self.labels_data.loc[(self.labels_data["Accession Number"] == int(accession_number))]
        missing_label = (df.shape[0] == 0)
        # screen_timepoint = series_data["study_yr"][0]
        # bad_label = not self.check_label(pt_metadata, screen_timepoint)

        # # invalid label
        # if not bad_label:
        #     y, _, _, time_at_event = self.get_label(pt_metadata, screen_timepoint)
        #     invalid_label = (y == -1) or (time_at_event < 0)
        # else:
        #     invalid_label = False

        # insufficient_slices = len(series_dict["paths"]) < self.args.min_num_images

        if (
            wrong_series
            or wrong_thickness
            or missing_label
            # or bad_label
            # or invalid_label
            # or insufficient_slices
        ):
            return True
        else:
            return False

    def get_volume_dict(self, series_id, series_dict, exam_dict, pt_metadata, pid, split):
        img_paths = series_dict["paths"]
        slice_locations = series_dict["slice_location"]
        series_data = series_dict["series_data"]
        # screen_timepoint = series_data["study_yr"][0]
        # assert screen_timepoint == exam_dict["screen_timepoint"]


        sorted_img_paths, sorted_slice_locs = self.order_slices(
            img_paths, slice_locations
        )

        # if not sorted_img_paths[0].startswith(self.args.img_dir):
        #     sorted_img_paths = [
        #         self.args.img_dir
        #         + path[path.find("nlst-ct-png") + len("nlst-ct-png") :]
        #         for path in sorted_img_paths
        #     ]
        # if self.args.img_file_type == "dicom":
        #     sorted_img_paths = [
        #         path.replace("nlst-ct-png", "nlst-ct").replace(".png", "")
        #         for path in sorted_img_paths
        #     ]

        y = self.get_label(exam_dict)

        sample = {
            "paths": sorted_img_paths,
            "slice_locations": sorted_slice_locs,
            "y": int(y),
            "exam_str": "{}_{}".format(exam_dict["exam"], series_id),
            "accession": exam_dict["accession_number"],
            "series": series_id,
            "pid": pid,
            "num_original_slices": len(series_dict["paths"]),
            "pixel_spacing": series_data["pixel_spacing"]
            + [series_data["slice_thickness"]],
            # "slice_thickness": self.get_slice_thickness_class( TODO: Update slice thickness class based on our images
            #     series_dict["slice_thickness"]
            # ),
            "slice_thickness": series_data["slice_thickness"]
        }


        if self.args.fit_to_length:
            sample["paths"] = fit_to_length(sorted_img_paths, self.args.num_images)
            sample["slice_locations"] = fit_to_length(
                sorted_slice_locs, self.args.num_images, "<PAD>"
            )

        return sample


    def get_label(self, exam_dict):
        """
        Args:
            pid
            exam_dict (dict): The input exam of the patient

        Returns:
            0 if the closest biopsy in the future has a Gleason score < 7
            1 if the closest biopsy in the future has a Gleason score >= 7
        """
        accession_number = exam_dict["accession_number"]
        df = self.labels_data.loc[(self.labels_data["Accession Number"] == int(accession_number))]
        
        # assume df only has one row
        assert df.shape[0] == 1, "There should only be one accession exam associated with the accession number."
        gleason_score = df.iloc[0]["GS"]
        if gleason_score < 7:
            return 0
        else:
            return 1


    def get_pixel_spacing(self, dcm_path):
        """Get slice thickness and row/col spacing

        Args:
            path (str): path to sample png file in the series

        Returns:
            pixel spacing: [thickness, spacing[0], spacing[1]]
                thickness (float): CT slice thickness
                spacing (list): spacing along x and y axes
        """
        dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        spacing = [float(d) for d in dcm.PixelSpacing] + [float(dcm.SliceThickness)]
        return spacing


    def order_slices(self, img_paths, slice_locations, reverse=False):
        sorted_ids = np.argsort(slice_locations)
        if reverse:
            sorted_ids = sorted_ids[::-1]
        sorted_img_paths = np.array(img_paths)[sorted_ids].tolist()
        sorted_slice_locs = np.sort(slice_locations).tolist()

        return sorted_img_paths, sorted_slice_locs


    def assign_splits(self, meta):
        if self.args.split_type == "random":
            for idx in range(len(meta)):
                meta[idx]["split"] = np.random.choice(
                    ["train", "dev", "test"], p=self.args.split_probs
                )


    def get_summary_statement(self, dataset, split_group):
        summary = "Contructed MGH Prostate Cancer Risk {} dataset with {} records, {} exams, {} patients, and the following class balance \n {}"
        class_balance = Counter([d["y"] for d in dataset])
        exams = set([d["accession"] for d in dataset])
        patients = set([d["pid"] for d in dataset])
        statement = summary.format(
            split_group, len(dataset), len(exams), len(patients), class_balance
        )
        return statement


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]

        try:
            item = {}
            input_dict = self.get_images(sample["paths"], sample)
            x = input_dict["input"]

            item["x"] = x
            item["y"] = sample["y"]
            for key in MR_ITEM_KEYS:
                if key in sample:
                    item[key] = sample[key]

            return item
        except Exception:
            warnings.warn(LOAD_FAIL_MSG.format(sample["accession"], traceback.print_exc()))

    def get_images(self, paths, sample):
        """
        Returns a stack of transformed images by their absolute paths.
        If cache is used - transformed images will be loaded if available,
        and saved to cache if not.
        """
        out_dict = {}
        # if self.args.fix_seed_for_multi_image_augmentations:
        #     sample["seed"] = np.random.randint(0, 2**32 - 1)

        # get images for multi image input
        s = copy.deepcopy(sample)
        input_dicts = []
        for e, path in enumerate(paths):
            input_dicts.append(self.input_loader.get_image(path, s))

        images = [i["input"] for i in input_dicts]
        input_arr = self.reshape_images(images)

        # TODO! 
        # resample pixel spacing
        resample_now = (
            self.args.resample_pixel_spacing_prob > np.random.uniform()
        ) and self.args.resample_pixel_spacing
        if self.always_resample_pixel_spacing or resample_now:
            spacing = torch.tensor(sample["pixel_spacing"] + [1])
            input_arr = tio.ScalarImage(
                affine=torch.diag(spacing),
                tensor=input_arr.permute(0, 2, 3, 1),
            )
            input_arr = self.resample_transform(input_arr)
            input_arr = self.padding_transform(input_arr.data)

            
        elif self.args.resample_pixel_spacing:
            input_arr = self.padding_transform(input_arr.permute(0, 2, 3, 1))
            mask_arr = self.padding_transform(mask_arr.permute(0, 2, 3, 1))

        if self.args.resample_pixel_spacing:
            out_dict["input"] = input_arr.data.permute(0, 3, 1, 2)
        else:
            out_dict["input"] = input_arr
        
        return out_dict

    def reshape_images(self, images):
        images = [im.unsqueeze(0) for im in images]
        images = torch.cat(images, dim=0)
        # print(images.shape)
        # Convert from (T, C, H, W) to (C, T, H, W)
        images = images.permute(1, 0, 2, 3)
        return images

    def get_slice_thickness_class(self, thickness):
        BINS = [1, 1.5, 2, 2.5]
        for i, tau in enumerate(BINS):
            if thickness <= tau:
                return i
        if self.args.slice_thickness_filter is not None:
            raise ValueError("THICKNESS > 2.5")
        return 4
