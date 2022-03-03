import os
from posixpath import split
import traceback, warnings
import pickle, json
import numpy as np
from sybilx import augmentations
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
)
from sybilx.utils.registry import register_object
from sybilx.datasets.nlst_risk_factors import NLSTRiskFactorVectorizer
import pydicom


CORRUPTED_PATHS = "/Mounts/rbg-storage1/datasets/ACRIN_XRAY/NLST/corrupted_img_paths.pkl"

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

RACE_ID_KEYS = {
    1: "white",
    2: "black",
    3: "asian",
    4: "american_indian_alaskan",
    5: "native_hawaiian_pacific",
    6: "hispanic",
}
ETHNICITY_KEYS = {1: "Hispanic or Latino", 2: "Neither Hispanic nor Latino"}
GENDER_KEYS = {1: "Male", 2: "Female"}
EDUCAT_LEVEL = {
    1: 1,  # 8th grade = less than HS
    2: 1,  # 9-11th = less than HS
    3: 2,  # HS Grade
    4: 3,  # Post-HS
    5: 4,  # Some College
    6: 5,  # Bachelors = College Grad
    7: 6,  # Graduate School = Postrad/Prof
}


@register_object("ct_localizers", "dataset")
class NLSTCTLocalizers(data.Dataset):
    def __init__(self, args, split_group):
        """
        NLST Xray Dataset
        params: args - config.
        params: transformer - A transformer object, takes in a PIL image, performs some transforms and returns a Tensor
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        """
        super(NLSTCTLocalizers, self).__init__()

        self.split_group = split_group
        self.args = args
        # self._num_images = args.num_images  # number of slices in each volume
        self._max_followup = args.max_followup

        try:
            self.metadata_json = json.load(open(args.dataset_file_path, "r"))
        except Exception as e:
            raise Exception(METAFILE_NOTFOUND_ERR.format(args.dataset_file_path, e))

        self.input_loader = get_sample_loader(split_group, args)

        if self.args.region_annotations_filepath:
            self.annotations_metadata = json.load(
                open(self.args.region_annotations_filepath, "r")
            )

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
        # self.corrupted_paths = self.CORRUPTED_PATHS["paths"]
        # self.corrupted_series = self.CORRUPTED_PATHS["series"]
        # self.risk_factor_vectorizer = NLSTRiskFactorVectorizer(self.args)

        if self.args.assign_splits:
            np.random.seed(self.args.cross_val_seed)
            self.assign_splits(self.metadata_json)

        dataset = []

        for mrn_row in tqdm(self.metadata_json, position=0):
            pid, split, exams, pt_metadata = (
                mrn_row["pid"],
                mrn_row["split"],
                mrn_row["accessions"],
                mrn_row["pt_metadata"],
            )
            
            # 3 that fail to load dicoms
            if pid in ['205566', '201397', '202619']:
                continue

            if not split == split_group:
                continue

            for exam_dict in exams:
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
        
        # if not localizer then skip (localizers are the projections)
        series_data = series_dict["series_data"]
        is_localizer = self.is_localizer(series_data)
        if not is_localizer:
            return True
        else:
            # is localizer, now check that is frontal screen (ie not lateral)
            # TODO: update json with ImageOrientationPatient
            # Here the test mutates the image paths
            num_images = len(series_dict['paths'])
            lateral_count = 0
            for path in series_dict['paths']:
                ds = pydicom.dcmread(path.replace("nlst-ct-png", "nlst-ct").replace(".png", ""))
                # TODO: here decision is to use arbitrary localizer in the paths if it is frontal
                if ds.ImageOrientationPatient[0] == 0:
                    lateral_count += 1
                elif ds.ImageOrientationPatient[0] != 0:
                    series_dict['paths'] = path
                    break
            # all localizer images are lateral, then skip this sample
            if lateral_count == num_images:
                return True

        # check if valid label (info is not missing)
        screen_timepoint = exam_dict["screen_timepoint"] # series_data["study_yr"][0]
        bad_label = not self.check_label(pt_metadata, screen_timepoint)

        # invalid label
        if not bad_label:
            y, _, _, time_at_event = self.get_label(pt_metadata, screen_timepoint)
            invalid_label = (y == -1) or (time_at_event < 0)
        else:
            invalid_label = False

        if (
            bad_label
            or invalid_label
        ):
            return True
        else:
            return False

    def get_volume_dict(self, series_id, series_dict, exam_dict, pt_metadata, pid, split):
        path = series_dict["paths"]
        # series_data = series_dict["series_data"]
        # device = series_data["manufacturer"][0]
        screen_timepoint = exam_dict["screen_timepoint"] # series_data["study_yr"][0]
        # assert screen_timepoint == exam_dict["screen_timepoint"]

        # if series_id in self.corrupted_series:
        #     if any([path in self.corrupted_paths for path in img_paths]):
        #         uncorrupted_imgs = np.where(
        #             [path not in self.corrupted_paths for path in img_paths]
        #         )[0]
        #         img_paths = np.array(img_paths)[uncorrupted_imgs].tolist()
        #         slice_locations = np.array(slice_locations)[uncorrupted_imgs].tolist()

        # if not path[0].startswith(self.args.img_dir):
          #  path = self.args.img_dir + path[path.find("nlst-xr-png") + len("nlst-xr-png") :]

        if self.args.img_file_type == "dicom":
           path = path.replace("nlst-ct-png", "nlst-ct").replace(".png", "")

        y, y_seq, y_mask, time_at_event = self.get_label(pt_metadata, screen_timepoint)

        exam_int = int(f"{int(pid)}{int(screen_timepoint)}{int(series_id.split('.')[-1][-3:])}")

        sample = {
            "path": path,
            "y": int(y),
            "time_at_event": time_at_event,
            "y_seq": y_seq,
            "y_mask": y_mask,
            "exam_str": "{}_{}".format(exam_dict["exam"], series_id),
            "exam": exam_int,
            "accession": exam_dict["accession_number"],
            "series": series_id,
            # "study": series_data["studyuid"][0],
            "screen_timepoint": screen_timepoint,
            "pid": pid,
            # "device": device,
            "institution": pt_metadata["cen"][0],
            "cancer_laterality": self.get_cancer_side(pt_metadata),
        }

        if self.args.use_risk_factors:
            sample["risk_factors"] = self.get_risk_factors(pt_metadata, screen_timepoint, return_dict=False)

        return sample

    def check_label(self, pt_metadata, screen_timepoint):
        valid_days_since_rand = (
            pt_metadata["scr_days{}".format(screen_timepoint)][0] > -1
        )
        valid_days_to_cancer = pt_metadata["candx_days"][0] > -1
        valid_followup = pt_metadata["fup_days"][0] > -1
        return (valid_days_since_rand) and (valid_days_to_cancer or valid_followup)

    def get_label(self, pt_metadata, screen_timepoint):
        days_since_rand = pt_metadata["scr_days{}".format(screen_timepoint)][0]
        days_to_cancer_since_rand = pt_metadata["candx_days"][0]
        days_to_cancer = days_to_cancer_since_rand - days_since_rand
        years_to_cancer = (
            int(days_to_cancer // 365) if days_to_cancer_since_rand > -1 else 100
        )
        days_to_last_followup = int(pt_metadata["fup_days"][0] - days_since_rand)
        years_to_last_followup = days_to_last_followup // 365
        y = years_to_cancer < self.args.max_followup
        y_seq = np.zeros(self.args.max_followup)
        cancer_timepoint = pt_metadata["cancyr"][0]
        if y:
            if years_to_cancer > -1:
                assert screen_timepoint <= cancer_timepoint
            time_at_event = years_to_cancer
            y_seq[years_to_cancer:] = 1
        else:
            time_at_event = min(years_to_last_followup, self.args.max_followup - 1)
        y_mask = np.array(
            [1] * (time_at_event + 1)
            + [0] * (self.args.max_followup - (time_at_event + 1))
        )
        assert len(y_mask) == self.args.max_followup
        return y, y_seq.astype("float64"), y_mask.astype("float64"), time_at_event

    def is_localizer(self, series_dict):
        is_localizer = (
            (series_dict["imageclass"][0] == 0)
            or ("LOCALIZER" in series_dict["imagetype"][0])
            or ("TOP" in series_dict["imagetype"][0])
        )
        return is_localizer

    def get_cancer_side(self, pt_metadata):
        """
        Return if cancer in left or right

        right: (rhil, right hilum), (rlow, right lower lobe), (rmid, right middle lobe), (rmsb, right main stem), (rup, right upper lobe),
        left: (lhil, left hilum),  (llow, left lower lobe), (lmsb, left main stem), (lup, left upper lobe), (lin, lingula)
        else: (med, mediastinum), (oth, other), (unk, unknown), (car, carina)
        """
        right_keys = ["locrhil", "locrlow", "locrmid", "locrmsb", "locrup"]
        left_keys = ["loclup", "loclmsb", "locllow", "loclhil", "loclin"]
        other_keys = ["loccar", "locmed", "locoth", "locunk"]

        right = any([pt_metadata[key][0] > 0 for key in right_keys])
        left = any([pt_metadata[key][0] > 0 for key in left_keys])
        other = any([pt_metadata[key][0] > 0 for key in other_keys])

        return np.array([int(right), int(left), int(other)])

    def get_risk_factors(self, pt_metadata, screen_timepoint, return_dict=False):
        age_at_randomization = pt_metadata["age"][0]
        days_since_randomization = pt_metadata["scr_days{}".format(screen_timepoint)][0]
        current_age = age_at_randomization + days_since_randomization // 365

        age_start_smoking = pt_metadata["smokeage"][0]
        age_quit_smoking = pt_metadata["age_quit"][0]
        years_smoking = pt_metadata["smokeyr"][0]
        is_smoker = pt_metadata["cigsmok"][0]

        years_since_quit_smoking = 0 if is_smoker else current_age - age_quit_smoking

        education = (
            pt_metadata["educat"][0]
            if pt_metadata["educat"][0] != -1
            else pt_metadata["educat"][0]
        )

        race = pt_metadata["race"][0] if pt_metadata["race"][0] != -1 else 0
        race = 6 if pt_metadata["ethnic"][0] == 1 else race
        ethnicity = pt_metadata["ethnic"][0]

        weight = pt_metadata["weight"][0] if pt_metadata["weight"][0] != -1 else 0
        height = pt_metadata["height"][0] if pt_metadata["height"][0] != -1 else 0
        bmi = weight / (height**2) * 703 if height > 0 else 0  # inches, lbs

        prior_cancer_keys = [
            "cancblad",
            "cancbrea",
            "canccerv",
            "canccolo",
            "cancesop",
            "canckidn",
            "canclary",
            "canclung",
            "cancoral",
            "cancnasa",
            "cancpanc",
            "cancphar",
            "cancstom",
            "cancthyr",
            "canctran",
        ]
        cancer_hx = any([pt_metadata[key][0] == 1 for key in prior_cancer_keys])
        family_hx = any(
            [pt_metadata[key][0] == 1 for key in pt_metadata if key.startswith("fam")]
        )

        risk_factors = {
            "age": current_age,
            "race": race,
            "race_name": RACE_ID_KEYS.get(pt_metadata["race"][0], "UNK"),
            "ethnicity": ethnicity,
            "ethnicity_name": ETHNICITY_KEYS.get(ethnicity, "UNK"),
            "education": education,
            "bmi": bmi,
            "cancer_hx": cancer_hx,
            "family_lc_hx": family_hx,
            "copd": pt_metadata["diagcopd"][0],
            "is_smoker": is_smoker,
            "smoking_intensity": pt_metadata["smokeday"][0],
            "smoking_duration": pt_metadata["smokeyr"][0],
            "years_since_quit_smoking": years_since_quit_smoking,
            "weight": weight,
            "height": height,
            "gender": GENDER_KEYS.get(pt_metadata["gender"][0], "UNK"),
        }

        if return_dict:
            return risk_factors
        else:
            return np.array(
                [v for v in risk_factors.values() if not isinstance(v, str)]
            )

    def assign_splits(self, meta):
        if self.args.split_type == "institution_split":
            self.assign_institutions_splits(meta)
        elif self.args.split_type == "random":
            for idx in range(len(meta)):
                meta[idx]["split"] = np.random.choice(
                    ["train", "dev", "test"], p=self.args.split_probs
                )

    def assign_institutions_splits(self, meta):
        institutions = set([m["pt_metadata"]["cen"][0] for m in meta])
        institutions = sorted(institutions)
        institute_to_split = {
            cen: np.random.choice(["train", "dev", "test"], p=self.args.split_probs)
            for cen in institutions
        }
        for idx in range(len(meta)):
            meta[idx]["split"] = institute_to_split[meta[idx]["pt_metadata"]["cen"][0]]

    @property
    def CORRUPTED_PATHS(self):
        return pickle.load(open(CORRUPTED_PATHS, "rb"))

    def get_summary_statement(self, dataset, split_group):
        summary = "Contructed NLST X-Ray Cancer Risk {} dataset with {} records, {} exams, {} patients, and the following class balance \n {}"
        class_balance = Counter([d["y"] for d in dataset])
        exams = set([d["exam"] for d in dataset])
        patients = set([d["pid"] for d in dataset])
        statement = summary.format(
            split_group, len(dataset), len(exams), len(patients), class_balance
        )
        statement += "\n" + "Censor Times: {}".format(
            Counter([d["time_at_event"] for d in dataset])
        )
        statement
        return statement

    def get_ct_annotations(self, sample):
        # correct empty lists of annotations
        if sample["series"] in self.annotations_metadata:
            self.annotations_metadata[sample["series"]] = {
                k: v
                for k, v in self.annotations_metadata[sample["series"]].items()
                if len(v) > 0
            }

        if sample["series"] in self.annotations_metadata:
            # check if there is an annotation in a slice
            sample["volume_annotations"] = np.array(
                [
                    int(
                        os.path.splitext(os.path.basename(path))[0]
                        in self.annotations_metadata[sample["series"]]
                    )
                    for path in sample["paths"]
                ]
            )
            # store annotation(s) data (x,y,width,height) for each slice
            sample["annotations"] = [
                {
                    "image_annotations": self.annotations_metadata[
                        sample["series"]
                    ].get(os.path.splitext(os.path.basename(path))[0], None)
                }
                for path in sample["paths"]
            ]
        else:
            sample["volume_annotations"] = np.array([0 for _ in sample["paths"]])
            sample["annotations"] = [
                {"image_annotations": None} for path in sample["paths"]
            ]
        return sample

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        # if self.args.use_annotations:
        #     sample = self.get_ct_annotations(sample)
        #     sample["annotation_areas"] = get_scaled_annotation_area(sample, self.args)
        #     sample["has_annotation"] = np.sum(sample["volume_annotations"]) > 0
        try:
            item = {}
            input_dict = self.get_image(sample["path"], sample)

            x, mask = input_dict["input"], input_dict["mask"]
            # if self.args.use_all_images:
            #     c, n, h, w = x.shape
            #     x = torch.nn.functional.interpolate(
            #         x.unsqueeze(0), (self._num_images, h, w), align_corners=True
            #     )[0]
            #     if mask is not None:
            #         mask = torch.nn.functional.interpolate(
            #             mask.unsqueeze(0), (self._num_images, h, w), align_corners=True
            #         )[0]

            # if self.args.use_annotations:
            #     # item['mask'] = mask
            #     # mask = item.pop('mask')
            #     mask = torch.abs(mask)
            #     mask_area = mask.sum(dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1)
            #     mask_area[mask_area == 0] = 1
            #     mask = mask / mask_area
            #     item["image_annotations"] = mask
            #     if self.args.use_all_images:
            #         t = torch.from_numpy(sample["annotation_areas"])
            #         item["annotation_areas"] = F.interpolate(
            #             t[None, None],
            #             (self._num_images),
            #             mode="linear",
            #             align_corners=True,
            #         )[0, 0]
            #         t = torch.from_numpy(sample["volume_annotations"]).float()
            #         item["volume_annotations"] = F.interpolate(
            #             t[None, None],
            #             (self._num_images),
            #             mode="linear",
            #             align_corners=True,
            #         )[0, 0]
            #     else:
            #         item["annotation_areas"] = sample["annotation_areas"]
            #         item["volume_annotations"] = sample["volume_annotations"]

            if self.args.use_risk_factors:
                item["risk_factors"] = sample["risk_factors"]

            item["x"] = x
            item["y"] = sample["y"]
            for key in CT_ITEM_KEYS:
                if key in sample:
                    item[key] = sample[key]

            return item
        except Exception:
            warnings.warn(LOAD_FAIL_MSG.format(sample["exam"], traceback.print_exc()))

    def get_image(self, path, sample):
        """
        Returns a stack of transformed images by their absolute paths.
        If cache is used - transformed images will be loaded if available,
        and saved to cache if not.
        """
        out_dict = {}
        if self.args.fix_seed_for_multi_image_augmentations:
            sample["seed"] = np.random.randint(0, 2**32 - 1)

        # get images for multi image input
        s = copy.deepcopy(sample)
        input_dict = self.input_loader.get_image(path, s)
        # if self.args.use_annotations:
        #     s["annotations"] = sample["annotations"][0]

        image = input_dict["input"]
        masks = input_dict["mask"]

        # out_dict["input"] = self.reshape_images(image)
        # out_dict["mask"] = (
        #     self.reshape_images(masks) if self.args.use_annotations else None
        # )

        # return out_dict
        return input_dict

    def reshape_images(self, images):
        images = [im.unsqueeze(0) for im in images]
        images = torch.cat(images, dim=0)
        # Convert from (T, C, H, W) to (C, T, H, W)
        images = images.permute(1, 0, 2, 3)
        return images