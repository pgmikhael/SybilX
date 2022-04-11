import os
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
    get_cum_label,
    get_scaled_annotation_area,
    METAFILE_NOTFOUND_ERR,
    LOAD_FAIL_MSG,
)
from sybilx.utils.registry import register_object
from sybilx.datasets.nlst_risk_factors import NLSTRiskFactorVectorizer


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


@register_object("nlst_xray", "dataset")
class NLST_XRay_Dataset(data.Dataset):
    def __init__(self, args, split_group):
        """
        NLST Xray Dataset
        params: args - config.
        params: transformer - A transformer object, takes in a PIL image, performs some transforms and returns a Tensor
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        """
        super(NLST_XRay_Dataset, self).__init__()

        self.split_group = split_group
        self.args = args
        self._max_followup = args.max_followup

        try:
            self.metadata_json = json.load(open(args.dataset_file_path, "r"))
        except Exception as e:
            raise Exception(METAFILE_NOTFOUND_ERR.format(args.dataset_file_path, e))

        self.input_loader = get_sample_loader(split_group, args)

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
            pid, split, exams, pt_metadata = (
                mrn_row["pid"],
                mrn_row["split"],
                mrn_row["accessions"],
                mrn_row["pt_metadata"],
            )

            for exam_dict in exams:
                for series_dict in exam_dict["image_series"]:
                    series_id = series_dict["series_id"]
                    if self.skip_sample(series_dict, pt_metadata, exam_dict, split, split_group):
                        continue

                    sample = self.get_volume_dict(
                        series_id, series_dict, exam_dict, pt_metadata, pid, split
                    )
                    if len(sample) == 0:
                        continue

                    dataset.append(sample)

        return dataset

    def skip_sample(self, series_dict, pt_metadata, exam_dict, pt_split, split_group):
        if not pt_split == split_group:
            return True

        # check if valid label (info is not missing)
        screen_timepoint = exam_dict["screen_timepoint"] 
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
        path = series_dict["path"]
        screen_timepoint = exam_dict["screen_timepoint"]

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
            "invert_pixels": series_dict["PhotometricInterpretation"] == "MONOCHROME1"
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
        cancer_timepoint = pt_metadata["cancyr"][0]

        if y:
            if years_to_cancer > -1:
                assert screen_timepoint <= cancer_timepoint
            time_at_event = years_to_cancer
        else:
            time_at_event = min(years_to_last_followup, self.args.max_followup - 1)

        if "corn" in self.args.loss_fns:
            y_seq, y_mask = get_cum_label(y, time_at_event, self.args.max_followup)
        else:
            y_seq = np.zeros(self.args.max_followup)
            if y:
                y_seq[years_to_cancer:] = 1

            y_mask = np.array(
                [1] * (time_at_event + 1)
                + [0] * (self.args.max_followup - (time_at_event + 1))
            )


        assert len(y_mask) == self.args.max_followup
        return y, y_seq.astype("float64"), y_mask.astype("float64"), time_at_event

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
        # is_smoker = pt_metadata["cigsmok"][0]

        # years_since_quit_smoking = 0 if is_smoker else current_age - age_quit_smoking

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
        
        if not pt_metadata["gender"][0] in [-1, 1, 2]:
            print(pt_metadata["gender"])
        # assert pt_metadata["gender"] in [-1, 1, 2], "unrecognized gender"

        risk_factors = {
            "age": current_age,
            # "race": race,
            # "race_name": RACE_ID_KEYS.get(pt_metadata["race"][0], "UNK"),
            # "ethnicity": ethnicity,
            # "ethnicity_name": ETHNICITY_KEYS.get(ethnicity, "UNK"),
            # "education": education,
            # "bmi": bmi,
            # "cancer_hx": cancer_hx,
            # "family_lc_hx": family_hx,
            # "copd": pt_metadata["diagcopd"][0],
            "is_smoker": int(pt_metadata["cigsmok"][0] == 1),
            "is_not_smoker": int(pt_metadata["cigsmok"][0] == 0),
            "smoking_status_unknown": int(pt_metadata["cigsmok"][0] == -1),
            # "smoking_intensity": pt_metadata["smokeday"][0],
            # "smoking_duration": pt_metadata["smokeyr"][0],
            # "years_since_quit_smoking": years_since_quit_smoking,
            # "weight": weight,
            # "height": height,
            # "gender": GENDER_KEYS.get(pt_metadata["gender"][0], "UNK"),
            "is_female": int(pt_metadata["gender"] == 2),
            "is_male": int(pt_metadata["gender"] == 1),
            "gender_unknown": int(pt_metadata["gender"] == -1)
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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]

        try:
            item = {}
            input_dict = self.get_image(sample["path"], sample)

            try:
                x, mask = input_dict["input"], input_dict["mask"]
            except KeyError:
                x = input_dict["input"]
                mask = None

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
        # get images for multi image input
        s = copy.deepcopy(sample)
        input_dict = self.input_loader.get_image(path, s)

        return input_dict


@register_object("nlst_xray_test", "dataset")
class NLST_XRay_Test_Dataset(NLST_XRay_Dataset):
    def __init__(self, args, split_group):
        assert args.test and not args.train, "This dataset is for testing only"
        super(NLST_XRay_Test_Dataset, self).__init__(args, split_group)

    def skip_sample(self, series_dict, pt_metadata, exam_dict, pt_split, split_group):
        # check if valid label (info is not missing)
        screen_timepoint = exam_dict["screen_timepoint"] 
        bad_label = not self.check_label(pt_metadata, screen_timepoint)

        # filter out non-frontal x-rays
        if type(series_dict["PatientOrientation"]) == list and series_dict["PatientOrientation"][0] == 'P':
            return True

        # optional filtering based on image type
        if self.args.filter_derived_images and 'DERIVED' in series_dict["ImageType"]:
            return True

        if self.args.filter_post_processed_images and 'POST_PROCESSED' in series_dict["ImageType"]:
            return True

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
