from curses.ascii import isdigit
import os
from posixpath import split
import traceback, warnings
import pickle, json
import numpy as np
#from sybilx import augmentations
from tqdm import tqdm
from collections import Counter
import copy
import torch
import pandas as pd
#import torch.nn.functional as F
from torch.utils import data
#from sybilx.serie import Serie
from sybilx.utils.loading import get_sample_loader
from sybilx.datasets.utils import (
    get_cum_label,
    METAFILE_NOTFOUND_ERR,
    LOAD_FAIL_MSG,
)
from sybilx.utils.registry import register_object
#from sybilx.datasets.nlst_risk_factors import NLSTRiskFactorVectorizer

CXR_LC_PATIENT_SPLITS_FILENAME = ("/Mounts/rbg-storage1/datasets/PLCO_XRAY/patient2split_cxr_lc.pkl")
CXR_LC_IMAGE_SPLITS_FILENAME = ("/Mounts/rbg-storage1/datasets/PLCO_XRAY/image2split_cxr_lc.pkl")

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
    1: "White, Non-Hispanic",
    2: "Black, Non-Hispanic",
    3: "Hispanic",
    4: "Asian",
    6: "Pacific Islander",
    6: "American Indian",
    7: "Missing",
}
ETHNICITY_KEYS = {0: "Not Hispanic", 1: "Hispanic"}
SEX_KEYS = {1: "Male", 2: "Female"}
EDUCAT_LEVEL = {
    1: 1,  # 8th grade = less than HS
    2: 1,  # 9-11th = less than HS
    3: 2,  # 12 Years Or Completed High School = HS Grade
    4: 3,  # Post-HS
    5: 4,  # Some College
    6: 5,  # College Graduate
    7: 6,  # Postgraduate = Postrad/Prof
}

@register_object("plco_xray", "dataset")
class PLCO_XRay_Dataset(data.Dataset):
    def __init__(self, args, split_group):
        """
        PLCO Xray Dataset
        params: args - config.
        params: transformer - A transformer object, takes in a PIL image, performs some transforms and returns a Tensor
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        """
        super(PLCO_XRay_Dataset, self).__init__()
        self.NUM_SKIPPED_FROM_CXR_LC_SPLIT = 0
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
        # self.risk_factor_vectorizer = NLSTRiskFactorVectorizer(self.args)

        if self.args.split_type == 'cxr_lc':
            CXR_LC_IMAGE_SPLITS = pickle.load(open(CXR_LC_IMAGE_SPLITS_FILENAME, "rb"))

        if self.args.assign_splits:
            np.random.seed(self.args.cross_val_seed)
            if self.args.split_type == 'cxr_lc':
                CXR_LC_PATIENT_SPLITS = pickle.load(open(CXR_LC_PATIENT_SPLITS_FILENAME, "rb"))
                self.assign_splits(self.metadata_json, CXR_LC_PATIENT_SPLITS)
            else:
                self.assign_splits(self.metadata_json)

        dataset = []

        for mrn_row in tqdm(self.metadata_json, position=0):
            pid, split, exams, pt_metadata = (
                mrn_row["pid"],
                mrn_row["split"],
                mrn_row["accessions"],
                mrn_row["pt_metadata"],
            )

            if not split == split_group:
                continue

            for exam_dict in exams:
                for series_dict in exam_dict["image_series"]:
                    filename = series_dict["filename"]
                    if self.args.split_type == 'cxr_lc':
                        if self.skip_sample(series_dict, pt_metadata, exam_dict, split_group, CXR_LC_IMAGE_SPLITS):
                            continue
                    elif self.skip_sample(series_dict, pt_metadata, exam_dict, split_group):
                        continue

                    sample = self.get_volume_dict(
                        filename, series_dict, exam_dict, pt_metadata, pid, split
                    )
                    if len(sample) == 0:
                        continue

                    dataset.append(sample)

        if self.args.split_type == 'cxr_lc':
            print(f"Number of samples skipped because they are not in PLCO splits: {self.NUM_SKIPPED_FROM_CXR_LC_SPLIT}")
        return dataset

    def skip_sample(self, series_dict, pt_metadata, exam_dict, split_group, split_dict=None):
        # check if valid label (info is not missing)
        study_yr = exam_dict["study_yr"] # series_data["study_yr"][0]
        visit_num = exam_dict["visit_num"] # series_data["study_yr"][0]
        days_since_rand = pt_metadata["xry_days{}".format(study_yr)]
        screen_timepoint = days_since_rand // 365
        followup_days = pt_metadata["lung_exitdays"]
        days_to_cancer_since_rand = pt_metadata["lung_cancer_diagdays"]
        years_to_cancer = (
            int(days_to_cancer_since_rand // 365) if days_to_cancer_since_rand > -1 else 100
        )

        if self.args.plco_train_study_yrs is not None and split_group == 'train' and study_yr not in self.args.plco_train_study_yrs:
            return True

        if self.args.plco_test_study_yrs is not None and split_group in ('dev','test') and study_yr not in self.args.plco_test_study_yrs:
            return True

        if self.args.plco_use_only_visitnum is not None and visit_num not in self.args.plco_use_only_visitnum:
            return True

        if self.args.plco_use_only_one_image and exam_dict['image_series'].index(series_dict) > 0:
            return True

        if self.args.min_img_width is not None and series_dict['Image ImageWidth'] < self.args.min_img_width:
            return True
        if self.args.max_img_width is not None and series_dict['Image ImageWidth'] > self.args.max_img_width:
            return True
        if self.args.min_img_height is not None and series_dict['Image ImageHeight'] < self.args.min_img_height:
            return True
        if self.args.max_img_height is not None and series_dict['Image ImageHeight'] > self.args.max_img_height:
            return True

        if self.args.split_type == 'cxr_lc':
            assert split_dict is not None
            if (not series_dict["filename"] in split_dict) or (split_group == 'skip'):
                self.NUM_SKIPPED_FROM_CXR_LC_SPLIT += 1
                return True

        # if scan happened after last followup date
        if days_since_rand > followup_days:
            return True

        # invalid label
        bad_label = not self.check_label(pt_metadata, study_yr)
        if not bad_label:
            y, _, _, time_at_event = self.get_label(pt_metadata, study_yr)
            invalid_label = (y == -1) or (time_at_event < 0)

            # ensure scan happened before cancer diagnosis
            if y and (days_since_rand > days_to_cancer_since_rand):
                return True
        else:
            invalid_label = False


        if bad_label or invalid_label:
            return True
        else:
            return False


    def get_volume_dict(self, series_id, series_dict, exam_dict, pt_metadata, pid, split):
        path = series_dict["path"]
        study_yr = exam_dict["study_yr"]

        y, y_seq, y_mask, time_at_event = self.get_label(pt_metadata, study_yr)

        # char no. of first char + all digits
        pid_int = int( str(ord(pid[0])) + ''.join(c for c in pid if c.isdigit()) )
        exam_int = int(f"{int(pid_int)}{int(study_yr)}")

        sample = {
            "path": path,
            "y": int(y),
            "time_at_event": time_at_event,
            "y_seq": y_seq,
            "y_mask": y_mask,
            "exam_str": "{}_{}".format(exam_dict["exam"], series_id),
            "exam": exam_int,
            "accession": exam_dict["study_yr"],
            "series": series_id,
            "screen_timepoint": study_yr,
            "pid": pid,
            # "study": series_data["studyuid"][0],
            # "device": device,
            #"institution": pt_metadata["cen"],
            #"cancer_laterality": self.get_cancer_side(pt_metadata),
        }

        if self.args.use_risk_factors:
            sample["risk_factors"] = self.get_risk_factors(pt_metadata, study_yr, return_dict=False)

        return sample

    def check_label(self, pt_metadata, study_yr):
        valid_days_since_rand = pt_metadata["xry_days{}".format(study_yr)] > -1
        valid_days_to_cancer = pt_metadata["lung_cancer_diagdays"] > -1
        #valid_followup = pt_metadata["fup_days"] > -1
        valid_followup_days = pt_metadata["lung_exitdays"] > -1
        return valid_days_since_rand and (valid_days_to_cancer or valid_followup_days)

    def get_label(self, pt_metadata, study_yr):
        days_since_rand = pt_metadata["xry_days{}".format(study_yr)]
        days_to_cancer_since_rand = pt_metadata["lung_cancer_diagdays"]
        days_to_cancer = days_to_cancer_since_rand - days_since_rand
        years_to_cancer = (
            int(days_to_cancer // 365) if days_to_cancer_since_rand > -1 else 100
        )
        days_to_last_followup = int(pt_metadata["lung_exitdays"] - days_since_rand)
        years_to_last_followup = days_to_last_followup // 365
        y = years_to_cancer < self.args.max_followup
        timepoint = days_since_rand // 365 # actual timepoint (year) of xray may be before study_yr
        cancer_timepoint = days_to_cancer_since_rand // 365
        if y:
            if years_to_cancer > -1:
                assert timepoint <= cancer_timepoint
            time_at_event = years_to_cancer
        else:
            time_at_event = min(years_to_last_followup, self.args.max_followup - 1)

        if "corn" in self.args.loss_fns:
            y_seq, y_mask = get_cum_label(y, time_at_event, self.args.max_followup, self.args.use_faux_binary)
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

    #def get_cancer_side(self, pt_metadata):
        #"""
        #Return if cancer in left or right

        #right: (rhil, right hilum), (rlow, right lower lobe), (rmid, right middle lobe), (rmsb, right main stem), (rup, right upper lobe),
        #left: (lhil, left hilum),  (llow, left lower lobe), (lmsb, left main stem), (lup, left upper lobe), (lin, lingula)
        #else: (med, mediastinum), (oth, other), (unk, unknown), (car, carina)
        #"""
        #right_keys = ["locrhil", "locrlow", "locrmid", "locrmsb", "locrup"]
        #left_keys = ["loclup", "loclmsb", "locllow", "loclhil", "loclin"]
        #other_keys = ["loccar", "locmed", "locoth", "locunk"]

        #right = any([pt_metadata[key] > 0 for key in right_keys])
        #left = any([pt_metadata[key] > 0 for key in left_keys])
        #other = any([pt_metadata[key] > 0 for key in other_keys])

        #return np.array([int(right), int(left), int(other)])

    def get_risk_factors(self, pt_metadata, screen_timepoint, return_dict=False):
        age_at_randomization = pt_metadata["age"]
        days_since_randomization = pt_metadata["xry_days{}".format(screen_timepoint)]
        current_age = age_at_randomization + days_since_randomization // 365

        age_start_smoking = pt_metadata["smokea_f"]
        age_quit_smoking = pt_metadata["ssmokea_f"]
        years_smoking = pt_metadata["cig_years"]
        cigarettes_per_day = pt_metadata["cigpd_f"] # divided into 8 intervals
        pack_years = pt_metadata["pack_years"] if pt_metadata["pack_years"] not in ('.F','.M') else -1

        is_smoker = pt_metadata["cig_stat"] == 1
        is_not_smoker = pt_metadata["cig_stat"] in (0, 2) #Never/former smoker
        smoking_status_unknown = pt_metadata["cig_stat"] in ('.A', '.F', '.M', -1)

        # years_since_quit_smoking = 0 if is_smoker else current_age - age_quit_smoking

        # TODO: ensure education level coding corresponds with NLST coding
        education = (
            pt_metadata["educat"]
            if pt_metadata["educat"] not in ('.F', '.M')
            else -1
        )

        race = pt_metadata["race7"] if pt_metadata["race7"] != -1 else 0
        #race = 6 if pt_metadata["ethnic"][0] == 1 else race # XXX: what does this do?

        ethnicity = pt_metadata["hispanic_f"] if pt_metadata["hispanic_f"] in (0, 1) else -1

        weight = pt_metadata["weight_f"] if pt_metadata["weight_f"] not in (-1, '.F', '.M', '.R',) else 0
        height = pt_metadata["height_f"] if pt_metadata["height_f"] not in (-1, '.F', '.M', '.R') else 0
        bmi = weight / (height**2) * 703 if height > 0 else 0  # inches, lbs

        # only lung cancer history considered. Not any cancer
        cancer_hx = pt_metadata["ph_lung_trial"]
        family_hx = pt_metadata["lung_fh"] in (1, 9) # 1="Yes, Immediate Family Member", 9="Possibly - Relative Or Cancer Type Not Clear"

        risk_factors = {
            "age": current_age,
            # "race": race,
            # "race_name": RACE_ID_KEYS.get(race, "UNK"),
            # "ethnicity": ethnicity,
            # "ethnicity_name": ETHNICITY_KEYS.get(ethnicity, "UNK"),
            # "education": education,
            # "bmi": bmi,
            # "cancer_hx": cancer_hx,
            # "family_lc_hx": family_hx,
            #"copd": pt_metadata["diagcopd"][0],
            "is_smoker": is_smoker,
            "is_not_smoker": is_not_smoker, # don't judge, this is to be true to CXR-LC paper
            "smoking_status_unknown": smoking_status_unknown,
            # "smoking_intensity": cigarettes_per_day,
            # "smoking_duration": years_smoking,
            # "years_since_quit_smoking": years_since_quit_smoking,
            # "weight": weight,
            # "height": height,
            # "sex": SEX_KEYS.get(pt_metadata["sex"], "UNK"),
            "is_female": int(pt_metadata["sex"] == 0),
            "is_male": int(pt_metadata["sex"] == 1),
            "gender_unknown": int(pt_metadata["sex"] == -1)
        }

        if return_dict:
            return risk_factors
        else:
            return np.array(
                [v for v in risk_factors.values() if not isinstance(v, str)]
            )

    def assign_splits(self, meta, split_dict=None):
        if self.args.split_type == "random":
            for idx in range(len(meta)):
                meta[idx]["split"] = np.random.choice(["train", "dev", "test"], p=self.args.split_probs)
        elif self.args.split_type == "cxr_lc":
            assert split_dict is not None
            self.assign_cxr_lc_splits(meta, split_dict)

    def assign_cxr_lc_splits(self, meta, split_dict):
        for idx in range(len(meta)):
            plco2sybilsplit = {'Tr': 'train', 'Te': 'test', 'Tu': 'dev', 'skip':'skip'}
            split_set = split_dict.get(meta[idx]["pid"], {'skip',})
            assert len(split_set) == 1, "patient is assigned multiple splits"
            (split_type,) = split_set 
            meta[idx]["split"] = plco2sybilsplit[split_type]

    def get_summary_statement(self, dataset, split_group):
        summary = "Contructed PLCO X-Ray Cancer Risk {} dataset with {} records, {} exams, {} patients, and the following class balance \n {}"
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
        s = copy.deepcopy(sample)
        input_dict = self.input_loader.get_image(path, s)
        
        return input_dict


@register_object("plco_xray_smokers", "dataset")
class PLCO_XRay_Smokers_Dataset(PLCO_XRay_Dataset):
    def skip_sample(self, series_dict, pt_metadata, exam_dict, split_group, split_dict=None):
        is_smoker = pt_metadata["cig_stat"] == 1
        if is_smoker:
            return True

        return super(PLCO_XRay_Smokers_Dataset, self).skip_sample(series_dict, pt_metadata, exam_dict, split_group, split_dict=None)


@register_object("plco_nlst_smokers", "dataset")
class PLCO_NLST_Smokers_Dataset(PLCO_XRay_Dataset):
    """
    Filters smokers using the same criteria as NLST
    """
    def skip_sample(self, series_dict, pt_metadata, exam_dict, split_group, split_dict=None):
        screen_timepoint = exam_dict["study_yr"]  
        age_at_randomization = pt_metadata["age"]
        days_since_randomization = pt_metadata["xry_days{}".format(screen_timepoint)]
        current_age = age_at_randomization + days_since_randomization // 365

        age_quit_smoking = pt_metadata["ssmokea_f"]
        pack_years = pt_metadata["pack_years"] if pt_metadata["pack_years"] not in ('.F','.M') else -1

        is_smoker = pt_metadata["cig_stat"] == 1
        years_since_quit_smoking = 0 if is_smoker else current_age - age_quit_smoking

        # NLST Critera
        # Current or former cigarette smokers within the past 15 years
        # 55 to 74 years of age
        # At least 30 pack-years of smoking 
        # [Pack-years = packs per day x number of years smoking].
        is_NLST_smoker = (55 <= current_age <= 74) and (pack_years >= 30) and (0 <= years_since_quit_smoking <= 15)
        if is_NLST_smoker:
            return True

        return super(PLCO_NLST_Smokers_Dataset, self).skip_sample(series_dict, pt_metadata, exam_dict, split_group, split_dict=None)