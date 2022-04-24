import numpy as np
from tqdm import tqdm
from ast import literal_eval
import copy
from sybilx.datasets.nlst import NLST_Survival_Dataset
from collections import Counter
from sybilx.datasets.utils import fit_to_length, get_scaled_annotation_area
from sybilx.utils.registry import register_object
import pickle 

@register_object("cgmh", "dataset")
class CGMH_Dataset(NLST_Survival_Dataset):
    """
    CGHM, Taiwan Eval Cohort
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
        assert not self.args.train, "Cohort 2 should not be used for training"
        empty_paths = pickle.load(open('/home/peter/empty_scans.p','rb'))
        dataset = []

        for mrn_row in tqdm(self.metadata_json):

            pid, exams = mrn_row["pid"], mrn_row["exams"]

            for exam_dict in exams:

                for series_dict in exam_dict["series"]:
                    if self.skip_sample(series_dict, exam_dict, empty_paths):
                        continue

                    sample = self.get_volume_dict(series_dict, exam_dict, mrn_row)
                    if len(sample) == 0:
                        continue

                    dataset.append(sample)

        return dataset

    def skip_sample(self, series_dict, exam_dict, empty_paths):
        if exam_dict["days_to_event"] < -1:
            return True
        # check if screen is localizer screen or not enough images
        if self.is_localizer(series_dict):
            return True

        if len(series_dict["paths"]) < self.args.min_num_images:
            return True
        
        if any([ p.replace("CGMH_LDCT", "ldct_pngs").replace(".dcm", ".png") in empty_paths for p in series_dict["paths"] ]):
            return True
        
        #if len(set([float(s[-1]) for s in series_dict["slice_position"]])) == 1:
        if len(set(series_dict["slice_position"])) == 1:
            return True

        slice_thickness = series_dict['slice_thickness']
        wrong_thickness = (self.args.slice_thickness_filter is not None) and (slice_thickness > self.args.slice_thickness_filter or (slice_thickness < 0))
        if wrong_thickness:
            return True
        return False

    def get_volume_dict(self, series_dict, exam_dict, mrn_row):

        img_paths = series_dict["paths"]
        img_paths = [
            p.replace("CGMH_LDCT", "ldct_pngs").replace(".dcm", ".png")
            for p in img_paths
        ]
        slice_locations = [float(s) for s in series_dict["slice_position"]]

        sorted_img_paths, sorted_slice_locs = self.order_slices(
            img_paths, slice_locations
        )

        y, y_seq, y_mask, time_at_event = self.get_label(exam_dict, mrn_row)

        series_id = series_dict["seriesid"]
        sample = {
            "paths": sorted_img_paths,
            "slice_locations": sorted_slice_locs,
            "y": int(y),
            "time_at_event": time_at_event,
            "y_seq": y_seq,
            "y_mask": y_mask,
            "exam": "{}{}{}".format(
                mrn_row["pid"][-5:] ,
                exam_dict["examid"][-5:],
                    series_id.replace(".", "")[-5:],
                ),
            "study": exam_dict["examid"],
            "series": series_id,
            "pid": mrn_row["pid"],
            "pixel_spacing": series_dict['pixel_spacing'] + [series_dict['slice_thickness'] ],
            "manufacturer": exam_dict["manufacturer"]
        }

        if self.args.fit_to_length:
            sample["paths"] = fit_to_length(sorted_img_paths, self.args.num_images)
            sample["slice_locations"] = fit_to_length(
                sorted_slice_locs, self.args.num_images, "<PAD>"
            )

        return sample

    def get_label(self, exam_dict, mrn_row):
        assert exam_dict["cancer"] in ["lung", "none", "other"]
        is_cancer_cohort = exam_dict["cancer"] == "lung"
        days_to_event = exam_dict["days_to_event"]

        y = False
        if is_cancer_cohort and (not np.isnan(days_to_event)) and (days_to_event > -1):
            years_to_cancer = int(days_to_event // 365)
            y = years_to_cancer < self.args.max_followup

        y_seq = np.zeros(self.args.max_followup)

        if y:
            time_at_event = years_to_cancer
            y_seq[years_to_cancer:] = 1
        else:
            if is_cancer_cohort:
                assert (days_to_event < 0) or (
                    years_to_cancer >= self.args.max_followup
                )
                time_at_event = self.args.max_followup - 1
            else:
                assert days_to_event > -1, "Days to last negative followup is < 0"
                years_to_last_neg_followup = days_to_event // 365
                time_at_event = min(
                    years_to_last_neg_followup, self.args.max_followup - 1
                )

        y_mask = np.array(
            [1] * (time_at_event + 1)
            + [0] * (self.args.max_followup - (time_at_event + 1))
        )
        y_mask = y_mask[: self.args.max_followup]
        return y, y_seq.astype("float64"), y_mask.astype("float64"), time_at_event

    def is_localizer(self, series_dict):
        is_localizer = "LOCALIZER" in series_dict["ImageType"]
        return is_localizer
    
    @staticmethod
    def set_args(args):
        args.num_classes = args.max_followup

    def get_summary_statement(self, dataset, split_group):
        summary = "Constructed CGHM CT Cancer Survival {} dataset with {} records, {} exams, {} patients, and the following class balance \n {}"
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
        statement += "\n" + "Cancer Censor Times: {}".format(
            Counter([d["time_at_event"] for d in dataset if d['y']])
        )
        annotation_msg = (
            self.annotation_summary_msg(dataset) if self.args.use_annotations else ""
        )
        statement += annotation_msg
        return statement

    def assign_splits(self, meta):
        for idx in range(len(meta)):
            meta[idx]["split"] = np.random.choice(
                ["train", "dev", "test"], p=self.args.split_probs
            )

@register_object("cgmh_exclude_other", "dataset")
class CGMH_ExcludeOther(CGMH_Dataset):
    def skip_sample(self, series_dict, exam_dict, empty_paths):
        if super().skip_sample(series_dict, exam_dict, empty_paths):
            return True
        if exam_dict["cancer"] == "other":
            return True
        return False
