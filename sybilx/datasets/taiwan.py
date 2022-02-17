import numpy as np
from tqdm import tqdm
from ast import literal_eval
import copy
from sybilx.datasets.nlst import NLST_Survival_Dataset
from collections import Counter
from sybilx.datasets.utils import fit_to_length, get_scaled_annotation_area
from sybilx.utils.registry import register_object


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

        dataset = []

        for mrn_row in tqdm(self.metadata_json):

            pid, exams = mrn_row["pid"], mrn_row["exams"]

            for exam_dict in exams:

                for series_dict in exam_dict["series"]:
                    if self.skip_sample(series_dict, exam_dict):
                        continue

                    sample = self.get_volume_dict(series_dict, exam_dict, mrn_row)
                    if len(sample) == 0:
                        continue

                    dataset.append(sample)

        return dataset

    def skip_sample(self, series_dict, exam_dict):

        # check if screen is localizer screen or not enough images
        if self.is_localizer(series_dict):
            return True

        if len(series_dict["paths"]) < self.args.min_num_images:
            return True

        return False

    def get_volume_dict(self, series_dict, exam_dict, mrn_row):

        img_paths = series_dict["paths"]
        img_paths = [
            p.replace("CGMH_LDCT", "ldct_pngs").replace(".dcm", ".png")
            for p in img_paths
        ]
        slice_locations = series_dict["slice_position"]

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
            "exam": int(
                "{}{}{}".format(
                    mrn_row["pid"],
                    exam_dict["examid"],
                    series_id.replace(".", "")[-5:],
                )
            ),  # last 5 of study id + last 5 of series id
            "series": series_id,
            "pid": mrn_row["pid"],
        }

        if not self.args.use_all_images:
            sample["paths"] = fit_to_length(sorted_img_paths, self.args.num_images)
            sample["slice_locations"] = fit_to_length(
                sorted_slice_locs, self.args.num_images, "<PAD>"
            )

        return sample

    def get_label(self, exam_dict, mrn_row):

        is_cancer_cohort = exam_dict["cancer"]
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
        is_localizer = "LOCALIZER" in literal_eval(series_dict["ImageType"])
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

    def get_images(self, paths, sample):
        """
        Returns a stack of transformed images by their absolute paths.
        If cache is used - transformed images will be loaded if available,
        and saved to cache if not.
        """
        out_dict = {}
        if self.args.fix_seed_for_multi_image_augmentations:
            sample["seed"] = np.random.randint(0, 2**32 - 1)

        # get images for multi image input
        input_dicts = [
            self.input_loader.get_image(path, sample) for e, path in enumerate(paths)
        ]

        images = [i["input"] for i in input_dicts]
        out_dict["input"] = self.reshape_images(images)
        out_dict["mask"] = None

        return out_dict
