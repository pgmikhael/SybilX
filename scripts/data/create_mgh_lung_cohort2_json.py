import json
import os
import numpy as np
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import time
from ast import literal_eval

"""
Script to create MGH Lung dataset json.

python scripts/dicom_to_png/dicom_to_png.py --dicom_dir /Mounts/rbg-storage1/datasets/MGH_Lung_Fintelmann/dicoms/ --png_dir /Mounts/rbg-storage1/datasets/MGH_Lung_Fintelmann/pngs/ --dcmtk --dicom_types generic --window

python scripts/dicom_metadata/dicom_metadata_to_json.py --directory /Mounts/rbg-storage1/datasets/MGH_Lung_Fintelmann/dicoms --results_path /Mounts/rbg-storage1/datasets/MGH_Lung_Fintelmann/dicom_metadata.json
"""

SPLIT_PROBS = [0.7, 0.15, 0.15]

# keys to extract from the DICOM files
EXAM_DICOMKEYS = ["StudyDescription", "StudyInstanceUID"]
SERIES_DICOMKEYS = [
    "SeriesDescription",
    "Manufacturer",
    "ManufacturerModelName",
    "WindowCenter",
    "WindowWidth",
    "SliceThickness",
    "ImageType",
]


def extract_from_dict(dictionary, keys, replace_missing_with=None):
    """Creates a new dict containing only the given keys."""
    temp_dict = {}
    for key in keys:
        temp_dict[key] = dictionary.get(key, replace_missing_with)
    return temp_dict


def extract_from_series(series, keys, replace_missing_with=None):
    """Creates a dict from a pandas Series containing only the given keys."""
    temp_dict = {}
    for key in keys:
        val = series[key]
        if val == np.nan:
            val = replace_missing_with
        elif isinstance(val, np.generic):
            # convert numpy types to Python types
            val = val.item()
        temp_dict[key] = val
    return temp_dict


parser = argparse.ArgumentParser()
parser.add_argument(
    "--source_json_path",
    type=str,
    default="/Mounts/rbg-storage1/datasets/MGH_Lung_Fintelmann/meta012022/mgh_screening_metadata.json",
)
parser.add_argument(
    "--cohort1_path",
    type=str,
    default="/Mounts/rbg-storage1/datasets/MGH_Lung_Fintelmann/mgh_metadata.json",
)
parser.add_argument(
    "--output_json_path",
    type=str,
    default="/Mounts/rbg-storage1/datasets/MGH_Lung_Fintelmann/mgh_lung_cancer_cohort2.json",
)
parser.add_argument(
    "--metadata_csv_path",
    type=str,
    default="/Mounts/rbg-storage1/datasets/MGH_Lung_Fintelmann/meta012022/MIT_LDCT_key_sheet_anonymized_updated_with_days_before_cancer_dx.csv",
)
parser.add_argument(
    "--png_path_replace_pattern",
    type=str,
    nargs=2,
    default=["MIT_Lung_Cancer_Screening", "screening_pngs"],
)
parser.add_argument(
    "--error_json_path",
    type=str,
    default="/Mounts/rbg-storage1/datasets/MGH_Lung_Fintelmann/meta012022/error.log",
)


def log_error(errors, category, **args):
    error_dict = args
    print(
        "{} ({})".format(
            category, ", ".join([key + "=" + args[key] for key in args.keys()])
        )
    )
    if category in errors:
        errors[category].append(error_dict)
    else:
        errors[category] = [error_dict]


if __name__ == "__main__":
    np.random.seed(2022)

    args = parser.parse_args()

    cohort1_data = json.load(open(args.cohort1_path, "r"))
    cohort1_pids = {d["pid"]: True for d in cohort1_data}

    errors = {}

    find_pattern, replace_with = (
        args.png_path_replace_pattern[0],
        args.png_path_replace_pattern[1],
    )

    patient_metadata = pd.read_csv(args.metadata_csv_path)
    # patient_metadata.fillna(-1, inplace = True)
    study_instance_uids = set(patient_metadata["Study Instance UID"])

    # Source json, output of OncoData ..
    print("> loading json")
    start = time.time()
    source_json = json.load(open(args.source_json_path, "r"))
    # source_json = ijson.items(open(args.source_json_path, 'rb'), prefix='item') # streaming - slower, but skips initial wait which is better for debugging
    print("Took", time.time() - start, "seconds")

    json_dataset, pid_list = [], []
    for row_dict in tqdm(
        source_json, desc="Processing metadata from {}".format(args.source_json_path)
    ):
        path_sections = row_dict["dicom_path"].split("/")
        filename = path_sections[-1]

        if filename.startswith("."):
            log_error(errors, "Skipped file starting with '.'", filename=filename)
            continue

        path = (
            row_dict["dicom_path"]
            .replace(find_pattern, replace_with)
            .replace(".dcm", ".png")
        )

        if not os.path.exists(path):
            continue 

        if not row_dict["dicom_metadata"].get("SeriesInstanceUID", False):
            log_error(
                errors, "Series Instance UID not present in DICOM metadata!", path=path
            )
            continue

        series_id = row_dict["dicom_metadata"]["SeriesInstanceUID"]
        study_instance_uid = row_dict["dicom_metadata"]["StudyInstanceUID"]
        if study_instance_uid not in study_instance_uids:
            log_error(
                errors,
                "Study Instance UID not present in Patient metadata CSV!",
                studyuid=study_instance_uid,
                path=path,
            )
            continue

        slice_location = float(row_dict["dicom_metadata"].get("SliceLocation", -1))
        image_posn = float(
            literal_eval(
                row_dict["dicom_metadata"].get("ImagePositionPatient", "[-1]")
            )[-1]
        )

        # match metadata by StudyInstanceUID
        exam_meta_rows = patient_metadata[
            patient_metadata["Study Instance UID"] == study_instance_uid
        ]
        if len(exam_meta_rows) == 0:
            log_error(
                errors,
                "No rows with given studyuid found in metadata CSV!",
                studyuid=study_instance_uid,
                path=path,
            )
            continue
        elif len(exam_meta_rows) > 1:
            log_error(
                errors,
                "Multiple rows with given studyuid found in metadata CSV! Choosing first!",
                studyuid=study_instance_uid,
                path=path,
            )

        exam_dict = exam_meta_rows.iloc[0].to_dict()

        exam_dict.update(
            extract_from_dict(
                row_dict["dicom_metadata"], EXAM_DICOMKEYS, replace_missing_with=-1
            )
        )

        pid = exam_dict["bridge_uid"]

        ## series data
        series_dict = extract_from_dict(
            row_dict["dicom_metadata"], SERIES_DICOMKEYS, replace_missing_with=-1
        )

        img_series_dict = {
            "paths": [path],
            "slice_location": [slice_location],
            "image_posn": [image_posn],
            "series_data": series_dict,
        }

        if pid in pid_list:
            # if we have already inserted an exam for a patient
            pt_idx = pid_list.index(pid)
            existing_exams = [
                exam["StudyInstanceUID"] for exam in json_dataset[pt_idx]["accessions"]
            ]
            # check if the exam for the current image already exists in the data
            if study_instance_uid in existing_exams:
                exam_idx = existing_exams.index(study_instance_uid)
                if series_id not in list(
                    json_dataset[pt_idx]["accessions"][exam_idx]["image_series"].keys()
                ):
                    # if there are no images for the given image series ID
                    json_dataset[pt_idx]["accessions"][exam_idx]["image_series"][
                        series_id
                    ] = img_series_dict
                elif (
                    path
                    not in json_dataset[pt_idx]["accessions"][exam_idx]["image_series"][
                        series_id
                    ]["paths"]
                ):
                    # if the image series doesn't have the current image (tested by path)
                    # append each value to list in the dict
                    json_dataset[pt_idx]["accessions"][exam_idx]["image_series"][
                        series_id
                    ]["paths"].append(path)
                    json_dataset[pt_idx]["accessions"][exam_idx]["image_series"][
                        series_id
                    ]["slice_location"].append(slice_location)
                    json_dataset[pt_idx]["accessions"][exam_idx]["image_series"][
                        series_id
                    ]["image_posn"].append(image_posn)
            else:
                # if the exam doesn't exists, create a new one
                exam_dict["image_series"] = {series_id: img_series_dict}
                json_dataset[pt_idx]["accessions"].append(exam_dict)

        else:
            # if the patient has no preexisting exams in the data, create one from scratch
            exam_dict["image_series"] = {series_id: img_series_dict}
            pt_dict = {
                "accessions": [exam_dict],
                "pid": pid,
                "split": np.random.choice(["train", "dev", "test"], p=SPLIT_PROBS),
                "in_cohort1": cohort1_pids.get(pid, False),
            }

            json_dataset.append(pt_dict)
            pid_list.append(pid)

    json.dump(json_dataset, open(args.output_json_path, "w"))
    print("Saved output json to ", args.output_json_path)

    print("Errors counts:")
    for error_category in errors.keys():
        print("- {}: {}".format(error_category, len(errors[error_category])))

    if args.error_json_path is not None:
        json.dump(errors, open(args.error_json_path, "w"))
        print("Saved error summary to", args.error_json_path)
