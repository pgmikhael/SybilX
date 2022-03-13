import json
import os
import pickle
import numpy as np
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import ijson
# import ijson
import time
from collections import defaultdict
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
    "PixelSpacing",
]

# keys to extract from the metadata csv
# The following should possibly be move to the series_metakeys: download, IV_contrast, comment_AT
EXAM_METAKEYS = [
    "patient_ID", 
    "bridge_uid", 
    "patient_IDs used in cohort 1",
    "LR Score",
    "Smodifier",
    "Lung RadsS Mod Findings",
    "NonACRS Mod",
    "Lung RadsS Mod Mass",
    "Other Interstitial Lung Disease",  
    "Other Interstitial Lung Disease Specified",    
    "exam Status",
    "indication",   
    "Smoking Status",   
    "Year Since Last Smoked",
    "Smoking Cessation Guidance Provided",
    "Packs Years",
    "gender",
    "age at the exam",
    "language",
    "race",
    "marital_status",
    "religion",
    "exam Year",    
    "procedurecode",
    "proceduredesc",
    "number of days after the oldest study of the patient",
    "Study Description",
    "Study Instance UID",   
    "Future_cancer",
    "days_before_cancer_dx",
    "days_to_last_follow_up",   
    "Age at Diagnosis",
    "Date of Initial Diagnosis",
    "Primary Site",
    "Laterality",
    "Histology/Behavior ICD-O-2",
    "Vital Status"
]
SERIES_METAKEYS = [
    "series downloaded"
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
    "--output_json_path",
    type=str,
    default="/Mounts/rbg-storage1/datasets/MGH_Lung_Fintelmann/mgh_lung_cancer_cohort2v2.json",
)
parser.add_argument(
    "--metadata_csv_path",
    type=str,
    default="/Mounts/rbg-storage1/datasets/MGH_Lung_Fintelmann/meta012022/MIT_LDCT_key_sheet_anonymized_update_with_days_before_cancer_dx_and_last_follow_up_v3.csv",
)
parser.add_argument(
    "--png_path_replace_pattern", type=str, nargs=2, default=["MIT_Lung_Cancer_Screening", "screening_pngs"]
)
parser.add_argument("--error_json_path", type=str, default=None)


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
    args = parser.parse_args()

    errors = {}

    find_pattern, replace_with = (
        args.png_path_replace_pattern[0],
        args.png_path_replace_pattern[1],
    )

    # Dataset json, create new or update existing
    if os.path.exists(args.output_json_path):
        json_dataset = json.load(open(args.output_json_path, "r"))
        pid_list = [d["pid"] for d in json_dataset]
    else:
        json_dataset, pid_list = [], []
    processed_len = len(pid_list)

    patient_metadata = pd.read_csv(args.metadata_csv_path)
    # patient_metadata.fillna(-1, inplace = True)
    study_instance_uids = set(patient_metadata['Study Instance UID'] )

    # Source json, output of OncoData ..
    print("> loading json")
    start = time.time()
    #source_json = json.load(open(args.source_json_path, "r"))
    source_json = ijson.items(open(args.source_json_path, 'rb'), prefix='item') # streaming - slower, but skips initial wait which is better for debugging
    print("Took", time.time() - start, "seconds")

    for row_dict in tqdm(
        source_json, desc="Processing metadata from {}".format(args.source_json_path)
    ):
        path_sections = row_dict["dicom_path"].split("/")
        filename = path_sections[-1]

        if filename.startswith("."):
            log_error(errors, "Skipped file starting with '.'", filename=filename)
            continue

        dcm_keys = row_dict["dicom_metadata"].keys()

        path = (
            row_dict["dicom_path"]
            .replace(find_pattern, replace_with)
            .replace(".dcm", ".png")
        )

        assert os.path.exists(path), f"Could not find path {path}!"

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

        # select specific columns
        exam_meta_rows = exam_meta_rows.loc[:, EXAM_METAKEYS]
        # TODO: Ensure that exam keys have the same values in all rows

        exam_dict = extract_from_dict(
            row_dict["dicom_metadata"], EXAM_DICOMKEYS, replace_missing_with=-1
        )
        exam_dict.update(extract_from_series(exam_meta_rows.iloc[0], EXAM_METAKEYS))

        pid = exam_dict["bridge_uid"]

        ## series data
        series_dict = extract_from_dict(
            row_dict["dicom_metadata"], SERIES_DICOMKEYS, replace_missing_with=-1
        )

        img_series_dict = {
            "paths": [path],
            "slice_location": [slice_location],
            "image_posn": [image_posn],
            "series_data": series_dict
            #'slice_number': # filled in by post-processing step
        }

        if pid in pid_list:
            # if we have already inserted an exam for a patient
            pt_idx = pid_list.index(pid)
            existing_exams = [
                exam["studyinstance_uid"] for exam in json_dataset[pt_idx]["accessions"]
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
            }

            json_dataset.append(pt_dict)
            pid_list.append(pid)

    for pt_idx, pt_dict in tqdm(enumerate(json_dataset), desc="Post-Processing"):
        cancer_cohort_yes_no = []
        diff_days = []
        diff_days_diagnosis = []
        for exam_idx, exam_dict in enumerate(pt_dict["accessions"]):
            for series_id in json_dataset[pt_idx]["accessions"][exam_idx][
                "image_series"
            ].keys():
                # calculate slice numbering
                slice_locations = json_dataset[pt_idx]["accessions"][exam_idx][
                    "image_series"
                ][series_id]["slice_location"]
                # TODO: should maybe be sorted in reverse direction instead
                json_dataset[pt_idx]["accessions"][exam_idx]["image_series"][series_id][
                    "slice_number"
                ] = np.argsort(slice_locations).tolist()

    json.dump(json_dataset, open(args.output_json_path, "w"))
    print("Saved output json to ", args.output_json_path)

    print("Errors counts:")
    for error_category in errors.keys():
        print("- {}: {}".format(error_category, len(errors[error_category])))

    if args.error_json_path is not None:
        json.dump(errors, open(args.error_json_path, "w"))
        print("Saved error summary to", args.error_json_path)
