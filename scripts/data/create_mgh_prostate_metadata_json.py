"""
Create prostate metadata json. Following create_nlst_xray_metadata_json.py
"""
import json
import os
import numpy as np
import re
from tqdm import tqdm
import argparse
import pandas as pd
import pydicom

import time 

SPLIT_PROBS = [0.7, 0.15, 0.15]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_json_path",
    type=str,
    default="/Mounts/rbg-storage1/users/janicey/prostate/prostate_dataset.json",
)
parser.add_argument(
    "--data_dir", type=str, default="/storage/prostate"
)
parser.add_argument(
    "--biop_csv",
    type=str,
    default="/Mounts/rbg-storage1/datasets/MGH_Prostate_Salari/reports/biop_output_anon.csv",
)
parser.add_argument(
    "--rad_csv",
    type=str,
    default="/Mounts/rbg-storage1/datasets/MGH_Prostate_Salari/reports/rad_output_anon.csv", # TODO: scp to server, add server path
)

def get_files_from_walk(dir, endings = tuple(), phrases = tuple()):

    def check_endings(f):
        if len(endings) == 0:
            return True

        for end in endings:
            if f.endswith(end):
                return True
        return False
    
    def check_phrases(root):
        if len(phrases) == 0:
            return True

        for phrase in phrases:
            if len(re.findall(phrase, root)) > 0:
                return True
        return False 

    outputs = []
    i = 0 # for logging/debugging purposes
    for root, _, files in os.walk(dir):
        # if i % 100 == 0:
        #     print("walk iteration ", i)
        # if i > 500:
        #     break
        if i%10000 == 0: 
            print("walk iteration: ", i)
            print("dicoms loaded: ", len(outputs))
        outputs.extend([os.path.join(root, f) for f in files if check_endings(f) and check_phrases(root)])
        i += 1
    return outputs

def make_reportdata_dict(
    dataframe,
    mrn_pid,
):
    """Taken from create_nlst_xray_metadat_json.py's make_metadata_dict method."""
    df = dataframe.loc[(dataframe.MRN == int(mrn_pid))]
    if df.shape[0] > 0:
        return df.to_dict("list") # dict where keys are column names, values is column list following row order
    else:
        return {}

if __name__ == "__main__":
    print("in name main section of code", flush=True)
    args = parser.parse_args()

    dicoms = get_files_from_walk(args.data_dir, (".dcm",), (r"T2.*[Aa][Xx](ial)?", r"[aA][Xx](ial)?.*T2"))

    biop_data = pd.read_csv(args.biop_csv, low_memory=True)
    biop_data.fillna(0, inplace=True) # na for gleason scores when benign

    rad_data = pd.read_csv(args.rad_csv, low_memory=True)
    rad_data.replace(to_replace="No Evidence", value=0, inplace=True) # occurs in pirads scores

    json_dataset = []
    pid2idx = {}
    peid2idx = {}
    print("collected dicoms, size is: ", len(dicoms))
    skipped = 0
    for path in tqdm(dicoms):

        dcm_meta = pydicom.dcmread(path, stop_before_pixels=True)

        dcm_keys = list(dcm_meta.keys())
        if (
            ("PatientID" not in dcm_keys)
            or ("StudyDate" not in dcm_keys)
            or ("AccessionNumber" not in dcm_keys)
        ):
            skipped += 1
            print("missing keys, skipped: ", skipped)
            continue
        pid = dcm_meta.PatientID
        study_date = dcm_meta.StudyDate
        accession_number = dcm_meta.AccessionNumber
        exam = "{}".format(accession_number)
        series_id = dcm_meta.SeriesInstanceUID
        sop_id = dcm_meta.SOPInstanceUID
        peid = "{}{}".format(pid, exam)

        try:
            slice_location = float(dcm_meta.get("SliceLocation", -1))
            image_position = [float(pos) for pos in dcm_meta.ImagePositionPatient]
            pixel_spacing = [float(space) for space in dcm_meta.PixelSpacing]
        except:
            continue
        
        exam_dict = {
            "exam": exam,
            "accession_number": accession_number,
            "study_date": study_date,
            "patient_age": dcm_meta.PatientAge,
        }

        series_dict = {
            "sop_id": sop_id,
            "series_id": series_id,
            "series_date": dcm_meta.SeriesDate,
            "series_time": dcm_meta.SeriesTime,
            "series_desc": dcm_meta.SeriesDescription,
            "slice_thickness": dcm_meta.SliceThickness,
            "instance_number": dcm_meta.InstanceNumber,
            "image_position": image_position,
            "window_center": dcm_meta.WindowCenter,
            "window_width": dcm_meta.WindowWidth,
            "pixel_spacing": pixel_spacing,
            # "image_type": dcm_meta.ImageType,
        }

        img_series_dict = {
            "paths": [path],
            "slice_location": [slice_location],
            "series_data": series_dict,
        }

        if pid in pid2idx:
            pt_idx = pid2idx[pid]

            # check if patient's exam already exists in dataset
            if peid in peid2idx:
                exam_idx = peid2idx[peid]
                # check if patient's series already exists in exam data
                if series_id not in list(
                    json_dataset[pt_idx]["accessions"][exam_idx]["image_series"].keys()
                ):
                    json_dataset[pt_idx]["accessions"][exam_idx]["image_series"][series_id] = img_series_dict
                # check if patient's image already exists in series data
                elif (
                    path
                    not in json_dataset[pt_idx]["accessions"][exam_idx]["image_series"][series_id]["paths"]
                ):
                    json_dataset[pt_idx]["accessions"][exam_idx]["image_series"][series_id]["paths"].append(path)
                    json_dataset[pt_idx]["accessions"][exam_idx]["image_series"][series_id]["slice_location"].append(slice_location)
            else:
                # case where exam doesn't exist. create a new one
                peid2idx[peid] = len(json_dataset[pt_idx]["accessions"])
                exam_dict["image_series"] = {series_id: img_series_dict}
                json_dataset[pt_idx]["accessions"].append(exam_dict)

        else:
            pid2idx[pid] = len(json_dataset)

            pt_dict = {
                "accessions": [exam_dict],
                "pid": pid,
                "split": np.random.choice(["train", "dev", "test"], p=SPLIT_PROBS),
                "birth_date": dcm_meta.PatientBirthDate,
                "rad_reports": make_reportdata_dict(rad_data, pid),
                "biop_reports": make_reportdata_dict(biop_data, pid),
            }
            pt_dict["accessions"][0]["image_series"] = {series_id: img_series_dict}

            peid2idx[peid] = 0
            json_dataset.append(pt_dict)

    img_series = json_dataset[0]["accessions"][0]["image_series"]
    
    # # debugging purposes
    # total_paths = 0
    # for series_id in img_series:
    #     num_paths = len(img_series[series_id]["paths"])
    #     print("num paths: ", num_paths)
    #     total_paths += num_paths
    # print("total paths: ", total_paths)

    print("parsed info, about to dump into dataset: ", len(json_dataset))
    json.dump(json_dataset, open(args.output_json_path, "w"))
