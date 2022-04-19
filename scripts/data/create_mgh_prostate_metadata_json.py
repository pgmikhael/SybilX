"""
Create prostate metadata json. Following create_nlst_xray_metadata_json.py
"""
import json
import os
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
import pydicom

import time 

SPLIT_PROBS = [0.7, 0.15, 0.15]

dcmTagsToNames = {
    "PatientID": 0x00100020,
    "StudyDate": 0x00080020,
    "AccessionNumber": 0x00080050,
    "ClinicalTrialTimePointID": 0x00120050,
    "SeriesInstanceUID": 0x0020000e,
    "SOPInstanceUID": 0x00080018,
    "ImageType": 0x00080008,
    "InstanceNumber": 0x00200013,
    "WindowCenter": 0x00281050,
    "WindowWidth": 0x00281051,
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_json_path",
    type=str,
    default="/Mounts/rbg-storage1/datasets/MGH_Prostate_Salari/prostate_dataset.json",
)
parser.add_argument(
    "--data_dir", type=str, default="/Mounts/rbg-storage1/datasets/MGH_Prostate_Salari"
)

if __name__ == "__main__":
    print("in name main section of code", flush=True)
    args = parser.parse_args()

    dicoms = []
    print("parsed args")
    i = 0
    for root, _, files in os.walk(args.data_dir):
        if i % 20 == 0:
            print("walk iteration ", i)
        if i > 100:
            break
        dicoms.extend([os.path.join(root, f) for f in files if f.endswith(".dcm")])
        i += 1

    json_dataset = []
    pid2idx = {}
    peid2idx = {}
    print("collected dicoms, size is: ", len(dicoms))
    i = 0
    for path in tqdm(dicoms):
        if i % 20 == 0:
            print("on dicom ", i)
        i += 1

        dcm_meta = pydicom.dcmread(path, stop_before_pixels=True)

        dcm_keys = list(dcm_meta.keys())
        skipped = 0
        if (
            (dcmTagsToNames["PatientID"] not in dcm_keys)
            or (dcmTagsToNames["StudyDate"] not in dcm_keys)
            or (dcmTagsToNames["AccessionNumber"] not in dcm_keys)
            or (dcmTagsToNames["ClinicalTrialTimePointID"] not in dcm_keys)
        ):
            skipped += 1
            print("missing keys, skipped: ", skipped)
            continue
        pid = dcm_meta[dcmTagsToNames["PatientID"]]
        date = dcm_meta[dcmTagsToNames["StudyDate"]]
        accession_number = dcm_meta[dcmTagsToNames["AccessionNumber"]]
        timepoint = int(dcm_meta[dcmTagsToNames["ClinicalTrialTimePointID"]][-1])  # convert from 'T1' to 1
        exam = "{}_T{}".format(accession_number, timepoint)
        series_id = dcm_meta["SeriesInstanceUID"]
        sop_id = dcm_meta["SOPInstanceUID"]

        peid = "{}{}".format(pid, exam)

        exam_dict = {
            "exam": exam,
            "accession_number": accession_number,
            "screen_timepoint": timepoint,
            "date": date,
        }

        img_dict = {
            "path": path,
            "image_type": dcm_meta["ImageType"],
            "sop_id": sop_id,
            "series_id": series_id,
            "instance_number": dcm_meta["InstanceNumber"],
            "window_center": dcm_meta["WindowCenter"],
            "window_width": dcm_meta["WindowWidth"],
        }

        if pid in pid2idx:
            pt_idx = pid2idx[pid]

            if peid in peid2idx:
                exam_idx = peid2idx[peid]
                json_dataset[pt_idx]["accessions"][exam_idx]["image_series"].append(
                    img_dict
                )

            else:
                peid2idx[peid] = len(json_dataset[pt_idx]["accessions"])
                exam_dict["image_series"] = [img_dict]
                json_dataset[pt_idx]["accessions"].append(exam_dict)

        else:
            pid2idx[pid] = len(json_dataset)

            pt_dict = {
                "accessions": [exam_dict],
                "pid": pid,
                "split": np.random.choice(["train", "dev", "test"], p=SPLIT_PROBS),
            }
            pt_dict["accessions"][0]["image_series"] = [img_dict]

            json_dataset.append(pt_dict)

    print("parsed info, about to dump into dataset: ", len(json_dataset))
    json.dump(json_dataset, open(args.output_json_path, "w"))
