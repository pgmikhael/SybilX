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
    start = time.time()
    for path in tqdm(dicoms):
        if i % 20 == 0:
            print("on dicom ", i)
            now = time.time()
            print(now - start, " time has passed since starting")
        i += 1

        dcm_meta = pydicom.dcmread(path, stop_before_pixels=True)

        dcm_keys = list(dcm_meta.keys())
        print("dcm_keys: ", dcm_keys)
        if (0x00100020 in dcm_keys):
            print("yay patient id spotted")
        elif (
            ("PatientID" not in dcm_keys)
            or ("StudyDate" not in dcm_keys)
            or ("AccessionNumber" not in dcm_keys)
            or ("ClinicalTrialTimePointID" not in dcm_keys)
        ):
            print("missing keys, skipped")
            continue
        try:
            pid = dcm_meta.PatientID
        except IndexError:
            pid = dcm_meta[0x0010020]
        print("has PatientID: ", pid)
        break
        date = dcm_meta.StudyDate
        accession_number = dcm_meta.AccessionNumber
        timepoint = int(dcm_meta.ClinicalTrialTimePointID[-1])  # convert from 'T1' to 1
        exam = "{}_T{}".format(accession_number, timepoint)
        series_id = dcm_meta.SeriesInstanceUID
        sop_id = dcm_meta.SOPInstanceUID

        peid = "{}{}".format(pid, exam)

        exam_dict = {
            "exam": exam,
            "accession_number": accession_number,
            "screen_timepoint": timepoint,
            "date": date,
        }

        img_dict = {
            "path": path,
            "image_type": dcm_meta.ImageType,
            "sop_id": sop_id,
            "series_id": series_id,
            "instance_number": dcm_meta.InstanceNumber,
            "window_center": dcm_meta.WindowCenter,
            "window_width": dcm_meta.WindowWidth,
            "kvp": dcm_meta.KVP,
            "exposure_time": dcm_meta.ExposureTime,
            "exposure": dcm_meta.Exposure,
            "acquisition_desc": dcm_meta.AcquisitionDeviceProcessingDescription,
            "tube_current": dcm_meta.XRayTubeCurrent,
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
