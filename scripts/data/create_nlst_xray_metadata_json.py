"""
Create X-Ray dataset
"""
import json
import os
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
import pydicom

SPLIT_PROBS = [0.7, 0.15, 0.15]


parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_json_path",
    type=str,
    default="/Mounts/rbg-storage1/datasets/ACRIN_XRAY/xray_dataset.json",
)
parser.add_argument(
    "--data_dir", type=str, default="/Mounts/rbg-storage1/datasets/ACRIN_XRAY"
)
parser.add_argument(
    "--nlst_metadata_csv",
    type=str,
    default="/Mounts/rbg-storage1/datasets/NLST/package-nlst-564.2020-01-30/NLST_564/nlst_564.delivery.010220/nlst_564_prsn_20191001.csv",
)

if __name__ == "__main__":
    args = parser.parse_args()

    dicoms = []
    for root, _, files in os.walk(args.data_dir):
        dicoms.extend([os.path.join(root, f) for f in files if f.endswith(".dcm")])

    meta_data = pd.read_csv(args.nlst_metadata_csv, low_memory=True)
    meta_data.fillna(-1, inplace=True)

    def make_metadata_dict(
        dataframe,
        pid,
    ):

        df = dataframe.loc[(dataframe.pid == int(pid))]
        if df.shape[0] > 0:
            return df.to_dict("list")
        else:
            return {}

    json_dataset = []
    pid2idx = {}
    peid2idx = {}
    for path in tqdm(dicoms):
        dcm_meta = pydicom.dcmread(path, stop_before_pixels=True)

        dcm_keys = list(dcm_meta.keys())
        if (
            ("PatientID" not in dcm_keys)
            or ("StudyDate" not in dcm_keys)
            or ("AccessionNumber" not in dcm_keys)
            or ("ClinicalTrialTimePointID" not in dcm_keys)
        ):
            continue
        pid = dcm_meta.PatientID
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
                "pt_metadata": make_metadata_dict(meta_data, pid),
            }
            pt_dict["accessions"][0]["image_series"] = [img_dict]

            json_dataset.append(pt_dict)

    json.dump(json_dataset, open(args.output_json_path, "w"))
