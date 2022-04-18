import json
import os
import pickle
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
import time
import pydicom
from collections import defaultdict
from ast import literal_eval

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_json_path",
    type=str,
    default="/Mounts/rbg-storage1/datasets/NLST/ct_localizers.json",
)
parser.add_argument(
    "--full_nlst_json_path",
    type=str,
    default="/Mounts/rbg-storage1/datasets/NLST/full_nlst_google.json",
)
parser.add_argument(
    "--path_replace_pattern",
    type=str,
    nargs=2,
    default=["/Mounts/rbg-storage1/datasets/NLST/nlst-ct-png","/storage/NLST/nlst-ct"]
)
parser.add_argument(
    "--extension_replace_pattern",
    type=str,
    nargs=2,
    default=[".png",""]
)

def is_localizer(series_dict):
    is_localizer = (
        (series_dict["imageclass"][0] == 0)
        or ("LOCALIZER" in series_dict["imagetype"][0])
        or ("TOP" in series_dict["imagetype"][0])
    )
    return is_localizer

def is_lateral(series_dict):
    num_images = len(series_dict['paths'])
    lateral_count = 0
    for path in series_dict['paths']:
        ds = pydicom.dcmread(path)
        # TODO: here decision is to use arbitrary localizer in the paths if it is frontal
        if ds.ImageOrientationPatient[0] == 0:
            lateral_count += 1
    # all localizer images are lateral, then skip this sample
    if lateral_count == num_images:
        return True
    else:
        return False

if __name__ == "__main__":
    args = parser.parse_args()
    print("Loading json...")
    meta = json.load(open(args.full_nlst_json_path,"r"))
    for p in tqdm(meta):
        for e in p['accessions']:
            for sid, s in e['image_series'].items():
                series_data = s['series_data']
                if not is_localizer(series_data):
                    continue
		
                paths = [ path.replace(args.path_replace_pattern[0], args.path_replace_pattern[1])\
                              .replace(args.extension_replace_pattern[0], args.extension_replace_pattern[1]) 
                          for path in s['paths'] ]
                s['paths'] = paths

                s['is_lateral'] = is_lateral(s)

    print("Saving json...")
    json.dump(meta, open(args.output_json_path, "w"))
