"""
Create X-Ray dataset
"""
import json
import argparse
from pathlib import Path

import exifread
import numpy as np
import pandas as pd
from tqdm import tqdm

DEFAULT_SPLIT_PROBS = [0.7, 0.15, 0.15]

parser = argparse.ArgumentParser()
parser.add_argument('--tifs_dir', type = Path, default="")
parser.add_argument('--link_csv', type = Path)
parser.add_argument('--person_csv', type = Path)
parser.add_argument('--output_json_path', type = Path)
parser.add_argument('--error_json_path', type = Path, default = None)
parser.add_argument('--split_probs', type = int, nargs = 3, default = DEFAULT_SPLIT_PROBS)

if __name__ == "__main__":
    args = parser.parse_args()

    tifs = list(args.tifs_dir.glob('**/*.tif'))

    print("Loading", args.link_csv)
    filename_to_id_map_df = pd.read_csv(args.link_csv)
    filename_to_id_map = filename_to_id_map_df.set_index('image_file_name').to_dict('index')

    print("Loading", args.person_csv)
    lung_persons = pd.read_csv(args.person_csv)
    lung_persons.fillna(-1, inplace=True)
    id_to_metadata_map = lung_persons.set_index('plco_id').to_dict('index')

    json_dataset = []
    pid2idx = {}
    peid2idx = {}
    loadingbar = tqdm(tifs)
    for path in loadingbar:
        loadingbar.set_description(f"Processing {path}")
        filename = path.name

        pid = filename_to_id_map[filename]['plco_id']
        years_from_baseline = filename_to_id_map[filename]['assoc_visit_syr']
        visit_num = filename_to_id_map[filename]['assoc_visit_visnum']
        # couldn't find: date, series_id, sop_id

        exam = '{}_T{}'.format(visit_num, years_from_baseline)
        peid = "{}{}".format(pid, exam)

        exam_dict = {
            "exam": exam,
            "visit_num": visit_num,
            "study_yr": years_from_baseline,
            #"accession_number": accession_number
            #"date": date
        }

        img_dict = {
            "path": str(path.absolute()),
            "filename": filename
        }
        # tif image metadata
        #with open(str(path), 'rb') as f:
        #    tags = exifread.process_file(f)
        #    for tag in tags.keys():
        #        if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 
        #                       'EXIF MakerNote', 'Image DateTime'): # DateTime is date of image-digitization, not of scan
        #            value = tags[tag].values
        #            if type(value[0]) not in (str, int, float, bool):
        #                continue
        #            if len(value) == 1:
        #                value = value[0]

        #            img_dict[tag] = value


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

            patient_metadata = id_to_metadata_map[pid]

            pt_dict = {
                "accessions": [exam_dict],
                "pid": pid,
                "split": np.random.choice(["train", "dev", "test"], p=args.split_probs),
                "pt_metadata": patient_metadata
                }
            pt_dict["accessions"][0]["image_series"] = [img_dict]

            json_dataset.append(pt_dict)

    json.dump(list(json_dataset), open(args.output_json_path, "w"))
