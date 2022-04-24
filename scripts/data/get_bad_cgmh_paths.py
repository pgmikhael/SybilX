import os
from tqdm import tqdm 
from p_tqdm import p_umap
import argparse
import json 
import pickle

def check_path(p):
    if not os.path.exists(p):
        return p
    return

def get_paths(dataset):
    paths = []
    for mrnrow in tqdm(dataset, total = len(dataset), desc = "Collecting paths"):
        exams = mrnrow["exams"]
        for exam_dict in exams:
            for series_dict in exam_dict["series"]:
                paths.extend([p.replace("CGMH_LDCT", "ldct_pngs").replace(".dcm", ".png") for p in series_dict['paths']])
    return paths

parser = argparse.ArgumentParser(description="Dispatcher.")
parser.add_argument(
    "--dataset_path",
    type=str,
    required=True,
    help="path to dataset json file",
)

if __name__ == "__main__":
    args = parser.parse_args()
    dataset = json.load(open(args.dataset_path, 'r'))
    paths = get_paths(dataset)
    notfound_paths = p_umap(check_path, paths)
    print()
    pickle.dump(notfound_paths, open('/home/peter/empty_scans.p', 'wb'))
