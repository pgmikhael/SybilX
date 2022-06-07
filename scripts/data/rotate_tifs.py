"""
Create X-Ray dataset
"""
import json
import argparse
from pathlib import Path
import os
import cv2
import skimage.io
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.svm import SVC

parser = argparse.ArgumentParser()
parser.add_argument('--tifs_dirs', type = Path, nargs='*', default="")
parser.add_argument('--rotate_method', type = str, choices=['consensus', 'quadrant','cxrlc'], default='consensus')
parser.add_argument('--replace_pattern', type = str, nargs=2)

mean_img = None
consensus_svm = None
def clf_consensus(img, args):
    global mean_img, consensus_svm
    if mean_img is None:
        df = pd.read_csv("/Mounts/rbg-storage1/datasets/PLCO_XRAY/rotations_ludvig.csv")
        annotated_filenames = df.filename.tolist()
        annotations = df.y.tolist()
        tifs = []
        for path in args.tifs_dirs:
            tifs.extend(path.glob('*.tif'))
        annotated_paths = [path for path in tifs if path.name in annotated_filenames]
        up_images = [skimage.io.imread(str(annotated_paths[i]), plugin='tifffile') for i in range(len(annotated_paths)) if annotations[i] == 0]
        up_images = [cv2.resize(img, (5,5)) for img in up_images]
        up_images = np.stack(up_images)
        mean_img = up_images.mean(axis=0)

        left_images = [skimage.io.imread(str(annotated_paths[i]), plugin='tifffile') for i in range(len(annotated_paths)) if annotations[i] == 1]
        left_images = [cv2.resize(img, (5,5)) for img in left_images]

        X_train = np.stack([(mean_img - img).reshape(-1) for img in up_images] + [(mean_img - img).reshape(-1) for img in left_images])
        y_train = np.array([0]*len(up_images) + [1]*len(left_images))

        consensus_svm = SVC()
        consensus_svm.fit(X_train, y_train)
        
    img = cv2.resize(img, (5,5))
    return consensus_svm.predict((mean_img - img).reshape((1, -1))).item()

def clf_simple(img):
    h,w = img.shape
    q2 = img[:h//2, w//2:]
    q3 = img[h//2:, :w//2]
    
    q2_counts, q2_intensities = np.histogram(q2, bins=np.unique(q2))
    q3_counts, q3_intensities = np.histogram(q3, bins=np.unique(q3))
    
    q2_14_count = (q2 == 14).sum()
    q3_14_count = (q3 == 14).sum()
    
    q2_check = q2_14_count > 200_000
    q3_check = q3_14_count < 250_000

    #is_left = q2_check and q3_check

    #is_ambiguous = not is_left and (q2_check or q3_check)

    #return is_left or is_ambiguous
    return q2_check or q3_check

if __name__ == "__main__":
    args = parser.parse_args()

    tifs = []
    for path in args.tifs_dirs:
        tifs.extend(path.glob('*.tif'))

    #print("Loading", args.rotated_csv)
    #rotated_filenames = pd.read_csv(args.rotated_csv).iloc[:,0]
    #rotated_filenames = set(filename.replace(".png",".tif") for filename in rotated_filenames)

    loadingbar = tqdm(tifs)
    for path in loadingbar:
        loadingbar.set_description(f"Processing {path}")
        filename = path.name

        #if filename in rotated_filenames:
        #    rotation = -90
        #else:
        #    rotation = 0
        
        new_filename = str(path.absolute()).replace(args.replace_pattern[0], args.replace_pattern[1])
        if not os.path.exists(new_filename):
            img = skimage.io.imread(str(path), plugin='tifffile')
            if not img.shape[0] > 0 :
                print(f"'None' image, path: {path}")

            if args.rotate_method == 'consensus':
                try:
                    is_rotated = clf_consensus(img, args)
                except:
                    print(f"Consensus failed. Image path: {path}")
                    is_rotated = False
            elif args.rotate_method == 'quadrant':
                is_rotated = clf_simple(img)
            else:
                assert args.rotate_method == 'cxrlc'
                assert False, "TODO"

            if is_rotated and img is not None:
                try:
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                except Exception as e:
                    print(f"Could not rotate image due to '{e}', path: {path}") 

            par_dir = Path(new_filename).parent
            if not par_dir.exists():
                par_dir.mkdir(parents=True)
            try:
                if img.shape[0] > 0:
                    cv2.imwrite(new_filename, img)
                else:
                    raise Exception("Image is empty")
            except:
                print(f"Image save failed, image path: {path}")

