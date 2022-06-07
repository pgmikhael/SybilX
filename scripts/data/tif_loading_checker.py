import json
import argparse
from pathlib import Path
import os
import cv2
import skimage.io
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--tifs_dirs', type = Path, nargs='*', default="")


if __name__ == "__main__":
    args = parser.parse_args()

    tifs = []
    for path in args.tifs_dirs:
        tifs.extend(path.glob('*.tif'))

    loadingbar = tqdm(tifs[26743:])
    for path in loadingbar:
        loadingbar.set_description(f"Processing {path}")
        filename = path.name
        
        img = skimage.io.imread(str(path), plugin='tifffile')
        if not img.shape[0] > 0 :
            print(f"'None' image, path: {path}")
