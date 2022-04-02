"""Recursively converts every file in a folder from .npy -> .npz 
containing one array with the name "image".
"""
import json
import os
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

import time

parser = argparse.ArgumentParser()
parser.add_argument(
    "source",
    help="Source Folder",
    type=Path,
)

parser.add_argument(
    "--output_suffix",
    help="Override output file extension",
    type=str,
    default='.npz'
)


if __name__ == "__main__":
    args = parser.parse_args()

    print(f"Collecting files...")
    files = list(args.source.glob("**/*.npy"))
    print(f"This will convert {len(files)} files.")

    if input("Are you sure you want to continue? (type yes)") == "yes": 
        loadingbar = tqdm(files)
        for path in loadingbar:
            loadingbar.set_description(str(path))
            img = np.load(str(path))

            new_path = path.with_suffix(args.output_suffix)
            np.savez(new_path, image=img)
            os.remove(str(path))

