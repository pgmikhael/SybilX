import json
import os
import pickle
import numpy as np
import math
from tqdm import tqdm
import argparse
import pandas as pd
import time
from collections import defaultdict
from ast import literal_eval
from torch.utils.data import DataLoader
from pathlib import Path
import hashlib



parser = argparse.ArgumentParser()
parser.add_argument(
    "--source",
    help="Source JSON metadata file",
    type=Path,
    #default='/Mounts/rbg-storage1/datasets/NLST/full_nlst_google.json',
    default="/Mounts/rbg-storage1/datasets/NLST/nlst_metadata_022020.json",
)
parser.add_argument(
    "--annotations",
    help="Source annotations JSON file",
    type=Path,
    default="/Mounts/rbg-storage1/datasets/NLST/annotations_122020.json",
)
parser.add_argument(
    "--output_metadata",
    help="Output JSON metadata file",
    type=Path,
    required=True,
)
parser.add_argument(
    "--output_img_dir",
    help="Output image dir",
    type=Path,
    required=True,
)
parser.add_argument(
    "--batch_size",
    help="Batch size",
    type=int,
    default=32
)
parser.add_argument(
    "--num_workers",
    help="Number of workers",
    type=int,
    default=0
)


def project(volumes: np.array, masks: np.array):
    """ Projects a batch of volumes and annotation masks
    Params:
    - volumes: (B, N, H, W)  - image volumes
    - masks: (B, N, H, W)  - annotation masks (sum to one)
    """
    images = np.mean(volume, axis=2)
    masks = np.mean(mask, axis=2)
    return images, masks

def convert_input_path(path):
    """Takes an original single slice png path, and returns a different path (png, cached scaled png)"""
    # convert to png path
    return path.replace("nlst-ct-png", "nlst-ct").replace(".png", "")
    # TODO: convert to cached png path


### SUPPORTING FUNCTIONS FROM SYBIL ###

def is_localizer(series_dict):
    return (series_dict["imageclass"][0] == 0) or ("LOCALIZER" in series_dict["imagetype"][0]) or ("TOP" in series_dict["imagetype"][0])

def order_slices(img_paths, slice_locations):
    sorted_ids = np.argsort(slice_locations)
    sorted_img_paths = np.array(img_paths)[sorted_ids].tolist()
    sorted_slice_locs = np.sort(slice_locations).tolist()

    return sorted_img_paths, sorted_slice_locs

def get_scaled_annotation_mask(additional, img_size, scale_annotation=True):
    """
    Construct bounding box masks for annotations
    Args:
        - additional['image_annotations']: list of dicts { 'x', 'y', 'width', 'height' }, where bounding box coordinates are scaled [0,1].
        - args
    Returns:
        - mask of same size as input image, filled in where bounding box was drawn. If additional['image_annotations'] = None, return empty mask. Values correspond to how much of a pixel lies inside the bounding box, as a fraction of the bounding box's area
    """
    H, W = img_size
    mask = np.zeros((H, W))
    if additional["image_annotations"] is None:
        return mask

    for annotation in additional["image_annotations"]:
        single_mask = np.zeros((H, W))
        x_left, y_top = annotation["x"] * W, annotation["y"] * H
        x_right, y_bottom = (
            x_left + annotation["width"] * W,
            y_top + annotation["height"] * H,
        )

        # pixels completely inside bounding box
        x_quant_left, y_quant_top = math.ceil(x_left), math.ceil(y_top)
        x_quant_right, y_quant_bottom = math.floor(x_right), math.floor(y_bottom)

        # excess area along edges
        dx_left = x_quant_left - x_left
        dx_right = x_right - x_quant_right
        dy_top = y_quant_top - y_top
        dy_bottom = y_bottom - y_quant_bottom

        # fill in corners first in case they are over-written later by greater true intersection
        # corners
        single_mask[math.floor(y_top), math.floor(x_left)] = dx_left * dy_top
        single_mask[math.floor(y_top), x_quant_right] = dx_right * dy_top
        single_mask[y_quant_bottom, math.floor(x_left)] = dx_left * dy_bottom
        single_mask[y_quant_bottom, x_quant_right] = dx_right * dy_bottom

        # edges
        single_mask[y_quant_top:y_quant_bottom, math.floor(x_left)] = dx_left
        single_mask[y_quant_top:y_quant_bottom, x_quant_right] = dx_right
        single_mask[math.floor(y_top), x_quant_left:x_quant_right] = dy_top
        single_mask[y_quant_bottom, x_quant_left:x_quant_right] = dy_bottom

        # completely inside
        single_mask[y_quant_top:y_quant_bottom, x_quant_left:x_quant_right] = 1

        # in case there are multiple boxes, add masks and divide by total later
        mask += single_mask

    if scale_annotation:
        mask /= mask.sum()
    return mask

def md5(key):
    """
    returns a hashed with md5 string of the key
    """
    return hashlib.md5(key.encode()).hexdigest()

### / END OF SUPPORTING FUNCS ###

class Data(object):
    def __init__(metadata, annotations):
        #self.volume_paths = ...]

	self.annotations_metadata = annotations

	self.sids = [] # series ids
	self.sid2paths = {} # map from series id to (sorted) list of paths

	for pt in metadata:
	    for exam_dict in pt["accessions"]:
		for sid, series_dict in exam_dict["image_series"].items():
		    if is_localizer(series_dict["series_data"]):
			continue

		    # load paths
		    img_paths = series_dict["paths"]
		    slice_locations = series_dict["img_position"]

		    img_paths, slice_locs = order_slices(
			    img_paths, slice_locations
			)

                    # png -> dicom paths
                    img_paths = [convert_input_path(path) for path in img_paths]

		    #self.sid2paths = dict( () )
                    self.sids.append(sid)
		    self.sid2paths[sid] = img_paths

    def load_volume(self, paths):
	# Note: doesn't perform histogram eq.
	imgs = [ pydicom.dcmread(path).pixel_array for path in paths]
	volume = np.stack(imgs)
	return volume

    def load_annotations(self, series_id, image_paths, volume_size):
	if series_id not in self.annotations_metadata:
	    return np.zeros(volume_size)
	
	img_size = volume_size[1:]
	
	annotations = [
	    {
		"image_annotations": annotations_metadata[
		    series_id
		].get(os.path.splitext(os.path.basename(path))[0], None)
	    }
	    for path in image_paths 
	]
	img_masks = [get_scaled_annotation_mask(additional, img_size=img_size) for additional in annotations]
	volume_mask = np.stack(img_masks)
	
	return volume_mask


    def __getitem__(self, index):
        """Load a single volume and annotations."""
	sid = self.sids[index]
        paths = self.volume_paths[sid]
        volume = self.load_volume(paths)
	mask = self.load_annotations(sid, paths, volume_size=volume.shape)
        return index, volume, mask

    def __len__(self):
        return len(self.sids)


if __name__ == "__main__":
    args = parser.parse_args()

    print(f"Loading {args.source}")
    ct_dataset = json.load(open(args.source, "r"))
    print(f"Loading {args.annotations}")
    ct_annotations = json.load(open(args.annotations, "r"))

    data = Data(metadata=ct_dataset, annotations=ct_annotations)

    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

    for batch in loader:
        indices = batch[0]
        volumes = batch[1]
        annotations = batch[2]
        imgs, masks = project(volumes, annotations)

	for i, index in enumerate(indices):
	    old_paths = data.sid2paths[data.sids[index]]
	    new_filename = md5(str(old_paths)) + '.npz'
            new_path = os.path.join(args.output_img_dir, new_filename)

	    # save new file
            np.savez(new_path, image=imgs[i], masks[i])
            

    # TODO: create and dump new dataset with new paths
    proj_dataset = deepcopy(ct_dataset)
    #json.dump(proj_dataset, open(args.output, "w"))

