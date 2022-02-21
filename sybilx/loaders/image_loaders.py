from sybilx.loaders.abstract_loader import abstract_loader
from sybilx.utils.registry import register_object
import cv2
import torch
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import numpy as np
from sybilx.datasets.utils import get_scaled_annotation_mask, IMG_PAD_TOKEN

LOADING_ERROR = "LOADING ERROR! {}"


@register_object("cv_loader", "input_loader")
class OpenCVLoader(abstract_loader):
    def configure_path(self, path, sample):
        return path

    def load_input(self, path, sample):
        """
        loads as grayscale image
        """
        return {"input": cv2.imread(path, 0)}

    @property
    def cached_extension(self):
        return ".png"


@register_object("ct_loader", "input_loader")
class CTLoader(abstract_loader):
    def configure_path(self, path, sample):
        return path

    def load_input(self, path, sample):
        """
        loads as grayscale image
        """
        mask = (
            get_scaled_annotation_mask(sample["annotations"], self.args)
            if self.args.use_annotations
            else None
        )
        if path == self.pad_token:
            shape = (self.args.num_chan, self.args.img_size[0], self.args.img_size[1])
            x = torch.zeros(*shape)
            mask = (
                torch.from_numpy(mask * 0).unsqueeze(0)
                if self.args.use_annotations
                else None
            )
        else:
            x = cv2.imread(path, 0)

        return {"input": x, "mask": mask}

    @property
    def cached_extension(self):
        return ".png"

@register_object("ct_16bit_loader", "input_loader")
class CT16Loader(abstract_loader):
    def configure_path(self, path, sample):
        return path

    def load_input(self, path, sample):
        """
        loads as grayscale image
        """
        mask = (
            get_scaled_annotation_mask(sample["annotations"], self.args)
            if self.args.use_annotations
            else None
        )
        if path == self.pad_token:
            shape = (self.args.num_chan, self.args.img_size[0], self.args.img_size[1])
            x = torch.zeros(*shape)
            mask = (
                torch.from_numpy(mask * 0).unsqueeze(0)
                if self.args.use_annotations
                else None
            )
        else:
            x = cv2.imread(path, -1)
            x = np.float16(x)
        return {"input": x, "mask": mask}

    @property
    def cached_extension(self):
        return ".png"

@register_object("dicom_loader", "input_loader")
class DicomLoader(abstract_loader):
    def __init__(self, cache_path, augmentations, args):
        super(DicomLoader, self).__init__(cache_path, augmentations, args)
        self.window_center = -600
        self.window_width = 1500

    def configure_path(self, path, sample):
        return path

    def load_input(self, path, sample):
        try:
            dcm = pydicom.dcmread(path)
            dcm = apply_modality_lut(dcm.pixel_array, dcm)
            arr = apply_windowing(dcm, self.window_center, self.window_width)
        except Exception:
            raise Exception(LOADING_ERROR.format("COULD NOT LOAD DICOM."))
        return {"input": arr}

    @property
    def cached_extension(self):
        return ""


def apply_windowing(image, center, width, bit_size=16):
    """Windowing function to transform image pixels for presentation.
    Must be run after a DICOM modality LUT is applied to the image.
    Windowing algorithm defined in DICOM standard:
    http://dicom.nema.org/medical/dicom/2020b/output/chtml/part03/sect_C.11.2.html#sect_C.11.2.1.2
    Reference implementation:
    https://github.com/pydicom/pydicom/blob/da556e33b/pydicom/pixel_data_handlers/util.py#L460
    Args:
        image (ndarray): Numpy image array
        center (float): Window center (or level)
        width (float): Window width
        bit_size (int): Max bit size of pixel
    Returns:
        ndarray: Numpy array of transformed images
    """
    y_min = 0
    y_max = 2**bit_size - 1
    y_range = y_max - y_min

    c = center - 0.5
    w = width - 1

    below = image <= (c - w / 2)  # pixels to be set as black
    above = image > (c + w / 2)  # pixels to be set as white
    between = np.logical_and(~below, ~above)

    image[below] = y_min
    image[above] = y_max
    if between.any():
        image[between] = ((image[between] - c) / w + 0.5) * y_range + y_min

    return image
