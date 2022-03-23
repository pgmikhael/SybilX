from sybilx.loaders.abstract_loader import abstract_loader
from sybilx.utils.registry import register_object
import cv2
import torch
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import numpy as np
from sybilx.datasets.utils import get_scaled_annotation_mask, IMG_PAD_TOKEN
import copy
from sybilx.utils.registry import md5

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
class FullCTLoader(abstract_loader):
    """Loads all CT slices as a volume"""
    
    def configure_path(self, paths, sample):
        return md5(str(paths))

    def load_input(self, input_path, sample):
        out_dict = {}
        if self.args.fix_seed_for_multi_image_augmentations:
            sample["seed"] = np.random.randint(0, 2**32 - 1)
        
        input_dicts = []
        for e, path in enumerate(sample["paths"]):
            x = cv2.imread(path, 0)

            # TODO: this is a dumb way to make the mask be the same size as the img
            annotation_mask_args = copy.deepcopy(self.args)
            annotation_mask_args.img_size = x.shape

            mask = (
                get_scaled_annotation_mask(sample["annotations"][e], annotation_mask_args)
                if self.args.use_annotations
                else None
            )
            
            input_dicts.append({"input": x, "mask": mask})

        images = [i["input"] for i in input_dicts]
        masks = [i["mask"] for i in input_dicts]

        out_dict["input"] = self.reshape_images(images)
        out_dict["mask"] = self.reshape_images(masks) if self.args.use_annotations else None

        return out_dict

    def reshape_images(self, images):
        if isinstance(images[0], np.ndarray):
            images = [np.expand_dims(im, axis=0) for im in images]
            images = np.concatenate(images, axis=0)
        elif torch.is_tensor(images[0]):
            images = [im.unsqueeze(0) for im in images]
            images = torch.cat(images, dim=0)
        return images

    @property
    def cached_extension(self):
        return ""


@register_object("dicom_transform_loader", "input_loader")
class DicomTransformLoader(abstract_loader):
    """
    MIMIC method of DICOM image loading
    source: https://physionet.org/content/mimic-cxr-jpg/2.0.0/
    Chest radiographs were converted from DICOM to a compressed JPG format. 
    
    First, the image pixels were extracted from the DICOM file using the pydicom library. 
    Pixel values were normalized to the range [0, 255] by subtracting the lowest value in the image, dividing by the highest value in the shifted image, truncating values, and converting the result to an unsigned integer. 
    The DICOM field PhotometricInterpretation was used to determine whether the pixel values were inverted, 
    and if necessary images were inverted such that air in the image appears white (highest pixel value), while the outside of the patient's body appears black (lowest pixel value). 
    The OpenCV library was then used to histogram equalize the image with the intention of enhancing contrast. 
    Histogram equalization involves shifting pixel values towards 0 or towards 255 such that all pixel values 0 through 255 have approximately equal frequency. 
    Images were then converted to JPG files using OpenCV with a quality factor of 95.
    """
    def __init__(self, cache_path, augmentations, args):
        super(DicomTransformLoader, self).__init__(cache_path, augmentations, args)

    def configure_path(self, path, sample):
        return path

    def load_input(self, path, sample):
        try:
            dcm = pydicom.dcmread(path)
            pixel_array = apply_modality_lut(dcm.pixel_array, dcm)
            # below should do the same as 'apply_modality_lut'
            # if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
            #     pixel_array = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
            # else:
            #     pixel_array = dcm.pixel_array

            min_max_pixel_array = self.transform_image(pixel_array)
            if hasattr(dcm, 'PhotometricInterpretation') and not 'MONOCHROME2' in dcm.PhotometricInterpretation:
                min_max_pixel_array = 255 - min_max_pixel_array

        except Exception:
            raise Exception(LOADING_ERROR.format("COULD NOT LOAD DICOM."))

        return {"input": min_max_pixel_array, "mask": None}

    def transform_image(self, pixel_array):
        min_val = np.min(pixel_array)
        min_max_pixel_array = pixel_array - min_val
        max_val = np.max(min_max_pixel_array)
        min_max_pixel_array = np.trunc(( min_max_pixel_array / max_val ) * 255).astype(np.uint8)
        min_max_pixel_array = cv2.equalizeHist(min_max_pixel_array)
        return min_max_pixel_array

    @property
    def cached_extension(self):
        return ""


@register_object("cv_transform_loader", "input_loader")
class CVTransformLoader(DicomTransformLoader):
    """
    MIMIC method of image loading, using OpenCV (see DicomTransformLoader)
    """
    def __init__(self, cache_path, augmentations, args):
        super(DicomTransformLoader, self).__init__(cache_path, augmentations, args)


    def load_input(self, path, sample):
        img = cv2.imread(path, 0)
        min_max_pixel_array = self.transform_image(img)

        return {"input": min_max_pixel_array, "mask": None}

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
            x = np.float32(x)
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
