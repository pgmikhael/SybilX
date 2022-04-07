from sybilx.loaders.abstract_loader import abstract_loader
from sybilx.utils.registry import register_object
import cv2
import torch
import os.path
import pydicom
import skimage.io
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

@register_object("tif_loader", "input_loader")
class TIFFLoader(abstract_loader):
    def configure_path(self, path, sample):
        return path

    def load_input(self, path, sample):
        """
        loads as grayscale image
        """
        return {"input": skimage.io.imread(path, plugin='tifffile')}

    @property
    def cached_extension(self):
        return ""

@register_object("ct_dicom_loader", "input_loader")
class FullCTDicomLoader(abstract_loader):
    """Loads all CT slices as a volume"""
    
    def configure_path(self, paths, sample):
        return str(sorted(paths))

    def load_input(self, input_path, sample):
        out_dict = {}
        if self.args.fix_seed_for_multi_image_augmentations:
            sample["seed"] = np.random.randint(0, 2**32 - 1)
        
        input_dicts = []
        for e, path in enumerate(sample["paths"]):
            dcm = pydicom.dcmread(path)
            try:
                x = apply_modality_lut(dcm.pixel_array, dcm)
            except:
                raise Exception("Could not apply modality lut")

            # if hasattr(dcm, 'PhotometricInterpretation') and not 'MONOCHROME2' in dcm.PhotometricInterpretation:
            #     sample['invert_pixels'] = True
            # else:
            #     sample['invert_pixels'] = False
                
            # TODO: this is a way to make the mask be the same size as the img
            annotation_mask_args = copy.deepcopy(self.args)
            annotation_mask_args.img_size = x.shape

            mask = (
                get_scaled_annotation_mask(sample["annotations"][e], annotation_mask_args, scale_annotation=annotation_mask_args.scale_annotations)
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


@register_object("ct_png_loader", "input_loader")
class FullCTPNGLoader(abstract_loader):
    """Loads all CT slices as a volume"""
    
    def configure_path(self, paths, sample):
        return md5(str(sorted(paths)))

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
                get_scaled_annotation_mask(sample["annotations"][e], annotation_mask_args, scale_annotation=annotation_mask_args.scale_annotations)
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


@register_object("cached_ct_loader", "input_loader")
class FullCTLoader_from_cached(FullCTPNGLoader):
    """
    NOTE: THIS WAS NEVER DEBUGGED
    This is a loader to allow us to load CT datasets that were previously cached for the CT (Sybil) project
    """
    def __init__(self, cache_path, augmentations, args):
        super(FullCTLoader_from_cached, self).__init__(cache_path, augmentations, args)

    def configure_path(self, paths, sample):
        # we still want to cache the projections, just removed double hashing
        return str(paths)
    
    def get_hashed_paths(self, image_paths, attr_key="#256#256"):
        DEFAULT_CACHE_DIR = "default/"
        # TODO: debug and double check that attr_key is correct, it might be the entire augmentation name like @scale2d#256#256
        hashed_image_paths = []
        for image_path in image_paths:
            hashed_key = md5(image_path)
            par_dir = os.path.basename(os.path.dirname(image_path))
            # this is the line to double check
            hashed_path = os.path.join(self.cache_path, DEFAULT_CACHE_DIR, attr_key, par_dir, hashed_key + '.npz')    
            hashed_image_paths.append(hashed_path)
        return hashed_image_paths

    def load_input(self, input_path, sample):
        sample['paths'] = self.get_hashed_paths(sample['paths'])
        out_dict = {}
        if self.args.fix_seed_for_multi_image_augmentations:
            sample["seed"] = np.random.randint(0, 2**32 - 1)
        
        input_dicts = []
        for e, path in enumerate(sample["paths"]):
            cached_arrays = np.load(path)
            cached_arrays = self.cache.get(input_path, base_key)
            x = cached_arrays["image"]
            if "mask" in cached_arrays:
                mask = cached_arrays["mask"]
            elif self.args.use_annotations:
                # if masks are correctly cached this should not be necessary
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


@register_object("dicom_transform_loader", "input_loader")
class DicomTransformLoader(abstract_loader):
    """
    Expects: single X-Ray

    MIMIC method of DICOM image loading
    source: https://physionet.org/content/mimic-cxr-jpg/2.0.0/
    Chest radiographs were converted from DICOM to a compressed JPG format. 
    """
    def __init__(self, cache_path, augmentations, args):
        super(DicomTransformLoader, self).__init__(cache_path, augmentations, args)

    def configure_path(self, path, sample):
        return str(path)

    def load_input(self, path, sample):
        if path == self.pad_token:
            mask = (
            get_scaled_annotation_mask(sample["annotations"], self.args)
            if self.args.use_annotations
            else None
        )
            shape = (self.args.num_chan, self.args.img_size[0], self.args.img_size[1])
            min_max_pixel_array = torch.zeros(*shape)
            mask = (
                torch.from_numpy(mask * 0).unsqueeze(0)
                if self.args.use_annotations
                else None
            )
            return {"input": min_max_pixel_array, "mask": mask}
        else:
            try:
                dcm = pydicom.dcmread(path)
                pixel_array = apply_modality_lut(dcm.pixel_array, dcm)
                # below should do the same as 'apply_modality_lut'
                # if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
                #     pixel_array = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
                # else:
                #     pixel_array = dcm.pixel_array

                min_max_pixel_array = self.transform_image(pixel_array)
                # if hasattr(dcm, 'PhotometricInterpretation') and not 'MONOCHROME2' in dcm.PhotometricInterpretation:
                #     min_max_pixel_array = 255 - min_max_pixel_array

                if self.args.use_annotations:
                    mask = get_scaled_annotation_mask(sample["annotations"], self.args, scale_annotation=self.args.scale_annotations)
                    return {"input": min_max_pixel_array, "mask": mask}
                else:
                    return {"input": min_max_pixel_array}

            except Exception:
                raise Exception(LOADING_ERROR.format("COULD NOT LOAD DICOM."))

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
        if self.args.use_annotations:
            mask = get_scaled_annotation_mask(sample["annotations"], self.args)
            return {"input": min_max_pixel_array, "mask": mask}
        else:
            return {"input": min_max_pixel_array}

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
            x = np.float32(x)

            return {"input": x, "mask": mask}

    @property
    def cached_extension(self):
        return ".png"


@register_object("dicom_loader", "input_loader")
class DicomLoader(abstract_loader):
    def __init__(self, cache_path, augmentations, args):
        super(DicomLoader, self).__init__(cache_path, augmentations, args)

    def configure_path(self, path, sample):
        return path

    def load_input(self, path, sample):
        try:
            dcm = pydicom.dcmread(path)
            arr = apply_modality_lut(dcm.pixel_array, dcm)

            if self.args.use_annotations:
                mask = get_scaled_annotation_mask(sample["annotations"], self.args)
                return {"input": arr, "mask": mask}
            else:
                return {"input": arr}
                
        except Exception:
            raise Exception(LOADING_ERROR.format("COULD NOT LOAD DICOM."))

    @property
    def cached_extension(self):
        return ""
