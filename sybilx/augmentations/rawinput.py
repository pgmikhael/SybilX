import numpy as np
import albumentations as A
import cv2
from sybilx.utils.registry import register_object
from sybilx.augmentations.abstract import Abstract_augmentation


@register_object("scale_2d", "augmentation")
class Scale_2d(Abstract_augmentation):
    """
    Given PIL image, enforce its some set size
    (can use for down sampling / keep full res)
    """

    def __init__(self, args, kwargs):
        super(Scale_2d, self).__init__()
        assert len(kwargs.keys()) == 0
        width, height = args.img_size
        self.set_cachable(width, height)
        self.transform = A.Resize(height, width)

    def __call__(self, input_dict, sample=None):
        out = self.transform(
            image=input_dict["input"], mask=input_dict.get("mask", None)
        )
        input_dict["input"] = out["image"]
        input_dict["mask"] = out["mask"]
        return input_dict


@register_object("rand_hor_flip", "augmentation")
class Random_Horizontal_Flip(Abstract_augmentation):
    """
    Randomly flips image horizontally
    """

    def __init__(self, args, kwargs):
        super(Random_Horizontal_Flip, self).__init__()
        self.args = args
        assert len(kwargs.keys()) == 0
        self.transform = A.HorizontalFlip()

    def __call__(self, input_dict, sample=None):
        if "seed" in sample:
            self.set_seed(sample["seed"])
        input_dict["input"] = self.transform(image=input_dict["input"])["image"]
        return input_dict


@register_object("rand_ver_flip", "augmentation")
class Random_Vertical_Flip(Abstract_augmentation):
    """
    Randomly flips image vertically
    """

    def __init__(self, args, kwargs):
        super(Random_Vertical_Flip, self).__init__()
        self.args = args
        assert len(kwargs.keys()) == 0
        self.transform = A.VerticalFlip()

    def __call__(self, input_dict, sample=None):
        if "seed" in sample:
            self.set_seed(sample["seed"])
        input_dict["input"] = self.transform(image=input_dict["input"])["image"]
        return input_dict


@register_object("random_resize_crop", "augmentation")
class RandomResizedCrop(Abstract_augmentation):
    """
    Randomly Resize and Crop: https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.RandomResizedCrop
    kwargs:
        h: output height
        w: output width
        min_scale: min size of the origin size cropped
        max_scale: max size of the origin size cropped
        min_ratio: min aspect ratio of the origin aspect ratio cropped
        max_ratio: max aspect ratio of the origin aspect ratio cropped
    """

    def __init__(self, args, kwargs):
        super(RandomResizedCrop, self).__init__()
        self.args = args
        kwargs_len = len(kwargs.keys())
        assert (kwargs_len >= 2) and (kwargs_len <= 6)
        h, w = (int(kwargs["h"]), int(kwargs["w"]))
        min_scale = float(kwargs["min_scale"]) if "min_scale" in kwargs else 0.08
        max_scale = float(kwargs["max_scale"]) if "max_scale" in kwargs else 1.0
        min_ratio = float(kwargs["min_ratio"]) if "min_ratio" in kwargs else 0.75
        max_ratio = float(kwargs["max_ratio"]) if "max_ratio" in kwargs else 1.33
        self.transform = A.RandomResizedCrop(
            height=h,
            width=w,
            scale=(min_scale, max_scale),
            ratio=(min_ratio, max_ratio),
        )

    def __call__(self, input_dict, sample=None):
        if "seed" in sample:
            self.set_seed(sample["seed"])
        input_dict["input"] = self.transform(image=input_dict["input"])["image"]
        return input_dict


@register_object("random_crop", "augmentation")
class Random_Crop(Abstract_augmentation):
    """
    Randomly Crop: https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.RandomCrop
    kwargs:
        h: output height
        w: output width
    """

    def __init__(self, args, kwargs):
        super(Random_Crop, self).__init__()
        self.args = args
        kwargs_len = len(kwargs.keys())
        assert kwargs_len in [2, 3]
        size = (int(kwargs["h"]), int(kwargs["w"]))
        self.transform = A.RandomCrop(*size)

    def __call__(self, input_dict, sample=None):
        if "seed" in sample:
            self.set_seed(sample["seed"])
        input_dict["input"] = self.transform(image=input_dict["input"])["image"]
        return input_dict


@register_object("blur", "augmentation")
class Blur(Abstract_augmentation):
    """
    Randomly blurs image with kernel size limit: https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Blur

    kwargs:
        limit: maximum kernel size for blurring the input image. Should be in range [3, inf)
    """

    def __init__(self, args, kwargs):
        super(Blur, self).__init__()
        limit = float(kwargs["limit"]) if "limit" in kwargs else 3
        self.transform = A.Blur(blur_limit=limit)

    def __call__(self, input_dict, sample=None):
        if "seed" in sample:
            self.set_seed(sample["seed"])
        input_dict["input"] = self.transform(image=input_dict["input"])["image"]
        return input_dict


@register_object("center_crop", "augmentation")
class Center_Crop(Abstract_augmentation):
    """
    Center crop: https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.CenterCrop

    kwargs:
        h: height
        w: width
    """

    def __init__(self, args, kwargs):
        super(Center_Crop, self).__init__()
        self.args = args
        kwargs_len = len(kwargs.keys())
        assert kwargs_len in [2, 3]
        size = (int(kwargs["h"]), int(kwargs["w"]))
        self.set_cachable(*size)
        self.transform = A.CenterCrop(*size)

    def __call__(self, input_dict, sample=None):
        input_dict["input"] = self.transform(image=input_dict["input"])["image"]
        return input_dict


@register_object("color_jitter", "augmentation")
class Color_Jitter(Abstract_augmentation):
    """
    Center crop: https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ColorJitter

    kwargs:
        brightness: default 0.2
        contrast: default 0.2
        saturation: default 0.2
        hue: default 0.2
    """

    def __init__(self, args, kwargs):
        super(Color_Jitter, self).__init__()
        self.args = args
        kwargs_len = len(kwargs.keys())
        assert kwargs_len <= 4
        b, c, s, h = (
            float(kwargs["brightness"]) if "brightness" in kwargs else 0.2,
            float(kwargs["contrast"]) if "contrast" in kwargs else 0.2,
            float(kwargs["saturation"]) if "saturation" in kwargs else 0.2,
            float(kwargs["hue"]) if "hue" in kwargs else 0.2,
        )
        self.transform = A.HueSaturationValue(
            brightness=b, contrast=c, saturation=s, hue=h
        )

    def __call__(self, input_dict, sample=None):
        if "seed" in sample:
            self.set_seed(sample["seed"])
        input_dict["input"] = self.transform(image=input_dict["input"])["image"]
        return input_dict


@register_object("random_hue_satur_val", "augmentation")
class Hue_Saturation_Value(Abstract_augmentation):
    """
        HueSaturationValue wrapper

    kwargs:
        val (val_shift_limit): default 0
        saturation (sat_shift_limit): default 0
        hue (hue_shift_limit): default 0
    """

    def __init__(self, args, kwargs):
        super(Hue_Saturation_Value, self).__init__()
        self.args = args
        kwargs_len = len(kwargs.keys())
        assert kwargs_len <= 3
        val, satur, hue = (
            int(kwargs["val"]) if "val" in kwargs else 0,
            int(kwargs["saturation"]) if "saturation" in kwargs else 0,
            int(kwargs["hue"]) if "hue" in kwargs else 0,
        )
        self.transform = A.HueSaturationValue(
            hue_shift_limit=hue, sat_shift_limit=satur, val_shift_limit=val, p=1
        )

    def __call__(self, input_dict, sample=None):
        if "seed" in sample:
            self.set_seed(sample["seed"])
        input_dict["input"] = self.transform(image=input_dict["input"])["image"]
        return input_dict


@register_object("random_brightness_contrast", "augmentation")
class Random_Brightness_Contrast(Abstract_augmentation):
    """
        RandomBrightnessContrast wrapper https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomBrightnessContrast

    kwargs:
        contrast (contrast_limit): default 0
        brightness (sat_shiftbrightness_limit_limit): default 0
    """

    def __init__(self, args, kwargs):
        super(Random_Brightness_Contrast, self).__init__()
        self.args = args
        kwargs_len = len(kwargs.keys())
        assert kwargs_len <= 2
        contrast = float(kwargs["contrast"]) if "contrast" in kwargs else 0
        brightness = float(kwargs["brightness"]) if "brightness" in kwargs else 0

        self.transform = A.RandomBrightnessContrast(
            brightness_limit=brightness, contrast_limit=contrast, p=0.5
        )

    def __call__(self, input_dict, sample=None):
        if "seed" in sample:
            self.set_seed(sample["seed"])
        input_dict["input"] = self.transform(image=input_dict["input"])["image"]
        return input_dict


@register_object("gaussian_blur", "augmentation")
class Gaussian_Blur(Abstract_augmentation):
    """
    wrapper for https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.GaussianBlur
    blur must odd and in range [3, inf). Default: (3, 7).

    kwargs:
        min_blur: default 3
        max_blur
    """

    def __init__(self, args, kwargs):
        super(Gaussian_Blur, self).__init__()
        self.args = args
        kwargs_len = len(kwargs.keys())
        assert kwargs_len >= 1
        min_blur = int(kwargs["min_blur"]) if "min_blur" in kwargs else 3
        max_blur = int(kwargs["max_blur"])
        assert (max_blur % 2 == 1) and (min_blur % 2 == 1)
        self.transform = A.GaussianBlur(blur_limit=(min_blur, max_blur), p=1)

    def __call__(self, input_dict, sample=None):
        if "seed" in sample:
            self.set_seed(sample["seed"])
        input_dict["input"] = self.transform(image=input_dict["input"])["image"]
        return input_dict


@register_object("rotate_range", "augmentation")
class Rotate_Range(Abstract_augmentation):
    """
    Rotate image counter clockwise by random degree https://albumentations.ai/docs/api_reference/augmentations/geometric/rotate/#albumentations.augmentations.geometric.rotate.Rotate

        kwargs
            deg: max degrees to rotate
    """

    def __init__(self, args, kwargs):
        super(Rotate_Range, self).__init__()
        assert len(kwargs.keys()) == 1
        self.max_angle = int(kwargs["deg"])
        self.transform = A.Rotate(limit=self.max_angle, p=0.5)

    def __call__(self, input_dict, sample=None):
        if "seed" in sample:
            self.set_seed(sample["seed"])
        out = self.transform(
            image=input_dict["input"], mask=input_dict.get("mask", None)
        )
        input_dict["input"] = out["image"]
        input_dict["mask"] = out["mask"]
        return input_dict


@register_object("grayscale", "augmentation")
class Grayscale(Abstract_augmentation):
    """
    Convert image to grayscale https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ToGray
    """

    def __init__(self, args, kwargs):
        super(Grayscale, self).__init__()
        assert len(kwargs.keys()) == 0
        self.set_cachable(args.num_chan)

        self.transform = A.ToGray()

    def __call__(self, input_dict, sample=None):
        input_dict["input"] = self.transform(image=input_dict["input"])["image"]
        return input_dict


@register_object("random_affine_transform", "augmentation")
class Random_Affine_Transform(Abstract_augmentation):
    """
        Affine wrapper https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/

        Can modify this to do many things, but is set here just to allow for scale (for zoom)
        - Scaling ("zoom" in/out)

        Could also do:
        - Translation ("move" image on the x-/y-axis)
        - Rotation
        - Shear (move one side of the image, turning a square into a trapezoid)

    kwargs:
        scale=None, 
        translate_percent=None, 
        translate_px=None, 
        rotate=None, 
        shear=None, 
        interpolation=1, 
        mask_interpolation=0, 
        cval=0, 
        cval_mask=0, 
        mode=0, 
        fit_output=False, 
        always_apply=False, 
        p=0.5
    """

    def __init__(self, args, kwargs):
        super(Random_Affine_Transform, self).__init__()
        self.args = args
        kwargs_len = len(kwargs.keys())
        assert kwargs_len == 1
        scale = (-float(kwargs["scale"]),float(kwargs["scale"]))  if "scale" in kwargs else 0


        self.transform = A.Affine(
            scale=scale
        )

    def __call__(self, input_dict, sample=None):
        if "seed" in sample:
            self.set_seed(sample["seed"])
        input_dict["input"] = self.transform(image=input_dict["input"])["image"]
        return input_dict


@register_object("min_max_8bit_scaler", "augmentation")
class MinMax8BitScaler(Abstract_augmentation):
    """
    Scales images to 8 bits (0-255)
    """

    def __init__(self, args, kwargs):
        super(MinMax8BitScaler, self).__init__()
        assert len(kwargs.keys()) == 0
        self.set_cachable()

    def transform_image(self, pixel_array):
        min_val = np.min(pixel_array)
        min_max_pixel_array = pixel_array - min_val
        max_val = np.max(min_max_pixel_array)
        min_max_pixel_array = np.trunc(( min_max_pixel_array / max_val ) * 255).astype(np.uint8)
        return min_max_pixel_array

    def __call__(self, input_dict, sample=None):
        input_dict["input"] = self.transform_image(input_dict["input"])
        return input_dict

@register_object("min_max_scaler", "augmentation")
class MinMaxScaler(Abstract_augmentation):
    """
    Scales images from 0 to kwargs['max'] 
    """

    def __init__(self, args, kwargs):
        super(MinMaxScaler, self).__init__()
        assert len(kwargs.keys()) == 1
        self.max_scale = kwargs["max"] if "max" in kwargs else 1
        self.set_cachable()

    def transform_image(self, pixel_array):
        min_val = np.min(pixel_array)
        min_max_pixel_array = pixel_array - min_val
        max_val = np.max(min_max_pixel_array)
        min_max_pixel_array = np.trunc(( min_max_pixel_array / max_val ) * self.max_scale).astype(np.uint8)
        return min_max_pixel_array

    def __call__(self, input_dict, sample=None):
        input_dict["input"] = self.transform_image(input_dict["input"])
        return input_dict


@register_object("histogram_equalize", "augmentation")
class HistogramEqualize(Abstract_augmentation):
    """
    Expects:
        - pixel_array to already have 'apply_modality_lut' applied to it
        - values scaled to 0-255 (min max scaler)
    """
    def __init__(self, args, kwargs):
        super(HistogramEqualize, self).__init__()
        assert len(kwargs.keys()) == 0
        self.transform = A.Equalize(always_apply=True)
        self.set_cachable()

    def __call__(self, input_dict, sample=None):
        input_dict["input"] = self.transform(image=input_dict["input"])["image"]
        return input_dict


@register_object("elastic_deformation", "augmentation")
class ElasticDeformation(Abstract_augmentation):
    """
    Elastic deformation of images as described in [Simard2003]_ (with modifications). 
    Based on https://gist.github.com/ernestum/601cdf56d2b424757de5
    [Simard2003] Simard, Steinkraus and Platt 
    "Best Practices for Convolutional Neural Networks applied to Visual Document Analysis"
    """
    def __init__(self, args, kwargs):
        super(ElasticDeformation, self).__init__()
        self.args = args
        kwargs_len = len(kwargs.keys())
        assert kwargs_len == 2
        alpha, sigma = float(kwargs["alpha"]) if "alpha" in kwargs else 1, float(kwargs["sigma"]) if "sigma" in kwargs else 50
        self.transform = A.ElasticTransform(
                    alpha=alpha, 
                    sigma=sigma, 
                    alpha_affine=50, 
                    interpolation=1, 
                    border_mode=4, 
                    value=None, 
                    approximate=False, 
                    same_dxdy=False, 
                    p=0.5)

    def __call__(self, input_dict, sample=None):
        if "seed" in sample:
            self.set_seed(sample["seed"])
        input_dict["input"] = self.transform(image=input_dict["input"])["image"]
        return input_dict

@register_object("invert_pixels", "augmentation")
class InvertPixels(Abstract_augmentation):
    """
    Subtracts all pixels from fixed max pixel value (default: 255)
    """

    def __init__(self, args, kwargs):
        super(InvertPixels, self).__init__()
        assert len(kwargs.keys()) == 2
        self.max_pixel = float(kwargs["max_pixel"]) if "max_pixel" in kwargs else 255
        self.all = bool(kwargs["all_images"]) if "all_images" in kwargs else True
        self.set_cachable()
        # eg invert_pixels/max_pixel=255/all_images=0

    def __call__(self, input_dict, sample=None):
        if self.all:
            input_dict["input"] = self.max_pixel - input_dict["input"]
        if sample.get('invert_pixels', False):
            input_dict["input"] = self.max_pixel - input_dict["input"]
        return input_dict

@register_object("invert_pixels_relative", "augmentation")
class InvertPixelsRelative(Abstract_augmentation):
    """
    Subtracts all pixels from max pixel value in image
    """

    def __init__(self, args, kwargs):
        super(InvertPixelsRelative, self).__init__()
        assert len(kwargs.keys()) == 1
        self.all = bool(kwargs["all_images"]) if "all_images" in kwargs else True
        self.set_cachable()
        # eg invert_pixels_relative/all_images=0

    def __call__(self, input_dict, sample=None):
        if self.all:
            input_dict["input"] = input_dict["input"].max() - input_dict["input"]
        if sample.get('invert_pixels', False):
            input_dict["input"] = input_dict["input"].max() - input_dict["input"]
        return input_dict
