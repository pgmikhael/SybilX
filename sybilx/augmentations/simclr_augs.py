import numpy as np
import cv2
from sybilx.utils.registry import register_object
from sybilx.augmentations.abstract import Abstract_augmentation
from torchvision.transforms import transforms
from sybilx.augmentations.rawinput import Gaussian_Blur
from torchvision import transforms, datasets

@register_object("simclr", "augmentation")
class SimCLRAugs(Abstract_augmentation):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    def __init__(self, args, kwargs):
        super(SimCLRAugs, self).__init__()
        self.args = args
        kwargs_len = len(kwargs.keys())
        assert kwargs_len >= 1
        crop_size = int(kwargs["size"]) if "size" in kwargs else 1
        s = 1
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.transform = transforms.Compose([
                                            transforms.RandomResizedCrop(size=crop_size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            transforms.GaussianBlur(kernel_size=int(0.1 * crop_size), sigma=(0.1, 2.0))
                                            ])

    def __call__(self, input_dict, sample=None):
        if "seed" in sample:
            self.set_seed(sample["seed"])
        input_dict["input"] = self.transform(image=input_dict["input"])["image"]
        return input_dict