import numpy as np
from sybilx.augmentations.abstract import Abstract_augmentation
from sybilx.utils.registry import register_object

@register_object("project_ct", "augmentation")
class ProjectCT(Abstract_augmentation):
    """
    Projects 3D CT volume to approximate X-Ray
    """

    def __init__(self, args, kwargs):
        super(ProjectCT, self).__init__()
        assert len(kwargs) == 0
        self.set_cachable()

    def __call__(self, input_dict, sample=None):
        volume = input_dict["input"]
        mask = input_dict["mask"]

        img = np.flipud(np.mean(volume, axis=1)).copy()
        mask = np.flipud(np.mean(mask, axis=1)).copy()

        input_dict["input"] = img
        input_dict["mask"] = mask

        return input_dict

